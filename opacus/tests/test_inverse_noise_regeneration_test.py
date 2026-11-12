"""Test-first contracts: replay must match an independently sampled FIR stream.

No production sampler or recurrence is used by the oracle. New optional APIs
are checked at call time so absent features produce assertions, not collection
errors. CPU tests are required even when accelerator cases are unavailable.
"""

import copy
import inspect
from collections import deque
from types import SimpleNamespace

import pytest
import torch
from opacus import NoiseMechanismConfig, PrivacyEngine
from opacus.noise_mechanisms import InverseBandNoiseMechanism
from opacus.optimizers import DPOptimizer


def coefficients(p):
    return [1.0] + [-0.3 / i for i in range(1, p)]


def mechanism(p=4, mode=None):
    kwargs = dict(inverse_coeffs=coefficients(p), z_std=0.2)
    if mode is not None:
        assert (
            "noise_history_mode"
            in inspect.signature(InverseBandNoiseMechanism).parameters
        ), "inverse runtime must expose regenerate/stored history selection"
        kwargs["noise_history_mode"] = mode
    return InverseBandNoiseMechanism(**kwargs)


def context(device="cpu", generator=True, shapes=((3, 2), (3,))):
    params = [torch.nn.Parameter(torch.zeros(s, device=device)) for s in shapes]
    return SimpleNamespace(
        params=params,
        loss_reduction="sum",
        expected_batch_size=4,
        accumulated_iterations=1,
        secure_mode=False,
        generator=(
            torch.Generator(device=params[0].device).manual_seed(19)
            if generator
            else None
        ),
    )


def release(noiser, ctx):
    for p in ctx.params:
        p.summed_grad = torch.zeros_like(p)
    noiser.add_noise(ctx)
    return torch.cat([p.grad.detach().flatten() for p in ctx.params])


def reference_draw(ctx, std):
    # Preserve per-parameter shapes/dtypes and zero-scale RNG consumption.
    return torch.cat(
        [
            (
                torch.zeros_like(p)
                if std == 0
                else torch.normal(
                    mean=0.0,
                    std=std,
                    size=p.shape,
                    dtype=p.dtype,
                    device=p.device,
                    generator=ctx.generator,
                )
            ).flatten()
            for p in ctx.params
        ]
    )


def reference_filter(history, coeffs):
    result = coeffs[0] * history[-1]
    for lag in range(1, min(len(history), len(coeffs))):
        result = result + coeffs[lag] * history[-1 - lag]
    return result


def inventory(noiser):
    read = getattr(noiser, "noise_state_inventory", None)
    assert callable(read), "noise mechanisms must expose a non-cloning byte inventory"
    # Contract schema: history/diagnostics/scratch maps from device to bytes.
    value = read()
    assert set(value) == {"history", "diagnostics", "scratch"}
    assert all(
        isinstance(n, int) and n >= 0 for v in value.values() for n in v.values()
    )
    return value


def retained_tensor_bytes(owner):
    """Walk reachable tensor backing storage without trusting production inventory.

    Opaque generator objects are intentionally excluded; saved state tensors are
    visible. Follow descriptor objects as well as containers and deduplicate
    aliases/cycles. Parameters are borrowed training state, not noise storage.
    """
    seen, storages = set(), {}

    def visit(value):
        if id(value) in seen:
            return
        seen.add(id(value))
        if isinstance(value, torch.nn.Parameter):
            return
        if isinstance(value, torch.Tensor):
            storage = value.untyped_storage()
            storages[(str(value.device), storage.data_ptr())] = storage.nbytes()
        elif isinstance(value, dict):
            for item in value.values():
                visit(item)
        elif isinstance(value, (list, tuple, deque)):
            for item in value:
                visit(item)
        elif hasattr(value, "__dict__") and not isinstance(value, torch.Generator):
            visit(vars(value))

    visit(owner)
    return sum(storages.values())


def assert_compact_actual_storage(noiser, ctx, p):
    # Permit three persistent O(d) work vectors and a small coefficient tensor,
    # but never p full vectors hidden behind a plausible reported inventory.
    vector_bytes = sum(param.numel() * param.element_size() for param in ctx.params)
    state = ctx.generator.get_state()
    snapshot_bytes = state.numel() * state.element_size()
    bound = (p - 1) * snapshot_bytes + 3 * vector_bytes + p * 8
    actual = retained_tensor_bytes(noiser)
    assert (
        actual <= bound
    ), f"actual retained tensor storage {actual} exceeds replay bound {bound}"


def assert_tree_equal(actual, expected):
    if isinstance(expected, torch.Tensor):
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    elif isinstance(expected, dict):
        assert actual.keys() == expected.keys()
        for key in expected:
            assert_tree_equal(actual[key], expected[key])
    elif isinstance(expected, (tuple, list)):
        assert len(actual) == len(expected)
        for a, b in zip(actual, expected):
            assert_tree_equal(a, b)
    else:
        assert actual == expected


def test_reference_filter_matches_hand_computed_convolution():
    h = [torch.tensor([2.0]), torch.tensor([3.0]), torch.tensor([5.0])]
    assert reference_filter(h, [1.0, -0.5, 0.25]).item() == 4.0


@pytest.mark.parametrize("p", [1, 2, 64])
@pytest.mark.parametrize("default_rng", [False, True])
def test_noise_matches_independent_filter_and_rng_continuation(p, default_rng):
    torch.manual_seed(92)
    actual = context(generator=not default_rng)
    expected = context(generator=not default_rng)
    noiser = mechanism(p)
    history = []
    default_state = torch.get_rng_state()
    for step in range(p + 3):
        std = [0.2, 0.0, 0.5][step % 3]
        noiser.z_std = std
        if default_rng:
            torch.set_rng_state(default_state)
        # Matching interleaved external draws on both streams are intentional.
        torch.rand(3, generator=actual.generator)
        result = release(noiser, actual)
        continuation = torch.rand(2, generator=actual.generator)
        if default_rng:
            torch.set_rng_state(default_state)
        torch.rand(3, generator=expected.generator)
        history.append(reference_draw(expected, std))
        torch.testing.assert_close(
            result, reference_filter(history, coefficients(p)), rtol=1e-6, atol=1e-7
        )
        torch.testing.assert_close(
            continuation, torch.rand(2, generator=expected.generator), rtol=0, atol=0
        )
        default_state = torch.get_rng_state()
        assert noiser.state_depth == min(step + 1, p - 1)


@pytest.mark.parametrize("p", [1, 2, 64])
def test_default_retains_no_full_noise_vectors_or_debug_copies(p):
    ctx = context(shapes=((8192,),))
    noiser = mechanism(p)
    for _ in range(p + 2):
        release(noiser, ctx)
    assert_compact_actual_storage(noiser, ctx, p)
    assert (
        noiser.last_flat_z is None
    ), "ordinary replay must not retain diagnostic noise"
    assert (
        noiser.last_flat_u is None
    ), "ordinary replay must not retain diagnostic release"
    payload = inventory(noiser)
    # CPU state is typically 5056 bytes; measure instead of assuming 16 bytes.
    assert (
        sum(payload["history"].values()) <= (p - 1) * ctx.generator.get_state().numel()
    )
    assert sum(payload["diagnostics"].values()) == 0


@pytest.mark.parametrize(
    "family,key",
    [
        ("bifr", "bifr_inv_coeffs"),
        ("bisr", "bisr_inv_coeffs"),
        ("bandinvmf", "bandinvmf_inv_coeffs"),
    ],
)
@pytest.mark.parametrize("mode", ["regenerate", "stored"])
def test_family_builder_preserves_requested_history_mode(family, key, mode):
    config = NoiseMechanismConfig(
        mechanism=family,
        accounting_mode="bsr_accountant",
        mechanism_state={
            key: coefficients(4),
            "z_std": 0.2,
            "noise_history_mode": mode,
        },
    )
    noiser = PrivacyEngine._build_noise_mechanism_from_config(config)
    assert (
        getattr(noiser, "noise_history_mode", None) == mode
    ), "family builder dropped history selection"


def test_replay_rejects_live_layout_change_without_advancing_rng():
    ctx = context()
    noiser = mechanism()
    release(noiser, ctx)
    before = ctx.generator.get_state().clone()
    ctx.params = list(reversed(ctx.params))  # Same flattened size, different order.
    with pytest.raises(ValueError, match="layout|parameter"):
        release(noiser, ctx)
    torch.testing.assert_close(ctx.generator.get_state(), before, rtol=0, atol=0)


def optimizer(p=4, default_rng=False, device="cpu", seed=31):
    param = torch.nn.Parameter(torch.zeros(3, device=device))
    return DPOptimizer(
        optimizer=torch.optim.SGD([param], lr=0.05, momentum=0.9),
        noise_multiplier=0.0,
        max_grad_norm=1.0,
        expected_batch_size=4,
        generator=(
            None
            if default_rng
            else torch.Generator(device=param.device).manual_seed(seed)
        ),
        noise_mechanism=mechanism(p),
    )


def update(opt, splits=(4,)):
    for i, size in enumerate(splits):
        opt.zero_grad()
        for param in opt.params:
            param.grad_sample = torch.zeros((size,) + param.shape, device=param.device)
        opt.signal_skip_step(i != len(splits) - 1)
        opt.step()
    return opt.params[0].detach().clone()


def test_microbatches_advance_noise_history_once_per_update():
    whole, split = optimizer(), optimizer()
    for _ in range(7):
        torch.testing.assert_close(
            update(whole), update(split, (3, 1)), rtol=1e-6, atol=1e-7
        )
    assert (
        whole.noise_mechanism.steps_with_noise
        == split.noise_mechanism.steps_with_noise
        == 7
    )


@pytest.mark.parametrize("steps", [1, 7])
@pytest.mark.parametrize("default_rng", [False, True])
def test_checkpoint_continuation_is_versioned_and_reproducible(steps, default_rng):
    torch.manual_seed(17)
    original = optimizer(default_rng=default_rng)
    for _ in range(steps):
        update(original)
    saved_params = original.params[0].detach().clone()
    state = copy.deepcopy(original.state_dict())
    expected = [update(original) for _ in range(5)]
    torch.rand(37)  # Disturb the global stream before restore.
    restored = optimizer(default_rng=default_rng)
    restored.params[0].data.copy_(saved_params)
    restored.load_state_dict(state)
    for want in expected:
        torch.testing.assert_close(update(restored), want, rtol=1e-6, atol=1e-7)
    assert (
        state["_dp_noise_mechanism_state"].get("schema_version") is not None
    ), "replay checkpoints need an explicit version"


def test_invalid_checkpoint_does_not_mutate_underlying_optimizer():
    source, target = optimizer(2, seed=13), optimizer(4, seed=91)
    for _ in range(2):
        update(source)
    for _ in range(7):
        update(target)
    assert not torch.equal(source.generator.get_state(), target.generator.get_state())
    assert not torch.equal(
        source.original_optimizer.state[source.params[0]]["momentum_buffer"],
        target.original_optimizer.state[target.params[0]]["momentum_buffer"],
    )
    target.original_optimizer.param_groups[0]["lr"] = 0.123
    before = copy.deepcopy(target.state_dict())
    with pytest.raises(ValueError):
        target.load_state_dict(source.state_dict())
    assert (
        target.original_optimizer.param_groups[0]["lr"] == 0.123
    ), "rejected checkpoint changed optimizer settings"
    torch.testing.assert_close(
        target.generator.get_state(),
        before["_dp_noise_generator_state"],
        rtol=0,
        atol=0,
    )
    assert_tree_equal(target.state_dict(), before)


def test_legacy_history_rejected_by_default_replay():
    noiser = mechanism()
    legacy = {
        "inverse_coeffs": coefficients(4),
        "z_std": 0.2,
        "history": [torch.ones(9)],
        "steps_with_noise": 1,
    }
    with pytest.raises(ValueError, match="legacy|stored|version"):
        noiser.load_state_dict(legacy)


def test_legacy_history_can_continue_in_explicit_stored_mode():
    noiser = mechanism(mode="stored")
    noiser.load_state_dict(
        {
            "inverse_coeffs": coefficients(4),
            "z_std": 0.2,
            "history": [torch.ones(9)],
            "steps_with_noise": 1,
        }
    )
    ctx, ref = context(), context()
    fresh = reference_draw(ref, 0.2)
    torch.testing.assert_close(release(noiser, ctx), fresh - 0.3, rtol=1e-6, atol=1e-7)


def test_unverified_secure_replay_rejected_before_rng_consumption():
    noiser, ctx = mechanism(), context()
    ctx.secure_mode = True
    before = ctx.generator.get_state().clone()
    with pytest.raises((ValueError, NotImplementedError), match="secure|generator"):
        release(noiser, ctx)
    torch.testing.assert_close(ctx.generator.get_state(), before, rtol=0, atol=0)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires a real accelerator")
def test_single_gpu_replay_and_resume():
    ctx, ref, noiser = context("cuda"), context("cuda"), mechanism(64)
    history = []
    for step in range(68):
        history.append(reference_draw(ref, 0.2))
        torch.testing.assert_close(
            release(noiser, ctx),
            reference_filter(history, coefficients(64)),
            rtol=1e-6,
            atol=1e-7,
        )
        if step == 64:
            saved = copy.deepcopy(noiser.state_dict())
            noiser = mechanism(64)
            noiser.load_state_dict(saved)
    assert (
        sum(inventory(noiser)["history"].values())
        <= 63 * ctx.generator.get_state().numel()
    )


@pytest.mark.parametrize("mode", ["regenerate", "stored"])
def test_explicit_modes_match_reference_with_mixed_parameter_dtypes(mode):
    noiser, ctx, ref = mechanism(mode=mode), context(), context()
    ctx.params[1].data = ctx.params[1].data.double()
    ref.params[1].data = ref.params[1].data.double()
    history = []
    for _ in range(7):
        history.append(reference_draw(ref, 0.2))
        torch.testing.assert_close(
            release(noiser, ctx),
            reference_filter(history, coefficients(4)),
            rtol=1e-6,
            atol=1e-7,
        )


def test_history_payload_independent_of_parameter_dimension():
    sizes = []
    for dimension in [64, 16384]:
        noiser, ctx = mechanism(64), context(shapes=((dimension,),))
        for _ in range(66):
            release(noiser, ctx)
        assert_compact_actual_storage(noiser, ctx, 64)
        sizes.append(sum(inventory(noiser)["history"].values()))
    assert sizes[0] == sizes[1] == 63 * torch.Generator().get_state().numel()


def test_incompatible_factor_family_history_override_rejected():
    config = NoiseMechanismConfig(
        mechanism="bsr",
        accounting_mode="bsr_accountant",
        mechanism_state={
            "coeffs": [1.0, 0.2],
            "z_std": 0.2,
            "noise_history_mode": "regenerate",
        },
    )
    with pytest.raises(ValueError, match="history|inverse"):
        PrivacyEngine._build_noise_mechanism_from_config(config)


@pytest.mark.parametrize(
    "key,value",
    [
        ("schema_version", 999),
        ("noise_history_mode", "unknown"),
        ("layout_signature", "invalid"),
    ],
)
def test_checkpoint_rejects_incompatible_replay_metadata_before_mutation(key, value):
    source, target = optimizer(seed=13), optimizer(seed=91)
    update(source)
    for _ in range(7):
        update(target)
    target.original_optimizer.param_groups[0]["lr"] = 0.123
    state = copy.deepcopy(source.state_dict())
    state["_dp_noise_mechanism_state"][key] = value
    before = copy.deepcopy(target.state_dict())
    with pytest.raises(ValueError, match="version|mode|layout|incompatible"):
        target.load_state_dict(state)
    assert_tree_equal(target.state_dict(), before)


def test_storage_walk_deduplicates_aliases_and_finds_descriptor_tensors():
    backing = torch.zeros(128)
    owner = SimpleNamespace(
        descriptor=SimpleNamespace(saved=backing[:3]), aliases=[backing[8:12]]
    )
    owner.cycle = owner
    assert retained_tensor_bytes(owner) == backing.untyped_storage().nbytes()


def test_storage_guard_rejects_hidden_vectors_despite_plausible_inventory():
    ctx = context(shapes=((16384,),))
    owner = SimpleNamespace(hidden_history=[torch.zeros(16384) for _ in range(63)])
    owner.noise_state_inventory = lambda: {
        "history": {"cpu": 63 * ctx.generator.get_state().numel()},
        "diagnostics": {},
        "scratch": {},
    }
    with pytest.raises(AssertionError, match="actual retained tensor storage"):
        assert_compact_actual_storage(owner, ctx, 64)


@pytest.mark.parametrize("source_default", [False, True])
@pytest.mark.parametrize("steps", [0, 3])
def test_checkpoint_rejects_changed_rng_ownership_without_mutation(
    source_default, steps
):
    torch.manual_seed(123)
    source = optimizer(default_rng=source_default, seed=13)
    target = optimizer(default_rng=not source_default, seed=91)
    for _ in range(steps):
        update(source)
    for _ in range(2):
        update(target)
    target.original_optimizer.param_groups[0]["lr"] = 0.123
    before = copy.deepcopy(target.state_dict())
    global_before = torch.get_rng_state().clone()
    with pytest.raises(ValueError, match="generator.*ownership|ownership.*generator"):
        target.load_state_dict(copy.deepcopy(source.state_dict()))
    assert_tree_equal(target.state_dict(), before)
    torch.testing.assert_close(torch.get_rng_state(), global_before, rtol=0, atol=0)
