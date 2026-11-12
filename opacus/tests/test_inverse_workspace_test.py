"""Live tensor payload regression; excludes inputs, final gradients and RNG state."""

import weakref

import pytest
import torch
from torch.utils._python_dispatch import TorchDispatchMode
from torch.utils._pytree import tree_flatten

from opacus.tests import test_inverse_noise_regeneration_test as reference

context = reference.context
mechanism = reference.mechanism
release = reference.release


class LiveStorage(TorchDispatchMode):
    def __init__(self):
        super().__init__()
        self.refs = []
        self.samples = []

    def __torch_dispatch__(self, func, types, args=(), kwargs=None):
        result = func(*args, **(kwargs or {}))
        for value in tree_flatten(result)[0]:
            if isinstance(value, torch.Tensor):
                self.refs.append(weakref.ref(value))
        live = {}
        self.refs = [ref for ref in self.refs if ref() is not None]
        for ref in self.refs:
            value = ref()
            storage = value.untyped_storage()
            live[storage.data_ptr()] = storage.nbytes()
        self.samples.append(live)
        return result

    def peak(self, excluded=()):
        return max(
            (
                sum(n for key, n in row.items() if key not in excluded)
                for row in self.samples
            ),
            default=0,
        )


def test_storage_probe_controls():
    with LiveStorage() as probe:
        a = torch.zeros(1000)
        b = a.view(-1)
        c = a.clone()
    assert probe.peak() == 8000
    assert b.data_ptr() == a.data_ptr()
    assert c.data_ptr() != a.data_ptr()


@pytest.mark.parametrize("p", [2, 4, 64])
def test_replay_workspace(p):
    ctx = context(shapes=((16384,),) * 16)
    noiser = mechanism(p=p)
    for _ in range(p + 1):
        release(noiser, ctx)
    for param in ctx.params:
        param.grad = None
        param.summed_grad = torch.zeros_like(param)
    with LiveStorage() as probe:
        noiser.add_noise(ctx)
    # Include final gradients in the measured peak: excluding their addresses
    # retroactively could hide temporaries whose freed storage was reused.
    peak = probe.peak()
    budget = 2 * 16 * 16384 * 4 + 4 * 16384 * 4 + 65536
    print(f"p={p}: additional live payload including outputs={peak} budget={budget}")
    assert peak <= budget


@pytest.mark.parametrize("p", [1, 2, 4, 64])
def test_blockwise_replay_matches_stored_with_diagnostics(p):
    replay = mechanism(p=p)
    stored = mechanism(p=p, mode="stored")
    replay.record_noise_diagnostics = stored.record_noise_diagnostics = True
    a, b = context(), context()
    a.params[1].data = a.params[1].data.double()
    b.params[1].data = b.params[1].data.double()
    for step in range(p + 2):
        replay.z_std = stored.z_std = [0.2, 0, 0.7][step % 3]
        for ctx in (a, b):
            for param in ctx.params:
                param.summed_grad = torch.ones_like(param)
        replay.add_noise(a)
        stored.add_noise(b)
        for x, y in zip(a.params, b.params):
            assert torch.equal(x.grad, y.grad)
            assert torch.equal(x.summed_grad, torch.ones_like(x))
        assert torch.equal(replay.last_flat_z, stored.last_flat_z)
        assert torch.equal(replay.last_flat_u, stored.last_flat_u)
        assert torch.equal(a.generator.get_state(), b.generator.get_state())


@pytest.mark.parametrize("steps", [1, 6])
def test_pre_workspace_checkpoint_continuation(steps):
    # Construct the existing checkpoint format independently of state_dict().
    # Parameter-shaped draws reproduce its saved RNG snapshots and continuation.
    source = context()
    snapshots, draws = [], []
    for _ in range(steps):
        snapshots.append((source.generator.get_state(), 0.2))
        draws.append(reference.reference_draw(source, 0.2))
    continuation = source.generator.get_state()
    coeffs = tuple(reference.coefficients(4))
    checkpoint = {
        "schema_version": 1,
        "noise_history_mode": "regenerate",
        "inverse_coeffs": coeffs,
        "coeffs": coeffs,
        "z_std": 0.2,
        "steps_with_noise": steps,
        "layout_signature": tuple(
            (tuple(p.shape), str(p.dtype), str(p.device)) for p in source.params
        ),
        "replay_history": list(reversed(snapshots[-3:])),
    }
    draws.append(reference.reference_draw(source, 0.2))
    expected = reference.reference_filter(draws, coeffs)

    noiser = mechanism(p=4)
    ctx = context()
    noiser.load_state_dict(checkpoint)
    ctx.generator.set_state(continuation)
    assert torch.equal(release(noiser, ctx), expected)
    assert torch.equal(ctx.generator.get_state(), source.generator.get_state())
    assert noiser.steps_with_noise == steps + 1


def test_noncontiguous_summed_gradients_match_stored():
    a, b = context(shapes=((3, 2),)), context(shapes=((3, 2),))
    replay, stored = mechanism(p=4), mechanism(p=4, mode="stored")
    for _ in range(7):
        for ctx, noiser in ((a, replay), (b, stored)):
            ctx.params[0].summed_grad = torch.arange(6.0).view(2, 3).t()
            assert not ctx.params[0].summed_grad.is_contiguous()
            noiser.add_noise(ctx)
        assert torch.equal(a.params[0].grad, b.params[0].grad)


@pytest.mark.parametrize("blocks", [1, 16])
def test_finite_checks_have_constant_scalar_reads(blocks):
    from torch.utils._python_dispatch import TorchDispatchMode

    class ScalarReads(TorchDispatchMode):
        count = 0

        def __torch_dispatch__(self, func, types, args=(), kwargs=None):
            if func == torch.ops.aten._local_scalar_dense.default:
                self.count += 1
            return func(*args, **(kwargs or {}))

    noiser = mechanism(p=4)
    ctx = context(shapes=((8,),) * blocks)
    for _ in range(5):
        release(noiser, ctx)
    with ScalarReads() as reads:
        release(noiser, ctx)
    assert reads.count <= 3


@pytest.mark.parametrize("bad", [float("nan"), float("inf")])
def test_nonfinite_late_gradient_rejects_before_assigning_outputs(bad):
    noiser = mechanism(p=4)
    ctx = context()
    for p in ctx.params:
        p.summed_grad = torch.zeros_like(p)
    ctx.params[-1].summed_grad[0] = bad
    with pytest.raises(ValueError, match="summed_flat"):
        noiser.add_noise(ctx)
    assert all(p.grad is None for p in ctx.params)
    assert noiser.first_non_finite_event["tensor"] == "summed_flat"


def test_nonfinite_fresh_noise_keeps_live_rng_continuation(monkeypatch):
    from opacus.noise_mechanisms import correlated

    original = correlated.generate_noise

    def bad_draw(**kwargs):
        block = original(**kwargs)
        if block.shape == (3,):
            block[0] = float("nan")
        return block

    ctx, oracle = context(), context()
    reference.reference_draw(oracle, 0.2)
    noiser = mechanism(p=4)
    for p in ctx.params:
        p.summed_grad = torch.zeros_like(p)
    monkeypatch.setattr(correlated, "generate_noise", bad_draw)
    with pytest.raises(ValueError, match="z_flat"):
        noiser.add_noise(ctx)
    assert torch.equal(ctx.generator.get_state(), oracle.generator.get_state())
    assert noiser.first_non_finite_event["non_finite"] == 1
    assert noiser.state_depth == 0
    assert all(p.grad is None for p in ctx.params)
