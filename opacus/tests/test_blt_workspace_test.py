"""Allocation and independent recurrence contracts for BLT noising."""

import pytest
import torch

from opacus.accountants.analysis.blt import blt_pair_from_theta_pair
from opacus.noise_mechanisms.blt import BufferedToeplitzNoiseMechanism
from opacus.tests import test_inverse_noise_regeneration_test as reference
from opacus.tests import test_inverse_workspace_test as workspace


def mechanism(buffers=4, diagnostics=False):
    pair = blt_pair_from_theta_pair(
        theta=[0.8 - i * 0.05 for i in range(buffers)],
        theta_hat=[0.7 - i * 0.05 for i in range(buffers)],
    )
    return BufferedToeplitzNoiseMechanism(
        pair=pair, z_std=0.2, record_noise_diagnostics=diagnostics
    )


@pytest.mark.parametrize("buffers", [0, 1, 4, 8])
def test_independent_recurrence(buffers):
    noiser = mechanism(buffers, diagnostics=True)
    ctx = reference.context()
    oracle = reference.context()
    state = [torch.zeros(9) for _ in range(buffers)]
    for step in range(7):
        scale = [0.2, 0, 0.7][step % 3]
        noiser.z_std = scale
        z = reference.reference_draw(oracle, scale)
        if state:
            weighted = noiser._omega[0] * state[0]
            for omega, buf in zip(noiser._omega[1:], state[1:]):
                weighted = weighted + omega * buf
            expected = z - weighted
        else:
            expected = z.clone()
        state = [theta * buf + expected for theta, buf in zip(noiser._theta, state)]
        actual = reference.release(noiser, ctx)
        assert torch.equal(actual, expected)
        assert torch.equal(noiser.last_flat_z, z)
        assert torch.equal(noiser.last_flat_u, expected)
        assert torch.equal(ctx.generator.get_state(), oracle.generator.get_state())
        for actual_buf, expected_buf in zip(noiser._buffers, state):
            assert torch.equal(actual_buf, expected_buf)


def test_buffer_storage_reused_and_checkpoint_independent():
    noiser = mechanism()
    ctx = reference.context()
    reference.release(noiser, ctx)
    pointers = [buf.data_ptr() for buf in noiser._buffers]
    saved = noiser.state_dict()
    copies = [buf.clone() for buf in saved["buffers"]]
    reference.release(noiser, ctx)
    assert pointers == [buf.data_ptr() for buf in noiser._buffers]
    assert all(torch.equal(a, b) for a, b in zip(saved["buffers"], copies))
    restored = mechanism()
    restored.load_state_dict(saved)
    restored._buffers[0].add_(1)
    assert torch.equal(saved["buffers"][0], copies[0])


@pytest.mark.parametrize("buffers", [1, 4, 8])
@pytest.mark.parametrize("warm", [False, True])
def test_blt_workspace(buffers, warm):
    noiser = mechanism(buffers)
    ctx = reference.context(shapes=((16384,),) * 16)
    if warm:
        reference.release(noiser, ctx)
    persistent = {buf.data_ptr() for buf in noiser._buffers}
    for param in ctx.params:
        param.grad = None
        param.summed_grad = torch.zeros_like(param)
    with workspace.LiveStorage() as probe:
        noiser.add_noise(ctx)
    # Existing state stays live throughout; exclude only these known allocations.
    # Include final gradients so address reuse cannot hide earlier temporaries.
    peak = probe.peak(persistent)
    vector_bytes = 16 * 16384 * 4
    budget = 2 * vector_bytes + 4 * 16384 * 4 + 65536
    if not warm:
        budget += buffers * vector_bytes
    print(f"b={buffers}: additional payload={peak} budget={budget}")
    assert peak <= budget


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
@pytest.mark.parametrize("reduction", ["sum", "mean"])
@pytest.mark.parametrize("secure", [False, True])
def test_noising_protocol_and_checkpoint(dtype, reduction, secure):
    from opacus.noise_mechanisms._utils import generate_noise

    noiser = mechanism(4, diagnostics=True)
    ctx = reference.context(shapes=((3, 2),))
    oracle = reference.context(shapes=((3, 2),))
    for current in (ctx, oracle):
        current.params[0].data = current.params[0].data.to(dtype)
        current.loss_reduction = reduction
        current.accumulated_iterations = 2
        current.secure_mode = secure
    state = [torch.zeros(6, dtype=dtype) for _ in range(4)]
    for _ in range(5):
        gradient = torch.arange(6, dtype=dtype).view(2, 3).t()
        ctx.params[0].summed_grad = gradient.clone().t().contiguous().t()
        assert not ctx.params[0].summed_grad.is_contiguous()
        std = 0.2 if reduction == "sum" else 0.2 * 4 * 2
        z = generate_noise(
            std, torch.zeros((3, 2), dtype=dtype), oracle.generator, secure
        ).flatten()
        weighted = state[0] * noiser._omega[0]
        for omega, buf in zip(noiser._omega[1:], state[1:]):
            weighted = weighted + omega * buf
        expected = z - weighted
        state = [theta * buf + expected for theta, buf in zip(noiser._theta, state)]
        noiser.add_noise(ctx)
        assert torch.equal(ctx.params[0].grad, gradient + expected.view_as(gradient))
        assert torch.equal(ctx.params[0].summed_grad, gradient)
        assert torch.equal(noiser.last_flat_z, z)
        assert torch.equal(ctx.generator.get_state(), oracle.generator.get_state())
        for actual, wanted in zip(noiser._buffers, state):
            assert torch.equal(actual, wanted)
        with pytest.raises(ValueError, match="cleared"):
            noiser.add_noise(ctx)
    # Build old-format state directly rather than retaining a binary fixture.
    pair = noiser.pair
    checkpoint = {
        "pair": {
            "forward": {
                "theta": list(pair.forward.theta_array()),
                "omega": list(pair.forward.omega_array()),
            },
            "inverse": {
                "theta": list(pair.inverse.theta_array()),
                "omega": list(pair.inverse.omega_array()),
            },
        },
        "z_std": 0.2,
        "buffers": [buf.clone() for buf in state],
        "steps_with_noise": 5,
    }
    restored = mechanism(4)
    restored.load_state_dict(checkpoint)
    assert restored.steps_with_noise == 5
    z = torch.ones(6, dtype=dtype)
    assert torch.equal(
        restored._apply_inverse_stream(z), noiser._apply_inverse_stream(z)
    )
    noiser.reset_state()
    assert noiser._buffers == []
