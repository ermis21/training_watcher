"""offload_to_cpu / reload_to_device — optimizer-state value AND device preservation.

The CUDA test is skipped unless a GPU is available; CPU variants exercise the
optimizer-state movers without any GPU.

Regression guard: AdamW keeps the per-parameter ``step`` on the CPU in the default
(non-capturable) mode even when the moments live on CUDA.  Reload must restore each
tensor to its *original* device, not blindly to the training device — so these tests
compare devices, not just ``.cpu()`` values (an earlier ``.cpu()``-only check masked the
bug entirely).
"""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from training_watcher.offload import (
    cuda_free_bytes,
    offload_to_cpu,
    reload_to_device,
)


def _train_two_steps(model, opt, device, in_features=64):
    for _ in range(2):
        opt.zero_grad()
        y = model(torch.randn(8, in_features, device=device)).sum()
        y.backward()
        opt.step()


def _snapshot(opt):
    """Record (device, value-on-cpu) for every optimizer-state tensor."""
    snap = []
    for state in opt.state.values():
        entry = {}
        for k, v in state.items():
            if torch.is_tensor(v):
                entry[k] = (v.device, v.detach().clone().cpu())
            else:
                entry[k] = (None, v)
        snap.append(entry)
    return snap


def _assert_restored(opt, snap):
    for entry, state in zip(snap, opt.state.values()):
        for k, (dev, ref) in entry.items():
            v = state[k]
            if torch.is_tensor(v):
                assert v.device == dev, f"{k}: device {v.device} != original {dev}"
                assert torch.equal(v.detach().cpu(), ref), f"{k}: value changed"
            else:
                assert v == ref, f"{k}: scalar changed"


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")
def test_offload_reload_roundtrip_cuda():
    device = "cuda"
    # Wide layer so params + grads + AdamW moments are a few hundred MB that the driver
    # reports as freed (a tiny Linear is below mem_get_info granularity).
    dim = 4096
    model = torch.nn.Linear(dim, dim).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3)
    _train_two_steps(model, opt, device, in_features=dim)

    states = list(opt.state.values())
    assert states and any("exp_avg" in s for s in states) and any("step" in s for s in states)

    # AdamW (non-capturable) keeps `step` on CPU while moments are on CUDA — assert that,
    # so the test meaningfully exercises mixed-device restore.
    step_dev = next(s["step"].device for s in states if "step" in s and torch.is_tensor(s["step"]))
    ea_dev = next(s["exp_avg"].device for s in states if "exp_avg" in s)
    assert ea_dev.type == "cuda"

    snap = _snapshot(opt)

    free_before, free_after, restore = offload_to_cpu(model, opt)
    assert free_before is not None and free_after is not None
    assert free_after > free_before                       # offload freed working-set VRAM
    assert next(model.parameters()).device.type == "cpu"  # model params now on CPU
    # every optimizer tensor (incl. cuda moments) parked on CPU during the pause
    for state in opt.state.values():
        for v in state.values():
            if torch.is_tensor(v):
                assert v.device.type == "cpu"

    reload_to_device(model, opt, device, restore)
    assert next(model.parameters()).device.type == "cuda"

    # values identical AND every tensor back on its ORIGINAL device (step stays on CPU)
    _assert_restored(opt, snap)
    new_step_dev = next(s["step"].device for s in opt.state.values()
                        if "step" in s and torch.is_tensor(s["step"]))
    assert new_step_dev == step_dev

    # a real optimizer step still works after the round-trip (no device mismatch)
    _train_two_steps(model, opt, device, in_features=dim)


def test_offload_reload_roundtrip_cpu():
    # No GPU: offload/reload on a CPU model is a value-preserving no-op move.
    model = torch.nn.Linear(64, 64)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3)
    _train_two_steps(model, opt, "cpu")
    snap = _snapshot(opt)

    assert cuda_free_bytes("cpu") is None
    before, after, restore = offload_to_cpu(model, opt)
    assert before is None and after is None               # no CUDA device
    reload_to_device(model, opt, "cpu", restore)
    _assert_restored(opt, snap)
    _train_two_steps(model, opt, "cpu")                   # still trainable
