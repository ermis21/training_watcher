"""Move a model + optimizer between GPU and CPU to free VRAM during a pause.

Torch is imported lazily so the rest of the library (config / window / smi / monitor /
controller decision logic) imports and unit-tests with no torch installed.

What gets freed
---------------
``offload_to_cpu`` moves model parameters/buffers and every optimizer-state tensor
(``exp_avg``, ``exp_avg_sq``, and the per-parameter ``step`` — which is a CUDA tensor
under PyTorch's default ``foreach`` AdamW) to the CPU, then synchronizes, drops Python
references, ``gc.collect()``s, and calls ``empty_cache()`` so the caching allocator
returns the blocks to the driver for another process.

Best-effort caveat: the process keeps its CUDA *context* (a few hundred MB) for as long
as it lives — that is not freed and cannot be without tearing down torch's CUDA state.
A competitor still gets the multi-GB working set back, which is the point.
"""

from __future__ import annotations

from typing import Any


def cuda_free_bytes(device: Any) -> int | None:
    """Free bytes on ``device`` per the driver, or ``None`` if unavailable."""
    try:
        import torch

        dev = torch.device(device)
        if dev.type != "cuda":
            return None
        return int(torch.cuda.mem_get_info(dev)[0])
    except Exception:
        return None


# Maps (id(param), state_key) -> original torch.device, so reload restores each optimizer
# tensor to where it really lived.  Crucial because AdamW keeps the per-parameter ``step``
# on the **CPU** in the default (non-capturable) mode even when the moments are on CUDA —
# blindly moving everything back to the GPU corrupts the step's device.
RestoreMap = dict


def _offload_optimizer_state(optimizer: Any) -> RestoreMap:
    """Move every CUDA tensor in ``optimizer.state`` to CPU, recording its origin device."""
    import torch

    restore: RestoreMap = {}
    for param, state in optimizer.state.items():
        for key, val in list(state.items()):
            if torch.is_tensor(val):
                restore[(id(param), key)] = val.device
                if val.device.type != "cpu":
                    state[key] = val.to("cpu")
            # non-tensor entries (e.g. a python-int ``step``) are left untouched
    return restore


def _reload_optimizer_state(optimizer: Any, fallback: Any, restore: RestoreMap | None) -> None:
    """Restore each optimizer tensor to its recorded device (or ``fallback`` if unknown)."""
    import torch

    for param, state in optimizer.state.items():
        for key, val in list(state.items()):
            if torch.is_tensor(val):
                target = restore.get((id(param), key)) if restore is not None else None
                if target is None:
                    target = fallback
                if val.device != torch.device(target):
                    state[key] = val.to(target)


def offload_to_cpu(model: Any, optimizer: Any) -> tuple[int | None, int | None, RestoreMap]:
    """Move model + optimizer state to CPU and release the cache.

    Order matters: move the model first (``nn.Module.to`` swaps ``.data`` in place, so the
    optimizer's parameter-keyed state stays valid), then the optimizer-state tensors, then
    synchronize / gc / empty_cache.  Returns ``(free_before, free_after, restore_map)`` —
    pass ``restore_map`` to :func:`reload_to_device` so each tensor returns to its origin.
    """
    import gc

    import torch

    device = next(model.parameters()).device
    free_before = cuda_free_bytes(device)

    model.to("cpu")
    restore = _offload_optimizer_state(optimizer)

    if device.type == "cuda":
        torch.cuda.synchronize(device)
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()

    free_after = cuda_free_bytes(device)
    return free_before, free_after, restore


def reload_to_device(model: Any, optimizer: Any, device: Any, restore: RestoreMap | None = None) -> None:
    """Move model + optimizer state back onto ``device``, honoring per-tensor origins.

    ``restore`` (from :func:`offload_to_cpu`) preserves the original device of each
    optimizer tensor — notably keeping AdamW's CPU ``step`` on the CPU.  Without it,
    optimizer tensors fall back to ``device``.

    May raise ``torch.cuda.OutOfMemoryError`` if a third process grabbed the VRAM while we
    were paused — the caller is expected to catch it, ``empty_cache()``, and stay paused.
    """
    model.to(device)
    _reload_optimizer_state(optimizer, device, restore)


def empty_cache(device: Any = None) -> None:
    """Best-effort ``torch.cuda.empty_cache`` (used after a failed reload)."""
    try:
        import torch

        if device is not None and torch.device(device).type != "cuda":
            return
        torch.cuda.empty_cache()
    except Exception:
        pass
