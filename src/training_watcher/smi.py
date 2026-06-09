"""GPU sensing via ``nvidia-smi`` (stdlib-only, no hard torch dependency).

Design notes
------------
* Every subprocess call has a **timeout** — a hung/blocked ``nvidia-smi`` must never
  stall the monitor thread (the old daemon's no-timeout ``check_output`` could hang
  forever).  Any failure/timeout yields ``GpuReading(ok=False)`` and the caller treats
  it as "unknown" (conservative: never resume, never falsely yield).
* Pure parsing is split from the subprocess call so the state machine can be unit-tested
  by injecting a fake ``smi_runner`` — no GPU required.
* Sensing is **device-specific**: we watch the physical GPU the trainer actually uses,
  resolved through ``CUDA_VISIBLE_DEVICES`` / the device UUID, not blindly GPU 0.
"""

from __future__ import annotations

import os
import subprocess
from dataclasses import dataclass, field
from typing import Callable, Sequence

# A runner takes nvidia-smi args (after the binary) + a timeout and returns parsed CSV
# rows (list of fields per line).  Raises on any failure.  Injectable for tests.
SmiRunner = Callable[[Sequence[str], float], list[list[str]]]


@dataclass(frozen=True)
class GpuReading:
    """One sensor reading for a single GPU.

    ``ok=False`` means we could not read the GPU (smi failure/timeout/parse error);
    the other fields are then meaningless and must be treated as "unknown".
    """

    ok: bool
    util_pct: int | None = None
    free_gb: float | None = None
    total_gb: float | None = None
    pids: frozenset[int] = field(default_factory=frozenset)

    @property
    def free_frac(self) -> float | None:
        if self.free_gb is None or not self.total_gb:
            return None
        return self.free_gb / self.total_gb


def _default_runner(args: Sequence[str], timeout: float) -> list[list[str]]:
    """Run ``nvidia-smi <args>`` with a hard timeout; return CSV rows split into fields."""
    out = subprocess.run(
        ["nvidia-smi", *args, "--format=csv,noheader,nounits"],
        capture_output=True,
        text=True,
        timeout=timeout,
        check=True,
    ).stdout
    rows: list[list[str]] = []
    for line in out.splitlines():
        line = line.strip()
        if line:
            rows.append([c.strip() for c in line.split(",")])
    return rows


def parse_gpu_stats(rows: list[list[str]]) -> tuple[int, float, float]:
    """Parse a ``utilization.gpu,memory.free,memory.total`` row (MiB → GB).

    Raises ``ValueError``/``IndexError`` on malformed input (e.g. ``[N/A]``).
    """
    util, free, total = rows[0][0], rows[0][1], rows[0][2]
    return int(util), float(free) / 1024.0, float(total) / 1024.0


def parse_pids(rows: list[list[str]]) -> frozenset[int]:
    """Parse the first column of ``--query-compute-apps`` rows into a PID set."""
    pids: set[int] = set()
    for r in rows:
        if r and r[0].isdigit():
            pids.add(int(r[0]))
    return frozenset(pids)


def read_gpu(
    gpu_index: int | None,
    timeout: float,
    runner: SmiRunner = _default_runner,
) -> GpuReading:
    """Read utilization, free/total VRAM, and compute PIDs for one GPU.

    ``gpu_index`` selects the physical GPU (``nvidia-smi -i``); ``None`` reads GPU 0.
    Any error returns ``GpuReading(ok=False)``.
    """
    sel = ["-i", str(gpu_index)] if gpu_index is not None else []
    try:
        stat_rows = runner([*sel, "--query-gpu=utilization.gpu,memory.free,memory.total"], timeout)
        util, free_gb, total_gb = parse_gpu_stats(stat_rows)
        app_rows = runner([*sel, "--query-compute-apps=pid"], timeout)
        pids = parse_pids(app_rows)
    except Exception:
        return GpuReading(ok=False)
    return GpuReading(ok=True, util_pct=util, free_gb=free_gb, total_gb=total_gb, pids=pids)


def resolve_physical_index(
    device: object | None,
    explicit_index: int | None,
    runner: SmiRunner = _default_runner,
    timeout: float = 10.0,
) -> int | None:
    """Resolve the physical ``nvidia-smi`` index for the trainer's GPU.

    Priority: explicit config index → torch device UUID matched against ``nvidia-smi``
    → ``CUDA_VISIBLE_DEVICES`` integer remap → ``None`` (read GPU 0).  Best-effort and
    never raises; logs nothing here (the controller logs the resolved value).
    """
    if explicit_index is not None:
        return explicit_index

    # torch device index within the *visible* set (cuda:0 == first visible GPU).
    visible_idx = 0
    try:  # pragma: no cover - depends on torch/runtime
        import torch  # noqa: WPS433 (lazy, optional)

        if device is not None:
            dev = torch.device(device)
            if dev.type == "cuda" and dev.index is not None:
                visible_idx = dev.index
        # Try to match by UUID — robust to any CUDA_VISIBLE_DEVICES form (indices or UUIDs).
        uuid = getattr(torch.cuda.get_device_properties(visible_idx), "uuid", None)
        if uuid is not None:
            target = f"GPU-{uuid}" if not str(uuid).startswith("GPU-") else str(uuid)
            rows = runner(["--query-gpu=index,uuid"], timeout)
            for r in rows:
                if len(r) >= 2 and r[1].strip() == target:
                    return int(r[0])
    except Exception:
        pass

    # Fall back to CUDA_VISIBLE_DEVICES integer remap: visible position -> physical index.
    cvd = os.environ.get("CUDA_VISIBLE_DEVICES")
    if cvd:
        parts = [p.strip() for p in cvd.split(",") if p.strip()]
        if visible_idx < len(parts) and parts[visible_idx].isdigit():
            return int(parts[visible_idx])
    return None
