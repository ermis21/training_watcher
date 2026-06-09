"""training_watcher — in-process cooperative GPU sharing for training loops.

Quickstart::

    from training_watcher import CoopController, CoopConfig, auto_stop

    coop = CoopController(CoopConfig(off_hours=(22, 8)), device=device)
    coop.register(model, checkpoint_cb)          # checkpoint_cb: zero-arg, reads live loop state
    coop.start()
    auto_stop(coop)
    for step in training_loop:
        ...
        coop.guard(global_step, epoch, optimizer)   # no-op unless a pause is due

See ``CoopConfig`` for the full policy and ``README.md`` for the design and caveats.
"""

from __future__ import annotations

from .checkpoint import atomic_checkpoint_dir, find_latest_checkpoint, is_complete
from .config import CoopConfig
from .controller import CoopController, auto_stop, autoyield, watcher
from .monitor import GpuMonitor, Snapshot
from .offload import offload_to_cpu, reload_to_device
from .smi import GpuReading, read_gpu, resolve_physical_index
from .window import in_owned_window

__version__ = "0.1.0"

__all__ = [
    "__version__",
    "CoopConfig",
    "CoopController",
    "watcher",
    "autoyield",
    "auto_stop",
    "GpuMonitor",
    "Snapshot",
    "GpuReading",
    "read_gpu",
    "resolve_physical_index",
    "in_owned_window",
    "offload_to_cpu",
    "reload_to_device",
    "atomic_checkpoint_dir",
    "find_latest_checkpoint",
    "is_complete",
]
