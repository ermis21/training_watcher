"""CoopController — the one object a training loop holds.

Wiring (register-once, lean hot path)::

    coop = CoopController(CoopConfig(...), device=device)
    coop.register(model, checkpoint_cb, reload_cb=None,
                  on_pause=..., on_resume=..., on_yield=...)
    coop.start()
    auto_stop(coop)                      # SIGTERM/SIGINT -> graceful stop (chained)
    ...
    for step in loop:
        ...
        coop.guard(global_step, epoch, optimizer)   # no-op unless a pause is due

The named primitives ``watcher`` / ``autoyield`` / ``auto_stop`` (bottom of this module)
are thin convenience wrappers over this class.
"""

from __future__ import annotations

import logging
import os
import signal
import threading
import time
from datetime import datetime
from typing import Any, Callable, Mapping

from .agenteval import AgentEval
from .config import CoopConfig
from .monitor import GpuMonitor, Snapshot
from .offload import cuda_free_bytes, empty_cache, offload_to_cpu, reload_to_device
from .smi import GpuReading, read_gpu, resolve_physical_index
from .window import in_owned_window

CheckpointCb = Callable[[], None]
LifecycleCb = Callable[[dict[str, Any]], None]

RUNNING = "RUNNING"
PAUSED = "PAUSED"
STOPPING = "STOPPING"


class CoopController:
    """In-process cooperative GPU sharing: pause→free VRAM→resume around competitors."""

    def __init__(
        self,
        config: CoopConfig,
        *,
        device: Any = "cuda",
        logger: logging.Logger | None = None,
        reader: Callable[[], GpuReading] | None = None,
        monotonic: Callable[[], float] = time.monotonic,
    ) -> None:
        from .logging_util import get_logger

        self._cfg = config
        self._device = device
        self._log = logger or get_logger(config.log_path)
        self._monotonic = monotonic

        self._gpu_index = config.gpu_index           # resolved fully in start()
        self._reader = reader                        # injectable; built in start() if None
        self._monitor: GpuMonitor | None = None
        self._agent = AgentEval(config, self._log)

        # registered training objects / callbacks
        self._model: Any = None
        self._optimizer_ref: Any = None
        self._restore_map: Any = None
        self._checkpoint_cb: CheckpointCb | None = None
        self._reload_cb: Callable[[], None] | None = None
        self._on_pause: LifecycleCb | None = None
        self._on_resume: LifecycleCb | None = None
        self._on_yield: LifecycleCb | None = None

        # runtime state
        self._state = RUNNING
        self._auto_yield_enabled = True
        self._stop_event = threading.Event()

        # counters
        self.pauses_total = 0
        self.seconds_paused = 0.0
        self.reload_oom_retries = 0

    # ── registration ──────────────────────────────────────────────────────────
    def register(
        self,
        model: Any,
        checkpoint_cb: CheckpointCb,
        *,
        reload_cb: Callable[[], None] | None = None,
        on_pause: LifecycleCb | None = None,
        on_resume: LifecycleCb | None = None,
        on_yield: LifecycleCb | None = None,
    ) -> "CoopController":
        """Register the model + a zero-arg checkpoint callback (reads live loop state)."""
        self._model = model
        self._checkpoint_cb = checkpoint_cb
        self._reload_cb = reload_cb
        self._on_pause = on_pause
        self._on_resume = on_resume
        self._on_yield = on_yield
        return self

    # ── lifecycle ─────────────────────────────────────────────────────────────
    def start(self) -> "CoopController":
        """Resolve the GPU, run the own-PID self-check, and start the monitor thread."""
        self._gpu_index = resolve_physical_index(
            self._device, self._cfg.gpu_index, timeout=self._cfg.smi_timeout_s
        )
        if self._reader is None:
            idx, to = self._gpu_index, self._cfg.smi_timeout_s
            self._reader = lambda: read_gpu(idx, to)

        self._monitor = GpuMonitor(self._cfg, self._reader, monotonic=self._monotonic)
        self._self_check()
        self._monitor.start()
        self._log.info(
            "started: gpu_index=%s auto_yield=%s headroom=%.1fGB poll=%.0fs "
            "yield_debounce=%.0fs resume_idle=%.0fs",
            self._gpu_index, self._auto_yield_enabled, self._cfg.vram_headroom_gb,
            self._cfg.poll_s, self._cfg.yield_debounce_s, self._cfg.resume_idle_s,
        )
        return self

    def _self_check(self) -> None:
        """Confirm we can see our own PID on the GPU; otherwise fail closed (no auto-yield).

        If ``nvidia-smi`` reports host PIDs while we live in an isolated PID namespace, our
        own PID would look like a competitor and we'd pause forever — so unless we can prove
        we see ourselves, we disable auto-yield (configurable via ``cfg.fail_closed``).
        """
        reading = self._reader() if self._reader else GpuReading(ok=False)
        if reading.ok and self._cfg.our_pid in reading.pids:
            self._auto_yield_enabled = True
            return
        msg = (
            "self-check FAILED: own PID %d not visible in nvidia-smi compute-apps "
            "(PID-namespace mismatch or CUDA not yet initialized)."
        )
        if self._cfg.fail_closed:
            self._auto_yield_enabled = False
            self._log.warning(msg + " fail_closed=True → auto-yield DISABLED (will not pause).",
                              self._cfg.our_pid)
        else:
            self._auto_yield_enabled = True
            self._log.warning(msg + " fail_closed=False → auto-yield kept ENABLED (risky).",
                              self._cfg.our_pid)

    # ── introspection ─────────────────────────────────────────────────────────
    @property
    def state(self) -> str:
        return self._state

    @property
    def stop_requested(self) -> bool:
        return self._stop_event.is_set()

    @property
    def auto_yield_enabled(self) -> bool:
        return self._auto_yield_enabled

    def request_stop(self) -> None:
        """Ask any in-progress pause-wait to return so the trainer can checkpoint + exit."""
        self._stop_event.set()
        if self._state == PAUSED:
            self._state = STOPPING

    def note_checkpoint(
        self, step: int, log_path: str | None = None, metrics: Mapping[str, Any] | None = None
    ) -> None:
        """Call right after the trainer writes a periodic checkpoint (drives agent-eval)."""
        self._agent.maybe_run(step, metrics)

    def stop(self) -> None:
        if self._monitor is not None:
            self._monitor.stop()

    # ── decision predicates ───────────────────────────────────────────────────
    def _is_stale(self, snap: Snapshot) -> bool:
        return (self._monotonic() - snap.taken_at) > self._cfg.stale_after_s

    def _should_yield(self, now_wall: datetime, snap: Snapshot) -> bool:
        cfg = self._cfg
        if not self._auto_yield_enabled:
            return False
        if in_owned_window(now_wall, cfg):
            return False                       # we own the GPU now → never yield
        if not snap.ok or self._is_stale(snap):
            return False                       # unknown → don't yield
        if not snap.competitor_present or snap.threat_since is None:
            return False
        return (self._monotonic() - snap.threat_since) >= cfg.yield_debounce_s

    def _can_resume(self, snap: Snapshot) -> bool:
        if not snap.ok or self._is_stale(snap):
            return False                       # unknown → never resume
        if not snap.is_clear or snap.clear_since is None:
            return False
        return (self._monotonic() - snap.clear_since) >= self._cfg.resume_idle_s

    # ── hot path ──────────────────────────────────────────────────────────────
    def guard(self, step: int, epoch: int, optimizer: Any) -> bool:
        """Call once per optimizer step. Returns True iff a pause happened this call.

        Fast path is a single atomic snapshot read + a couple of comparisons.
        """
        if self._monitor is None:
            return False
        snap = self._monitor.snapshot
        if not self._should_yield(datetime.now(), snap):
            return False
        self._do_pause(step, epoch, optimizer)
        return True

    def _do_pause(self, step: int, epoch: int, optimizer: Any) -> None:
        assert self._monitor is not None
        cfg = self._cfg
        self._optimizer_ref = optimizer            # reachable by _try_reload during the wait
        snap = self._monitor.snapshot
        self._log.warning(
            "competitor on GPU outside owned window (pids=%s free=%.1fGB) — pausing at step %d",
            sorted(snap.other_pids), snap.free_gb or -1.0, step,
        )
        self._emit(self._on_yield, {"step": step, "epoch": epoch, "other_pids": sorted(snap.other_pids)})

        # 1) checkpoint (load-bearing insurance if we're killed while paused)
        if self._checkpoint_cb is not None:
            try:
                self._checkpoint_cb()
            except Exception as exc:                       # offload anyway; state is live in RAM
                self._log.warning("pause checkpoint failed (%r) — continuing to offload", exc)

        # 2) free VRAM (restore map keeps e.g. AdamW's CPU `step` on the CPU on reload)
        before, after, self._restore_map = offload_to_cpu(self._model, optimizer)
        self._state = PAUSED
        self.pauses_total += 1
        pause_started = self._monotonic()
        self._log.warning(
            "PAUSED at step %d — freed VRAM %s → %s",
            step, _fmt_gb(before), _fmt_gb(after),
        )
        self._emit(self._on_pause, {"step": step, "epoch": epoch,
                                    "free_before_gb": _gb(before), "free_after_gb": _gb(after)})

        # 3) wait until the GPU is clear+idle for resume_idle_s, then reload (OOM-safe)
        wait_tick = min(cfg.poll_s, 5.0)
        while not self._stop_event.is_set():
            if self._can_resume(self._monitor.snapshot):
                fresh = self._monitor.fresh_snapshot()     # re-sense right before committing
                if not self._can_resume(fresh):
                    continue
                if self._try_reload(step):
                    break
            self._stop_event.wait(wait_tick)

        # 4) resumed (or stopping)
        paused_for = self._monotonic() - pause_started
        self.seconds_paused += paused_for
        if self._stop_event.is_set():
            self._state = STOPPING
            self._log.warning("stop requested while paused at step %d (paused %.0fs) — "
                              "trainer will checkpoint and exit", step, paused_for)
            return
        self._state = RUNNING
        self._log.warning("RESUMED at step %d after %.0fs paused (%d total pauses)",
                          step, paused_for, self.pauses_total)
        self._emit(self._on_resume, {"step": step, "epoch": epoch, "paused_seconds": paused_for})
        if self._reload_cb is not None:
            try:
                self._reload_cb()
            except Exception as exc:
                self._log.warning("reload_cb failed: %r", exc)

    def _try_reload(self, step: int) -> bool:
        """Attempt to move state back to the GPU; on OOM stay paused and retry later."""
        try:
            reload_to_device(self._model, self._optimizer_ref, self._device, self._restore_map)
            return True
        except Exception as exc:  # torch.cuda.OutOfMemoryError and friends
            if _is_oom(exc):
                self.reload_oom_retries += 1
                empty_cache(self._device)
                if self._monitor is not None:
                    self._monitor.reset_clear()            # restart the idle countdown
                self._log.warning(
                    "reload OOM at step %d (a process grabbed the VRAM) — staying paused, "
                    "retry %d", step, self.reload_oom_retries,
                )
                return False
            raise

    # ── helpers ───────────────────────────────────────────────────────────────
    def _emit(self, cb: LifecycleCb | None, payload: dict[str, Any]) -> None:
        if cb is None:
            return
        try:
            cb(payload)
        except Exception as exc:
            self._log.warning("lifecycle callback failed: %r", exc)


def _gb(n: int | None) -> float | None:
    return None if n is None else round(n / 1e9, 2)


def _fmt_gb(n: int | None) -> str:
    return "?" if n is None else f"{n / 1e9:.1f}GB"


def _is_oom(exc: BaseException) -> bool:
    if exc.__class__.__name__ == "OutOfMemoryError":
        return True
    return "out of memory" in str(exc).lower()


# ── named primitives (thin wrappers the user asked for) ───────────────────────
def watcher(config: CoopConfig, **kwargs: Any) -> CoopController:
    """Build a controller and start its monitor. (Register the model before guarding.)"""
    return CoopController(config, **kwargs).start()


def autoyield(controller: CoopController, step: int, epoch: int, optimizer: Any) -> bool:
    """Free-function form of :meth:`CoopController.guard` — the per-step reservation check."""
    return controller.guard(step, epoch, optimizer)


class auto_stop:
    """Install SIGTERM/SIGINT handlers that chain to any existing handler then stop the run.

    Usable as a one-shot installer (``auto_stop(coop)``) or a context manager
    (``with auto_stop(coop): ...``) which restores the previous handlers on exit.
    """

    def __init__(self, controller: CoopController, signals: tuple[int, ...] = (signal.SIGTERM, signal.SIGINT)):
        self._controller = controller
        self._signals = signals
        self._prev: dict[int, Any] = {}
        self._install()

    def _install(self) -> None:
        for sig in self._signals:
            try:
                prev = signal.getsignal(sig)
                self._prev[sig] = prev

                def handler(signum: int, frame: Any, _prev: Any = prev) -> None:
                    if callable(_prev) and _prev not in (signal.SIG_DFL, signal.SIG_IGN):
                        try:
                            _prev(signum, frame)        # preserve trainer's _STOP_REQUESTED path
                        except Exception:
                            pass
                    self._controller.request_stop()

                signal.signal(sig, handler)
            except (ValueError, OSError):
                # signal.signal only works on the main thread; skip elsewhere.
                self._controller._log.warning(
                    "auto_stop: could not install handler for signal %s (not main thread?)", sig
                )

    def __enter__(self) -> "auto_stop":
        return self

    def __exit__(self, *exc: Any) -> None:
        for sig, prev in self._prev.items():
            try:
                signal.signal(sig, prev)
            except (ValueError, OSError):
                pass
