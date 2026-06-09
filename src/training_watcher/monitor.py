"""Background GPU monitor: polls the sensor and publishes an immutable snapshot.

Concurrency model
-----------------
A single daemon thread is the *only* writer.  Each poll it builds a fresh, frozen
:class:`Snapshot` and publishes it with one attribute assignment (atomic under the GIL).
The hot training path reads ``monitor.snapshot`` — a single attribute read of an
immutable object — with **no lock**.  The debounce/idle timers (`threat_since`,
`clear_since`) live on the monitor thread and are carried inside each snapshot, so the
reader never sees a half-updated timer.
"""

from __future__ import annotations

import threading
import time
from dataclasses import dataclass, field
from typing import Callable

from .config import CoopConfig
from .smi import GpuReading


@dataclass(frozen=True)
class Snapshot:
    """A point-in-time view the controller acts on.

    ``ok=False`` (or a stale snapshot) means "unknown" and is handled conservatively by
    the controller: never resume, never falsely yield.
    """

    ok: bool
    taken_at: float                       # monotonic timestamp
    util_pct: int | None = None
    free_gb: float | None = None
    total_gb: float | None = None
    other_pids: frozenset[int] = field(default_factory=frozenset)
    competitor_present: bool = False      # foreign process AND free < headroom
    is_clear: bool = False                # no foreign process AND idle AND plenty of free VRAM
    threat_since: float | None = None     # monotonic time the competitor became continuously present
    clear_since: float | None = None      # monotonic time the GPU became continuously clear

    @staticmethod
    def unknown(now: float) -> "Snapshot":
        return Snapshot(ok=False, taken_at=now)


class GpuMonitor:
    """Polls a sensor function and maintains the published :class:`Snapshot`."""

    def __init__(
        self,
        cfg: CoopConfig,
        reader: Callable[[], GpuReading],
        *,
        monotonic: Callable[[], float] = time.monotonic,
    ) -> None:
        self._cfg = cfg
        self._reader = reader
        self._monotonic = monotonic
        # timers owned exclusively by the polling thread
        self._threat_since: float | None = None
        self._clear_since: float | None = None
        self._snap: Snapshot = Snapshot.unknown(monotonic())
        self._thread: threading.Thread | None = None
        self._stop = threading.Event()

    # ── published state (lock-free read) ──────────────────────────────────────
    @property
    def snapshot(self) -> Snapshot:
        return self._snap

    def fresh_snapshot(self) -> Snapshot:
        """Force an immediate synchronous poll and return it (used right before a reload)."""
        return self.poll_once()

    def reset_clear(self) -> None:
        """Restart the idle countdown (called after a failed/OOM reload)."""
        self._clear_since = None

    # ── the state update (pure-ish; unit-tested directly) ─────────────────────
    def poll_once(self, reading: GpuReading | None = None) -> Snapshot:
        """Take one reading, advance the debounce/idle timers, publish and return."""
        cfg = self._cfg
        now = self._monotonic()
        r = reading if reading is not None else self._reader()

        other = frozenset(r.pids - {cfg.our_pid}) if r.ok else frozenset()
        competitor = bool(r.ok and other and r.free_gb is not None and r.free_gb < cfg.vram_headroom_gb)
        is_clear = bool(
            r.ok
            and not other
            and r.util_pct is not None
            and r.util_pct < cfg.idle_util_pct
            and r.free_frac is not None
            and r.free_frac >= cfg.resume_free_frac
        )

        # monotonic "continuously true since" timers, reset the instant the condition breaks.
        # Use explicit None checks (not truthiness): a legitimate timestamp can be 0.0.
        if competitor:
            self._threat_since = now if self._threat_since is None else self._threat_since
        else:
            self._threat_since = None
        if is_clear:
            self._clear_since = now if self._clear_since is None else self._clear_since
        else:
            self._clear_since = None

        snap = Snapshot(
            ok=r.ok,
            taken_at=now,
            util_pct=r.util_pct,
            free_gb=r.free_gb,
            total_gb=r.total_gb,
            other_pids=other,
            competitor_present=competitor,
            is_clear=is_clear,
            threat_since=self._threat_since,
            clear_since=self._clear_since,
        )
        self._snap = snap            # atomic publish
        return snap

    # ── thread lifecycle ──────────────────────────────────────────────────────
    def start(self) -> None:
        if self._thread is not None:
            return
        self._stop.clear()
        self._thread = threading.Thread(target=self._loop, name="gpu-monitor", daemon=True)
        self._thread.start()

    def _loop(self) -> None:
        while not self._stop.is_set():
            try:
                self.poll_once()
            except Exception:
                # A failed poll must never kill the monitor; publish "unknown".
                self._threat_since = None
                self._clear_since = None
                self._snap = Snapshot.unknown(self._monotonic())
            self._stop.wait(self._cfg.poll_s)

    def stop(self) -> None:
        self._stop.set()
        t = self._thread
        if t is not None:
            t.join(timeout=self._cfg.poll_s + self._cfg.smi_timeout_s + 1.0)
        self._thread = None
