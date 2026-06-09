"""Shared fixtures for the training_watcher test suite.

Everything here is pure-Python and deterministic: a fake monotonic clock the tests
advance by hand, a ``GpuReading`` factory, and helpers to wire a ``GpuMonitor`` /
``CoopController`` with an injected reader + clock.  No real GPU, no torch, no sleeps.
"""

from __future__ import annotations

import logging
from typing import Callable

import pytest

from training_watcher.config import CoopConfig
from training_watcher.controller import CoopController
from training_watcher.monitor import GpuMonitor
from training_watcher.smi import GpuReading

# A sentinel "our own" PID used across the suite.  It is deliberately not a real PID so
# tests are independent of the actual process id; configs pin ``our_pid`` to this.
OUR_PID = 424242


@pytest.fixture
def fake_clock() -> dict:
    """A mutable monotonic clock.

    Returns a dict with::

        clock["t"]      -> current value (mutate to advance)
        clock["now"]()  -> a monotonic getter to pass as ``monotonic=`` / inject
        clock["advance"](dt) -> advance the clock by ``dt`` seconds
    """
    state = {"t": 1000.0}

    def now() -> float:
        return state["t"]

    def advance(dt: float) -> None:
        state["t"] += dt

    state["now"] = now
    state["advance"] = advance
    return state


@pytest.fixture
def reading() -> Callable[..., GpuReading]:
    """Factory for ``GpuReading`` with convenient defaults.

    ``reading()``                       -> a clean, idle GPU holding only OUR_PID.
    ``reading(competitor=True)``        -> a foreign PID + low free VRAM.
    Any field can be overridden by keyword (``ok``, ``util_pct``, ``free_gb``,
    ``total_gb``, ``pids``).
    """

    def make(
        *,
        ok: bool = True,
        util_pct: int = 0,
        free_gb: float = 44.0,
        total_gb: float = 46.0,
        pids=(OUR_PID,),
        competitor: bool = False,
    ) -> GpuReading:
        if competitor:
            pids = (OUR_PID, 99999)
            free_gb = 2.0
            util_pct = 95
        if not ok:
            return GpuReading(ok=False)
        return GpuReading(
            ok=True,
            util_pct=util_pct,
            free_gb=free_gb,
            total_gb=total_gb,
            pids=frozenset(pids),
        )

    return make


@pytest.fixture
def make_config() -> Callable[..., CoopConfig]:
    """Build a CoopConfig pinned to OUR_PID with fast (test-sized) timers by default."""

    def make(**overrides) -> CoopConfig:
        base = dict(
            our_pid=OUR_PID,
            off_hours=None,            # default: no recurring window (tests opt in)
            reservation_end=None,
            vram_headroom_gb=10.0,
            yield_debounce_s=60.0,
            resume_idle_s=600.0,
            idle_util_pct=10,
            resume_free_frac=0.90,
            poll_s=20.0,
            stale_after_s=60.0,
        )
        base.update(overrides)
        return CoopConfig(**base)

    return make


@pytest.fixture
def make_monitor(fake_clock, make_config) -> Callable[..., GpuMonitor]:
    """Build a GpuMonitor wired to the fake clock; reader is a no-op placeholder.

    Tests drive it via ``monitor.poll_once(reading=...)`` so the reader is unused.
    """

    def make(reader=None, **cfg_overrides) -> GpuMonitor:
        cfg = make_config(**cfg_overrides)
        reader = reader or (lambda: GpuReading(ok=False))
        return GpuMonitor(cfg, reader, monotonic=fake_clock["now"])

    return make


@pytest.fixture
def make_controller(fake_clock, make_config) -> Callable[..., CoopController]:
    """Build a CoopController with an injected reader + fake clock, no thread started.

    A silent logger keeps test output clean.
    """

    def make(reader=None, **cfg_overrides) -> CoopController:
        cfg = make_config(**cfg_overrides)
        log = logging.getLogger("training_watcher.test")
        log.addHandler(logging.NullHandler())
        log.propagate = False
        reader = reader or (lambda: GpuReading(ok=False))
        return CoopController(
            cfg,
            device="cpu",
            logger=log,
            reader=reader,
            monotonic=fake_clock["now"],
        )

    return make
