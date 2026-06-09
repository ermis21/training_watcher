"""CoopController._self_check: own-PID visibility decides auto-yield (fail-closed)."""

from __future__ import annotations

from training_watcher.smi import GpuReading

from conftest import OUR_PID


def test_self_check_sees_own_pid_enables(make_controller):
    reader = lambda: GpuReading(ok=True, util_pct=0, free_gb=44.0, total_gb=46.0,
                                pids=frozenset({OUR_PID, 111}))
    ctrl = make_controller(reader=reader, fail_closed=True)
    ctrl._self_check()
    assert ctrl.auto_yield_enabled is True


def test_self_check_missing_pid_fail_closed_disables(make_controller):
    # our PID NOT in the reading -> with fail_closed, auto-yield disabled + a warning
    reader = lambda: GpuReading(ok=True, util_pct=0, free_gb=44.0, total_gb=46.0,
                                pids=frozenset({111, 222}))
    ctrl = make_controller(reader=reader, fail_closed=True)

    # The controller logs to its own (non-propagating) logger; attach a capturing
    # handler directly so we can assert the warning was emitted.
    import logging

    records = []

    class _Cap(logging.Handler):
        def emit(self, record):
            records.append(record)

    cap = _Cap(level=logging.WARNING)
    ctrl._log.addHandler(cap)
    try:
        ctrl._self_check()
    finally:
        ctrl._log.removeHandler(cap)

    assert ctrl.auto_yield_enabled is False
    assert any(rec.levelno == logging.WARNING and "self-check FAILED" in rec.getMessage()
               for rec in records)


def test_self_check_missing_pid_fail_open_stays_enabled(make_controller):
    reader = lambda: GpuReading(ok=True, util_pct=0, free_gb=44.0, total_gb=46.0,
                                pids=frozenset({111}))
    ctrl = make_controller(reader=reader, fail_closed=False)
    ctrl._self_check()
    assert ctrl.auto_yield_enabled is True


def test_self_check_failed_read_fail_closed_disables(make_controller):
    reader = lambda: GpuReading(ok=False)
    ctrl = make_controller(reader=reader, fail_closed=True)
    ctrl._self_check()
    assert ctrl.auto_yield_enabled is False


def test_self_check_failed_read_fail_open_stays_enabled(make_controller):
    reader = lambda: GpuReading(ok=False)
    ctrl = make_controller(reader=reader, fail_closed=False)
    ctrl._self_check()
    assert ctrl.auto_yield_enabled is True
