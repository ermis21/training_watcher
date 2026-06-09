"""CoopController decision predicates: _should_yield / _can_resume / staleness."""

from __future__ import annotations

from datetime import datetime, timedelta

from training_watcher.monitor import GpuMonitor

from conftest import OUR_PID


# Wall-clock "now" used for the owned-window check; far from any off_hours window.
NOON = datetime(2026, 1, 1, 12, 0, 0)


def _wire_monitor(controller, fake_clock, reader):
    """Attach a real GpuMonitor (same fake clock) to the controller for _is_stale."""
    mon = GpuMonitor(controller._cfg, reader, monotonic=fake_clock["now"])
    controller._monitor = mon
    return mon


def _fresh_competitor_snap(mon, reading):
    return mon.poll_once(reading=reading(competitor=True))


def _fresh_clear_snap(mon, reading):
    return mon.poll_once(reading=reading())


# ── _should_yield ─────────────────────────────────────────────────────────────
def test_should_yield_true_after_debounce(make_controller, fake_clock, reading):
    ctrl = make_controller(yield_debounce_s=60.0)
    ctrl._auto_yield_enabled = True
    mon = _wire_monitor(ctrl, fake_clock, lambda: reading(competitor=True))
    snap = _fresh_competitor_snap(mon, reading)
    # before debounce elapses -> no yield
    assert ctrl._should_yield(NOON, snap) is False
    fake_clock["advance"](60.0)
    # re-poll so the snapshot is fresh (not stale) but threat_since unchanged
    snap2 = _fresh_competitor_snap(mon, reading)
    assert snap2.threat_since == snap.threat_since
    assert ctrl._should_yield(NOON, snap2) is True


def test_should_yield_false_before_debounce(make_controller, fake_clock, reading):
    ctrl = make_controller(yield_debounce_s=60.0)
    ctrl._auto_yield_enabled = True
    mon = _wire_monitor(ctrl, fake_clock, lambda: reading(competitor=True))
    snap = _fresh_competitor_snap(mon, reading)
    fake_clock["advance"](59.0)
    snap2 = _fresh_competitor_snap(mon, reading)
    assert ctrl._should_yield(NOON, snap2) is False


def test_should_yield_false_inside_reservation(make_controller, fake_clock, reading):
    ctrl = make_controller(yield_debounce_s=0.0, reservation_end=NOON + timedelta(hours=1))
    ctrl._auto_yield_enabled = True
    mon = _wire_monitor(ctrl, fake_clock, lambda: reading(competitor=True))
    snap = _fresh_competitor_snap(mon, reading)
    fake_clock["advance"](100.0)
    snap2 = _fresh_competitor_snap(mon, reading)
    assert ctrl._should_yield(NOON, snap2) is False   # we own the GPU


def test_should_yield_false_inside_off_hours(make_controller, fake_clock, reading):
    ctrl = make_controller(yield_debounce_s=0.0, off_hours=(22, 8))
    ctrl._auto_yield_enabled = True
    mon = _wire_monitor(ctrl, fake_clock, lambda: reading(competitor=True))
    snap = _fresh_competitor_snap(mon, reading)
    fake_clock["advance"](100.0)
    snap2 = _fresh_competitor_snap(mon, reading)
    night = datetime(2026, 1, 1, 23, 0, 0)
    assert ctrl._should_yield(night, snap2) is False
    # outside the window the same competitor *would* yield
    assert ctrl._should_yield(NOON, snap2) is True


def test_should_yield_false_when_disabled(make_controller, fake_clock, reading):
    ctrl = make_controller(yield_debounce_s=0.0)
    ctrl._auto_yield_enabled = False
    mon = _wire_monitor(ctrl, fake_clock, lambda: reading(competitor=True))
    snap = _fresh_competitor_snap(mon, reading)
    fake_clock["advance"](100.0)
    snap2 = _fresh_competitor_snap(mon, reading)
    assert ctrl._should_yield(NOON, snap2) is False


def test_should_yield_false_on_unknown_snapshot(make_controller, fake_clock, reading):
    ctrl = make_controller(yield_debounce_s=0.0)
    ctrl._auto_yield_enabled = True
    mon = _wire_monitor(ctrl, fake_clock, lambda: reading(ok=False))
    snap = mon.poll_once(reading=reading(ok=False))
    assert ctrl._should_yield(NOON, snap) is False


def test_should_yield_false_on_stale_snapshot(make_controller, fake_clock, reading):
    ctrl = make_controller(yield_debounce_s=0.0, stale_after_s=60.0)
    ctrl._auto_yield_enabled = True
    mon = _wire_monitor(ctrl, fake_clock, lambda: reading(competitor=True))
    snap = _fresh_competitor_snap(mon, reading)
    # advance well past stale_after_s WITHOUT re-polling -> snapshot is stale
    fake_clock["advance"](120.0)
    assert ctrl._is_stale(snap) is True
    assert ctrl._should_yield(NOON, snap) is False


def test_should_yield_false_no_competitor(make_controller, fake_clock, reading):
    ctrl = make_controller(yield_debounce_s=0.0)
    ctrl._auto_yield_enabled = True
    mon = _wire_monitor(ctrl, fake_clock, lambda: reading())
    snap = _fresh_clear_snap(mon, reading)
    assert ctrl._should_yield(NOON, snap) is False


# ── _can_resume ───────────────────────────────────────────────────────────────
def test_can_resume_true_after_idle(make_controller, fake_clock, reading):
    ctrl = make_controller(resume_idle_s=600.0)
    mon = _wire_monitor(ctrl, fake_clock, lambda: reading())
    snap = _fresh_clear_snap(mon, reading)
    assert ctrl._can_resume(snap) is False    # not idle long enough yet
    fake_clock["advance"](600.0)
    snap2 = _fresh_clear_snap(mon, reading)
    assert snap2.clear_since == snap.clear_since
    assert ctrl._can_resume(snap2) is True


def test_can_resume_false_before_idle(make_controller, fake_clock, reading):
    ctrl = make_controller(resume_idle_s=600.0)
    mon = _wire_monitor(ctrl, fake_clock, lambda: reading())
    snap = _fresh_clear_snap(mon, reading)
    fake_clock["advance"](599.0)
    snap2 = _fresh_clear_snap(mon, reading)
    assert ctrl._can_resume(snap2) is False


def test_can_resume_false_on_unknown(make_controller, fake_clock, reading):
    ctrl = make_controller(resume_idle_s=0.0)
    mon = _wire_monitor(ctrl, fake_clock, lambda: reading(ok=False))
    snap = mon.poll_once(reading=reading(ok=False))
    assert ctrl._can_resume(snap) is False


def test_can_resume_false_on_stale(make_controller, fake_clock, reading):
    ctrl = make_controller(resume_idle_s=0.0, stale_after_s=60.0)
    mon = _wire_monitor(ctrl, fake_clock, lambda: reading())
    snap = _fresh_clear_snap(mon, reading)
    fake_clock["advance"](120.0)    # stale, no re-poll
    assert ctrl._is_stale(snap) is True
    assert ctrl._can_resume(snap) is False


def test_can_resume_false_if_clear_since_reset(make_controller, fake_clock, reading):
    ctrl = make_controller(resume_idle_s=10.0)
    mon = _wire_monitor(ctrl, fake_clock, lambda: reading())
    _fresh_clear_snap(mon, reading)
    fake_clock["advance"](100.0)
    # reset_clear() restarts the idle countdown -> clear_since becomes "now"
    mon.reset_clear()
    snap = _fresh_clear_snap(mon, reading)
    assert snap.clear_since == snap.taken_at
    assert ctrl._can_resume(snap) is False    # countdown restarted


def test_can_resume_false_when_competitor_present(make_controller, fake_clock, reading):
    ctrl = make_controller(resume_idle_s=0.0)
    mon = _wire_monitor(ctrl, fake_clock, lambda: reading(competitor=True))
    snap = _fresh_competitor_snap(mon, reading)
    assert ctrl._can_resume(snap) is False
