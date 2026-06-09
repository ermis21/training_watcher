"""GpuMonitor.poll_once: competitor/clear detection and the continuity timers."""

from __future__ import annotations

from training_watcher.smi import GpuReading

from conftest import OUR_PID


def test_competitor_sets_present_and_threat_since(make_monitor, fake_clock, reading):
    mon = make_monitor()
    snap = mon.poll_once(reading=reading(competitor=True))
    assert snap.ok is True
    assert snap.competitor_present is True
    assert snap.other_pids == frozenset({99999})
    assert snap.threat_since == fake_clock["t"]
    assert snap.is_clear is False


def test_threat_since_persists_across_polls(make_monitor, fake_clock, reading):
    mon = make_monitor()
    first = mon.poll_once(reading=reading(competitor=True))
    start = first.threat_since
    fake_clock["advance"](30.0)
    second = mon.poll_once(reading=reading(competitor=True))
    # continuity timer is the *start* time, unchanged while the competitor persists
    assert second.threat_since == start
    assert second.competitor_present is True


def test_threat_since_resets_when_competitor_leaves(make_monitor, fake_clock, reading):
    mon = make_monitor()
    mon.poll_once(reading=reading(competitor=True))
    fake_clock["advance"](30.0)
    snap = mon.poll_once(reading=reading())   # back to clean/idle
    assert snap.competitor_present is False
    assert snap.threat_since is None


def test_clear_sets_is_clear_and_clear_since(make_monitor, fake_clock, reading):
    mon = make_monitor()
    snap = mon.poll_once(reading=reading(util_pct=0, free_gb=45.0, total_gb=46.0))
    assert snap.is_clear is True
    assert snap.clear_since == fake_clock["t"]
    assert snap.competitor_present is False


def test_clear_since_persists_then_resets_on_competitor(make_monitor, fake_clock, reading):
    mon = make_monitor()
    first = mon.poll_once(reading=reading())
    start = first.clear_since
    fake_clock["advance"](100.0)
    second = mon.poll_once(reading=reading())
    assert second.clear_since == start
    # a competitor breaks the clear streak
    snap = mon.poll_once(reading=reading(competitor=True))
    assert snap.is_clear is False
    assert snap.clear_since is None


def test_clear_resets_on_high_util(make_monitor, reading):
    mon = make_monitor(idle_util_pct=10)
    mon.poll_once(reading=reading())
    snap = mon.poll_once(reading=reading(util_pct=50))   # busy but no foreign pid
    assert snap.is_clear is False
    assert snap.clear_since is None


def test_clear_resets_on_low_free_frac(make_monitor, reading):
    mon = make_monitor(resume_free_frac=0.90)
    mon.poll_once(reading=reading())
    # free_frac = 20/46 < 0.90 -> not clear
    snap = mon.poll_once(reading=reading(util_pct=0, free_gb=20.0, total_gb=46.0))
    assert snap.is_clear is False
    assert snap.clear_since is None


def test_clear_resets_on_failed_read(make_monitor, reading):
    mon = make_monitor()
    mon.poll_once(reading=reading())
    snap = mon.poll_once(reading=GpuReading(ok=False))
    assert snap.ok is False
    assert snap.is_clear is False
    assert snap.clear_since is None


def test_own_pid_only_is_empty_others(make_monitor, reading):
    mon = make_monitor()
    # only our pid present -> other_pids empty, not a competitor, and (idle) clear
    snap = mon.poll_once(reading=reading(pids=(OUR_PID,)))
    assert snap.other_pids == frozenset()
    assert snap.competitor_present is False
    assert snap.is_clear is True


def test_own_pid_only_with_low_free_is_not_competitor(make_monitor, reading):
    mon = make_monitor()
    # only our pid but free < headroom -> still NOT a competitor (it's us)
    snap = mon.poll_once(reading=reading(pids=(OUR_PID,), free_gb=2.0, util_pct=90))
    assert snap.competitor_present is False
    assert snap.other_pids == frozenset()


def test_reset_clear_nulls_clear_since(make_monitor, reading):
    mon = make_monitor()
    snap = mon.poll_once(reading=reading())
    assert snap.clear_since is not None
    mon.reset_clear()
    # internal timer cleared; next clear poll restarts it fresh
    snap2 = mon.poll_once(reading=reading())
    assert snap2.clear_since == snap2.taken_at


def test_snapshot_atomic_publish(make_monitor, reading):
    mon = make_monitor()
    returned = mon.poll_once(reading=reading(competitor=True))
    # .snapshot returns exactly the last poll result
    assert mon.snapshot is returned


def test_failed_read_when_competitor_present_field(make_monitor, reading):
    # competitor requires r.ok; a failed read can never be competitor_present
    mon = make_monitor()
    snap = mon.poll_once(reading=GpuReading(ok=False, pids=frozenset({99999})))
    assert snap.competitor_present is False
    assert snap.other_pids == frozenset()
