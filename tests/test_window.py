"""Ownership-window logic: in_off_hours + in_owned_window."""

from __future__ import annotations

from datetime import datetime, timedelta

import pytest

from training_watcher.config import CoopConfig
from training_watcher.window import in_off_hours, in_owned_window


# ── in_off_hours: wrapping window (22, 8) ─────────────────────────────────────
@pytest.mark.parametrize("hour", [22, 23, 0, 7])
def test_off_hours_wrap_inside(hour):
    assert in_off_hours(hour, (22, 8)) is True


@pytest.mark.parametrize("hour", [8, 9, 21])
def test_off_hours_wrap_outside(hour):
    assert in_off_hours(hour, (22, 8)) is False


# ── in_off_hours: non-wrapping window (8, 18) ─────────────────────────────────
@pytest.mark.parametrize("hour", [8, 12, 17])
def test_off_hours_nonwrap_inside(hour):
    assert in_off_hours(hour, (8, 18)) is True


@pytest.mark.parametrize("hour", [7, 18, 19, 0, 23])
def test_off_hours_nonwrap_outside(hour):
    assert in_off_hours(hour, (8, 18)) is False


# ── in_off_hours: equal endpoints means empty window ──────────────────────────
@pytest.mark.parametrize("hour", range(0, 24))
def test_off_hours_equal_always_false(hour):
    assert in_off_hours(hour, (8, 8)) is False


# ── CoopConfig rejects out-of-range hours ─────────────────────────────────────
@pytest.mark.parametrize("bad", [(24, 8), (22, 24), (-1, 8), (22, -1)])
def test_config_rejects_out_of_range_hours(bad):
    with pytest.raises(ValueError):
        CoopConfig(off_hours=bad)


def test_config_accepts_valid_hours():
    cfg = CoopConfig(off_hours=(22, 8))
    assert cfg.off_hours == (22, 8)


# ── in_owned_window ───────────────────────────────────────────────────────────
def test_owned_reservation_in_future():
    now = datetime(2026, 1, 1, 12, 0, 0)
    cfg = CoopConfig(off_hours=None, reservation_end=now + timedelta(hours=1))
    assert in_owned_window(now, cfg) is True


def test_owned_reservation_in_past():
    now = datetime(2026, 1, 1, 12, 0, 0)
    cfg = CoopConfig(off_hours=None, reservation_end=now - timedelta(hours=1))
    assert in_owned_window(now, cfg) is False


def test_owned_off_hours_only():
    # 23:00 is inside the wrapping (22,8) window.
    now = datetime(2026, 1, 1, 23, 0, 0)
    cfg = CoopConfig(off_hours=(22, 8), reservation_end=None)
    assert in_owned_window(now, cfg) is True


def test_not_owned_off_hours_outside():
    now = datetime(2026, 1, 1, 12, 0, 0)
    cfg = CoopConfig(off_hours=(22, 8), reservation_end=None)
    assert in_owned_window(now, cfg) is False


def test_not_owned_no_reservation_offhours_none():
    now = datetime(2026, 1, 1, 12, 0, 0)
    cfg = CoopConfig(off_hours=None, reservation_end=None)
    assert in_owned_window(now, cfg) is False


def test_reservation_wins_even_outside_offhours():
    now = datetime(2026, 1, 1, 12, 0, 0)
    cfg = CoopConfig(off_hours=(22, 8), reservation_end=now + timedelta(hours=2))
    assert in_owned_window(now, cfg) is True
