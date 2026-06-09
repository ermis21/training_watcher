"""Ownership-window logic: are we inside a slot where we may keep the GPU?"""

from __future__ import annotations

from datetime import datetime

from .config import CoopConfig


def in_off_hours(hour: int, off_hours: tuple[int, int]) -> bool:
    """True if ``hour`` falls in the recurring window ``(start, end)``.

    The window is half-open ``[start, end)`` and wraps midnight when ``start > end``
    (e.g. ``(22, 8)`` covers 22:00–07:59).  ``start == end`` means an empty window.
    """
    s, e = off_hours
    if s == e:
        return False
    if s > e:                      # wraps midnight
        return hour >= s or hour < e
    return s <= hour < e


def in_owned_window(now: datetime, cfg: CoopConfig) -> bool:
    """True when we own the GPU and must not yield.

    Owned if inside an explicit reservation (``now < reservation_end``) or inside the
    recurring ``off_hours`` window.  Uses wall-clock time on purpose — reservations are
    wall-clock facts.  (Elapsed-time countdowns elsewhere use a monotonic clock.)
    """
    if cfg.reservation_end is not None and now < cfg.reservation_end:
        return True
    if cfg.off_hours is not None and in_off_hours(now.hour, cfg.off_hours):
        return True
    return False
