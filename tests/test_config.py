"""CoopConfig: our_pid resolves to the real PID (not a sentinel), plus validation."""

from __future__ import annotations

import os

import pytest

from training_watcher.config import CoopConfig


def test_our_pid_defaults_to_real_pid():
    # default_factory=os.getpid runs at instance creation; it is NOT a sentinel.
    assert CoopConfig().our_pid == os.getpid()


def test_our_pid_is_overridable():
    assert CoopConfig(our_pid=4242).our_pid == 4242


def test_validation_off_hours_range():
    with pytest.raises(ValueError):
        CoopConfig(off_hours=(22, 25))


def test_validation_positive_headroom():
    with pytest.raises(ValueError):
        CoopConfig(vram_headroom_gb=0)


def test_validation_resume_free_frac_bounds():
    with pytest.raises(ValueError):
        CoopConfig(resume_free_frac=1.5)


def test_validation_positive_poll():
    with pytest.raises(ValueError):
        CoopConfig(poll_s=0)
