"""checkpoint_cb arity handling: register() detects zero-arg vs (step, epoch) and
_do_pause calls the matching form. No torch/GPU — offload/reload are monkeypatched.

The pause-moment checkpoint is the kill-safety net, so the two-arg form must receive the
exact step/epoch passed to guard()/_do_pause().
"""

from __future__ import annotations

import training_watcher.controller as ctrl_mod
from training_watcher.controller import _cb_wants_args


# ── pure arity detection ──────────────────────────────────────────────────────
def test_cb_wants_args_zero_arg():
    assert _cb_wants_args(lambda: None) is False


def test_cb_wants_args_two_positional():
    assert _cb_wants_args(lambda step, epoch: None) is True


def test_cb_wants_args_var_positional():
    assert _cb_wants_args(lambda *a: None) is True


def test_cb_wants_args_one_positional_treated_as_zero():
    # A single-arg callback is ambiguous; we do NOT pass args (avoids surprising a
    # closure that happens to take one bound default). Documented zero-arg default.
    assert _cb_wants_args(lambda only: None) is False


def test_cb_wants_args_keyword_only_is_zero():
    assert _cb_wants_args(lambda *, foo=1: None) is False


def test_cb_wants_args_uninspectable_falls_back_to_zero():
    # A callable whose signature() raises ValueError must default to zero-arg, not crash.
    class NoSig:
        def __call__(self):  # pragma: no cover - never actually called
            pass

    obj = NoSig()
    import inspect

    def fake_signature(_):
        raise ValueError("no signature for you")

    real = inspect.signature
    inspect.signature = fake_signature
    try:
        assert _cb_wants_args(obj) is False  # must not raise
    finally:
        inspect.signature = real


def test_register_records_arity(make_controller):
    c0 = make_controller()
    c0.register(model=object(), checkpoint_cb=lambda: None)
    assert c0._checkpoint_cb_wants_args is False

    c2 = make_controller()
    c2.register(model=object(), checkpoint_cb=lambda step, epoch: None)
    assert c2._checkpoint_cb_wants_args is True


# ── _do_pause dispatch (offload/reload stubbed; resumes immediately) ──────────
def _stub_offload(monkeypatch, controller):
    """Make _do_pause torch-free and resume on the first wait."""
    monkeypatch.setattr(ctrl_mod, "offload_to_cpu", lambda m, o: (None, None, {}))
    monkeypatch.setattr(ctrl_mod, "reload_to_device", lambda m, o, d, r: None)
    monkeypatch.setattr(ctrl_mod, "empty_cache", lambda d=None: None)
    # _do_pause loops until _can_resume; force an immediate resume.
    monkeypatch.setattr(controller, "_can_resume", lambda snap: True)


class _FakeMonitor:
    def __init__(self, snap):
        self._snap = snap

    @property
    def snapshot(self):
        return self._snap

    def fresh_snapshot(self):
        return self._snap

    def reset_clear(self):
        pass


def _wire_pause(controller, fake_clock, reading):
    snap = _FakeMonitor(
        # any snapshot object; _can_resume is stubbed so its contents don't matter
        type("S", (), {"other_pids": frozenset(), "free_gb": 40.0})()
    )
    controller._monitor = snap
    return snap


def test_do_pause_calls_zero_arg_cb(make_controller, fake_clock, reading, monkeypatch):
    ctrl = make_controller()
    seen = {"called": 0}
    ctrl.register(model=object(), checkpoint_cb=lambda: seen.__setitem__("called", seen["called"] + 1))
    _wire_pause(ctrl, fake_clock, reading)
    _stub_offload(monkeypatch, ctrl)

    ctrl._do_pause(step=123, epoch=4, optimizer=object())
    assert seen["called"] == 1


def test_do_pause_calls_two_arg_cb_with_step_epoch(make_controller, fake_clock, reading, monkeypatch):
    ctrl = make_controller()
    captured = {}
    ctrl.register(
        model=object(),
        checkpoint_cb=lambda step, epoch: captured.update(step=step, epoch=epoch),
    )
    _wire_pause(ctrl, fake_clock, reading)
    _stub_offload(monkeypatch, ctrl)

    ctrl._do_pause(step=777, epoch=9, optimizer=object())
    assert captured == {"step": 777, "epoch": 9}


def test_do_pause_two_arg_cb_exception_does_not_abort_offload(make_controller, fake_clock, reading, monkeypatch):
    ctrl = make_controller()
    offloaded = {"n": 0}

    def boom(step, epoch):
        raise RuntimeError("checkpoint blew up")

    ctrl.register(model=object(), checkpoint_cb=boom)
    _wire_pause(ctrl, fake_clock, reading)
    monkeypatch.setattr(
        ctrl_mod, "offload_to_cpu",
        lambda m, o: (offloaded.__setitem__("n", offloaded["n"] + 1) or (None, None, {})),
    )
    monkeypatch.setattr(ctrl_mod, "reload_to_device", lambda m, o, d, r: None)
    monkeypatch.setattr(ctrl_mod, "empty_cache", lambda d=None: None)
    monkeypatch.setattr(ctrl, "_can_resume", lambda snap: True)

    ctrl._do_pause(step=1, epoch=0, optimizer=object())   # must not raise
    assert offloaded["n"] == 1                             # offload still happened


def test_do_pause_logs_zero_free_gb_not_minus_one(make_controller, fake_clock, reading, monkeypatch):
    # free_gb==0.0 (a full GPU) must log as 0.0, not -1.0 ("unknown"): is-None vs truthiness.
    import logging

    ctrl = make_controller()
    ctrl.register(model=object(), checkpoint_cb=lambda: None)
    mon = type("S", (), {"other_pids": frozenset({99999}), "free_gb": 0.0})()
    ctrl._monitor = _FakeMonitor(mon)
    _stub_offload(monkeypatch, ctrl)

    records = []

    class _Cap(logging.Handler):
        def emit(self, record):
            records.append(record.getMessage())

    cap = _Cap(level=logging.WARNING)
    ctrl._log.addHandler(cap)
    try:
        ctrl._do_pause(step=5, epoch=0, optimizer=object())
    finally:
        ctrl._log.removeHandler(cap)

    pausing = [m for m in records if "pausing at step" in m]
    assert pausing and "free=0.0GB" in pausing[0]
    assert "free=-1.0GB" not in pausing[0]
