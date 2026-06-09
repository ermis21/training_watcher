"""auto_stop: SIGTERM/SIGINT handler installs, chains to the previous handler, restores.

Runs the handler in-process (we never actually kill the test process) and must run on
the main thread (signal.signal requirement).
"""

from __future__ import annotations

import logging
import signal

import pytest

from training_watcher.controller import auto_stop


class DummyController:
    """Minimal stand-in exposing the surface auto_stop touches."""

    def __init__(self):
        self.stopped = False
        self._log = logging.getLogger("training_watcher.test")
        self._log.addHandler(logging.NullHandler())

    def request_stop(self):
        self.stopped = True


def test_auto_stop_installs_and_calls_request_stop():
    prev = signal.getsignal(signal.SIGTERM)
    try:
        ctrl = DummyController()
        auto_stop(ctrl, signals=(signal.SIGTERM,))
        handler = signal.getsignal(signal.SIGTERM)
        assert handler is not prev          # a new handler is installed
        assert callable(handler)
        # invoke the handler directly (no real signal raised)
        handler(signal.SIGTERM, None)
        assert ctrl.stopped is True
    finally:
        signal.signal(signal.SIGTERM, prev)


def test_auto_stop_chains_to_previous_handler():
    sentinel = {"prev_called": False}

    def prev_handler(signum, frame):
        sentinel["prev_called"] = True

    original = signal.getsignal(signal.SIGTERM)
    signal.signal(signal.SIGTERM, prev_handler)
    try:
        ctrl = DummyController()
        auto_stop(ctrl, signals=(signal.SIGTERM,))
        handler = signal.getsignal(signal.SIGTERM)
        handler(signal.SIGTERM, None)
        # chained to the pre-existing handler AND requested stop
        assert sentinel["prev_called"] is True
        assert ctrl.stopped is True
    finally:
        signal.signal(signal.SIGTERM, original)


def test_auto_stop_context_manager_restores():
    prev = signal.getsignal(signal.SIGTERM)
    ctrl = DummyController()
    with auto_stop(ctrl, signals=(signal.SIGTERM,)) as guard:
        assert guard is not None
        installed = signal.getsignal(signal.SIGTERM)
        assert installed is not prev
    # on exit the previous handler is restored
    assert signal.getsignal(signal.SIGTERM) == prev


def test_auto_stop_chained_prev_exception_isolated():
    # if the previous handler raises, request_stop must still run
    def boom(signum, frame):
        raise RuntimeError("trainer handler blew up")

    original = signal.getsignal(signal.SIGTERM)
    signal.signal(signal.SIGTERM, boom)
    try:
        ctrl = DummyController()
        auto_stop(ctrl, signals=(signal.SIGTERM,))
        handler = signal.getsignal(signal.SIGTERM)
        handler(signal.SIGTERM, None)   # should not raise
        assert ctrl.stopped is True
    finally:
        signal.signal(signal.SIGTERM, original)
