"""AgentEval: fires every Nth checkpoint, failure-isolated, <=1 concurrent."""

from __future__ import annotations

import logging
import os
import subprocess
import time

import pytest

from training_watcher.agenteval import AgentEval
from training_watcher.config import CoopConfig


def _logger():
    log = logging.getLogger("training_watcher.test.agent")
    log.addHandler(logging.NullHandler())
    log.propagate = False
    return log


def _wait_worker_done(agent, timeout=3.0):
    """Block (bounded) until the agent worker thread finishes."""
    deadline = time.monotonic() + timeout
    w = agent._worker
    if w is None:
        return
    while w.is_alive() and time.monotonic() < deadline:
        time.sleep(0.01)
    w.join(timeout=0.5)


def _wait_for_file(path, timeout=3.0):
    deadline = time.monotonic() + timeout
    while not os.path.exists(path) and time.monotonic() < deadline:
        time.sleep(0.01)
    return os.path.exists(path)


def test_fires_only_on_nth_and_logs(tmp_path, monkeypatch):
    log_path = str(tmp_path / "agent.log")
    cfg = CoopConfig(agent_enabled=True, agent_every_n_ckpts=3, agent_log_path=log_path)

    class FakeCompleted:
        returncode = 0
        stdout = "looks healthy"
        stderr = ""

    calls = {"n": 0}

    def fake_run(cmd, **kwargs):
        calls["n"] += 1
        return FakeCompleted()

    monkeypatch.setattr("training_watcher.agenteval.subprocess.run", fake_run)

    agent = AgentEval(cfg, _logger())
    assert agent.maybe_run(1) is False     # count 1
    assert agent.maybe_run(2) is False     # count 2
    fired = agent.maybe_run(3)             # count 3 -> fires
    assert fired is True
    _wait_worker_done(agent)
    assert _wait_for_file(log_path)
    body = (tmp_path / "agent.log").read_text()
    assert "rc=0" in body
    assert "looks healthy" in body
    assert calls["n"] == 1


def test_disabled_never_fires(tmp_path, monkeypatch):
    cfg = CoopConfig(agent_enabled=False, agent_every_n_ckpts=1,
                     agent_log_path=str(tmp_path / "a.log"))

    def fake_run(cmd, **kwargs):  # pragma: no cover - must not be called
        raise AssertionError("should not run when disabled")

    monkeypatch.setattr("training_watcher.agenteval.subprocess.run", fake_run)
    agent = AgentEval(cfg, _logger())
    for s in range(5):
        assert agent.maybe_run(s) is False


def test_cli_missing_latches(tmp_path, monkeypatch):
    log_path = str(tmp_path / "agent.log")
    cfg = CoopConfig(agent_enabled=True, agent_every_n_ckpts=1, agent_log_path=log_path)

    def fake_run(cmd, **kwargs):
        raise FileNotFoundError("no such CLI")

    monkeypatch.setattr("training_watcher.agenteval.subprocess.run", fake_run)
    agent = AgentEval(cfg, _logger())

    # first call fires a worker that hits FileNotFoundError (caller never sees it)
    assert agent.maybe_run(1) is True
    _wait_worker_done(agent)
    assert agent._cli_missing is True
    assert _wait_for_file(log_path)
    assert "not found" in (tmp_path / "agent.log").read_text()
    # subsequent calls return False (latched off)
    assert agent.maybe_run(2) is False
    assert agent.maybe_run(3) is False


def test_timeout_isolated(tmp_path, monkeypatch):
    log_path = str(tmp_path / "agent.log")
    cfg = CoopConfig(agent_enabled=True, agent_every_n_ckpts=1, agent_log_path=log_path)

    def fake_run(cmd, **kwargs):
        raise subprocess.TimeoutExpired("x", 1)

    monkeypatch.setattr("training_watcher.agenteval.subprocess.run", fake_run)
    agent = AgentEval(cfg, _logger())
    assert agent.maybe_run(1) is True     # caller never sees the timeout
    _wait_worker_done(agent)
    assert _wait_for_file(log_path)
    assert "timed out" in (tmp_path / "agent.log").read_text()
    # timeout does NOT latch off -> can still fire again
    assert agent.maybe_run(2) is True
    _wait_worker_done(agent)


def test_generic_exception_isolated(tmp_path, monkeypatch):
    log_path = str(tmp_path / "agent.log")
    cfg = CoopConfig(agent_enabled=True, agent_every_n_ckpts=1, agent_log_path=log_path)

    def fake_run(cmd, **kwargs):
        raise RuntimeError("kaboom")

    monkeypatch.setattr("training_watcher.agenteval.subprocess.run", fake_run)
    agent = AgentEval(cfg, _logger())
    assert agent.maybe_run(1) is True
    _wait_worker_done(agent)
    assert _wait_for_file(log_path)
    assert "ERROR" in (tmp_path / "agent.log").read_text()


def test_at_most_one_concurrent(tmp_path, monkeypatch):
    log_path = str(tmp_path / "agent.log")
    cfg = CoopConfig(agent_enabled=True, agent_every_n_ckpts=1, agent_log_path=log_path)

    import threading
    release = threading.Event()

    class FakeCompleted:
        returncode = 0
        stdout = ""
        stderr = ""

    def fake_run(cmd, **kwargs):
        release.wait(2.0)   # hold the worker "alive"
        return FakeCompleted()

    monkeypatch.setattr("training_watcher.agenteval.subprocess.run", fake_run)
    agent = AgentEval(cfg, _logger())

    assert agent.maybe_run(1) is True
    # worker is alive (blocked in fake_run) -> next call is skipped
    assert agent._worker.is_alive()
    assert agent.maybe_run(2) is False
    release.set()
    _wait_worker_done(agent)
    # once the worker finishes, a new run can fire again
    assert agent.maybe_run(3) is True
    release.set()
    _wait_worker_done(agent)
