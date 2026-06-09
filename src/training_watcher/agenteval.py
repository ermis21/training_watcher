"""Optional hook: every N checkpoints, ask a coding-agent CLI to review the run.

Fully failure-isolated and non-blocking:

* runs in its own thread (``subprocess.run`` — exec, never a bare ``fork`` into a live
  CUDA context; never ``multiprocessing`` here);
* at most **one** agent invocation in flight — if the previous one is still running we
  skip this checkpoint rather than stack threads;
* every outcome (success / missing CLI / timeout / error) is logged, never raised into
  the training thread.
"""

from __future__ import annotations

import json
import logging
import os
import subprocess
import threading
from typing import Any, Mapping

from .config import CoopConfig


class AgentEval:
    """Fires ``cfg.agent_cmd`` headlessly every ``cfg.agent_every_n_ckpts`` checkpoints."""

    def __init__(self, cfg: CoopConfig, logger: logging.Logger) -> None:
        self._cfg = cfg
        self._log = logger
        self._ckpt_count = 0
        self._worker: threading.Thread | None = None
        self._cli_missing = False
        self._agent_log = cfg.agent_log_path or os.path.join(os.getcwd(), "training_watcher_agent.log")

    def maybe_run(self, step: int, metrics: Mapping[str, Any] | None = None) -> bool:
        """Count a checkpoint; on every Nth, launch the agent if enabled and idle."""
        if not self._cfg.agent_enabled or self._cli_missing:
            return False
        self._ckpt_count += 1
        if self._ckpt_count % self._cfg.agent_every_n_ckpts != 0:
            return False
        if self._worker is not None and self._worker.is_alive():
            self._log.info("agent-eval still running from a previous checkpoint — skipping")
            return False
        self._worker = threading.Thread(
            target=self._run, args=(step, dict(metrics or {})), name="agent-eval", daemon=True
        )
        self._worker.start()
        return True

    # ── internals ─────────────────────────────────────────────────────────────
    def _build_prompt(self, step: int, metrics: Mapping[str, Any]) -> str:
        cfg = self._cfg
        tail = self._log_tail(cfg.train_log_path, cfg.agent_log_tail_lines)
        return (
            f"{cfg.agent_prompt_prefix}\n\n"
            f"=== latest metrics (step {step}) ===\n{json.dumps(metrics, default=str)}\n\n"
            f"=== recent training log (last {cfg.agent_log_tail_lines} lines) ===\n{tail}\n"
        )

    @staticmethod
    def _log_tail(path: str | None, n: int) -> str:
        if not path or not os.path.isfile(path):
            return "(training log unavailable)"
        try:
            with open(path, "r", encoding="utf-8", errors="replace") as fh:
                return "".join(fh.readlines()[-n:])
        except Exception:
            return "(could not read training log)"

    def _run(self, step: int, metrics: Mapping[str, Any]) -> None:
        cfg = self._cfg
        prompt = self._build_prompt(step, metrics)
        cmd = [cfg.agent_cmd, cfg.agent_flag, prompt]
        try:
            res = subprocess.run(
                cmd, capture_output=True, text=True, timeout=cfg.agent_timeout_s, check=False
            )
            self._append(step, f"rc={res.returncode}\n{res.stdout}\n{res.stderr}")
            self._log.info("agent-eval at step %d → rc=%d (logged to %s)", step, res.returncode, self._agent_log)
        except FileNotFoundError:
            self._log.warning("agent-eval CLI '%s' not found — disabling further runs", cfg.agent_cmd)
            self._cli_missing = True
            self._append(step, f"ERROR: CLI '{cfg.agent_cmd}' not found")
        except subprocess.TimeoutExpired:
            self._log.warning("agent-eval at step %d timed out after %.0fs", step, cfg.agent_timeout_s)
            self._append(step, f"ERROR: timed out after {cfg.agent_timeout_s}s")
        except Exception as exc:  # never let the hook crash anything
            self._log.warning("agent-eval at step %d failed: %r", step, exc)
            self._append(step, f"ERROR: {exc!r}")

    def _append(self, step: int, body: str) -> None:
        try:
            with open(self._agent_log, "a", encoding="utf-8") as fh:
                fh.write(f"\n{'=' * 60}\n[step {step}]\n{body}\n")
        except Exception:
            pass
