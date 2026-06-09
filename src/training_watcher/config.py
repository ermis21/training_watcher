"""Configuration for cooperative GPU sharing.

A single frozen dataclass holds the whole policy.  Defaults match the shared-machine
policy ported from the old external ``gpu_watcher.py`` (poll 20 s, 10 GB headroom,
60 s yield debounce) but everything is overridable.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from datetime import datetime


@dataclass(frozen=True)
class CoopConfig:
    """Policy for :class:`~training_watcher.controller.CoopController`.

    Ownership window
    ----------------
    We *own* the GPU (never yield) when either is true:

    * ``reservation_end`` is set and ``now < reservation_end`` (an explicit slot), or
    * ``off_hours`` is set and the current hour falls inside it (a recurring slot;
      the tuple ``(start, end)`` wraps midnight when ``start > end``, e.g. ``(22, 8)``).

    Outside the owned window we are an opportunistic guest: we keep training while the
    GPU is otherwise free, and we yield (pause) only when a real competitor shows up.

    Yield / resume
    --------------
    A competitor is *present* when another process holds the GPU **and** free VRAM has
    dropped below ``vram_headroom_gb``.  We pause only after the competitor persists for
    ``yield_debounce_s`` (filters transient inference blips).  We resume only after the
    GPU has been continuously empty (no foreign process) **and** idle (low utilization,
    high free VRAM) for ``resume_idle_s``.
    """

    # ── ownership window ──────────────────────────────────────────────────────
    reservation_end: datetime | None = None
    off_hours: tuple[int, int] | None = (22, 8)

    # ── competitor / yield ────────────────────────────────────────────────────
    vram_headroom_gb: float = 10.0       # floor 8 GB on the shared box; never lower
    yield_debounce_s: float = 60.0       # competitor must persist this long before we pause

    # ── resume ────────────────────────────────────────────────────────────────
    resume_idle_s: float = 600.0         # GPU empty+idle this long before resuming (the ">10 min")
    idle_util_pct: int = 10              # utilization below this counts as "inactive"
    resume_free_frac: float = 0.90       # require free VRAM >= this fraction of total to resume

    # ── sensing ───────────────────────────────────────────────────────────────
    poll_s: float = 20.0                 # monitor poll cadence
    smi_timeout_s: float = 10.0          # per nvidia-smi call; timeout => "unknown", never hangs
    stale_after_s: float = 60.0          # a snapshot older than this is treated as "unknown"
    # Resolved to the real PID at instance-creation time (NOT a sentinel): the
    # default_factory runs os.getpid() in __init__, so CoopConfig().our_pid == os.getpid().
    # Overridable only for tests; production code should leave it at the default.
    our_pid: int = field(default_factory=os.getpid)
    gpu_index: int | None = None         # physical GPU to watch; None => resolve from the torch device

    # ── failure posture ───────────────────────────────────────────────────────
    fail_closed: bool = True             # if we can't see our own PID, disable auto-yield (never pause)

    # ── agent-eval hook ───────────────────────────────────────────────────────
    agent_enabled: bool = False
    agent_every_n_ckpts: int = 10
    agent_cmd: str = "pi"                # CLI to invoke (e.g. "pi" or "claude")
    agent_flag: str = "-p"              # headless/print flag
    agent_timeout_s: float = 120.0
    agent_log_path: str | None = None    # default: ./training_watcher_agent.log
    agent_log_tail_lines: int = 200
    agent_prompt_prefix: str = (
        "You are reviewing a live ML training run. Below is the recent log tail and the "
        "latest checkpoint metrics. In <=5 terse bullets, flag any divergence, NaN/Inf, "
        "stalled or exploding loss, or learning-rate problems. If healthy, say so in one line."
    )

    # ── logging ───────────────────────────────────────────────────────────────
    train_log_path: str | None = None    # source for the agent-eval log tail
    log_path: str | None = None          # controller's own log file (else stderr only)

    def __post_init__(self) -> None:
        if self.off_hours is not None:
            s, e = self.off_hours
            for h in (s, e):
                if not (0 <= h <= 23):
                    raise ValueError(f"off_hours hours must be in 0..23, got {self.off_hours}")
        if self.vram_headroom_gb <= 0:
            raise ValueError("vram_headroom_gb must be positive")
        if not (0.0 < self.resume_free_frac <= 1.0):
            raise ValueError("resume_free_frac must be in (0, 1]")
        if self.poll_s <= 0:
            raise ValueError("poll_s must be positive")
