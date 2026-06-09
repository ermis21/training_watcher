# training_watcher

**In-process cooperative GPU sharing for training loops** — checkpoint, free VRAM,
yield to a competitor, and auto-resume, with no external daemon. The watcher lives
*inside* your training process as a single object you `guard()` once per step.

## What it is and the policy

`training_watcher` lets a long training run share a GPU politely on a shared machine.
You **own** the GPU during your reserved slot (an explicit `reservation_end`) or during
a recurring `off_hours` window (e.g. `(22, 8)` = 22:00–07:59); inside an owned window the
controller never yields and trains freely. **Outside** the owned window you are an
opportunistic guest: you keep training while the GPU is otherwise free, but when a real
competitor appears — another process holding the GPU **and** free VRAM dropping below
`vram_headroom_gb` — that persists for `yield_debounce_s` (filtering transient inference
blips), the controller **pauses**: it checkpoints, offloads the model + optimizer state to
CPU to free the multi-GB working set, and waits. It **auto-resumes** only after the GPU has
been continuously empty (no foreign process) and idle (low utilization, high free VRAM) for
`resume_idle_s` (the ">10 min" default), reloading onto the GPU (OOM-safe: if a process
grabbed the VRAM first, it stays paused and retries). This loop continues until the run is
killed (SIGTERM/SIGINT → graceful stop) or training completes.

## Quickstart

```python
from training_watcher import CoopController, CoopConfig, auto_stop

coop = CoopController(CoopConfig(off_hours=(22, 8)), device=device)
coop.register(model, checkpoint_cb)          # checkpoint_cb: zero-arg, reads live loop state
coop.start()
auto_stop(coop)
for step in training_loop:
    ...
    coop.guard(global_step, epoch, optimizer)   # no-op unless a pause is due
```

`guard()` is the lean hot path: one atomic snapshot read plus a couple of comparisons; it
returns `True` only on the (rare) call where a pause actually happened. The background
monitor thread does all the sensing on its own cadence. Call `coop.note_checkpoint(step,
metrics=...)` right after you write a periodic checkpoint to drive the optional agent-eval
hook.

The named convenience primitives are thin wrappers over the controller:

- `watcher(config, **kwargs)` — build a controller and start its monitor.
- `autoyield(controller, step, epoch, optimizer)` — free-function form of `.guard(...)`.
- `auto_stop(controller)` — install SIGTERM/SIGINT handlers (chained), or use it as a
  context manager (`with auto_stop(coop): ...`) to restore the previous handlers on exit.

## `CoopConfig` fields

| Field | Default | Meaning |
|---|---|---|
| `reservation_end` | `None` | Explicit owned slot: own the GPU while `now < reservation_end` (wall-clock). |
| `off_hours` | `(22, 8)` | Recurring owned window `(start, end)`, half-open, wraps midnight when `start > end`; `start == end` ⇒ empty. Hours must be in `0..23`. |
| `vram_headroom_gb` | `10.0` | A competitor counts only when free VRAM drops below this. Keep ≥ 8 GB on the shared box. |
| `yield_debounce_s` | `60.0` | Competitor must persist this long before we pause (filters inference blips). |
| `resume_idle_s` | `600.0` | GPU must be continuously empty + idle this long before resuming (the ">10 min"). |
| `idle_util_pct` | `10` | Utilization below this counts as "inactive" for the resume check. |
| `resume_free_frac` | `0.90` | Require free VRAM ≥ this fraction of total to resume. Must be in `(0, 1]`. |
| `poll_s` | `20.0` | Monitor poll cadence. Must be positive. |
| `smi_timeout_s` | `10.0` | Hard timeout per `nvidia-smi` call; a timeout yields an "unknown" reading (never hangs). |
| `stale_after_s` | `60.0` | A snapshot older than this is treated as "unknown" (conservative). |
| `our_pid` | `os.getpid()` | Our own PID; excluded from the competitor set (see CUDA-context caveat). |
| `gpu_index` | `None` | Physical GPU to watch; `None` ⇒ resolve from the torch device / `CUDA_VISIBLE_DEVICES`. |
| `fail_closed` | `True` | If we can't see our own PID in `nvidia-smi`, disable auto-yield (never pause). |
| `agent_enabled` | `False` | Enable the agent-eval hook. |
| `agent_every_n_ckpts` | `10` | Fire the agent every Nth `note_checkpoint`. |
| `agent_cmd` | `"pi"` | CLI to invoke (e.g. `"pi"` or `"claude"`). |
| `agent_flag` | `"-p"` | Headless/print flag passed to the CLI. |
| `agent_timeout_s` | `120.0` | Hard timeout per agent invocation. |
| `agent_log_path` | `None` | Agent output log (default `./training_watcher_agent.log`). |
| `agent_log_tail_lines` | `200` | How many lines of the training log to include in the agent prompt. |
| `agent_prompt_prefix` | (built-in) | Prompt prefix the agent reviews the run with. |
| `train_log_path` | `None` | Source training log for the agent-eval tail. |
| `log_path` | `None` | The controller's own log file (else stderr only). |

## CUDA-context caveat (important)

`offload_to_cpu` frees the **multi-GB working set** — model parameters, buffers, and every
optimizer-state tensor (`exp_avg`, `exp_avg_sq`, and the per-parameter `step`, which is a
CUDA tensor under PyTorch's default `foreach` AdamW) — then synchronizes, drops references,
`gc.collect()`s and `empty_cache()`s so the caching allocator returns those blocks to the
driver for a competitor. **But the process keeps its CUDA *context* — a few hundred MB — for
as long as it lives.** It cannot be released without tearing down torch's CUDA state. A
competitor still gets the large working set back, which is the point.

A direct consequence: **our own PID stays listed in `nvidia-smi` compute-apps** even while
paused. So the monitor always **excludes `cfg.our_pid`** from the competitor set — otherwise
we'd see ourselves as a competitor and pause forever. To make that exclusion safe, `start()`
runs a **self-check**: it confirms our PID is actually visible in `nvidia-smi`. If it isn't
(e.g. an isolated PID namespace where `nvidia-smi` reports host PIDs, or CUDA not yet
initialized), with `fail_closed=True` the controller **fails closed** — it disables
auto-yield (it will never pause) and logs a warning, rather than risk pausing forever on a
phantom competitor. Set `fail_closed=False` to keep auto-yield enabled in that case (risky).

## Agent-eval hook

Optionally, every `agent_every_n_ckpts` checkpoints (counted via `note_checkpoint`), the
watcher shells out to a coding-agent CLI (`agent_cmd agent_flag <prompt>`) to review the run
— it feeds the latest metrics plus a tail of `train_log_path` and asks for a terse health
read (divergence, NaN/Inf, stalled/exploding loss, lr problems). It is fully
failure-isolated and non-blocking: it runs in its own thread via `subprocess.run` (never a
bare `fork` into a live CUDA context, never `multiprocessing`); at most **one** invocation is
in flight at a time (a still-running prior call means this checkpoint is skipped); a missing
CLI latches the hook off; timeouts and errors are logged to `agent_log_path`, never raised
into the training thread. Disabled by default (`agent_enabled=False`).

## Limitations / when to prefer an external process

- **In-process by design.** The watcher shares the training process. If that process is
  `SIGKILL`ed or OOM-killed, the watcher dies with it — the load-bearing safety net is the
  **atomic pause-moment checkpoint** (`atomic_checkpoint_dir` writes to a temp dir and
  `os.replace`s it into `step-N` only after dropping a `COMPLETE` sentinel, so a torn write is
  invisible to `find_latest_checkpoint`). Resume from the latest complete checkpoint on
  restart.
- **CUDA context is not freed** (see caveat): if a competitor needs *every* last MB, the
  few-hundred-MB context we retain is not given back. For that, prefer an **external** watcher
  that can fully tear down / restart the training process.
- **PID-namespace mismatches** (containers where `nvidia-smi` shows host PIDs) trip the
  self-check; with `fail_closed=True` the watcher won't yield. An external host-side daemon
  that owns the namespace is more robust there.
- **Sensing is `nvidia-smi`-based** (subprocess, timed out per call) — coarse-grained at the
  `poll_s` cadence and blind to MIG slice topology beyond the resolved physical index.
- **Single GPU per controller.** It watches one resolved physical GPU; multi-GPU jobs need a
  controller per device (or an external scheduler).

If you need preemption that survives a process kill, hard VRAM guarantees, or cross-process
scheduling policy, prefer an external daemon (e.g. a host-side `gpu_watcher.py` that
SIGTERM-checkpoints and restarts jobs) — `training_watcher` is the lightweight, in-loop
option for cooperative, best-effort sharing.

## Running the tests

```bash
PYTHONPATH=src python -m pytest tests/ -v
```

The core is stdlib-only and tests with no GPU and no torch (fake `nvidia-smi` runner,
injected monotonic clock, injected reader). The offload tests `importorskip("torch")` and the
CUDA round-trip is skipped unless a CUDA device is available.
