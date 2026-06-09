# training_watcher — guidance for Claude Code

In-process cooperative GPU sharing for training loops. A training process calls this
library directly instead of relying on an external watcher daemon: when a competitor
appears on the GPU outside the owned window it checkpoints, offloads the model + optimizer
to CPU to free the VRAM, waits until the GPU has been idle past a threshold, then resumes
from live in-RAM state. No second process; nothing to misconfigure.

## Layout
- `src/training_watcher/` — `config.py` (`CoopConfig`), `window.py` (ownership window),
  `smi.py` (nvidia-smi sensing), `monitor.py` (`GpuMonitor` daemon + immutable `Snapshot`),
  `offload.py` (CPU offload/reload), `controller.py` (`CoopController` + `watcher`/`autoyield`/
  `auto_stop`), `checkpoint.py` (atomic dirs), `agenteval.py` (CLI eval hook), `logging_util.py`.
- `tests/` — pytest, mocked nvidia-smi + injected clock, **no GPU required**; one CUDA-gated
  offload round-trip (`test_offload_gpu.py`).

## How to work here
- Run tests: `pytest -q` (from the repo root, with the package importable). On a box without
  a venv on PATH, use the project interpreter, e.g. `PYTHONPATH=src python -m pytest tests/ -q`.
- The **core is stdlib-only**; `torch` is an optional, *lazily imported* dependency used only by
  `offload.py`. Never add a top-level `import torch` to the core — it must import and unit-test
  on a machine with no torch. Keep new torch usage inside functions.
- Determinism in tests comes from **dependency injection**: pass a fake `reader` (returns
  `GpuReading`), a fake `monotonic` clock, and an injected `runner` for `smi`. Don't sleep on
  real wall-clock or call real `nvidia-smi` in tests.

## Public API (keep stable)
```python
coop = CoopController(CoopConfig(off_hours=(22, 8)), device=device)
coop.register(model, checkpoint_cb, on_pause=?, on_resume=?, on_yield=?)  # checkpoint_cb is ZERO-arg
coop.start()                 # resolves the physical GPU, runs the self-check, starts the monitor
auto_stop(coop)              # chains SIGTERM/SIGINT -> graceful stop (preserves any prior handler)
for step in loop:
    ...
    coop.guard(global_step, epoch, optimizer)   # hot path; no-op unless a pause is due
    if step % N == 0: coop.note_checkpoint(step, metrics={...})
```
`checkpoint_cb` is **zero-arg** and is expected to close over live loop state (so it sees the
current `global_step`/`optimizer`). `guard()` is passed the optimizer each call because trainers
often rebuild it (e.g. at a freeze→unfreeze boundary) — never cache a stale optimizer reference.

## Invariants that must not regress (each has a test; keep them)
1. **Per-tensor device restoration on reload.** AdamW (non-capturable, the default) keeps the
   per-parameter `step` tensor on the **CPU** while moments are on CUDA. `offload_to_cpu` records
   each tensor's origin device and `reload_to_device` restores to it. A `.cpu()`-only equality
   test masks a device bug — the round-trip test asserts **device** equality too.
2. **Own-PID exclusion + fail-closed self-check.** The paused process keeps its CUDA context, so
   our own PID stays in `nvidia-smi` compute-apps. Every "GPU empty" predicate excludes `our_pid`.
   If `start()` can't see our own PID (PID-namespace mismatch), it disables auto-yield rather than
   pausing forever (`fail_closed`, default True).
3. **Conservative on "unknown".** Any nvidia-smi failure/timeout or a stale snapshot is treated as
   unknown: never resume, never falsely yield. smi calls always pass a timeout.
4. **Monotonic timers use explicit `is None` checks**, never `x or now` truthiness — a real
   timestamp can be `0.0`.
5. **Device-specific sensing.** `resolve_physical_index` matches the torch device UUID against
   nvidia-smi, because torch ordering (fastest-first) and nvidia-smi ordering (PCI) can differ.
   The monitor must watch the card the trainer actually trains on.
6. **Atomic checkpoints.** `atomic_checkpoint_dir` writes a temp dir, drops a `COMPLETE` sentinel,
   then `os.replace`s into `step-N`. `find_latest_checkpoint` only returns complete dirs.
7. **In-process pause needs no reseek** — only valid with a single-thread loader (`num_workers=0`);
   the loop is simply blocked inside `guard()`. The pause-moment checkpoint is the kill-safety net.

## Commit / publish
- Repo: github.com/ermis21/training_watcher (default branch `main`).
- End commit messages with the `Co-Authored-By: Claude ...` trailer; commit/push only when asked.
- Bump `__version__` (in `__init__.py`) and `CHANGELOG.md` together on any release.
