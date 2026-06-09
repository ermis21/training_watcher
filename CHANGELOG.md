# Changelog

All notable changes to `training_watcher` are documented here.
The format follows [Keep a Changelog](https://keepachangelog.com/), and the project
adheres to [Semantic Versioning](https://semver.org/).

## [0.1.1] — 2026-06-09

Ergonomics + a small log-correctness fix. No change to the pause/resume state machine.

### Added
- `register()` now accepts a `checkpoint_cb(step, epoch)` callback **in addition to** the
  classic zero-arg form. The arity is inspected once at registration (`inspect.signature`)
  and the matching call form is used at the pause site, so a consumer can pass a stable
  module-level function instead of a fragile closure. Backward-compatible: a zero-arg cb is
  still called as `cb()`.

### Fixed
- `_do_pause` pause log formatted `free_gb` with `free_gb or -1.0`, which mislabels a
  legitimately full GPU (`free_gb == 0.0`) as "unknown" (`-1.0`). Now uses an explicit
  `is None` check (log-only; no decision logic was affected).

### Docs
- Documented the `num_workers=0` requirement directly on `register()` and `guard()`.
- Documented that `guard()` intentionally takes the optimizer (rebuilt at freeze→unfreeze
  boundaries) but **not** the scheduler (holds no GPU tensors, never offloaded).
- Documented that `note_checkpoint()` takes no checkpoint path on purpose (the trainer owns
  its layout; agent-eval reads the log tail + metrics, not the checkpoint files).
- Noted that `CoopConfig.our_pid` resolves to the real PID at instance creation (not a
  sentinel) via `default_factory=os.getpid`.

## [0.1.0] — 2026-06-09

Initial release.

### Added
- `CoopController` — in-process cooperative GPU sharing for a training loop:
  pause + checkpoint + free VRAM when a competitor appears outside the owned window,
  then auto-resume once the GPU is idle for `resume_idle_s` (default 10 min).
- Named primitives `watcher`, `autoyield`, `auto_stop`.
- `CoopConfig` policy: reservation window + off-hours ownership, VRAM headroom, yield
  debounce, resume-idle countdown, device-specific sensing, agent-eval hook.
- `GpuMonitor` daemon thread publishing an immutable, lock-free `Snapshot`.
- `offload_to_cpu` / `reload_to_device` with **per-tensor device restoration** (keeps
  AdamW's CPU `step` on the CPU) and OOM-safe reload.
- `nvidia-smi` sensing with a hard timeout and device-specific resolution
  (`CUDA_VISIBLE_DEVICES` / UUID aware); torch is an optional, lazily-imported dependency.
- Atomic checkpoint directories (`atomic_checkpoint_dir`, `find_latest_checkpoint`) with a
  `COMPLETE` sentinel so a torn write is never resumed.
- Start-time own-PID self-check that **fails closed** (disables auto-yield) under a PID
  namespace mismatch.
- Non-blocking agent-eval hook (`pi`/`claude` headless) with ≤1 concurrent run and full
  failure isolation.
- Lifecycle callbacks (`on_pause`/`on_resume`/`on_yield`) and counters
  (`pauses_total`, `seconds_paused`, `reload_oom_retries`).
- Test suite (mocked `nvidia-smi` + fake clock, no GPU required) plus a CUDA-gated
  optimizer offload/reload round-trip that asserts value **and** device preservation.
