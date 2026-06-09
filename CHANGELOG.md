# Changelog

All notable changes to `training_watcher` are documented here.
The format follows [Keep a Changelog](https://keepachangelog.com/), and the project
adheres to [Semantic Versioning](https://semver.org/).

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
