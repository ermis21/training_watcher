"""Atomic checkpoint directories.

The pause-moment checkpoint is *load-bearing* (it is the only safety net if the process
is SIGKILLed or OOM-killed while paused), so it must never be observed half-written.

``atomic_checkpoint_dir`` yields a temporary directory to write into; on clean exit it
drops a ``COMPLETE`` sentinel, fsyncs, and ``os.replace``s the temp dir into its final
``step-N`` name in one move.  ``find_latest_checkpoint`` only ever returns a directory
that carries the sentinel, so a torn write is invisible to resume.
"""

from __future__ import annotations

import os
import shutil
from contextlib import contextmanager
from typing import Iterator

SENTINEL = "COMPLETE"


def _fsync_dir(path: str) -> None:
    try:
        fd = os.open(path, os.O_RDONLY)
        try:
            os.fsync(fd)
        finally:
            os.close(fd)
    except OSError:
        pass


@contextmanager
def atomic_checkpoint_dir(out_dir: str, step: int) -> Iterator[str]:
    """Yield a temp dir to populate; atomically promote it to ``out_dir/step-{step}``."""
    os.makedirs(out_dir, exist_ok=True)
    final = os.path.join(out_dir, f"step-{step}")
    tmp = os.path.join(out_dir, f".step-{step}.tmp-{os.getpid()}")
    if os.path.exists(tmp):
        shutil.rmtree(tmp, ignore_errors=True)
    os.makedirs(tmp)

    yield tmp                                   # caller writes checkpoint files here

    # mark complete, flush, then swap into place
    with open(os.path.join(tmp, SENTINEL), "w", encoding="utf-8") as fh:
        fh.write("ok\n")
        fh.flush()
        os.fsync(fh.fileno())
    _fsync_dir(tmp)

    if os.path.exists(final):                   # rare: re-saving the same step
        shutil.rmtree(final, ignore_errors=True)
    os.replace(tmp, final)
    _fsync_dir(out_dir)


def is_complete(ckpt_dir: str) -> bool:
    """True if ``ckpt_dir`` carries the completion sentinel."""
    return os.path.isfile(os.path.join(ckpt_dir, SENTINEL))


def find_latest_checkpoint(out_dir: str) -> str | None:
    """Return the highest-step **complete** ``step-N`` directory, or ``None``."""
    if not os.path.isdir(out_dir):
        return None
    best: tuple[int, str] | None = None
    for name in os.listdir(out_dir):
        if not name.startswith("step-") or name.endswith(".tmp"):
            continue
        path = os.path.join(out_dir, name)
        if not os.path.isdir(path) or not is_complete(path):
            continue
        try:
            step = int(name.split("-", 1)[1])
        except ValueError:
            continue
        if best is None or step > best[0]:
            best = (step, path)
    return best[1] if best else None
