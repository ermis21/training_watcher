"""Atomic checkpoint directories: sentinel, torn-write invisibility, latest selection."""

from __future__ import annotations

import os

import pytest

from training_watcher.checkpoint import (
    SENTINEL,
    atomic_checkpoint_dir,
    find_latest_checkpoint,
    is_complete,
)


def test_atomic_commit_creates_complete(tmp_path):
    out = str(tmp_path)
    final = os.path.join(out, "step-100")
    with atomic_checkpoint_dir(out, 100) as work:
        # inside the context the final dir does NOT yet exist; work happens in a temp dir
        assert os.path.isdir(work)
        assert os.path.realpath(work) != os.path.realpath(final)
        assert not os.path.exists(final)
        with open(os.path.join(work, "weights.bin"), "w") as fh:
            fh.write("w")
    # after clean exit: promoted + sentinel present
    assert os.path.isdir(final)
    assert os.path.isfile(os.path.join(final, SENTINEL))
    assert is_complete(final) is True
    assert os.path.isfile(os.path.join(final, "weights.bin"))
    assert find_latest_checkpoint(out) == final


def test_torn_write_ignored(tmp_path):
    out = str(tmp_path)
    # a clean step-100
    with atomic_checkpoint_dir(out, 100):
        pass
    final_100 = os.path.join(out, "step-100")
    # simulate a torn step-200: directory exists but NO sentinel
    torn = os.path.join(out, "step-200")
    os.makedirs(torn)
    with open(os.path.join(torn, "weights.bin"), "w") as fh:
        fh.write("partial")
    assert is_complete(torn) is False
    # find_latest ignores the torn dir despite its higher step number
    assert find_latest_checkpoint(out) == final_100


def test_multiple_complete_returns_highest(tmp_path):
    out = str(tmp_path)
    for step in (50, 100, 75):
        with atomic_checkpoint_dir(out, step):
            pass
    assert find_latest_checkpoint(out) == os.path.join(out, "step-100")


def test_exception_inside_context_no_final(tmp_path):
    out = str(tmp_path)
    final = os.path.join(out, "step-300")

    class Boom(Exception):
        pass

    with pytest.raises(Boom):
        with atomic_checkpoint_dir(out, 300) as work:
            with open(os.path.join(work, "weights.bin"), "w") as fh:
                fh.write("partial")
            raise Boom()

    # the final dir must NOT exist (promotion never happened)
    assert not os.path.exists(final)
    assert find_latest_checkpoint(out) is None


def test_find_latest_empty_dir(tmp_path):
    assert find_latest_checkpoint(str(tmp_path)) is None


def test_find_latest_missing_dir(tmp_path):
    missing = os.path.join(str(tmp_path), "nope")
    assert find_latest_checkpoint(missing) is None


def test_resave_same_step(tmp_path):
    out = str(tmp_path)
    with atomic_checkpoint_dir(out, 100) as work:
        with open(os.path.join(work, "a.txt"), "w") as fh:
            fh.write("first")
    with atomic_checkpoint_dir(out, 100) as work:
        with open(os.path.join(work, "b.txt"), "w") as fh:
            fh.write("second")
    final = os.path.join(out, "step-100")
    assert is_complete(final)
    # the re-save replaced the directory
    assert os.path.isfile(os.path.join(final, "b.txt"))
    assert not os.path.isfile(os.path.join(final, "a.txt"))
