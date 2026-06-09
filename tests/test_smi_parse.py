"""nvidia-smi parsing + read_gpu with an injected runner (no real GPU)."""

from __future__ import annotations

import subprocess

import pytest

from training_watcher.smi import GpuReading, parse_gpu_stats, parse_pids, read_gpu


# ── parse_gpu_stats ───────────────────────────────────────────────────────────
def test_parse_gpu_stats_normal_row():
    # util=37, free=10240 MiB, total=46080 MiB
    util, free_gb, total_gb = parse_gpu_stats([["37", "10240", "46080"]])
    assert util == 37
    assert free_gb == pytest.approx(10240 / 1024.0)   # MiB -> GB
    assert total_gb == pytest.approx(46080 / 1024.0)
    assert free_gb == pytest.approx(10.0)
    assert total_gb == pytest.approx(45.0)


@pytest.mark.parametrize(
    "rows",
    [
        [["[N/A]", "10240", "46080"]],   # N/A util
        [["37", "[N/A]", "46080"]],      # N/A free
        [[]],                            # empty inner row -> IndexError
        [["37", "10240"]],               # short row -> IndexError
        [],                              # no rows -> IndexError
    ],
)
def test_parse_gpu_stats_malformed_raises(rows):
    with pytest.raises((ValueError, IndexError)):
        parse_gpu_stats(rows)


# ── parse_pids ────────────────────────────────────────────────────────────────
def test_parse_pids_filters_and_dedups():
    rows = [["123"], ["456"], ["123"], ["[N/A]"], [""], ["notapid"], []]
    pids = parse_pids(rows)
    assert pids == frozenset({123, 456})


def test_parse_pids_empty():
    assert parse_pids([]) == frozenset()


# ── read_gpu with injected runner ─────────────────────────────────────────────
def test_read_gpu_ok():
    calls = []

    def runner(args, timeout):
        calls.append(list(args))
        if "--query-compute-apps=pid" in args:
            return [["123"], ["456"]]
        return [["20", "20480", "46080"]]

    r = read_gpu(3, 10.0, runner=runner)
    assert r.ok is True
    assert r.util_pct == 20
    assert r.free_gb == pytest.approx(20480 / 1024.0)
    assert r.total_gb == pytest.approx(46080 / 1024.0)
    assert r.pids == frozenset({123, 456})
    assert r.free_frac == pytest.approx((20480 / 1024.0) / (46080 / 1024.0))
    # index passed via -i on every call
    for args in calls:
        assert args[:2] == ["-i", "3"]


def test_read_gpu_no_index_omits_dash_i():
    seen = []

    def runner(args, timeout):
        seen.append(list(args))
        if "--query-compute-apps=pid" in args:
            return [["123"]]
        return [["5", "40000", "46080"]]

    r = read_gpu(None, 10.0, runner=runner)
    assert r.ok is True
    for args in seen:
        assert "-i" not in args


def test_read_gpu_called_process_error():
    def runner(args, timeout):
        raise subprocess.CalledProcessError(1, "nvidia-smi")

    r = read_gpu(0, 10.0, runner=runner)
    assert r == GpuReading(ok=False)
    assert r.ok is False


def test_read_gpu_timeout():
    def runner(args, timeout):
        raise subprocess.TimeoutExpired("nvidia-smi", timeout)

    r = read_gpu(0, 5.0, runner=runner)
    assert r.ok is False


def test_read_gpu_garbage_rows():
    def runner(args, timeout):
        # first call (gpu stats) returns garbage -> parse_gpu_stats raises -> ok=False
        return [["garbage"]]

    r = read_gpu(0, 10.0, runner=runner)
    assert r.ok is False


def test_read_gpu_passes_timeout_through():
    captured = {}

    def runner(args, timeout):
        captured["timeout"] = timeout
        if "--query-compute-apps=pid" in args:
            return [["1"]]
        return [["0", "46000", "46080"]]

    read_gpu(0, 7.5, runner=runner)
    assert captured["timeout"] == 7.5
