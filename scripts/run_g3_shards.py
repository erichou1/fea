#!/usr/bin/env python3
"""Run G3 deterministic shards with auditable resource sampling.

This dispatcher does not perform K6 coverage or adjudication.  It records the
number of durable trajectory-case files and macOS memory pressure about every
five minutes while the requested independent G3 shard processes run.
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import os
import re
import subprocess
import sys
import time
from pathlib import Path
from typing import Any


def _utc_now() -> str:
    return dt.datetime.now(dt.timezone.utc).isoformat()


def _append(path: Path, payload: dict[str, Any]) -> None:
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, sort_keys=True, allow_nan=False) + "\n")
        handle.flush()
        os.fsync(handle.fileno())


def _vm_snapshot() -> dict[str, Any]:
    result: dict[str, Any] = {}
    try:
        raw = subprocess.check_output(["vm_stat"], text=True, stderr=subprocess.STDOUT)
        for label, key in (("Pages free", "pages_free"), ("Pages active", "pages_active"),
                           ("Pages inactive", "pages_inactive"), ("Pages speculative", "pages_speculative"),
                           ("Pages wired down", "pages_wired")):
            match = re.search(r"^" + re.escape(label) + r":\s+(\d+)", raw, flags=re.MULTILINE)
            if match:
                result[key] = int(match.group(1))
    except (OSError, subprocess.CalledProcessError) as error:
        result["vm_stat_error"] = str(error)
    try:
        result["swapusage"] = subprocess.check_output(["sysctl", "-n", "vm.swapusage"], text=True, stderr=subprocess.STDOUT).strip()
    except (OSError, subprocess.CalledProcessError) as error:
        result["swapusage_error"] = str(error)
    if "pages_free" not in result:
        # Linux fallback: without it the free-pages floor is silently INERT on any
        # non-macOS host, disabling the memory stop condition. Derive pages_free
        # from /proc/meminfo MemAvailable using the same 16 KiB page unit macOS
        # reports, so floor values mean the same amount of memory on both hosts.
        try:
            meminfo = Path("/proc/meminfo").read_text()
            match = re.search(r"^MemAvailable:\s+(\d+)\s+kB", meminfo, flags=re.MULTILINE)
            if match:
                mem_available_bytes = int(match.group(1)) * 1024
                result["pages_free"] = mem_available_bytes // 16384
                result["mem_available_gb"] = round(mem_available_bytes / 1e9, 2)
                result["pages_free_source"] = "proc_meminfo_memavailable_16k_units"
        except OSError as error:
            result["meminfo_error"] = str(error)
    return result


def _case_counts(root: Path) -> dict[str, int]:
    development = sum(1 for _ in root.glob("trajectory-development-*.json"))
    calibration = sum(1 for _ in root.glob("trajectory-calibration-*.json"))
    return {"development_completed": development, "calibration_completed": calibration, "completed_total": development + calibration}


def _tail(path: Path, limit: int = 12000) -> str:
    try:
        with path.open("rb") as handle:
            handle.seek(0, os.SEEK_END)
            handle.seek(max(0, handle.tell() - limit))
            return handle.read().decode("utf-8", errors="replace")
    except OSError:
        return ""


def _terminate(children: dict[int, subprocess.Popen[str]], *, reason: str, audit_path: Path) -> None:
    for shard, child in children.items():
        if child.poll() is None:
            child.terminate()
            _append(audit_path, {"event": "worker_terminate_requested", "timestamp_utc": _utc_now(), "shard": shard, "reason": reason})
    deadline = time.monotonic() + 60.0
    while time.monotonic() < deadline and any(child.poll() is None for child in children.values()):
        time.sleep(0.25)
    for shard, child in children.items():
        if child.poll() is None:
            child.kill()
            _append(audit_path, {"event": "worker_kill_requested_after_grace", "timestamp_utc": _utc_now(), "shard": shard, "reason": reason})


def main() -> int:
    parser = argparse.ArgumentParser(description="G3 shard dispatcher with resource audit log")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument("--split-manifest", required=True)
    parser.add_argument("--expected-split-manifest-sha256", required=True)
    parser.add_argument("--archive", required=True)
    parser.add_argument("--expected-archive-sha256", required=True)
    parser.add_argument("--g1b-root", required=True)
    parser.add_argument("--expected-cohort-manifest-sha256", required=True)
    parser.add_argument("--expected-cluster-role-manifest-sha256", required=True)
    parser.add_argument("--ensemble-root", required=True)
    parser.add_argument("--normalization-path", required=True)
    parser.add_argument("--preregistration-path", required=True)
    parser.add_argument("--expected-preregistration-sha256", required=True)
    parser.add_argument("--audit-file", default="resource-throughput-final.jsonl")
    parser.add_argument("--log-dir", default="shards-final")
    parser.add_argument("--workers", type=int, default=4, help="deterministic shard count (one worker is the low-memory default)")
    parser.add_argument("--sample-seconds", type=float, default=300.0)
    parser.add_argument("--free-pages-floor", type=int, default=3000)
    parser.add_argument("--first-trajectory-timeout-seconds", type=float, default=900.0)
    args = parser.parse_args()
    if args.sample_seconds <= 0.0 or args.free_pages_floor < 1 or args.workers < 1 or args.first_trajectory_timeout_seconds <= 0.0:
        raise SystemExit("workers, sample-seconds, free-pages-floor, and first-trajectory-timeout-seconds must be positive")

    root = args.output.resolve()
    audit_path = root / args.audit_file
    if not root.is_dir():
        raise SystemExit("output root must already exist; run G3 initialize first")
    if audit_path.exists():
        raise SystemExit("requested audit file already exists; refuse to mingle run histories")

    common = [
        "-m", "sasto.g3_trajectory_calibration", "--mode", "run-shard", "--output", str(root),
        "--split-manifest", args.split_manifest, "--expected-split-manifest-sha256", args.expected_split_manifest_sha256,
        "--archive", args.archive, "--expected-archive-sha256", args.expected_archive_sha256,
        "--g1b-root", args.g1b_root, "--expected-cohort-manifest-sha256", args.expected_cohort_manifest_sha256,
        "--expected-cluster-role-manifest-sha256", args.expected_cluster_role_manifest_sha256,
        "--ensemble-root", args.ensemble_root, "--normalization-path", args.normalization_path,
        "--preregistration-path", args.preregistration_path,
        "--expected-preregistration-sha256", args.expected_preregistration_sha256,
        "--device", "cpu",
    ]
    env = dict(os.environ)
    env["PYTHONPATH"] = "src" + (os.pathsep + env["PYTHONPATH"] if env.get("PYTHONPATH") else "")
    started = time.monotonic()
    _append(audit_path, {"event": "run_start", "timestamp_utc": _utc_now(), "workers": args.workers, "device": "cpu",
                         "sample_seconds": args.sample_seconds, "free_pages_floor": args.free_pages_floor,
                         "first_trajectory_timeout_seconds": args.first_trajectory_timeout_seconds,
                         "command_prefix": [args.python, *common]})
    children: dict[int, subprocess.Popen[str]] = {}
    logs: dict[int, Path] = {}
    log_root = root / args.log_dir
    if log_root.exists():
        raise SystemExit("requested shard log directory already exists; refuse to mingle run histories")
    log_root.mkdir()
    for shard in range(1, args.workers + 1):
        log = log_root / f"shard-{shard}.log"
        logs[shard] = log
        command = [args.python, *common, "--shard", f"{shard}/{args.workers}"]
        handle = log.open("x", encoding="utf-8")
        children[shard] = subprocess.Popen(command, stdin=subprocess.DEVNULL, stdout=handle, stderr=subprocess.STDOUT, text=True, env=env)
        handle.close()
        _append(audit_path, {"event": "worker_started", "timestamp_utc": _utc_now(), "shard": shard, "pid": children[shard].pid,
                             "command": command})

    prior_count = _case_counts(root)["completed_total"]
    initial_count = prior_count
    prior_at = started
    consecutive_low_free = 0
    reported_exits: set[int] = set()
    stopped = False
    stop_reason = ""
    while True:
        now = time.monotonic()
        counts = _case_counts(root)
        seconds = now - prior_at
        delta = counts["completed_total"] - prior_count
        snapshot = _vm_snapshot()
        pages_free = snapshot.get("pages_free")
        if isinstance(pages_free, int) and pages_free < args.free_pages_floor:
            consecutive_low_free += 1
        else:
            consecutive_low_free = 0
        statuses = {str(shard): child.poll() for shard, child in children.items()}
        _append(audit_path, {"event": "sample", "timestamp_utc": _utc_now(), "elapsed_seconds": now - started,
                             **counts, "interval_completed_cases": delta, "interval_seconds": seconds,
                             "interval_cases_per_hour": (delta * 3600.0 / seconds) if seconds else 0.0,
                             "worker_returncodes": statuses, "consecutive_low_free_samples": consecutive_low_free, **snapshot})
        prior_count = counts["completed_total"]
        prior_at = now
        for shard, child in children.items():
            code = child.poll()
            if code is None or shard in reported_exits:
                continue
            reported_exits.add(shard)
            tail = _tail(logs[shard])
            _append(audit_path, {"event": "worker_exit", "timestamp_utc": _utc_now(), "shard": shard, "returncode": code,
                                 "log_tail": tail[-4000:]})
            if code != 0 and re.search(r"REJECTED:.*(?:digest|role)", tail, flags=re.IGNORECASE):
                stopped = True
                stop_reason = "digest_or_role_correctness_signal"
        # A completed worker set is never a capacity event; retain its nonzero
        # result for explicit controller review instead of misclassifying CLI use.
        if all(child.poll() is not None for child in children.values()) and not stopped:
            break
        if now - started >= args.first_trajectory_timeout_seconds and counts["completed_total"] == initial_count and not stopped:
            stopped = True
            stop_reason = "no_trajectory_within_first_15_minutes"
        if consecutive_low_free >= 3 and not stopped:
            stopped = True
            stop_reason = "sustained_pages_free_below_floor"
        if stopped:
            _append(audit_path, {"event": "hard_stop", "timestamp_utc": _utc_now(), "reason": stop_reason})
            _terminate(children, reason=stop_reason, audit_path=audit_path)
            return 3
        if all(child.poll() is not None for child in children.values()):
            break
        time.sleep(args.sample_seconds)

    final_counts = _case_counts(root)
    returncodes = {str(shard): child.returncode for shard, child in children.items()}
    outcome = 0 if all(code == 0 for code in returncodes.values()) else 4
    _append(audit_path, {"event": "run_complete", "timestamp_utc": _utc_now(), "elapsed_seconds": time.monotonic() - started,
                         **final_counts, "worker_returncodes": returncodes, "outcome": "success" if outcome == 0 else "worker_failure_without_forced_stop"})
    print(json.dumps({"output": str(root), **final_counts, "worker_returncodes": returncodes, "exit_code": outcome}, sort_keys=True))
    return outcome


if __name__ == "__main__":
    raise SystemExit(main())
