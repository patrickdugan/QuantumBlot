#!/usr/bin/env python3
"""Preflight TRMStoryworld QFT experiment settings with RAM-aware defaults."""

from __future__ import annotations

import argparse
import ctypes
import json
import os
import platform
from pathlib import Path
from typing import Dict, List


def detect_total_ram_bytes() -> int:
    if os.name == "nt":
        try:
            class MEMORYSTATUSEX(ctypes.Structure):
                _fields_ = [
                    ("dwLength", ctypes.c_ulong),
                    ("dwMemoryLoad", ctypes.c_ulong),
                    ("ullTotalPhys", ctypes.c_ulonglong),
                    ("ullAvailPhys", ctypes.c_ulonglong),
                    ("ullTotalPageFile", ctypes.c_ulonglong),
                    ("ullAvailPageFile", ctypes.c_ulonglong),
                    ("ullTotalVirtual", ctypes.c_ulonglong),
                    ("ullAvailVirtual", ctypes.c_ulonglong),
                    ("ullAvailExtendedVirtual", ctypes.c_ulonglong),
                ]

            statex = MEMORYSTATUSEX()
            statex.dwLength = ctypes.sizeof(MEMORYSTATUSEX)
            if ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(statex)):
                return int(statex.ullTotalPhys)
        except Exception:
            pass

    if hasattr(os, "sysconf"):
        try:
            page_size = int(os.sysconf("SC_PAGE_SIZE"))
            pages = int(os.sysconf("SC_PHYS_PAGES"))
            return page_size * pages
        except Exception:
            pass
    return 0


def suggest_batch_mb(total_ram_mb: float) -> Dict[str, int]:
    # Keep allocations conservative for mixed Python/Node workloads.
    if total_ram_mb <= 0:
        return {"batch_size": 8, "chunk_limit_mb": 64}
    if total_ram_mb < 4096:
        return {"batch_size": 8, "chunk_limit_mb": 64}
    if total_ram_mb < 8192:
        return {"batch_size": 16, "chunk_limit_mb": 128}
    if total_ram_mb < 16384:
        return {"batch_size": 32, "chunk_limit_mb": 256}
    return {"batch_size": 64, "chunk_limit_mb": 384}


def find_candidate_entrypoints(repo: Path) -> List[str]:
    candidates = []
    names = [
        "qft-runner.js",
        "qft-runner.ts",
        "qft_one.py",
        "run_layered_qft.py",
        "run_qft_demo.py",
    ]
    for name in names:
        found = list(repo.rglob(name))
        for path in found[:5]:
            candidates.append(str(path))
    return sorted(set(candidates))


def main() -> None:
    parser = argparse.ArgumentParser(description="TRMStoryworld preflight and RAM batch planner.")
    parser.add_argument("--repo", required=True, help="Path to TRMStoryworld repo.")
    parser.add_argument("--out", default="preflight_qft_experiment.json", help="Output JSON path.")
    args = parser.parse_args()

    repo = Path(args.repo)
    repo_exists = repo.exists() and repo.is_dir()
    entrypoints = find_candidate_entrypoints(repo) if repo_exists else []

    total_ram_bytes = detect_total_ram_bytes()
    total_ram_mb = float(total_ram_bytes) / (1024.0 * 1024.0) if total_ram_bytes else 0.0
    plan = suggest_batch_mb(total_ram_mb)

    ibm_key_present = bool(os.environ.get("IBM_CLOUD_API_KEY"))
    ibm_crn_present = bool(os.environ.get("IBM_QUANTUM_CRN"))

    result = {
        "repo": str(repo),
        "repo_exists": repo_exists,
        "entrypoint_candidates": entrypoints,
        "ibm_credentials": {
            "IBM_CLOUD_API_KEY_present": ibm_key_present,
            "IBM_QUANTUM_CRN_present": ibm_crn_present,
        },
        "host": {
            "platform": platform.platform(),
            "python": platform.python_version(),
            "total_ram_mb": round(total_ram_mb, 2),
        },
        "recommended": {
            "batch_size": plan["batch_size"],
            "chunk_limit_mb": plan["chunk_limit_mb"],
            "max_steps": 32,
            "max_episodes": 1,
            "rss_threshold": 0.85,
            "telemetry_out": "artifacts/sc1028_telemetry.jsonl",
            "lockfile": "run.lock",
        },
        "notes": [
            "Credentials are only presence-checked; values are never printed.",
            "Run small smoke tests before long IBM jobs.",
            "Keep rollout and analysis in separate processes.",
        ],
    }

    out = Path(args.out)
    out.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(f"[preflight] wrote {out}")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
