#!/usr/bin/env python3
"""Offline analyzer for SC1028 telemetry JSONL."""

from __future__ import annotations

import argparse
import json
import math
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Tuple


def load_jsonl(path: str) -> Iterable[dict]:
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Telemetry file not found: {path}")
    with p.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError:
                continue


def jensen_shannon(p: Dict[str, float], q: Dict[str, float]) -> float:
    keys = set(p.keys()) | set(q.keys())
    m = {k: 0.5 * p.get(k, 0.0) + 0.5 * q.get(k, 0.0) for k in keys}

    def _kl(a: Dict[str, float], b: Dict[str, float]) -> float:
        total = 0.0
        for k in keys:
            av = a.get(k, 0.0)
            bv = b.get(k, 0.0)
            if av > 0.0 and bv > 0.0:
                total += av * math.log2(av / bv)
        return total

    return 0.5 * _kl(p, m) + 0.5 * _kl(q, m)


def normalize_counter(counter: Counter) -> Dict[str, float]:
    total = float(sum(counter.values())) or 1.0
    return {k: float(v) / total for k, v in counter.items()}


def analyze(path: str) -> dict:
    primitive_freq: Counter = Counter()
    transition_matrix: Counter = Counter()
    seed_distributions: Dict[str, Counter] = defaultdict(Counter)
    early_failure_signatures: Counter = Counter()

    per_episode_chunks: Dict[Tuple[str, int], List[dict]] = defaultdict(list)

    for row in load_jsonl(path):
        run_id = str(row.get("run_id", "unknown"))
        episode_id = int(row.get("episode_id", 0))
        per_episode_chunks[(run_id, episode_id)].append(row)

        symbols = row.get("sc1028_symbols", []) or []
        primitive_freq.update(symbols)

        seed_key = str(row.get("seed", "none"))
        seed_distributions[seed_key].update(symbols)

    for (_, _), chunks in per_episode_chunks.items():
        chunks_sorted = sorted(chunks, key=lambda x: int(x.get("chunk_id", 0)))
        for i in range(1, len(chunks_sorted)):
            prev_syms = chunks_sorted[i - 1].get("sc1028_symbols", []) or []
            curr_syms = chunks_sorted[i].get("sc1028_symbols", []) or []
            for a in prev_syms:
                for b in curr_syms:
                    transition_matrix[(a, b)] += 1

        if chunks_sorted:
            first_five = chunks_sorted[:5]
            for row in first_five:
                term = str(row.get("termination", ""))
                if term in {"failure", "watchdog_exit", "max_steps_hit", "max_episodes_hit"}:
                    for sym in row.get("sc1028_symbols", []) or []:
                        early_failure_signatures[sym] += 1

    seed_keys = sorted(seed_distributions.keys())
    divergence = []
    for i, a in enumerate(seed_keys):
        for b in seed_keys[i + 1 :]:
            pa = normalize_counter(seed_distributions[a])
            pb = normalize_counter(seed_distributions[b])
            divergence.append(
                {
                    "seed_a": a,
                    "seed_b": b,
                    "js_divergence": jensen_shannon(pa, pb),
                }
            )

    transitions = [
        {"from": a, "to": b, "count": c}
        for (a, b), c in transition_matrix.most_common()
    ]

    return {
        "primitive_frequency": primitive_freq.most_common(),
        "transition_matrix": transitions,
        "seed_divergence": divergence,
        "early_failure_signatures": early_failure_signatures.most_common(),
        "episodes_analyzed": len(per_episode_chunks),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Analyze SC1028 telemetry JSONL.")
    parser.add_argument("telemetry_jsonl", type=str, help="Path to SC1028 telemetry JSONL sidecar.")
    parser.add_argument(
        "--out",
        type=str,
        default="sc1028_analysis.json",
        help="Output JSON file for analysis results.",
    )
    args = parser.parse_args()

    result = analyze(args.telemetry_jsonl)
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2)

    print(f"[analyze_sc1028] Wrote analysis to {args.out}")
    print(f"[analyze_sc1028] Episodes: {result['episodes_analyzed']}")
    print(f"[analyze_sc1028] Primitive count entries: {len(result['primitive_frequency'])}")
    print(f"[analyze_sc1028] Transitions: {len(result['transition_matrix'])}")


if __name__ == "__main__":
    main()

