# SC1028 Schema

SC1028 is a fixed-length symbolic DSL for explicit chunk-level telemetry.

## Bit Layout

- Semantic width: `1028` bits
- Storage width: `129` bytes (`1032` bits)
- Reserved padding: high nibble of byte `128` (`4` bits), must be `0`
- Byte bit-order: little-endian inside byte

The storage format is explicit and deterministic. No hidden channels are used.

## Encoding

- Source registry: `sc1028_symbols.py`
- Encoder/decoder: `sc1028.py`
- Log token format: base64url string (`sc1028_b64`)

## Chunk Telemetry Record

Each chunk boundary emits one JSONL line:

```json
{
  "run_id": "9d1d0b...",
  "episode_id": 0,
  "chunk_id": 3,
  "obs_hash": "e2f4a9...",
  "action_hash": "c11bd7...",
  "sc1028_b64": "AAAB...",
  "sc1028_symbols": ["meta.chunk_boundary", "action.qft_run", "term.chunk_ok"],
  "sc1028_version": "1.0.0",
  "scalars": {
    "rss_mb": 332.8,
    "rss_ratio": 0.41,
    "entropy": 4.91,
    "entropy_norm": 0.74,
    "topk_gap": 0.09
  },
  "termination": "running",
  "seed": 7,
  "ts_unix": 1770760000.123
}
```

## Primitive Groups

- Meta: `meta.*`
- Action/tool boundary: `action.*`, `tool.*`
- Termination: `term.*`
- Uncertainty proxies: `uncert.*`
- Constraint checks: `constraint.*`
- Loop/cycle heuristics: `loop.*`
- Runtime mode: `mode.*`
- Resource guardrails: `resource.*`

## Safety and Compliance

- Telemetry is explicit sidecar JSONL, streamed to disk per chunk.
- No covert watermarking or steganography.
- Optional provenance is explicit and opt-in (`--provenance_tag`).
- Rollout and analysis are separate phases:
  - Rollout: `qft-runner.js`
  - Offline analysis: `analyze_sc1028.py`

