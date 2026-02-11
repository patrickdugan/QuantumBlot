# RAM Batching Guide

Use conservative batch sizing for mixed embedding + PCA + QFT workflows.

## Baseline Heuristic

- `<4 GB RAM`: batch `8`, chunk limit `64 MB`
- `4-8 GB RAM`: batch `16`, chunk limit `128 MB`
- `8-16 GB RAM`: batch `32`, chunk limit `256 MB`
- `>=16 GB RAM`: batch `64`, chunk limit `384 MB`

## Safety Defaults

- `max_steps=32`
- `max_episodes=1`
- `rss_threshold=0.85`
- stream telemetry to JSONL and flush each chunk
- run short smoke test before large runs

## IBM Execution Notes

- Require `IBM_CLOUD_API_KEY` and `IBM_QUANTUM_CRN` in environment.
- Presence-check only; never print secret values.
- If credentials are missing, stop before job submission.

