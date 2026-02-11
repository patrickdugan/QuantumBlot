---
name: qft-experiments
description: Plan and run bounded-memory QFT experiments for TRM/StoryWorld-style repos, including PCA/RAM batch sizing, IBM credential preflight, and explicit telemetry-first execution. Use when users ask to run or test QFT/IBM experiments in `C:\projects\TRMStoryworld` (or similar repos), tune batching to available RAM, or prepare safe experiment commands.
---

# QFT Experiments

Run QFT experiment workflows with deterministic preflight and memory-safe defaults.
Prefer explicit artifact emission and dry-run checks before expensive cloud execution.

## Workflow

1) Run preflight and RAM plan
- Execute:
```bash
python scripts/preflight_trmstoryworld.py --repo C:\projects\TRMStoryworld
```
- This checks:
  - repo path exists
  - likely runner/entrypoint files
  - IBM credentials presence (`IBM_CLOUD_API_KEY`, `IBM_QUANTUM_CRN`) without printing secrets
  - host RAM and recommended bounded batch settings

2) Build experiment command from plan
- Use the emitted recommendation JSON to form commands with:
  - explicit `max_steps`, `max_episodes`, and RSS threshold
  - sidecar telemetry path
  - optional explicit provenance tag
- Never rely on hidden/proprietary output markers.

3) Execute safely
- Start with a small dry-run or status command if supported.
- Then run a minimal-size QFT test before large parameter sweeps.
- Keep rollout and analysis separate processes.

4) Analyze offline
- Run analyzer scripts in a fresh process on JSONL artifacts.
- Report:
  - primitive frequencies
  - transitions
  - seed divergence
  - early-failure signatures

## Guardrails

- Do not print credential values.
- Require explicit labels for any provenance tag.
- Stream telemetry to JSONL; do not accumulate unbounded traces in RAM.
- Use lockfile semantics when runner supports it.
- Use inference-only defaults for playthrough/test loops.

## Resources

- Script: `scripts/preflight_trmstoryworld.py`
  - Generates preflight + RAM-batch recommendations for experiment setup.
- Reference: `references/ram_batching.md`
  - Heuristics for selecting safe batch sizes from available RAM.

## Example Trigger Phrases

- "Run QFT experiments in TRMStoryworld with safe RAM batching."
- "Use IBM keys to test one small run first."
- "Tune PCA/context batching to this machine's memory."
- "Prepare a guarded experiment command with explicit telemetry output."
