# QFT Runner

TypeScript/JavaScript orchestrator for your Quantum Fourier Transform pipeline. Clean interface to your Python QFT workflow with proper error handling, logging, and workflow management.

## Features

- 🌊 **Full QFT Pipeline**: Embed → PCA → QFT → Decode
- 📊 **Multiple Embedding Models**: E5, Qwen support
- 🎯 **Theme-based Interference**: Layered QFT with theme-gated phases
- 🔧 **Flexible CLI**: Run individual steps or full workflow
- ✅ **Status Tracking**: Monitor pipeline artifacts and configuration

## Semantic Codex (SC1028)

This runner emits explicit chunk-level telemetry records with:

- `run_id`, `episode_id`, `chunk_id`
- `obs_hash`, `action_hash`
- `sc1028_b64`, `sc1028_symbols`, `sc1028_version`
- scalar metrics (`rss_mb`, `entropy`, `topk_gap`, ...)

Reference files:

- `docs/schema_sc1028.md`
- `sc1028_symbols.py`
- `sc1028.py`
- `analyze_sc1028.py`
- `semantic_proof.py`

Attestation modes:

- `crypto` (default): Merkle root over chunk records + optional HMAC signature from `SEMANTIC_ATTESTATION_KEY`
- `spectral`: explicit PCA -> FFT signature over a sidecar trace artifact (`.npy`)
- `both`: run crypto and spectral attestation in one pass

## Quick Start

### 1. Setup Environment

Create `qblot.env`:
```bash
export IBM_CLOUD_API_KEY="your_api_key_here"
export IBM_QUANTUM_CRN="crn:v1:bluemix:public:quantum-computing:..."
DEFAULT_BACKEND=ibm_torino
DEFAULT_SHOTS=8000
```

### 2. Install Dependencies

```bash
# For JavaScript (no build required)
npm install

# For TypeScript
npm install
npm run build
```

### 3. Run Pipeline

```bash
# JavaScript (recommended - no build step)
node qft-runner.js full --input data.txt --theme-id 2 --layered

# TypeScript
npx tsx qft-runner.ts full --input data.txt --theme-id 2 --layered

# Using npm scripts
npm run full -- --input data.txt --theme-id 2
```

### Memory-safe defaults

The runner now enforces operational guardrails by default:

- `--lockfile run.lock`
- `--telemetry_out sc1028_telemetry.jsonl`
- `--max_steps 32`
- `--max_episodes 1`
- `--rss_threshold 0.85`

Example:

```bash
node qft-runner.js full \
  --input data.txt \
  --theme-id 2 \
  --max_steps 32 \
  --max_episodes 1 \
  --rss_threshold 0.85 \
  --telemetry_out artifacts/sc1028_telemetry.jsonl
```

## Commands

### Runtime guardrail options

All commands accept:

- `--lockfile <path>`: single-instance lockfile path
- `--telemetry_out <path>`: SC1028 JSONL sidecar output
- `--max_steps <n>`: max chunk steps per episode
- `--max_episodes <n>`: max episodes per run
- `--episodes <n>`: number of episodes to run for `full`
- `--rss_threshold <f>`: memory watchdog threshold in `[0.80, 0.95]`
- `--seed <n>`: optional run seed tag (telemetry metadata)
- `--attestation_mode <off|crypto|spectral|both>`: explicit reasoning attestation mode
- `--attestation_out <path>`: attestation JSON sidecar
- `--attestation_key_env <env>`: env var for crypto attestation key
- `--attestation_components <n>`: PCA components for spectral mode
- `--attestation_frequency <n>`: spectral signature frequency
- `--attestation_strength <f>`: spectral signature strength
- `--attestation_seed <n>`: spectral signature seed
- `--spectral_trace_out <path>`: explicit spectral trace artifact path
- `--provenance_tag`: explicit visible provenance block

### `embed` - Generate Embeddings

Convert text to vector embeddings using E5 or Qwen models.

```bash
# E5 embeddings (local)
node qft-runner.js embed --input conversations.txt --model e5

# Qwen embeddings (API)
node qft-runner.js embed --input data.jsonl --model qwen --token YOUR_HF_TOKEN
```

**Output**: `{input}_e5.npy`, `{input}_e5.jsonl`

### `run` - Execute QFT

Run quantum circuit on existing embeddings.

```bash
node qft-runner.js run \
  --vectors data_e5.npy \
  --theme-id 2 \
  --backend ibm_brisbane \
  --shots 8192 \
  --layered
```

**Options**:
- `--vectors`: Input .npy file with embeddings
- `--theme-id`: Theme ID for interference pattern (0-N)
- `--backend`: IBM Quantum backend (default: ibm_torino)
- `--shots`: Number of measurements (default: 8000)
- `--row`: Which embedding vector to process (default: 0)
- `--pos`: Position for RoPE encoding (default: 0)
- `--rope`: Path to RoPE hint JSON
- `--layered`: Use layered QFT circuit (recommended)
- `--force`: Recompute existing outputs

**Output**: `qft_counts.json`, `decoded_evidence.json`

### `full` - Complete Workflow

Run end-to-end: embedding → QFT → decode.

```bash
node qft-runner.js full \
  --input conversations.txt \
  --theme-id 3 \
  --model e5 \
  --backend ibm_brisbane \
  --shots 8192 \
  --layered
```

**Workflow**:
1. Clean chat history (if `conversations.json`)
2. Generate embeddings
3. Run PCA dimensionality reduction
4. Sparsify and prepare vectors
5. Execute QFT circuit on IBM Quantum
6. Decode results to spectrum/NN
7. Generate payload (if RoPE provided)

### `status` - Pipeline Status

Check configuration and artifact status.

```bash
node qft-runner.js status
```

**Output**:
```
📊 QFT Pipeline Status

Backend: ibm_torino
Shots: 8000
Target Dim: 768
Qubits: 17
Sparsity: 70%

📁 Pipeline Artifacts:
   ✅ qft_Z.npy
   ✅ vectors_pca_topk.npy
   ✅ qft_counts.json
   ✅ decoded_evidence.json
   ❌ request_skeleton.json
```

## Configuration

Default configuration in `qft-runner.js`:

```javascript
{
  defaultBackend: 'ibm_torino',
  defaultShots: 8000,
  targetDim: 768,        // PCA target dimension
  sparsity: 0.7,         // 70% zeros per vector
  nqubits: 17,           // Qubits = log2(targetDim)
  optimizationLevel: 1,  // Qiskit transpiler optimization
}
```

Override in `qblot.env`:
```bash
DEFAULT_BACKEND=ibm_brisbane
DEFAULT_SHOTS=4096
```

## Pipeline Architecture

```
Input Text
    ↓
[embed] E5/Qwen Embeddings
    ↓
[clean_chat.py] Optional cleaning
    ↓
[qft_one.py pca] PCA projection → 768D
    ↓
[qft_one.py prep] Sparsify (70%) + pad pow2
    ↓
[qft_one.py prerank] Theme-based preselection (optional)
    ↓
[layered_qft.py] Build QFT circuit
    ├─ RoPE phase encoding
    ├─ State preparation
    ├─ Hadamard spread
    ├─ QFT forward
    ├─ Theme-gated RZ phases
    ├─ QFT inverse
    └─ Measurement
    ↓
[IBM Quantum Runtime] Execute circuit
    ↓
[decode_qft_hist.py] Decode counts → spectrum
    ↓
[emit_prompt.py] Generate payload (optional)
    ↓
Results: counts, decoded evidence, payload
```

## Examples

### Example 1: Process Chat History

```bash
# 1. Generate embeddings from conversations
node qft-runner.js embed --input conversations.txt --model e5

# 2. Run QFT with theme 5
node qft-runner.js run \
  --vectors conversations_e5.npy \
  --theme-id 5 \
  --shots 8192

# 3. Check results
node qft-runner.js status
```

### Example 2: Full Pipeline with RoPE

```bash
# Create RoPE hint
echo '{"hint": "trading patterns", "focus": "volatility"}' > rope.json

# Run full pipeline
node qft-runner.js full \
  --input market_data.txt \
  --theme-id 2 \
  --rope rope.json \
  --layered \
  --shots 16384
```

### Example 3: Multiple Theme Analysis

```bash
# Test different themes
for theme in {0..5}; do
  node qft-runner.js run \
    --vectors data_e5.npy \
    --theme-id $theme \
    --shots 8192
  mv qft_counts.json theme_${theme}_counts.json
done
```

## Python Dependencies

Required Python scripts (included in your upload):
- `embed_e5.py` - E5 embeddings
- `embed_qwen_api.py` - Qwen embeddings
- `qft_one.py` - Main QFT pipeline
- `layered_qft.py` - Circuit builder
- `clean_chat.py` - Chat cleaning
- `decode_qft_hist.py` - Result decoder (optional)
- `emit_prompt.py` - Payload generator (optional)

Python packages:
```bash
pip install numpy qiskit qiskit-ibm-runtime sentence-transformers requests
```

## Offline SC1028 Analysis

Rollout and analysis are intentionally separate phases.

```bash
python analyze_sc1028.py artifacts/sc1028_telemetry.jsonl --out artifacts/sc1028_analysis.json
```

Outputs:

- primitive frequency
- primitive transition matrix
- divergence across seeds
- early-failure signatures

## Reasoning Attestation

Generate explicit cryptographic and/or spectral attestation from telemetry:

```bash
node qft-runner.js status \
  --telemetry_out artifacts/sc1028_telemetry.jsonl \
  --attestation_mode both \
  --attestation_out artifacts/semantic_attestation.json \
  --provenance_tag
```

Verify attestation offline:

```bash
python semantic_proof.py verify --attestation artifacts/semantic_attestation.json
```

## Troubleshooting

### IBM Quantum Connection Issues

```bash
# Verify credentials
echo $IBM_CLOUD_API_KEY
echo $IBM_QUANTUM_CRN

# Test with IBM CLI
qiskit-ibm-runtime --version
```

### Module Not Found

```bash
# Ensure Python scripts are in same directory
ls -la *.py

# Or set custom path in qft-runner.js:
pythonScripts: '/path/to/scripts',
```

### Memory Issues with Large Embeddings

Reduce dimensions or sparsity:
```javascript
targetDim: 512,  // instead of 768
sparsity: 0.8,   // 80% zeros
```

## API Reference

### QFTRunner Class

```javascript
import { QFTRunner } from './qft-runner.js';

const runner = new QFTRunner('./qblot.env');

// Embed
const vectors = await runner.embed('data.txt', 'e5');

// Run QFT
const results = await runner.runQFT({
  vectors: vectors,
  themeId: 2,
  backend: 'ibm_brisbane',
  shots: 8192,
  layered: true,
});

// Full workflow
await runner.full('data.txt', {
  themeId: 3,
  model: 'e5',
  layered: true,
});

// Status
await runner.status();
```

## Performance Notes

- **E5 embeddings**: ~500 texts/sec on CPU
- **Qwen embeddings**: ~100 texts/sec via API
- **QFT execution**: 1-5 minutes on IBM Quantum (varies by backend queue)
- **Target dimension**: 768D = 17 qubits (optimal for most backends)

## License

MIT

## Author

TradeLayer

---

For issues or questions, check the Python script logs or IBM Quantum dashboard.
