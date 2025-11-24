# QFT Runner - Quick Reference

## Installation
```bash
chmod +x setup.sh
./setup.sh
```

## Configuration
```bash
cp qblot.env.example qblot.env
# Edit qblot.env with your IBM credentials
source qblot.env
```

## Common Commands

### Generate Embeddings
```bash
# E5 (local)
node qft-runner.js embed --input data.txt --model e5

# Qwen (API)
node qft-runner.js embed --input data.txt --model qwen --token YOUR_TOKEN
```

### Run QFT
```bash
# Basic
node qft-runner.js run --vectors data_e5.npy --theme-id 2

# Advanced
node qft-runner.js run \
  --vectors data_e5.npy \
  --theme-id 5 \
  --backend ibm_brisbane \
  --shots 16384 \
  --layered
```

### Full Pipeline
```bash
node qft-runner.js full \
  --input conversations.txt \
  --theme-id 2 \
  --layered
```

### Check Status
```bash
node qft-runner.js status
```

## Makefile Shortcuts

```bash
make help           # Show all commands
make install        # Install dependencies
make status         # Pipeline status

# Quick runs
make embed INPUT=data.txt MODEL=e5
make run THEME=2 SHOTS=8192
make full INPUT=data.txt THEME=3

# Utilities
make test           # Test pipeline
make clean          # Clean artifacts
make clean-all      # Deep clean
make themes         # Multi-theme analysis
```

## Pipeline Flow

```
Text Input
    ↓
[embed] → vectors.npy
    ↓
[PCA] → 768D projection
    ↓
[prep] → sparsify + pad
    ↓
[QFT] → quantum circuit
    ↓
[IBM] → execute
    ↓
[decode] → results
```

## Theme IDs

Theme IDs control interference patterns:
- `0` - Baseline (no theme gates)
- `1-5` - Different phase patterns
- Higher IDs = more randomization

## Backend Selection

```bash
# Fast but smaller
--backend ibm_torino    # 127 qubits

# Larger circuit capacity
--backend ibm_brisbane  # 133 qubits
```

## Shot Counts

- `1024` - Quick test
- `4096` - Standard
- `8192` - Good quality
- `16384` - High precision

## Typical Workflows

### Workflow 1: Quick Test
```bash
make test
```

### Workflow 2: Production Run
```bash
# 1. Generate embeddings
make embed INPUT=production_data.txt

# 2. Run with optimal settings
make run THEME=3 SHOTS=16384 BACKEND=ibm_brisbane

# 3. Check results
cat qft_counts.json
```

### Workflow 3: Multi-Theme Analysis
```bash
make themes
ls theme_*_counts.json
```

### Workflow 4: Batch Processing
```bash
make batch
ls results/
```

## Troubleshooting

### "IBM credentials not set"
```bash
source qblot.env
make check-env
```

### "Module not found"
```bash
# Python scripts must be in same directory
ls -la *.py
```

### "Memory error"
Reduce target dimension in qft-runner.js:
```javascript
targetDim: 512,  // instead of 768
```

## File Outputs

| File | Description |
|------|-------------|
| `qft_Z.npy` | PCA-projected vectors |
| `vectors_pca_topk.npy` | Sparsified vectors |
| `qft_counts.json` | Measurement counts |
| `decoded_evidence.json` | Decoded spectrum |
| `request_skeleton.json` | Payload (if RoPE) |

## Environment Variables

```bash
IBM_CLOUD_API_KEY        # Required: IBM API key
IBM_QUANTUM_CRN          # Required: Instance CRN
DEFAULT_BACKEND          # Optional: Default backend
DEFAULT_SHOTS            # Optional: Default shots
HF_TOKEN                 # Optional: For Qwen
```

## Performance Tips

1. **Use E5 locally** - Faster than API
2. **Start with low shots** - Test with 1024
3. **Cache embeddings** - Reuse .npy files
4. **Theme prerank** - Use for large datasets
5. **Monitor queue** - Check IBM backend status

## Resources

- IBM Quantum: https://quantum.ibm.com/
- Qiskit Docs: https://docs.quantum.ibm.com/
- E5 Model: https://huggingface.co/intfloat/e5-base-v2
- Qwen Embeddings: https://huggingface.co/Qwen/Qwen3-Embedding-8B

## Support

Check logs:
```bash
make logs
```

Verify environment:
```bash
make check-env
```

Show info:
```bash
make info
```
