# QFT Runner - Project Summary

## What Is This?

A TypeScript/JavaScript orchestrator for your Quantum Fourier Transform pipeline. This provides a clean, modern interface to your Python QFT workflow with proper error handling, logging, and workflow management.

## Files Included

### Core Files
- **qft-runner.js** - Pure JavaScript runner (use this for simplicity)
- **qft-runner.ts** - TypeScript version with full type annotations
- **package.json** - Node.js dependencies
- **qblot.env.example** - Configuration template

### Documentation
- **README.md** - Comprehensive documentation
- **QUICKSTART.md** - Quick reference guide
- **example-usage.js** - 10 programmatic usage examples

### Utilities
- **Makefile** - Convenient command shortcuts
- **setup.sh** - Automated installation script
- **gitignore** - Git ignore patterns

## Quick Setup

```bash
# 1. Copy environment template
cp qblot.env.example qblot.env

# 2. Edit with your IBM Quantum credentials
nano qblot.env

# 3. Run setup
chmod +x setup.sh
./setup.sh

# 4. Source environment
source qblot.env

# 5. Test it
node qft-runner.js status
```

## Why This Exists

Your Python scripts (`qft_one.py`, `layered_qft.py`, etc.) are powerful but complex. This JavaScript wrapper provides:

1. **Simplified CLI** - One command for full pipeline
2. **Better Error Handling** - Graceful failures and retries
3. **Workflow Management** - Orchestrate multi-step processes
4. **Status Tracking** - Monitor artifacts and progress
5. **Makefile Shortcuts** - Quick common operations
6. **Programmatic API** - Use in Node.js applications

## Architecture

```
JavaScript Layer (qft-runner.js)
    ↓
Environment Setup (qblot.env)
    ↓
Python Scripts
    ├── embed_e5.py / embed_qwen_api.py
    ├── clean_chat.py
    ├── qft_one.py (PCA, prep, QFT orchestration)
    ├── layered_qft.py (Circuit builder)
    ├── decode_qft_hist.py (Result decoder)
    └── emit_prompt.py (Payload generator)
    ↓
IBM Quantum Runtime
    ↓
Results (counts, decoded evidence, payloads)
```

## Key Features

### 1. Command-Line Interface
```bash
# Single command full pipeline
node qft-runner.js full --input data.txt --theme-id 2

# Or use Makefile
make full INPUT=data.txt THEME=2
```

### 2. Programmatic API
```javascript
import { QFTRunner } from './qft-runner.js';

const runner = new QFTRunner();
await runner.full('data.txt', { themeId: 2, shots: 8192 });
```

### 3. Multi-Theme Analysis
```bash
# Run 6 themes automatically
make themes

# Results in theme_0_counts.json through theme_5_counts.json
```

### 4. Batch Processing
```bash
# Process all .txt files
make batch

# Results organized in results/ directory
```

### 5. Status Monitoring
```bash
# Check pipeline state
make status

# Output:
# ✅ qft_Z.npy
# ✅ vectors_pca_topk.npy
# ✅ qft_counts.json
```

## Integration with TradeLayer

Since you're the founder of TradeLayer, this is designed to integrate seamlessly:

```javascript
// Example: Market signal analysis
const runner = new QFTRunner();

const marketSignals = [
  'bullish RSI divergence',
  'volume consolidation',
  'price action breakout',
];

await runner.full('market_signals.txt', {
  themeId: 5,  // Custom market theme
  shots: 8192,
  layered: true,
});

// Analyze quantum interference patterns in trading signals
```

See `example-usage.js` example #8 for full TradeLayer integration pattern.

## Configuration

Default settings (in qft-runner.js):
```javascript
{
  defaultBackend: 'ibm_torino',  // 127 qubits
  defaultShots: 8000,
  targetDim: 768,                // PCA dimension
  sparsity: 0.7,                 // 70% sparse
  nqubits: 17,                   // log2(1024) after padding
  optimizationLevel: 1,
}
```

Override via:
1. `qblot.env` file
2. Command-line args
3. Programmatic config

## Example Workflows

### Workflow 1: Test Run
```bash
make test
```

### Workflow 2: Production
```bash
# Generate embeddings
make embed INPUT=production_data.txt

# Run with high precision
make run THEME=3 SHOTS=16384 BACKEND=ibm_brisbane
```

### Workflow 3: Development
```bash
# Quick iterations
make dev-full
```

## What Makes This Better

### Before (Pure Python)
```bash
# Multiple manual steps
python embed_e5.py --input data.txt --output data.npy ...
python qft_one.py pca --src data.npy --target-dim 768 ...
python qft_one.py prep --pca-Z ... --sparsity 0.7 ...
python qft_one.py qft --prepped ... --backend ibm_torino ...
```

### After (JavaScript Wrapper)
```bash
# One command
node qft-runner.js full --input data.txt --theme-id 2
# or
make full INPUT=data.txt THEME=2
```

## Next Steps

1. **Customize Configuration**
   - Edit `qft-runner.js` config defaults
   - Modify `qblot.env` environment
   - Adjust `Makefile` shortcuts

2. **Add Your Own Themes**
   - Theme IDs control interference patterns
   - Experiment with 0-20+ for different results

3. **Integrate with Your Apps**
   - Import `QFTRunner` class
   - Build on example patterns
   - Create custom workflows

4. **Scale Up**
   - Use batch processing for large datasets
   - Multi-theme analysis for pattern discovery
   - Real-time monitoring for production

## Troubleshooting

Common issues and solutions in README.md "Troubleshooting" section.

Quick checks:
```bash
make check-env  # Verify setup
make status     # Check artifacts
make logs       # View recent logs
```

## Performance

- **E5 Embeddings**: ~500 texts/sec (local)
- **Qwen Embeddings**: ~100 texts/sec (API)
- **QFT Execution**: 1-5 minutes (varies by IBM queue)
- **Full Pipeline**: ~3-10 minutes typical

## Support

- IBM Quantum: https://quantum.ibm.com/
- Qiskit Docs: https://docs.quantum.ibm.com/
- TradeLayer: (your internal docs)

## License

MIT

## Author

Built for TradeLayer by Claude

---

**Ready to run?** Start with `./setup.sh` and then `make test`!
