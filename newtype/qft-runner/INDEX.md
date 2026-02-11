# 🌊 QFT Runner - Quantum Fourier Transform Pipeline Orchestrator

A modern TypeScript/JavaScript wrapper for your Python-based Quantum Fourier Transform pipeline.

## 📦 Package Contents

```
qft-runner/
├── qft-runner.js           # Main JavaScript runner (use this!)
├── qft-runner.ts           # TypeScript version (optional)
├── package.json            # Node.js dependencies
├── qblot.env.example       # Configuration template
├── setup.sh                # Automated setup script
├── Makefile                # Convenient shortcuts
├── example-usage.js        # 10 usage examples
├── gitignore               # Git ignore patterns
├── README.md               # Full documentation
├── QUICKSTART.md           # Quick reference
└── PROJECT_SUMMARY.md      # This overview
```

## 🚀 Getting Started (30 seconds)

```bash
# 1. Setup
cp qblot.env.example qblot.env
nano qblot.env  # Add your IBM credentials
./setup.sh

# 2. Test
source qblot.env
make test

# 3. Real run
node qft-runner.js full --input data.txt --theme-id 2
```

## 📚 Documentation Guide

**Start Here:**
1. Read **PROJECT_SUMMARY.md** (this file) - Overview
2. Run `./setup.sh` - Installation
3. Read **QUICKSTART.md** - Quick commands
4. Check **example-usage.js** - Code examples
5. Reference **README.md** - Full details

## 🎯 What You Need

### Environment
- Node.js 18+
- Python 3.8+
- IBM Quantum account

### Your Python Scripts (Required)
These should be in the same directory:
- `qft_one.py` - Main orchestrator
- `layered_qft.py` - Circuit builder
- `embed_e5.py` - E5 embeddings
- `embed_qwen_api.py` - Qwen embeddings
- `clean_chat.py` - Chat cleaner

Optional:
- `decode_qft_hist.py` - Result decoder
- `emit_prompt.py` - Payload generator

## ⚡ Quick Commands

```bash
# Status
node qft-runner.js status
make status

# Embed
node qft-runner.js embed --input data.txt
make embed INPUT=data.txt

# Run QFT
node qft-runner.js run --vectors data_e5.npy --theme-id 2
make run THEME=2

# Full pipeline
node qft-runner.js full --input data.txt --theme-id 3
make full INPUT=data.txt THEME=3

# Help
node qft-runner.js
make help
```

## 🎨 Key Features

### 1. Simplified Workflow
One command instead of multiple Python scripts:
```bash
make full INPUT=conversations.txt THEME=2
```

### 2. Multi-Theme Analysis
Test different interference patterns automatically:
```bash
make themes  # Runs themes 0-5
```

### 3. Batch Processing
Process multiple files:
```bash
make batch  # Processes all .txt files
```

### 4. Programmatic API
Use in your Node.js apps:
```javascript
import { QFTRunner } from './qft-runner.js';
const runner = new QFTRunner();
await runner.full('data.txt', { themeId: 2 });
```

### 5. Error Handling
Graceful failures, retries, helpful messages

### 6. Status Tracking
Monitor pipeline artifacts and progress

## 🏗️ Pipeline Architecture

```
Input (text/JSONL)
    ↓
[JavaScript Orchestrator]
    ↓
embed_e5.py → vectors.npy
    ↓
qft_one.py pca → 768D projection
    ↓
qft_one.py prep → sparsify + pad
    ↓
layered_qft.py → build circuit
    ↓
IBM Quantum Runtime → execute
    ↓
decode_qft_hist.py → analyze
    ↓
Results (counts, evidence, payload)
```

## 💡 Use Cases

### Research
- Quantum interference pattern analysis
- Fourier transform experiments
- Embedding space exploration

### Trading (TradeLayer)
- Market signal analysis via quantum interference
- Pattern recognition in financial data
- Multi-dimensional correlation discovery

### Data Science
- High-dimensional vector analysis
- Batch document processing
- Theme-based clustering

## 🎓 Examples

### Example 1: Simple Test
```bash
make test
```

### Example 2: Production Run
```bash
node qft-runner.js full \
  --input production_data.txt \
  --theme-id 5 \
  --backend ibm_brisbane \
  --shots 16384 \
  --layered
```

### Example 3: Multi-Theme
```bash
for theme in {0..5}; do
  make run THEME=$theme SHOTS=8192
  mv qft_counts.json theme_${theme}_counts.json
done
```

### Example 4: Programmatic
```javascript
// See example-usage.js for 10 detailed examples
const runner = new QFTRunner();

// Market analysis
await runner.full('market_signals.txt', {
  themeId: 5,
  shots: 8192,
});

// Read results
const results = JSON.parse(
  readFileSync('qft_counts.json', 'utf-8')
);
```

## 🔧 Configuration

Edit `qft-runner.js` for defaults:
```javascript
{
  defaultBackend: 'ibm_torino',
  defaultShots: 8000,
  targetDim: 768,
  sparsity: 0.7,
  nqubits: 17,
}
```

Or use environment variables in `qblot.env`:
```bash
DEFAULT_BACKEND=ibm_brisbane
DEFAULT_SHOTS=4096
```

## 📊 Performance

- Embeddings: 100-500 texts/sec
- QFT Execution: 1-5 minutes
- Full Pipeline: 3-10 minutes typical

Scale with:
- Batch processing
- Lower shots for testing
- Parallel theme analysis

## ❓ Troubleshooting

### "IBM credentials not set"
```bash
source qblot.env
make check-env
```

### "Python script not found"
```bash
ls -la *.py  # Ensure scripts are present
```

### Need help?
```bash
make help        # Show all commands
make check-env   # Verify setup
make info        # Show configuration
```

## 📁 File Reference

| File | Purpose | Start Here? |
|------|---------|-------------|
| PROJECT_SUMMARY.md | Overview | ✅ Yes |
| QUICKSTART.md | Quick reference | ✅ Yes |
| README.md | Full documentation | Later |
| qft-runner.js | Main script | Use it |
| example-usage.js | Code examples | Learn from |
| setup.sh | Installation | Run first |
| Makefile | Shortcuts | Use often |

## 🎯 Next Steps

1. **Setup** - Run `./setup.sh`
2. **Configure** - Edit `qblot.env`
3. **Test** - Run `make test`
4. **Explore** - Try examples in `example-usage.js`
5. **Build** - Integrate into your workflow

## 💼 TradeLayer Integration

As founder of TradeLayer, you can use this for:
- Quantum analysis of market signals
- Pattern discovery in trading data
- Multi-dimensional correlation analysis
- Real-time market interference patterns

See `example-usage.js` example #8 for TradeLayer-specific integration.

## 📞 Support

- Check `README.md` troubleshooting section
- Run `make check-env` for diagnostics
- View logs with `make logs`
- IBM Quantum: https://quantum.ibm.com/

## 📄 License

MIT

---

**Ready?** → `./setup.sh && make test`

**Questions?** → Check QUICKSTART.md

**Deep dive?** → Read README.md

**Code?** → See example-usage.js
