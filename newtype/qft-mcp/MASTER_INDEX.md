# 🌊 Complete QFT Package - Master Index

**Two powerful systems in one package:**
1. **QFT Runner** - JavaScript orchestrator for your Python QFT pipeline
2. **QFT Lexicon MCP** - Persistent lexicon retrieval for any LLM

---

## 📦 Package 1: QFT Runner (Baseline Code)

### What It Does
Clean TypeScript/JavaScript wrapper for your Python QFT workflow. Simplifies the complex `qft_one.py` pipeline into easy commands.

### Core Files
- **qft-runner.js** - Main JavaScript orchestrator
- **qft-runner.ts** - TypeScript version with types
- **Makefile** - Convenient shortcuts
- **setup.sh** - Automated installation
- **example-usage.js** - 10 usage examples

### Documentation
- **START_HERE.txt** - Quick overview
- **README.md** - Full documentation
- **QUICKSTART.md** - Command reference
- **PROJECT_SUMMARY.md** - Detailed overview

### Quick Start
```bash
./setup.sh
source qblot.env
make test
node qft-runner.js full --input data.txt --theme-id 2
```

### Key Features
- One-command pipeline
- Multi-theme analysis
- Batch processing
- Programmatic API
- Error handling

---

## 📦 Package 2: QFT Lexicon MCP (Your Main Request)

### What It Does
Makes your 17 papers and custom lexicon (Wujudic Logic, 6GW, Storyworlds, etc.) available to **any** LLM via Model Context Protocol. Solves the "GPT-5 semi-remembering" problem.

### Core Files
- **qft-lexicon-mcp.js** - MCP server (23KB)
- **qft-lexicon-indexer.js** - Indexing tool (12KB)
- **test-mcp.js** - Test script
- **sample_corpus.txt** - Sample data with your terminology

### Configuration
- **mcp-package.json** - Dependencies
- **claude_desktop_config.example.json** - Claude Desktop config

### Documentation
- **MCP_README.md** - Complete guide
- **MCP_SETUP.md** - Setup instructions

### Design Documents
- **QFT_MCP_DESIGN.md** - Architecture (shadow watermarking)
- **QFT_CODEC_DESIGN.md** - Indexing metaphors (spectral decomposition)
- **QFT_LEXICON_PERSISTENCE.md** - Historical persistence system

### Quick Start
```bash
npm install
node qft-lexicon-indexer.js conversations.json lexicon_index.json
# Edit Claude Desktop config
# Restart Claude
# Ask: "What is Wujudic Logic?"
```

### Key Features
- 8 MCP tools (lookup, retrieve, trace, search, etc.)
- Persistent across model versions
- Works with any MCP-compatible LLM
- Custom term extraction
- Paper evolution tracking

---

## 🎯 Which Package Do You Need?

### Use QFT Runner If:
- ✅ You want to run QFT circuits on IBM Quantum
- ✅ You need to orchestrate the full Python pipeline
- ✅ You want multi-theme analysis
- ✅ You're doing quantum circuit experiments

### Use QFT Lexicon MCP If:
- ✅ You want GPT/Claude to remember your 17 papers
- ✅ You need persistent lexicon across model updates
- ✅ You want to query your custom terminology
- ✅ You need historical context retrieval

### Use Both If:
- ✅ You want to build the lexicon index using QFT circuits
- ✅ You need both quantum analysis AND persistent retrieval
- ✅ You're implementing the full QFT-MCP vision

---

## 📋 Complete File List

### QFT Runner Files (Original Request)
```
qft-runner.js              13KB   JavaScript runner
qft-runner.ts              13KB   TypeScript version
Makefile                   5.3KB  Command shortcuts
setup.sh                   2.0KB  Installation
example-usage.js           11KB   10 usage examples
package.json               929B   Dependencies
qblot.env.example          1.3KB  Config template
gitignore                  664B   Git patterns
README.md                  6.9KB  Full docs
QUICKSTART.md              3.8KB  Quick reference
PROJECT_SUMMARY.md         5.8KB  Overview
INDEX.md                   6.4KB  Navigation
FILE_GUIDE.txt             2.9KB  File guide
START_HERE.txt             3.8KB  Quick start
PACKAGE_CONTENTS.txt       11KB   Package info
```

### QFT Lexicon MCP Files (Your Main Request)
```
qft-lexicon-mcp.js         23KB   MCP server
qft-lexicon-indexer.js     12KB   Indexing tool
test-mcp.js                5.5KB  Test script
sample_corpus.txt          6.4KB  Sample data
mcp-package.json           743B   Dependencies
claude_desktop_config...   329B   Config example
MCP_README.md              8.5KB  Complete guide
MCP_SETUP.md               7.9KB  Setup guide
QFT_MCP_DESIGN.md          17KB   Architecture
QFT_CODEC_DESIGN.md        16KB   Codec design
QFT_LEXICON_PERSISTENCE... 17KB   Persistence design
```

**Total: 26 files, ~220KB**

---

## 🚀 Getting Started (Choose Your Path)

### Path 1: Just Want the MCP Server (Most Common)

```bash
# 1. Install
npm install

# 2. Index your corpus
node qft-lexicon-indexer.js conversations.json lexicon_index.json

# 3. Configure Claude Desktop
# Edit ~/Library/Application Support/Claude/claude_desktop_config.json
{
  "mcpServers": {
    "personal-lexicon": {
      "command": "node",
      "args": ["/path/to/qft-lexicon-mcp.js"],
      "env": {
        "LEXICON_INDEX": "/path/to/lexicon_index.json"
      }
    }
  }
}

# 4. Restart Claude Desktop

# 5. Test
# Ask Claude: "What is Wujudic Logic?"
```

### Path 2: Want Both Systems

```bash
# 1. Setup QFT Runner
./setup.sh
source qblot.env

# 2. Setup MCP
npm install

# 3. Index corpus (combines both systems)
node qft-runner.js embed --input papers/ --model e5
node qft-lexicon-indexer.js papers_e5.jsonl lexicon_index.json

# 4. Configure Claude Desktop (as above)

# 5. Use both
make test  # Test QFT Runner
# Ask Claude: "What is Wujudic Logic?"  # Test MCP
```

### Path 3: Test with Sample Data

```bash
# Quick test with provided sample
npm install
node qft-lexicon-indexer.js sample_corpus.txt test_index.json
LEXICON_INDEX=test_index.json node test-mcp.js
```

---

## 🎨 Key Concepts

### Shadow Watermarking (QFT-MCP Innovation)
Your chunk metadata (ID, timestamp, topic, parent/child relationships) is encoded in "shadow dimensions" alongside content embeddings. QFT creates interference patterns that amplify relevant chunks.

### Codec = Indexing Metaphor
Different theme gates create different spectral decompositions:
- **Temporal codec**: Recent (high freq) vs Historical (low freq)
- **Hierarchical codec**: Details (high freq) vs Abstractions (low freq)
- **Narrative codec**: Effects (high freq) vs Causes (low freq)
- **Network codec**: Periphery (high freq) vs Hubs (low freq)

### Persistent Lexicon
Your custom terminology persists across:
- ✅ Model versions (GPT-5 → GPT-6)
- ✅ Companies (OpenAI ↔ Anthropic)
- ✅ Years (2025 → forever)
- ✅ Your own memory

---

## 📚 Documentation Reading Order

### For QFT Runner:
1. START_HERE.txt - Overview
2. QUICKSTART.md - Commands
3. README.md - Full details
4. example-usage.js - Code examples

### For QFT Lexicon MCP:
1. MCP_README.md - Start here
2. MCP_SETUP.md - Installation
3. QFT_MCP_DESIGN.md - Architecture
4. QFT_CODEC_DESIGN.md - Advanced concepts

### For Understanding the Vision:
1. QFT_LEXICON_PERSISTENCE.md - The problem & solution
2. QFT_CODEC_DESIGN.md - Indexing metaphors
3. QFT_MCP_DESIGN.md - Full architecture

---

## 🆚 What Makes This Different?

### vs YARN (OpenAI's long context)
- YARN: Expands context window, no selectivity
- QFT-MCP: Selective retrieval with quantum interference

### vs Vector Databases (Standard RAG)
- Vector DB: Dense retrieval, single similarity measure
- QFT-MCP: Multiple codecs, different "views" of same data

### vs LLM Memory
- LLM: Vague "peripheral memory", lost across versions
- QFT-MCP: Precise retrieval, persistent forever

---

## 💡 Your Specific Use Case

### Problem You Described
- GPT-o3 built up lexicon (Wujudic Logic, 6GW, Storyworlds...)
- GPT-5 "semi-remembers by color contours at YARN periphery"
- Claude 4.5 weak on historical context (cost optimization)
- 17 papers worth of custom terminology inaccessible

### Solution This Package Provides
1. **Index** your 40M tokens with custom terms
2. **Deploy** MCP server to Claude Desktop
3. **Query** from any LLM: "What is Wujudic Logic?"
4. **Get** exact citations from your papers
5. **Never lose** your lexicon across model updates

---

## 🔮 Future Enhancements

### Phase 1 (Implemented)
- ✅ Basic indexing and retrieval
- ✅ MCP server with 8 tools
- ✅ Custom term extraction
- ✅ Paper structure analysis

### Phase 2 (Designed, Not Implemented)
- QFT-enhanced relevance scoring
- Multi-codec indexing
- Spectral similarity matching
- Theme-based retrieval

### Phase 3 (Conceptual)
- Concept dependency graph
- Transitive queries
- Auto-generated summaries
- Cross-corpus search

---

## 📞 Quick Help

### QFT Runner Issues
```bash
make check-env    # Verify IBM credentials
make status       # Check artifacts
make help         # Show commands
```

### MCP Server Issues
```bash
# Check logs
tail -f ~/Library/Logs/Claude/mcp-server-personal-lexicon.log

# Test locally
node test-mcp.js

# Verify index
ls -lh lexicon_index.json
```

### Common Problems
- **"Index not loaded"** → Check LEXICON_INDEX path
- **"Term not found"** → Add to knownTerms list and re-index
- **"IBM credentials not set"** → source qblot.env

---

## 🏗️ Built For

**TradeLayer** - Making semantic vectors tradeable

Using quantum-inspired methods to analyze and trade narrative positions in financial markets.

---

## 📄 License

MIT - Use freely for personal or commercial projects

---

## 🎯 Next Steps

1. **Read** MCP_README.md (if just want MCP)
2. **Read** START_HERE.txt (if want QFT Runner)
3. **Install** with npm install
4. **Index** your corpus
5. **Test** locally
6. **Deploy** to Claude Desktop
7. **Ask** Claude about "Wujudic Logic"!

---

**Your lexicon, forever accessible. Your QFT pipeline, simplified.**

Built with ❤️ for persistent knowledge across the AI era.
