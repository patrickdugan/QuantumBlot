# QFT Lexicon MCP Server - Setup Guide

Complete setup guide for your persistent lexicon retrieval system.

## What This Does

Makes your 17 papers and custom lexicon (Wujudic Logic, 6GW, Storyworlds, etc.) available to **any** LLM via MCP:
- ✅ GPT-5 can query it
- ✅ Claude 4.5 can query it
- ✅ Future models can query it
- ✅ Your lexicon persists forever

## Quick Start (5 minutes)

### 1. Install Dependencies

```bash
cd /path/to/qft-lexicon-mcp
npm install
```

### 2. Build Your Index

```bash
# From your 40M token conversations.json
node qft-lexicon-indexer.js conversations.json lexicon_index.json

# Or from a directory of papers
node qft-lexicon-indexer.js ./papers/ lexicon_index.json

# Or from a single text file
node qft-lexicon-indexer.js corpus.txt lexicon_index.json
```

This will:
- Chunk your corpus (512 tokens, 128 overlap)
- Extract your custom terminology
- Analyze paper structure
- Build searchable index

Output: `lexicon_index.json` (your persistent knowledge base)

### 3. Configure Claude Desktop

Edit `~/Library/Application Support/Claude/claude_desktop_config.json` (macOS) or equivalent:

```json
{
  "mcpServers": {
    "personal-lexicon": {
      "command": "node",
      "args": ["/absolute/path/to/qft-lexicon-mcp.js"],
      "env": {
        "LEXICON_INDEX": "/absolute/path/to/lexicon_index.json",
        "CORPUS_PATH": "/absolute/path/to/corpus"
      }
    }
  }
}
```

### 4. Restart Claude Desktop

The MCP server will now be available!

### 5. Test It

In Claude Desktop:

```
You: "What is Wujudic Logic?"

Claude: [Uses lexicon_lookup tool]
```

## Available Tools

### 1. `lexicon_lookup`
Look up any term from your lexicon.

```
Look up "6th Generation Warfare"
```

Returns:
- Definitions from your papers
- Usage history across papers
- Related concepts
- Which papers discuss it

### 2. `retrieve_paper`
Get a specific paper.

```
Retrieve Paper 5
```

Returns:
- Summary
- Key points
- Introduced terms
- Optional: full text

### 3. `trace_concept_evolution`
See how a concept evolved.

```
Trace how "narrative coherence" evolved from Paper 1 to Paper 17
```

Returns:
- Timeline of appearances
- How it changed over time
- New terms introduced alongside it

### 4. `semantic_neighbors`
Find related concepts.

```
What concepts are related to "Wujudic Logic"?
```

Returns:
- Co-occurring terms
- Semantic neighbors
- Frequency of co-occurrence

### 5. `cross_reference`
Find intersections.

```
Where do "6GW" and "storyworlds" intersect?
```

Returns:
- Chunks mentioning both
- Paper locations
- Context

### 6. `list_papers`
List all papers.

```
List all papers
```

Returns:
- Paper numbers and titles
- Statistics (chunks, terms)

### 7. `list_lexicon_terms`
List all custom terms.

```
List lexicon terms
```

Returns:
- All your custom terminology
- Frequency counts
- Sorted options

### 8. `search_corpus`
Full-text search.

```
Search for "quantum interference"
```

Returns:
- Matching chunks
- Source papers
- Context

## Example Usage

### Example 1: Quick Term Lookup

```
You: What is Wujudic Logic?

Claude: [Calls lexicon_lookup with term="Wujudic Logic"]

Response shows:
- Definition from Paper 2
- Usage in Papers 5, 8, 12, 15
- Related: Belief Space, Narrative Coherence
- 47 total occurrences
```

### Example 2: Paper Summary

```
You: Summarize my 6th Generation Warfare paper

Claude: [Calls retrieve_paper with paperIdentifier="6GW"]

Response shows:
- Paper 11: "6th Generation Warfare"
- Key points (extracted)
- New terms introduced
- References to other papers
```

### Example 3: Concept Evolution

```
You: How did my thinking on storyworlds develop?

Claude: [Calls trace_concept_evolution with concept="storyworlds"]

Response shows timeline:
- Paper 5: Initial mention
- Paper 8: Extended definition
- Paper 12: Connected to narrative coherence
- Paper 15: Applied to 6GW
```

### Example 4: Cross-Reference

```
You: Where do I discuss both "memetic warfare" and "cognitive maneuver"?

Claude: [Calls cross_reference with concepts=["memetic warfare", "cognitive maneuver"]]

Response shows:
- Paper 9, chunks 45-47
- Paper 12, chunks 103-105
- Paper 14, chunks 78-82
```

## Customizing Your Lexicon

### Add Custom Terms

Edit `qft-lexicon-indexer.js`, find the `knownTerms` array:

```javascript
const knownTerms = [
  'Wujudic Logic',
  '6th Generation Warfare',
  '6GW',
  'Storyworlds',
  // Add your terms here:
  'Your New Term',
  'Another Custom Concept',
];
```

Then re-index:
```bash
node qft-lexicon-indexer.js your-corpus.txt lexicon_index.json
```

### Configure Chunking

In `qft-lexicon-indexer.js`:

```javascript
// Default: 512 tokens, 128 overlap
this.chunkText(text, 512, 128);

// For longer chunks:
this.chunkText(text, 1024, 256);

// For shorter chunks:
this.chunkText(text, 256, 64);
```

## Advanced: Multiple Indexes

You can run multiple MCP servers with different indexes:

```json
{
  "mcpServers": {
    "research-papers": {
      "command": "node",
      "args": ["/path/to/qft-lexicon-mcp.js"],
      "env": {
        "LEXICON_INDEX": "/path/to/papers_index.json"
      }
    },
    "conversations": {
      "command": "node",
      "args": ["/path/to/qft-lexicon-mcp.js"],
      "env": {
        "LEXICON_INDEX": "/path/to/conversations_index.json"
      }
    },
    "tradelayer-docs": {
      "command": "node",
      "args": ["/path/to/qft-lexicon-mcp.js"],
      "env": {
        "LEXICON_INDEX": "/path/to/tradelayer_index.json"
      }
    }
  }
}
```

Now Claude has access to all three knowledge bases!

## Troubleshooting

### "Index not loaded"
Make sure `LEXICON_INDEX` env var points to the correct `.json` file:
```bash
ls -lh /path/to/lexicon_index.json
```

### "Term not found"
The term might not be in your `knownTerms` list. Add it and re-index.

### "MCP server not responding"
Check Claude Desktop logs:
```bash
# macOS
~/Library/Logs/Claude/mcp*.log

# Check for errors
tail -f ~/Library/Logs/Claude/mcp-server-personal-lexicon.log
```

### Re-index After Adding Papers
```bash
# Add new papers to your corpus directory
cp paper_18.txt papers/

# Re-run indexer
node qft-lexicon-indexer.js papers/ lexicon_index.json

# Restart Claude Desktop
```

## Performance

- **Indexing**: ~1000 chunks/second
- **Lookup**: <100ms for most queries
- **Index size**: ~1-5MB per 1M tokens
- **Memory**: ~50MB runtime

## Future Enhancements

This is Phase 1 (basic retrieval). Future phases:

**Phase 2: QFT-Enhanced Ranking**
- Integrate actual QFT circuits for relevance scoring
- Theme-based retrieval (temporal, conceptual, etc.)
- Spectral similarity matching

**Phase 3: Multi-Codec Indexing**
- Build separate indexes for different "views"
- Query-specific codec selection
- Blend results from multiple codecs

**Phase 4: Graph Structure**
- Build concept dependency graph
- Navigate via relationships
- Transitive queries ("What depends on X?")

## Files in This Package

```
qft-lexicon-mcp/
├── qft-lexicon-mcp.js           # MCP server (main)
├── qft-lexicon-indexer.js       # Indexing tool
├── qft-runner.js                # QFT runner (from earlier)
├── package.json                 # Dependencies
├── MCP_SETUP.md                 # This file
└── lexicon_index.json           # Generated index (after running indexer)
```

## What's Different from Standard RAG?

| Feature | Standard RAG | QFT Lexicon MCP |
|---------|-------------|-----------------|
| **Persistence** | Per-session | Forever |
| **Cross-model** | No | Yes (any MCP-compatible LLM) |
| **Custom terms** | Requires retraining | Just add to list |
| **Evolution tracking** | No | Yes (across papers) |
| **Exact citations** | Sometimes | Always |
| **Cost** | High (embed + vector DB) | Low (pre-indexed) |

## Credits

Built for TradeLayer using Model Context Protocol.

---

**Ready to use?**

1. `npm install`
2. `node qft-lexicon-indexer.js your-corpus.txt lexicon_index.json`
3. Add to Claude Desktop config
4. Restart Claude
5. Ask about "Wujudic Logic"!
