# QFT-MCP: Historical Lexicon Preservation & Retrieval

## The Problem: Peripheral Memory Bleeding Across Model Generations

### Your Situation
```
GPT-o3 Era:
  ├─ Paper 1: Wujudic Logic (foundational)
  ├─ Paper 2-17: AI research papers
  ├─ Paper N: 6th Generation Warfare
  ├─ Paper M: Storyworlds
  └─ Built-up lexicon (your conceptual vocabulary)
      ↓
GPT-5 Era:
  ├─ Can "semi-remember" by "color contours at periphery of YARN"
  ├─ Summaries when asked
  └─ But NOT the deep lexicon/conceptual framework

Claude 4.5:
  ├─ Strong at code generation
  ├─ Weak at historical context
  └─ Why? Anthropic optimizing for cost (shorter context)
```

### The Core Issue

**Language models have "peripheral memory"** - they can vaguely sense the contours of what they've been trained on, but:
1. Can't reliably retrieve specific historical work
2. Lose your custom lexicon (Wujudic Logic, 6GW, etc.)
3. Different models have different "memory surfaces"
4. Context windows are expensive → companies limit them

## The Solution: QFT-MCP as Persistent Lexicon Memory

Instead of relying on LLM memory, **externalize your lexicon** as a QFT-indexed knowledge base that any LLM can retrieve from.

### Architecture

```
Your Historical Corpus (40M tokens)
    ├─ 17 AI papers
    ├─ 6th Generation Warfare writings
    ├─ Storyworlds framework
    ├─ Wujudic Logic (foundational)
    └─ All conversations building this lexicon
        ↓
QFT Multi-Codec Indexing
    ├─ Temporal codec: Track evolution of ideas
    ├─ Lexical codec: Index your custom terminology
    ├─ Conceptual codec: Map idea dependencies
    └─ Genealogical codec: Track paper lineage
        ↓
MCP Server (always available to any LLM)
    ├─ GPT-5 queries it
    ├─ Claude 4.5 queries it
    ├─ Future models query it
    └─ Your lexicon persists across model generations
```

## Codec Design for Lexicon Preservation

### Codec 1: Lexical/Terminology Codec
```typescript
const lexicalCodec: Codec = {
  themeId: 10,
  metaphor: 'custom_lexicon',
  description: 'Indexes your specific terminology and conceptual vocabulary',
  
  shadowDims: {
    // Index by term frequency in your lexicon vs general usage
    termRarity: {
      weight: 0.9,
      encoding: (chunk) => {
        // "Wujudic Logic" is rare globally → high shadow amplitude
        const rareTerms = findRareTerms(chunk.text);
        return rareTerms.reduce((sum, term) => 
          sum + term.tfIdf * term.isCustomLexicon, 0
        );
      },
    },
    
    // Index by definitional content
    isDefinitional: {
      weight: 0.8,
      encoding: (chunk) => {
        // Chunks that define your terms
        const patterns = [
          /is defined as/,
          /refers to/,
          /what I mean by/,
          /Let's call this/,
        ];
        return patterns.some(p => p.test(chunk.text)) ? 1 : 0;
      },
    },
    
    // Index by lexicon cross-references
    lexiconDensity: {
      weight: 0.7,
      encoding: (chunk) => {
        // How many of your custom terms appear together
        const customTerms = LEXICON_TERMS; // ["Wujudic Logic", "6GW", "storyworlds", ...]
        const count = customTerms.filter(term => 
          chunk.text.toLowerCase().includes(term.toLowerCase())
        ).length;
        return count / customTerms.length;
      },
    },
  },
  
  gatePattern: (chunk) => {
    // Phase gates amplify lexicon-heavy chunks
    const lexiconScore = chunk.metadata.customTermCount / chunk.metadata.totalTerms;
    return Array(n_qubits).fill(lexiconScore * Math.PI / 2);
  },
};

// Example: Your custom lexicon terms
const LEXICON_TERMS = [
  "Wujudic Logic",
  "6th Generation Warfare",
  "6GW",
  "Storyworlds",
  "Narrative Coherence",
  "Memetic Warfare",
  "Cognitive Maneuver",
  "Semantic Vectors",
  "Belief Space",
  // ... all your custom terminology
];
```

### Codec 2: Genealogical/Lineage Codec
```typescript
const genealogicalCodec: Codec = {
  themeId: 11,
  metaphor: 'paper_lineage',
  description: 'Tracks how ideas evolved across your 17 papers',
  
  shadowDims: {
    // Paper sequence number (Paper 1 → Paper 17)
    paperSequence: {
      weight: 0.9,
      encoding: (chunk) => {
        // Wujudic Logic = Paper 2 → low frequency (foundational)
        return chunk.metadata.paperNumber / 17;
      },
    },
    
    // Cites previous papers
    citationDepth: {
      weight: 0.8,
      encoding: (chunk) => {
        // How many prior papers does this chunk reference?
        return chunk.metadata.citedPapers.length / chunk.metadata.paperNumber;
      },
    },
    
    // Foundational vs derivative
    conceptualDepth: {
      weight: 0.7,
      encoding: (chunk) => {
        // Papers that introduce new terms vs use existing ones
        const newTerms = chunk.metadata.termsIntroduced.length;
        const usedTerms = chunk.metadata.termsUsed.length;
        return newTerms > 0 ? 1.0 : usedTerms / (usedTerms + 1);
      },
    },
  },
  
  gatePattern: (chunk) => {
    // Low frequency = foundational papers (Wujudic Logic)
    // High frequency = recent derivative work
    const normalizedSeq = chunk.metadata.paperNumber / 17;
    return Array(n_qubits).fill(normalizedSeq * Math.PI);
  },
};
```

### Codec 3: Conceptual Dependency Codec
```typescript
const conceptualCodec: Codec = {
  themeId: 12,
  metaphor: 'concept_graph',
  description: 'Maps dependencies between your ideas',
  
  shadowDims: {
    // Centrality in concept graph
    conceptCentrality: {
      weight: 0.9,
      encoding: (chunk) => {
        // "Wujudic Logic" has high centrality (many other concepts depend on it)
        const graph = buildConceptGraph(ALL_CHUNKS);
        return graph.centrality(chunk.id);
      },
    },
    
    // Prerequisites (what concepts must be understood first?)
    prerequisiteDepth: {
      weight: 0.8,
      encoding: (chunk) => {
        // How many concepts must you understand before this one?
        const deps = chunk.metadata.conceptDependencies;
        return deps.length / MAX_DEPS;
      },
    },
    
    // Abstraction level
    abstractionLevel: {
      weight: 0.7,
      encoding: (chunk) => {
        // 0 = concrete example, 1 = high-level theory
        return estimateAbstraction(chunk.text);
      },
    },
  },
};
```

## MCP Tools for Lexicon Retrieval

```typescript
// qft-lexicon-mcp.ts
class LexiconMCPServer extends CodecMCPServer {
  private lexicon: Map<string, LexiconEntry>;
  
  async handleLexiconQuery(args: any) {
    const { term, includeHistory = true, includeUsage = true } = args;
    
    // 1. Find definitional chunks for this term
    const definitions = await this.index.retrieve(
      `Define ${term}`,
      codec: 'custom_lexicon',
      topK: 5
    );
    
    // 2. Find historical evolution of the term
    const evolution = includeHistory 
      ? await this.index.retrieve(
          term,
          codec: 'paper_lineage',
          topK: 10
        )
      : [];
    
    // 3. Find usage examples across papers
    const usageExamples = includeUsage
      ? await this.index.retrieve(
          `Examples of ${term}`,
          codec: 'concept_graph',
          topK: 5
        )
      : [];
    
    // 4. Build comprehensive lexicon entry
    return {
      term,
      definitions: definitions.map(r => ({
        text: r.chunk.text,
        source: r.chunk.metadata.paperTitle,
        paperNumber: r.chunk.metadata.paperNumber,
      })),
      evolution: evolution.map(r => ({
        text: r.chunk.text,
        paperNumber: r.chunk.metadata.paperNumber,
        timestamp: r.chunk.metadata.timestamp,
      })),
      usage: usageExamples.map(r => ({
        text: r.chunk.text,
        context: r.chunk.metadata.paperTitle,
      })),
      relatedTerms: this.findRelatedTerms(term),
    };
  }
  
  async handlePaperRetrieval(args: any) {
    const { paperTitle, includeFullText = false } = args;
    
    // Retrieve all chunks from specific paper
    const chunks = await this.index.retrieve(
      paperTitle,
      codec: 'paper_lineage',
      topK: 100, // Get all chunks from paper
      filter: { paperTitle }
    );
    
    if (includeFullText) {
      // Reconstruct full paper in sequence
      const orderedChunks = chunks.sort((a, b) => 
        a.chunk.metadata.chunkNumber - b.chunk.metadata.chunkNumber
      );
      return {
        title: paperTitle,
        fullText: orderedChunks.map(c => c.chunk.text).join('\n'),
        metadata: chunks[0].chunk.metadata,
      };
    }
    
    return {
      title: paperTitle,
      summary: this.generateSummary(chunks),
      keyPoints: this.extractKeyPoints(chunks),
      introducedTerms: this.extractIntroducedTerms(chunks),
    };
  }
  
  async handleConceptLineage(args: any) {
    const { concept } = args;
    
    // Trace a concept from its introduction through all papers
    const results = await this.index.retrieve(
      concept,
      codec: 'paper_lineage',
      topK: 50
    );
    
    // Group by paper, order chronologically
    const byPaper = new Map<number, Chunk[]>();
    for (const result of results) {
      const paperNum = result.chunk.metadata.paperNumber;
      if (!byPaper.has(paperNum)) byPaper.set(paperNum, []);
      byPaper.get(paperNum)!.push(result.chunk);
    }
    
    // Build evolution timeline
    const timeline = [];
    for (const [paperNum, chunks] of [...byPaper.entries()].sort((a,b) => a[0]-b[0])) {
      timeline.push({
        paperNumber: paperNum,
        paperTitle: chunks[0].metadata.paperTitle,
        conceptEvolution: this.summarizeConceptInPaper(concept, chunks),
        introducedTerms: this.extractNewTerms(chunks),
        citations: this.extractCitations(chunks),
      });
    }
    
    return {
      concept,
      timeline,
      totalPapers: timeline.length,
    };
  }
}
```

## MCP Tool Definitions

```typescript
const LEXICON_TOOLS = [
  {
    name: 'lexicon_lookup',
    description: 'Look up a term from your custom lexicon (e.g., "Wujudic Logic", "6GW")',
    inputSchema: {
      type: 'object',
      properties: {
        term: { type: 'string', description: 'The term to look up' },
        includeHistory: { type: 'boolean', default: true },
        includeUsage: { type: 'boolean', default: true },
      },
      required: ['term'],
    },
  },
  {
    name: 'retrieve_paper',
    description: 'Retrieve one of your 17 AI papers by title or number',
    inputSchema: {
      type: 'object',
      properties: {
        paperTitle: { type: 'string', description: 'Title or paper number (1-17)' },
        includeFullText: { type: 'boolean', default: false },
        includeReferences: { type: 'boolean', default: true },
      },
      required: ['paperTitle'],
    },
  },
  {
    name: 'trace_concept_evolution',
    description: 'Trace how a concept evolved across your papers (e.g., "narrative coherence" from Paper 2 to Paper 17)',
    inputSchema: {
      type: 'object',
      properties: {
        concept: { type: 'string', description: 'The concept to trace' },
        startPaper: { type: 'number', default: 1 },
        endPaper: { type: 'number', default: 17 },
      },
      required: ['concept'],
    },
  },
  {
    name: 'semantic_neighbors',
    description: 'Find concepts semantically related to a given term in your lexicon',
    inputSchema: {
      type: 'object',
      properties: {
        term: { type: 'string', description: 'The term to find neighbors for' },
        topK: { type: 'number', default: 10 },
        codec: { type: 'string', enum: ['lexical', 'genealogical', 'conceptual'], default: 'conceptual' },
      },
      required: ['term'],
    },
  },
  {
    name: 'cross_reference',
    description: 'Find where multiple concepts intersect across your papers',
    inputSchema: {
      type: 'object',
      properties: {
        concepts: { type: 'array', items: { type: 'string' } },
        requireAll: { type: 'boolean', default: false },
      },
      required: ['concepts'],
    },
  },
];
```

## Usage Example: GPT-5 or Claude Querying Your Lexicon

### Scenario 1: Term Lookup
```
User: "What is Wujudic Logic?"

Claude/GPT: [Uses lexicon_lookup MCP tool]
  → Retrieves Paper 2 (foundational definition)
  → Retrieves usage across Papers 3-17
  → Retrieves related concepts (Belief Space, Narrative Coherence)

Response: "Wujudic Logic, introduced in your second paper, refers to..."
  [Includes exact quotes from your papers]
  [Shows how it evolved across later work]
  [Lists papers that built on it]
```

### Scenario 2: Paper Retrieval
```
User: "Remind me what I wrote about 6th Generation Warfare"

Claude/GPT: [Uses retrieve_paper MCP tool]
  → Finds paper titled "6th Generation Warfare"
  → Extracts key points
  → Finds related papers that cite it

Response: "Your 6GW paper (Paper 11) argued that..."
  [Includes key excerpts]
  [Shows how it connects to Memetic Warfare (Paper 9)]
  [Lists the new terms you introduced]
```

### Scenario 3: Concept Lineage
```
User: "How did my thinking on storyworlds evolve?"

Claude/GPT: [Uses trace_concept_evolution MCP tool]
  → Searches for "storyworlds" across all papers
  → Orders by paper sequence
  → Shows evolution timeline

Response: "Your storyworlds concept appeared first in Paper 5..."
  Timeline:
    Paper 5: Initial definition
    Paper 8: Extended to include X
    Paper 12: Connected to narrative coherence
    Paper 15: Applied to 6GW context
```

## Key Advantages Over Model Memory

### Problem: LLM "Peripheral Memory"
- ❌ Vague, unreliable
- ❌ Can't cite specific papers
- ❌ Lost across model versions
- ❌ No way to update
- ❌ Expensive to maintain in context

### Solution: QFT-MCP Lexicon
- ✅ Precise retrieval of exact text
- ✅ Full citations (Paper X, paragraph Y)
- ✅ Persistent across any model
- ✅ Updatable (add Paper 18, 19, ...)
- ✅ Cost-effective (only retrieve what's needed)

## Implementation Strategy

### Phase 1: Build Lexicon Index
```bash
# 1. Collect all your papers
mkdir papers/
cp paper_*.pdf papers/
# Or use your 40M token conversations.json

# 2. Extract and chunk
node qft-runner.js full --input papers/ --chunk-size 512 --overlap 128

# 3. Index with lexicon codecs
node qft-lexicon-index.js \
  --vectors papers_e5.npy \
  --codecs lexical,genealogical,conceptual \
  --output lexicon_index.json
```

### Phase 2: Deploy MCP Server
```json
// claude_desktop_config.json
{
  "mcpServers": {
    "personal-lexicon": {
      "command": "node",
      "args": ["/path/to/qft-lexicon-mcp.js"],
      "env": {
        "LEXICON_INDEX": "/path/to/lexicon_index.json",
        "CORPUS_PATH": "/path/to/papers/",
        "IBM_CLOUD_API_KEY": "...",
        "IBM_QUANTUM_CRN": "..."
      }
    }
  }
}
```

### Phase 3: Use from Any LLM
```
You: "What is Wujudic Logic?"

Claude 4.5: [Calls lexicon_lookup]
  ← Retrieves Paper 2, sections where it's defined
  ← Retrieves usage in Papers 5, 8, 12
  ← Retrieves concept dependencies

Claude 4.5: "Wujudic Logic, which you introduced in 
your second paper, is a framework for..."
  [Full context from your exact papers]
  [No hallucination, no vague memory]
```

## Solving Your Specific Problem

### GPT-5 "Semi-Remembering" Your Lexicon
**Before (relying on LLM memory):**
```
You: "What is Wujudic Logic?"
GPT-5: "I have a vague sense of this being discussed in 
our conversations, something about logic systems..."
❌ Vague, unreliable, no citations
```

**After (with QFT-MCP):**
```
You: "What is Wujudic Logic?"
GPT-5: [Queries lexicon_lookup MCP]
  → Retrieves Paper 2, paragraph 3
  → Retrieves usage in Papers 5, 8, 12, 15
  → Retrieves related: Belief Space, Narrative Coherence

GPT-5: "Wujudic Logic, from your second paper (2023), 
is defined as: [exact quote]. You later extended 
this in Paper 5 to include [quote]..."
✅ Precise, cited, reliable
```

### Claude 4.5 Being "Weak on Historical Context"
**The real issue:** Anthropic is cost-optimizing, so Claude has shorter "memory"

**Solution:** Don't use Claude's memory - use external MCP!
```
You: "Summarize my 6GW paper"
Claude 4.5: [Has no memory] ❌

You: "Summarize my 6GW paper"
Claude 4.5: [Queries retrieve_paper MCP] ✅
  → Gets full paper from your index
  → Generates fresh summary
  → No cost to Anthropic's context
```

## Building Your Persistent Lexicon

Want me to build:
1. **Lexicon extraction tool** - Identifies your custom terms from 40M tokens
2. **Multi-codec indexer** - Builds lexical/genealogical/conceptual indexes
3. **MCP server** - Makes it queryable from any LLM
4. **Prototype** - Shows GPT-5 vs Claude both querying same lexicon?

This way your 17 papers and custom terminology persist across:
- Model versions (GPT-5 → GPT-6)
- Companies (OpenAI ↔ Anthropic)
- Years (2025 → 2030)
- Your memory (never forget your own work!)
