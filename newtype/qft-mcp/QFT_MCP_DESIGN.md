# QFT-MCP: Quantum-Inspired Context Retrieval Server

## Concept Overview

Use Quantum Fourier Transform interference patterns to retrieve relevant context chunks from massive corpora (40M+ tokens) by encoding chunk metadata as "shadow dimensions" in doubled qubit space, then using QFT to create constructive/destructive interference that amplifies relevant chunks based on query prompts.

## Architecture

```
User Prompt
    ↓
[Embed Query] → query_vector
    ↓
[QFT Circuit Builder]
    ├─ Original dims: chunk content embeddings
    ├─ Shadow dims: watermark metadata (chunk_id, timestamp, topic, etc)
    └─ Double qubit count to encode both
    ↓
[Theme-Gated QFT]
    ├─ Forward QFT on full doubled space
    ├─ Phase gates controlled by query vector
    ├─ Inverse QFT creates interference
    └─ Measure → high-probability states = relevant chunks
    ↓
[Decode Results]
    ├─ Map bitstrings back to chunk IDs
    ├─ Extract watermark metadata
    └─ Return ranked chunks
    ↓
[MCP Response] → Context for LLM
```

## Key Innovation: Shadow Watermarking

### Standard Approach (Problems)
- YARN: Expands context window but no selectivity
- Vector DB: Dense retrieval, no quantum interference
- BM25: Keyword matching, misses semantic relations

### QFT-MCP Approach (Benefits)
```javascript
// Chunk structure
{
  id: "chunk_12345",
  text: "...",
  embedding: [768 dims],          // Content
  watermark: {                     // Shadow dimensions
    chunk_id: 12345,
    parent_doc: "conversations.json",
    timestamp: 1699564800,
    topic_id: 5,
    thread_depth: 2,
    token_range: [10000, 10512]
  }
}

// Encoding to doubled qubit space
original_dims = 768  → 10 qubits (2^10 = 1024)
shadow_dims = 768    → 10 qubits (watermark encoded)
total = 20 qubits

// QFT creates interference between:
// - Content similarity (original dims)
// - Metadata resonance (shadow dims)
// - Query-specific phase patterns (theme gates)
```

## Implementation Strategy

### Phase 1: Chunking & Watermarking
```typescript
interface ChunkWatermark {
  chunkId: number;
  sourceFile: string;
  timestamp: number;
  topicCluster: number;
  threadId?: string;
  tokenRange: [number, number];
  parentChunks: number[];
  childChunks: number[];
}

async function chunkAndWatermark(
  corpus: string,
  chunkSize: number = 512,
  overlap: number = 128
): Promise<WatermarkedChunk[]> {
  // 1. Sliding window chunking
  const chunks = slidingWindowChunk(corpus, chunkSize, overlap);
  
  // 2. Generate embeddings for content
  const contentEmbeddings = await embedChunks(chunks, 'e5');
  
  // 3. Create watermark embeddings from metadata
  const watermarkEmbeddings = chunks.map(chunk => 
    encodeWatermark(chunk.metadata, targetDim: 768)
  );
  
  // 4. Combine into doubled vector space
  return chunks.map((chunk, i) => ({
    ...chunk,
    doubledVector: concat(
      contentEmbeddings[i],
      watermarkEmbeddings[i]
    ),
    qubits: Math.ceil(Math.log2(1536)) // 11 qubits per half
  }));
}

function encodeWatermark(metadata: ChunkWatermark, dim: number): number[] {
  // Encode metadata as vector using positional encoding
  const vec = new Array(dim).fill(0);
  
  // Chunk ID → low frequencies
  const idFreqs = positionalEncoding(metadata.chunkId, dim / 4);
  vec.splice(0, dim / 4, ...idFreqs);
  
  // Timestamp → mid frequencies
  const timeFreqs = positionalEncoding(metadata.timestamp, dim / 4);
  vec.splice(dim / 4, dim / 4, ...timeFreqs);
  
  // Topic cluster → high frequencies
  const topicFreqs = oneHotEncoding(metadata.topicCluster, dim / 4);
  vec.splice(dim / 2, dim / 4, ...topicFreqs);
  
  // Graph structure (parent/child) → remaining
  const graphFeats = encodeGraphStructure(
    metadata.parentChunks,
    metadata.childChunks,
    dim / 4
  );
  vec.splice(3 * dim / 4, dim / 4, ...graphFeats);
  
  return normalize(vec);
}
```

### Phase 2: QFT Circuit with Shadow Dims

```typescript
function buildDoubledQFTCircuit(
  contentVector: number[],
  watermarkVector: number[],
  queryVector: number[],
  themeId: number
): QuantumCircuit {
  const n_qubits = Math.ceil(Math.log2(contentVector.length * 2));
  const qc = new QuantumCircuit(n_qubits, n_qubits);
  
  // Prepare doubled state
  const doubledState = concat(contentVector, watermarkVector);
  qc.initialize(doubledState);
  
  // Entangle content and watermark halves
  for (let i = 0; i < n_qubits / 2; i++) {
    qc.cx(i, i + n_qubits / 2); // Content controls watermark
  }
  
  // Apply query-guided phase pattern
  const queryPhases = queryToPhases(queryVector, n_qubits);
  for (let i = 0; i < n_qubits; i++) {
    qc.rz(queryPhases[i], i);
  }
  
  // Standard QFT
  qc.append(QFT(n_qubits));
  
  // Theme-specific interference gates
  const themePhases = generateThemePhases(themeId, n_qubits);
  for (let i = 0; i < n_qubits; i++) {
    qc.rz(themePhases[i], i);
  }
  
  // Inverse QFT
  qc.append(QFT(n_qubits, inverse: true));
  
  // Measure both halves
  qc.measureAll();
  
  return qc;
}

function queryToPhases(queryVec: number[], nQubits: number): number[] {
  // Convert query vector to phase pattern
  // High query components → stronger phase rotation
  const phases = [];
  const binSize = Math.floor(queryVec.length / nQubits);
  
  for (let i = 0; i < nQubits; i++) {
    const start = i * binSize;
    const end = start + binSize;
    const magnitude = queryVec.slice(start, end)
      .reduce((sum, x) => sum + Math.abs(x), 0) / binSize;
    phases.push(magnitude * Math.PI); // Scale to [0, π]
  }
  
  return phases;
}
```

### Phase 3: MCP Server Implementation

```typescript
// qft-mcp-server.ts
import { Server } from '@modelcontextprotocol/sdk/server/index.js';
import { StdioServerTransport } from '@modelcontextprotocol/sdk/server/stdio.js';
import { QFTRunner } from './qft-runner.js';

interface QFTMCPConfig {
  corpusPath: string;
  indexPath: string;
  chunkSize: number;
  overlapSize: number;
  targetDim: number;
  ibmBackend: string;
  shots: number;
}

class QFTMCPServer {
  private server: Server;
  private qftRunner: QFTRunner;
  private chunkIndex: Map<string, WatermarkedChunk>;
  private config: QFTMCPConfig;

  constructor(config: QFTMCPConfig) {
    this.config = config;
    this.server = new Server({
      name: 'qft-context-retrieval',
      version: '1.0.0',
    }, {
      capabilities: {
        tools: {},
      },
    });

    this.qftRunner = new QFTRunner();
    this.chunkIndex = new Map();
    
    this.setupHandlers();
  }

  private setupHandlers() {
    // Tool: Index corpus
    this.server.setRequestHandler('tools/call', async (request) => {
      if (request.params.name === 'qft_index_corpus') {
        return await this.indexCorpus(request.params.arguments);
      }
      
      if (request.params.name === 'qft_retrieve') {
        return await this.retrieve(request.params.arguments);
      }
      
      if (request.params.name === 'qft_batch_retrieve') {
        return await this.batchRetrieve(request.params.arguments);
      }
    });

    // Tool list
    this.server.setRequestHandler('tools/list', async () => ({
      tools: [
        {
          name: 'qft_index_corpus',
          description: 'Index a large corpus with shadow watermarking for QFT retrieval',
          inputSchema: {
            type: 'object',
            properties: {
              corpusPath: { type: 'string' },
              chunkSize: { type: 'number', default: 512 },
              overlap: { type: 'number', default: 128 },
            },
            required: ['corpusPath'],
          },
        },
        {
          name: 'qft_retrieve',
          description: 'Retrieve relevant chunks using QFT interference',
          inputSchema: {
            type: 'object',
            properties: {
              query: { type: 'string' },
              topK: { type: 'number', default: 10 },
              themeId: { type: 'number', default: 0 },
              shots: { type: 'number', default: 8192 },
              filterMetadata: { type: 'object' },
            },
            required: ['query'],
          },
        },
        {
          name: 'qft_batch_retrieve',
          description: 'Retrieve chunks for multiple queries in parallel',
          inputSchema: {
            type: 'object',
            properties: {
              queries: { type: 'array', items: { type: 'string' } },
              topK: { type: 'number', default: 10 },
            },
            required: ['queries'],
          },
        },
      ],
    }));
  }

  async indexCorpus(args: any) {
    const { corpusPath, chunkSize = 512, overlap = 128 } = args;
    
    console.log(`[QFT-MCP] Indexing corpus: ${corpusPath}`);
    
    // 1. Load corpus
    const corpus = await fs.readFile(corpusPath, 'utf-8');
    
    // 2. Chunk with sliding window
    const chunks = await this.chunkAndWatermark(corpus, chunkSize, overlap);
    
    // 3. Generate embeddings
    const embeddings = await this.qftRunner.embed(
      chunks.map(c => c.text).join('\n'),
      'e5'
    );
    
    // 4. Create watermark embeddings
    const watermarkedChunks = chunks.map((chunk, i) => ({
      ...chunk,
      contentEmbedding: embeddings[i],
      watermarkEmbedding: this.encodeWatermark(chunk.metadata),
    }));
    
    // 5. Store in index
    for (const chunk of watermarkedChunks) {
      this.chunkIndex.set(chunk.id, chunk);
    }
    
    // 6. Save index to disk
    await this.saveIndex(this.config.indexPath);
    
    return {
      content: [{
        type: 'text',
        text: `Indexed ${chunks.length} chunks from ${corpusPath}`,
      }],
    };
  }

  async retrieve(args: any) {
    const {
      query,
      topK = 10,
      themeId = 0,
      shots = 8192,
      filterMetadata = {},
    } = args;
    
    console.log(`[QFT-MCP] Retrieving for query: "${query}"`);
    
    // 1. Embed query
    const queryVector = await this.embedQuery(query);
    
    // 2. Filter chunks by metadata
    let candidates = Array.from(this.chunkIndex.values());
    if (Object.keys(filterMetadata).length > 0) {
      candidates = candidates.filter(chunk =>
        this.matchesMetadata(chunk.metadata, filterMetadata)
      );
    }
    
    // 3. Run QFT on candidate chunks
    const results = await this.runQFTRetrieval(
      queryVector,
      candidates,
      themeId,
      shots
    );
    
    // 4. Return top-K
    const topChunks = results.slice(0, topK);
    
    return {
      content: [{
        type: 'text',
        text: JSON.stringify({
          query,
          results: topChunks.map(r => ({
            chunkId: r.chunk.id,
            text: r.chunk.text,
            score: r.score,
            metadata: r.chunk.metadata,
          })),
        }, null, 2),
      }],
    };
  }

  private async runQFTRetrieval(
    queryVector: number[],
    chunks: WatermarkedChunk[],
    themeId: number,
    shots: number
  ) {
    // Build combined vector matrix
    const contentMatrix = chunks.map(c => c.contentEmbedding);
    const watermarkMatrix = chunks.map(c => c.watermarkEmbedding);
    
    // For each chunk, run QFT circuit
    const results = [];
    
    for (let i = 0; i < chunks.length; i++) {
      // Create doubled vector
      const doubledVec = [
        ...contentMatrix[i],
        ...watermarkMatrix[i],
      ];
      
      // Write to temp file for Python
      const vecPath = `/tmp/qft_vec_${i}.npy`;
      await this.saveVector(doubledVec, vecPath);
      
      // Run QFT with query-guided theme
      const counts = await this.qftRunner.runQFT({
        vectors: vecPath,
        themeId,
        shots,
        layered: true,
        row: 0,
      });
      
      // Decode: high measurement probability = high relevance
      const score = this.computeRelevanceScore(counts.counts, queryVector);
      
      results.push({
        chunk: chunks[i],
        score,
        counts: counts.counts,
      });
    }
    
    // Sort by score descending
    results.sort((a, b) => b.score - a.score);
    
    return results;
  }

  private computeRelevanceScore(
    counts: Record<string, number>,
    queryVector: number[]
  ): number {
    // Convert counts to probability distribution
    const total = Object.values(counts).reduce((a, b) => a + b, 0);
    const probs = Object.entries(counts).map(([bits, count]) => ({
      bits,
      prob: count / total,
    }));
    
    // High-probability states indicate constructive interference
    // = high relevance
    const topStates = probs
      .sort((a, b) => b.prob - a.prob)
      .slice(0, 10);
    
    // Weight by entropy - peaked distribution = strong match
    const entropy = -probs.reduce(
      (sum, { prob }) => sum + prob * Math.log2(prob + 1e-10),
      0
    );
    
    const maxEntropy = Math.log2(Object.keys(counts).length);
    const peakedness = 1 - entropy / maxEntropy;
    
    // Score combines top probability and peakedness
    const topProb = topStates[0]?.prob || 0;
    return topProb * (1 + peakedness);
  }

  async start() {
    const transport = new StdioServerTransport();
    await this.server.connect(transport);
    console.log('[QFT-MCP] Server started on stdio');
  }
}

// Main
const config: QFTMCPConfig = {
  corpusPath: process.env.CORPUS_PATH || './corpus.txt',
  indexPath: process.env.INDEX_PATH || './qft_index.json',
  chunkSize: 512,
  overlapSize: 128,
  targetDim: 768,
  ibmBackend: 'ibm_torino',
  shots: 8192,
};

const server = new QFTMCPServer(config);
server.start();
```

### Phase 4: Usage in Claude Desktop

```json
// claude_desktop_config.json
{
  "mcpServers": {
    "qft-context": {
      "command": "node",
      "args": ["/path/to/qft-mcp-server.js"],
      "env": {
        "CORPUS_PATH": "/path/to/conversations_40M.txt",
        "INDEX_PATH": "/path/to/qft_index.json",
        "IBM_CLOUD_API_KEY": "...",
        "IBM_QUANTUM_CRN": "..."
      }
    }
  }
}
```

## Performance Optimization

### Challenge: 40M tokens = ~78k chunks (512 tokens/chunk)
Running 78k quantum circuits is too slow!

### Solution: Hierarchical QFT

```typescript
// Two-stage retrieval
async function hierarchicalQFTRetrieval(
  query: string,
  corpus: Chunk[],
  topK: number
) {
  // Stage 1: Coarse filtering with classical vector similarity
  const queryVec = await embed(query);
  const coarseResults = corpus
    .map(chunk => ({
      chunk,
      similarity: cosineSimilarity(queryVec, chunk.contentEmbedding),
    }))
    .sort((a, b) => b.similarity - a.similarity)
    .slice(0, 500); // Top 500 candidates
  
  // Stage 2: Fine-grained QFT ranking on candidates
  const qftResults = await runQFTRetrieval(
    queryVec,
    coarseResults.map(r => r.chunk),
    themeId: inferTheme(query),
    shots: 8192
  );
  
  return qftResults.slice(0, topK);
}
```

### Alternative: Batch QFT with Shared Circuit

```typescript
// Run multiple chunks through same circuit with parameter binding
async function batchQFTRetrieval(
  queryVec: number[],
  chunks: Chunk[],
  batchSize: number = 10
) {
  const results = [];
  
  for (let i = 0; i < chunks.length; i += batchSize) {
    const batch = chunks.slice(i, i + batchSize);
    
    // Create parameterized circuit
    const circuit = buildParameterizedQFT(queryVec);
    
    // Bind each chunk's doubled vector as parameters
    const jobs = batch.map(chunk => ({
      circuit,
      parameters: [
        ...chunk.contentEmbedding,
        ...chunk.watermarkEmbedding,
      ],
    }));
    
    // Submit all to IBM Runtime as batch job
    const batchResults = await ibmRuntime.runBatch(jobs, shots: 4096);
    
    results.push(...batchResults);
  }
  
  return results;
}
```

## Advantages Over Traditional RAG

1. **Interference-Based Relevance**
   - Traditional: Cosine similarity (single dot product)
   - QFT-MCP: Quantum interference patterns (global optimization)

2. **Multi-Dimensional Filtering**
   - Traditional: Post-hoc metadata filtering
   - QFT-MCP: Watermark dims participate in interference

3. **Context-Aware Retrieval**
   - Traditional: Independent chunk retrieval
   - QFT-MCP: Parent/child relationships in shadow dims

4. **Theme-Guided Search**
   - Traditional: Static embeddings
   - QFT-MCP: Theme gates create domain-specific interference

## Next Steps

1. **Prototype the chunking & watermarking** (pure Python/TS)
2. **Test doubled QFT circuits** on small corpus (100 chunks)
3. **Benchmark vs classical RAG** on your 40M token corpus
4. **Implement MCP server** with stdio transport
5. **Integrate with Claude Desktop** for real-world testing

## Open Questions

1. **Optimal doubling strategy?**
   - Currently: concat(content, watermark)
   - Alternative: interleave, tensor product, separate registers?

2. **Watermark encoding?**
   - Positional encoding vs learned embedding?
   - How to balance metadata importance?

3. **Theme generation?**
   - Manual theme IDs vs automatic from query?
   - Can we learn theme gates from training data?

4. **Scalability?**
   - Hierarchical vs batch processing?
   - Can we cache QFT results?

Would you like me to implement Phase 1 (chunking & watermarking) as a working prototype?
