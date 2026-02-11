# QFT Codec: Indexing Metaphors as Spectral Patterns

## Core Concept: The Codec as Indexing System

A **codec** (encoder-decoder) isn't just compression - it's a *way of seeing*. Different codecs emphasize different patterns. In QFT-MCP, we use multiple "spectral metaphors" to create different indexing schemes for the same corpus.

```
Same 40M token corpus
    ↓
Multiple QFT codecs (different theme patterns)
    ↓
Each creates different "frequency spectrum" view
    ↓
Query can be "tuned" to specific metaphor/codec
```

## The Indexing Metaphor Concept

### Traditional Indexing (Single Metaphor)
```
Document → [word1, word2, ...] → Inverted Index → Retrieve
                ↓
           One way to see the text (lexical)
```

### QFT Spectral Indexing (Multiple Metaphors)
```
Document → [embedding] → Multiple QFT Decompositions → Retrieve
              ↓              ↓          ↓          ↓
           Metaphor 1    Metaphor 2  Metaphor 3  ...
          (Technical)   (Temporal)   (Narrative)
```

Each "metaphor" is a different way to decompose the information into frequency components.

## Codec = Theme-Gated Spectral Decomposition

### What is a Metaphor/Spectra?

Think of your corpus like white light through a prism:
- **White light** = Raw embedding space (all information mixed)
- **Prism** = QFT with specific theme gates
- **Spectrum** = How information spreads across frequency modes
- **Color** = Which frequency components are strong

Different prisms (theme gates) create different spectrums (metaphors).

### Example Metaphors/Codecs

#### Codec 1: Temporal Spectrum
```javascript
// Theme gates emphasize time-based frequency patterns
const temporalCodec = {
  themeId: 1,
  gatePattern: 'RZ rotations proportional to chunk timestamp',
  metaphor: 'Time as frequency',
  
  // Encodes:
  shadowDims: {
    timestamp: { weight: 0.8, encoding: 'positional' },
    recency: { weight: 0.6, encoding: 'exponential_decay' },
    sequence: { weight: 0.5, encoding: 'sequential_position' },
  },
  
  // Results in spectrum where:
  lowFreq: 'Long-term patterns, trends',
  midFreq: 'Medium-term developments',
  highFreq: 'Recent events, immediate context',
};

// Query: "What changed recently?"
// → Amplifies high-frequency components
// → Retrieves recent chunks with strong temporal signal
```

#### Codec 2: Semantic Hierarchy Spectrum
```javascript
const hierarchicalCodec = {
  themeId: 2,
  gatePattern: 'RZ rotations based on topic tree depth',
  metaphor: 'Abstraction as frequency',
  
  shadowDims: {
    topicDepth: { weight: 0.9, encoding: 'tree_depth' },
    abstractness: { weight: 0.7, encoding: 'concept_level' },
    parentLinks: { weight: 0.6, encoding: 'graph_centrality' },
  },
  
  // Results in spectrum where:
  lowFreq: 'High-level concepts, summaries',
  midFreq: 'Mid-level explanations',
  highFreq: 'Specific details, examples',
};

// Query: "Give me the big picture"
// → Amplifies low-frequency components
// → Retrieves abstract, high-level chunks
```

#### Codec 3: Narrative/Causal Spectrum
```javascript
const narrativeCodec = {
  themeId: 3,
  gatePattern: 'RZ based on causal graph position',
  metaphor: 'Causality as frequency',
  
  shadowDims: {
    causalPriority: { weight: 0.8, encoding: 'topological_order' },
    threadPosition: { weight: 0.7, encoding: 'conversation_turn' },
    responseDepth: { weight: 0.6, encoding: 'reply_chain_length' },
  },
  
  // Results in spectrum where:
  lowFreq: 'Root causes, initiating events',
  midFreq: 'Intermediate steps, reasoning',
  highFreq: 'Consequences, outcomes',
};

// Query: "Why did X happen?"
// → Amplifies low-to-mid frequency
// → Retrieves causal predecessors
```

#### Codec 4: Associative/Network Spectrum
```javascript
const associativeCodec = {
  themeId: 4,
  gatePattern: 'RZ based on graph connectivity',
  metaphor: 'Connection as frequency',
  
  shadowDims: {
    degree: { weight: 0.8, encoding: 'node_degree' },
    betweenness: { weight: 0.7, encoding: 'betweenness_centrality' },
    clustering: { weight: 0.6, encoding: 'local_clustering' },
  },
  
  // Results in spectrum where:
  lowFreq: 'Central hubs, key concepts',
  midFreq: 'Bridges between topics',
  highFreq: 'Peripheral details',
};

// Query: "What connects A and B?"
// → Amplifies mid-frequency bridge patterns
// → Retrieves linking chunks
```

## Multi-Codec Retrieval Architecture

```typescript
interface Codec {
  themeId: number;
  metaphor: string;
  shadowDimConfig: ShadowDimConfig;
  gatePattern: (chunk: Chunk) => number[]; // Returns phase angles
}

class MultiCodecQFTIndex {
  private codecs: Map<string, Codec>;
  private spectrums: Map<string, Map<string, SpectrumData>>;
  
  constructor() {
    this.codecs = new Map();
    this.spectrums = new Map();
  }
  
  async indexCorpus(chunks: Chunk[], codecs: Codec[]) {
    for (const codec of codecs) {
      console.log(`Indexing with ${codec.metaphor} codec...`);
      
      const spectrumMap = new Map<string, SpectrumData>();
      
      for (const chunk of chunks) {
        // 1. Create doubled vector (content + shadow dims for this codec)
        const contentVec = chunk.embedding;
        const shadowVec = this.encodeWithCodec(chunk, codec);
        const doubledVec = [...contentVec, ...shadowVec];
        
        // 2. Run QFT with codec-specific theme gates
        const circuit = buildCodecQFT(doubledVec, codec);
        const counts = await runCircuit(circuit, shots: 8192);
        
        // 3. Extract spectrum (frequency decomposition)
        const spectrum = countsToSpectrum(counts);
        
        // 4. Store: chunk → spectrum under this codec
        spectrumMap.set(chunk.id, {
          chunkId: chunk.id,
          spectrum,
          dominantFreqs: spectrum.peaks,
          energy: spectrum.totalEnergy,
        });
      }
      
      this.spectrums.set(codec.metaphor, spectrumMap);
    }
  }
  
  async retrieve(query: string, codecMetaphor: string, topK: number = 10) {
    // 1. Detect which codec to use (or use specified)
    const codec = codecMetaphor 
      ? this.codecs.get(codecMetaphor)
      : await this.detectCodec(query);
    
    if (!codec) throw new Error(`Codec ${codecMetaphor} not found`);
    
    // 2. Get query spectrum using same codec
    const queryVec = await embed(query);
    const querySpectrum = await this.getQuerySpectrum(queryVec, codec);
    
    // 3. Compare query spectrum to all chunk spectrums
    const spectrumMap = this.spectrums.get(codec.metaphor)!;
    const similarities = [];
    
    for (const [chunkId, chunkSpectrum] of spectrumMap.entries()) {
      // Spectral similarity (frequency domain matching)
      const similarity = this.spectralSimilarity(
        querySpectrum,
        chunkSpectrum
      );
      
      similarities.push({ chunkId, similarity });
    }
    
    // 4. Return top-K by spectral similarity
    return similarities
      .sort((a, b) => b.similarity - a.similarity)
      .slice(0, topK);
  }
  
  private spectralSimilarity(
    querySpec: Spectrum,
    chunkSpec: Spectrum
  ): number {
    // Compare frequency distributions
    const freqCorrelation = this.correlate(
      querySpec.frequencies,
      chunkSpec.frequencies
    );
    
    // Penalize frequency offset (want matching peaks)
    const peakAlignment = this.peakOverlap(
      querySpec.peaks,
      chunkSpec.peaks
    );
    
    // Combine
    return 0.7 * freqCorrelation + 0.3 * peakAlignment;
  }
  
  private async detectCodec(query: string): Promise<Codec> {
    // Analyze query to determine best metaphor
    const queryFeatures = await analyzeQuery(query);
    
    if (queryFeatures.temporal > 0.7) {
      return this.codecs.get('temporal')!;
    }
    if (queryFeatures.hierarchical > 0.7) {
      return this.codecs.get('hierarchical')!;
    }
    if (queryFeatures.causal > 0.7) {
      return this.codecs.get('narrative')!;
    }
    
    // Default to associative
    return this.codecs.get('associative')!;
  }
}

interface Spectrum {
  frequencies: number[];  // Frequency bins
  amplitudes: number[];   // Strength at each frequency
  phases: number[];       // Phase at each frequency
  peaks: FrequencyPeak[]; // Dominant frequencies
  totalEnergy: number;    // Total power
}

interface FrequencyPeak {
  frequency: number;
  amplitude: number;
  bandwidth: number;
}

function countsToSpectrum(counts: Record<string, number>): Spectrum {
  // Convert measurement counts to frequency spectrum
  const N = Object.keys(counts).length;
  const frequencies = [];
  const amplitudes = [];
  const phases = [];
  
  // Interpret bitstrings as frequency modes
  for (const [bitstring, count] of Object.entries(counts)) {
    const freq = parseInt(bitstring, 2) / Math.pow(2, bitstring.length);
    const amp = count / Object.values(counts).reduce((a,b) => a+b, 0);
    
    frequencies.push(freq);
    amplitudes.push(amp);
    phases.push(0); // Can extract from complex amplitudes if available
  }
  
  // Find peaks
  const peaks = findPeaks(amplitudes).map(idx => ({
    frequency: frequencies[idx],
    amplitude: amplitudes[idx],
    bandwidth: estimateBandwidth(amplitudes, idx),
  }));
  
  return {
    frequencies,
    amplitudes,
    phases,
    peaks,
    totalEnergy: amplitudes.reduce((a,b) => a + b*b, 0),
  };
}
```

## Practical Example: Your 40M Token Corpus

### Setup: Index with Multiple Codecs

```typescript
// 1. Define your codecs
const myCodecs: Codec[] = [
  {
    themeId: 1,
    metaphor: 'temporal',
    shadowDimConfig: {
      timestamp: 0.8,
      recency: 0.6,
      sequence: 0.5,
    },
  },
  {
    themeId: 2,
    metaphor: 'technical_depth',
    shadowDimConfig: {
      abstractness: 0.9,
      codeVsText: 0.8,
      conceptLevel: 0.7,
    },
  },
  {
    themeId: 3,
    metaphor: 'conversational',
    shadowDimConfig: {
      threadPosition: 0.9,
      responseDepth: 0.8,
      userVsAssistant: 0.7,
    },
  },
  {
    themeId: 4,
    metaphor: 'semantic_clusters',
    shadowDimConfig: {
      topicId: 0.9,
      crossTopicLinks: 0.7,
      centrality: 0.6,
    },
  },
];

// 2. Index corpus
const index = new MultiCodecQFTIndex();
await index.indexCorpus(myChunks, myCodecs);

// Now you have 4 different "views" of the same 40M tokens!
```

### Query with Codec Selection

```typescript
// Temporal query
const recentResults = await index.retrieve(
  "What did we discuss about quantum computing last week?",
  codec: 'temporal',
  topK: 10
);
// → Amplifies high-frequency (recent) components

// Hierarchical query
const overviewResults = await index.retrieve(
  "Give me a high-level summary of our conversations",
  codec: 'technical_depth',
  topK: 10
);
// → Amplifies low-frequency (abstract) components

// Conversational query
const threadResults = await index.retrieve(
  "What was the context of our discussion about X?",
  codec: 'conversational',
  topK: 10
);
// → Amplifies thread structure patterns

// Semantic query
const topicalResults = await index.retrieve(
  "Find all mentions of machine learning",
  codec: 'semantic_clusters',
  topK: 10
);
// → Amplifies topic clustering patterns
```

## Why This is Powerful

### 1. Same Data, Multiple Perspectives
```
40M tokens → [temporal view] → See evolution over time
          → [hierarchical view] → See abstraction levels
          → [narrative view] → See causal chains
          → [network view] → See connections
```

### 2. Frequency = Structure
- **Low frequencies** = Slow-varying patterns (trends, themes, high-level)
- **High frequencies** = Fast-varying patterns (details, specifics, recent)
- **Peaks** = Dominant structures in that metaphor

### 3. Codec Selection = Query Intent
The codec you choose determines *what kind of answer* you get:
- "Why?" → Causal codec (emphasize cause-effect structure)
- "When?" → Temporal codec (emphasize time structure)
- "What connects?" → Network codec (emphasize relationships)
- "Overall?" → Hierarchical codec (emphasize abstractions)

## Implementation: Codec-Specific Shadow Encoding

```typescript
function encodeWithCodec(chunk: Chunk, codec: Codec): number[] {
  const shadowVec = new Array(768).fill(0);
  
  switch (codec.metaphor) {
    case 'temporal':
      // Encode timestamp as low-frequency sinusoids
      const timeSince = Date.now() - chunk.metadata.timestamp;
      for (let i = 0; i < 256; i++) {
        const freq = i / 256;
        shadowVec[i] = Math.sin(2 * Math.PI * freq * timeSince);
      }
      
      // Encode recency as exponential decay
      const recency = Math.exp(-timeSince / (30 * 24 * 3600 * 1000)); // 30-day half-life
      shadowVec.fill(recency, 256, 512);
      
      // Encode sequence position
      for (let i = 512; i < 768; i++) {
        shadowVec[i] = chunk.metadata.sequencePosition / totalChunks;
      }
      break;
      
    case 'technical_depth':
      // Encode abstraction level (0 = concrete, 1 = abstract)
      const abstractness = estimateAbstractness(chunk.text);
      shadowVec.fill(abstractness, 0, 256);
      
      // Encode code vs prose ratio
      const codeRatio = countCodeBlocks(chunk.text) / chunk.text.length;
      shadowVec.fill(codeRatio, 256, 512);
      
      // Encode concept complexity
      const complexity = estimateConceptComplexity(chunk.text);
      shadowVec.fill(complexity, 512, 768);
      break;
      
    case 'conversational':
      // Encode thread depth (reply chain length)
      const depth = chunk.metadata.replyDepth;
      for (let i = 0; i < 256; i++) {
        shadowVec[i] = depth / maxDepth;
      }
      
      // Encode turn position in conversation
      const turnRatio = chunk.metadata.turnNumber / chunk.metadata.totalTurns;
      shadowVec.fill(turnRatio, 256, 512);
      
      // Encode speaker (user vs assistant)
      const speakerEncoding = chunk.metadata.speaker === 'user' ? 1 : -1;
      shadowVec.fill(speakerEncoding, 512, 768);
      break;
      
    case 'semantic_clusters':
      // One-hot encode topic cluster
      const topicId = chunk.metadata.topicCluster;
      shadowVec[topicId % 256] = 1;
      
      // Encode graph centrality
      const centrality = chunk.metadata.graphCentrality;
      shadowVec.fill(centrality, 256, 512);
      
      // Encode cross-topic links
      const crossLinks = chunk.metadata.crossTopicLinks.length;
      shadowVec.fill(crossLinks / maxLinks, 512, 768);
      break;
  }
  
  return normalize(shadowVec);
}
```

## MCP Integration: Multi-Codec Server

```typescript
// qft-codec-mcp.ts
class CodecMCPServer {
  private index: MultiCodecQFTIndex;
  
  async handleRetrieve(args: any) {
    const {
      query,
      codec = 'auto', // Auto-detect or specify
      topK = 10,
      blendCodecs = false, // Blend multiple codecs?
    } = args;
    
    if (blendCodecs) {
      // Run query through multiple codecs and blend results
      const results = await Promise.all([
        this.index.retrieve(query, 'temporal', topK),
        this.index.retrieve(query, 'technical_depth', topK),
        this.index.retrieve(query, 'conversational', topK),
      ]);
      
      // Weighted blend (learn weights from query features)
      return this.blendResults(results, query);
    } else {
      // Single codec
      const codecName = codec === 'auto' 
        ? await this.detectBestCodec(query)
        : codec;
      
      return await this.index.retrieve(query, codecName, topK);
    }
  }
}
```

## Visual Metaphor

Think of your 40M tokens as a musical recording:

1. **Raw audio** = Original text
2. **Spectrogram** = Frequency decomposition (QFT)
3. **Different EQ settings** = Different codecs
   - Bass boost = Emphasize low-freq (abstractions)
   - Treble boost = Emphasize high-freq (details)
4. **Query** = What you want to hear
5. **Codec selection** = Which EQ curve amplifies what you want

The same song sounds different with different EQ. The same corpus reveals different structures with different codecs.

## Next Steps

Want me to build:
1. **Codec library** with 5-10 useful metaphors?
2. **Spectral similarity functions** for each codec?
3. **Automatic codec detection** from query analysis?
4. **Prototype** showing temporal vs hierarchical retrieval on sample data?
