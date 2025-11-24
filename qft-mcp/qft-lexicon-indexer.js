#!/usr/bin/env node
/**
 * qft-lexicon-indexer.js - Build lexicon index from your corpus
 * 
 * Processes your 40M tokens, extracts custom terminology,
 * and creates multi-codec spectral indexes.
 */

import { readFileSync, writeFileSync, existsSync, readdirSync } from 'fs';
import { join, extname } from 'path';

class LexiconIndexer {
  constructor(config) {
    this.config = config;
    this.chunks = [];
    this.lexicon = new Map();
    this.papers = new Map();
  }

  async buildIndex() {
    console.log('[Indexer] Building lexicon index...\n');

    // Step 1: Load corpus
    await this.loadCorpus();

    // Step 2: Extract custom terms
    await this.extractLexicon();

    // Step 3: Analyze paper structure
    await this.analyzePapers();

    // Step 4: Build metadata
    await this.enrichMetadata();

    // Step 5: Save index
    await this.saveIndex();

    console.log('\n[Indexer] Index building complete!');
  }

  async loadCorpus() {
    console.log('[Indexer] Loading corpus...');
    
    const corpusPath = this.config.corpusPath;

    if (!existsSync(corpusPath)) {
      throw new Error(`Corpus path not found: ${corpusPath}`);
    }

    // Check if it's conversations.json
    if (corpusPath.endsWith('conversations.json')) {
      await this.loadConversations(corpusPath);
    } 
    // Check if it's a directory
    else if (existsSync(corpusPath) && !corpusPath.includes('.')) {
      await this.loadDirectory(corpusPath);
    }
    // Single text file
    else {
      await this.loadTextFile(corpusPath);
    }

    console.log(`[Indexer] Loaded ${this.chunks.length} chunks`);
  }

  async loadConversations(path) {
    console.log('[Indexer] Loading conversations.json...');
    
    const data = JSON.parse(readFileSync(path, 'utf-8'));
    let chunkId = 0;

    for (const convo of data) {
      const title = convo.title || 'Untitled';
      const createTime = convo.create_time;
      const mapping = convo.mapping || {};

      for (const node of Object.values(mapping)) {
        const msg = node.message;
        if (!msg) continue;

        const author = msg.author?.role || 'unknown';
        const content = msg.content;

        if (content?.content_type === 'text' && content.parts) {
          for (const part of content.parts) {
            if (!part || !part.trim()) continue;

            // Chunk long messages
            const textChunks = this.chunkText(part, 512, 128);
            
            for (let i = 0; i < textChunks.length; i++) {
              this.chunks.push({
                id: `chunk_${chunkId++}`,
                text: textChunks[i],
                metadata: {
                  source: 'conversations',
                  conversationTitle: title,
                  timestamp: createTime,
                  author,
                  chunkNumber: i,
                },
              });
            }
          }
        }
      }
    }
  }

  async loadDirectory(dirPath) {
    console.log(`[Indexer] Loading directory: ${dirPath}`);
    
    const files = readdirSync(dirPath);
    let paperNumber = 1;

    for (const file of files) {
      const filePath = join(dirPath, file);
      const ext = extname(file);

      if (ext === '.txt' || ext === '.md') {
        const text = readFileSync(filePath, 'utf-8');
        const paperTitle = file.replace(ext, '');
        
        // Chunk the paper
        const textChunks = this.chunkText(text, 512, 128);
        
        for (let i = 0; i < textChunks.length; i++) {
          this.chunks.push({
            id: `chunk_${this.chunks.length}`,
            text: textChunks[i],
            metadata: {
              source: 'paper',
              paperTitle,
              paperNumber,
              chunkNumber: i,
              totalChunks: textChunks.length,
            },
          });
        }

        this.papers.set(paperNumber, {
          number: paperNumber,
          title: paperTitle,
          file,
        });

        paperNumber++;
      }
    }
  }

  async loadTextFile(path) {
    console.log(`[Indexer] Loading text file: ${path}`);
    
    const text = readFileSync(path, 'utf-8');
    const textChunks = this.chunkText(text, 512, 128);

    for (let i = 0; i < textChunks.length; i++) {
      this.chunks.push({
        id: `chunk_${i}`,
        text: textChunks[i],
        metadata: {
          source: path,
          chunkNumber: i,
        },
      });
    }
  }

  chunkText(text, chunkSize = 512, overlap = 128) {
    const words = text.split(/\s+/);
    const chunks = [];
    
    for (let i = 0; i < words.length; i += (chunkSize - overlap)) {
      const chunk = words.slice(i, i + chunkSize).join(' ');
      if (chunk.length > 50) { // Skip very short chunks
        chunks.push(chunk);
      }
    }

    return chunks;
  }

  async extractLexicon() {
    console.log('\n[Indexer] Extracting custom lexicon...');

    // Define your known terms (you can expand this)
    const knownTerms = [
      'Wujudic Logic',
      '6th Generation Warfare',
      '6GW',
      'Storyworlds',
      'Narrative Coherence',
      'Memetic Warfare',
      'Cognitive Maneuver',
      'Semantic Vectors',
      'Belief Space',
      'TradeLayer',
      'Quantum Fourier Transform',
      'QFT',
    ];

    // Extract terms from chunks
    for (const chunk of this.chunks) {
      const foundTerms = [];

      for (const term of knownTerms) {
        if (chunk.text.toLowerCase().includes(term.toLowerCase())) {
          foundTerms.push(term);
          
          if (!this.lexicon.has(term)) {
            this.lexicon.set(term, []);
          }
          this.lexicon.get(term).push(chunk.id);
        }
      }

      if (foundTerms.length > 0) {
        chunk.metadata.customTerms = foundTerms;
      }

      // Mark definitional chunks
      const definitionalPatterns = [
        /is defined as/i,
        /refers to/i,
        /what (?:I|we) mean by/i,
        /let'?s call this/i,
        /this is (?:a|an|the)/i,
      ];

      chunk.metadata.isDefinitional = definitionalPatterns.some(p => 
        p.test(chunk.text)
      ) && foundTerms.length > 0;
    }

    console.log(`[Indexer] Found ${this.lexicon.size} unique terms`);
    console.log(`[Indexer] Terms: ${Array.from(this.lexicon.keys()).join(', ')}`);
  }

  async analyzePapers() {
    console.log('\n[Indexer] Analyzing paper structure...');

    // Group chunks by paper
    const byPaper = new Map();
    
    for (const chunk of this.chunks) {
      const paperNum = chunk.metadata.paperNumber;
      if (!paperNum) continue;

      if (!byPaper.has(paperNum)) {
        byPaper.set(paperNum, []);
      }
      byPaper.get(paperNum).push(chunk);
    }

    // Analyze each paper
    for (const [paperNum, chunks] of byPaper.entries()) {
      const paper = this.papers.get(paperNum);
      if (!paper) continue;

      // Extract introduced terms (terms that appear here but not in earlier papers)
      const termsInPaper = new Set();
      for (const chunk of chunks) {
        if (chunk.metadata.customTerms) {
          chunk.metadata.customTerms.forEach(t => termsInPaper.add(t));
        }
      }

      // Check which terms appeared in earlier papers
      const introducedTerms = [];
      for (const term of termsInPaper) {
        const chunkIds = this.lexicon.get(term) || [];
        const earlierOccurrence = chunkIds.some(id => {
          const c = this.chunks.find(ch => ch.id === id);
          return c && c.metadata.paperNumber && c.metadata.paperNumber < paperNum;
        });

        if (!earlierOccurrence) {
          introducedTerms.push(term);
        }
      }

      paper.introducedTerms = introducedTerms;
      paper.chunkCount = chunks.length;

      // Mark chunks with introduced terms
      for (const chunk of chunks) {
        if (chunk.metadata.customTerms) {
          chunk.metadata.introducedInThisPaper = chunk.metadata.customTerms
            .filter(t => introducedTerms.includes(t));
        }
      }
    }

    console.log(`[Indexer] Analyzed ${byPaper.size} papers`);
  }

  async enrichMetadata() {
    console.log('\n[Indexer] Enriching metadata...');

    // Add cross-references, citations, etc.
    for (const chunk of this.chunks) {
      // Simple citation detection
      const citationPatterns = [
        /Paper (\d+)/gi,
        /\b[A-Z][a-z]+ \(\d{4}\)/g, // Author (Year)
      ];

      const citations = [];
      for (const pattern of citationPatterns) {
        const matches = chunk.text.matchAll(pattern);
        for (const match of matches) {
          citations.push(match[0]);
        }
      }

      if (citations.length > 0) {
        chunk.metadata.citedPapers = [...new Set(citations)];
      }

      // Estimate abstraction level (simple heuristic)
      const abstractWords = [
        'framework', 'concept', 'theory', 'approach', 'methodology',
        'paradigm', 'principle', 'model', 'system',
      ];
      const concreteWords = [
        'example', 'specifically', 'instance', 'such as', 'for example',
        'demonstrated', 'shows', 'implementation',
      ];

      const abstractCount = abstractWords.reduce((sum, word) => 
        sum + (chunk.text.toLowerCase().match(new RegExp(word, 'g')) || []).length, 0
      );
      const concreteCount = concreteWords.reduce((sum, word) =>
        sum + (chunk.text.toLowerCase().match(new RegExp(word, 'g')) || []).length, 0
      );

      chunk.metadata.abstractness = abstractCount / (abstractCount + concreteCount + 1);
    }

    console.log('[Indexer] Metadata enriched');
  }

  async saveIndex() {
    const outputPath = this.config.outputPath;
    console.log(`\n[Indexer] Saving index to ${outputPath}...`);

    const index = {
      version: '1.0.0',
      created: new Date().toISOString(),
      stats: {
        totalChunks: this.chunks.length,
        totalTerms: this.lexicon.size,
        totalPapers: this.papers.size,
      },
      chunks: this.chunks,
      lexicon: Object.fromEntries(this.lexicon),
      papers: Object.fromEntries(this.papers),
    };

    writeFileSync(outputPath, JSON.stringify(index, null, 2));
    console.log('[Indexer] Index saved successfully');
    
    // Stats
    console.log('\n📊 Index Statistics:');
    console.log(`   Chunks: ${index.stats.totalChunks}`);
    console.log(`   Terms: ${index.stats.totalTerms}`);
    console.log(`   Papers: ${index.stats.totalPapers}`);
    console.log(`   Size: ${(JSON.stringify(index).length / 1024 / 1024).toFixed(2)} MB`);
  }
}

// CLI
async function main() {
  const args = process.argv.slice(2);
  
  if (args.length === 0) {
    console.log(`
Usage: node qft-lexicon-indexer.js <corpus_path> [output_path]

Examples:
  # Index conversations.json
  node qft-lexicon-indexer.js conversations.json lexicon_index.json

  # Index directory of papers
  node qft-lexicon-indexer.js ./papers/ lexicon_index.json

  # Index single text file
  node qft-lexicon-indexer.js corpus.txt lexicon_index.json
    `);
    process.exit(0);
  }

  const config = {
    corpusPath: args[0],
    outputPath: args[1] || 'lexicon_index.json',
  };

  const indexer = new LexiconIndexer(config);
  await indexer.buildIndex();
}

main().catch(error => {
  console.error('Error:', error);
  process.exit(1);
});
