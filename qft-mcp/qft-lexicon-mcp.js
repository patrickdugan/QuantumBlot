#!/usr/bin/env node
/**
 * qft-lexicon-mcp.js - MCP Server for Historical Lexicon Retrieval
 * 
 * Provides persistent access to your custom lexicon across any LLM.
 * Uses QFT-based spectral indexing to retrieve from your 17 papers and 40M tokens.
 */

import { Server } from '@modelcontextprotocol/sdk/server/index.js';
import { StdioServerTransport } from '@modelcontextprotocol/sdk/server/stdio.js';
import {
  CallToolRequestSchema,
  ListToolsRequestSchema,
} from '@modelcontextprotocol/sdk/types.js';
import { readFileSync, existsSync, writeFileSync } from 'fs';
import { QFTRunner } from './qft-runner.js';

class LexiconMCPServer {
  constructor(config) {
    this.config = config;
    this.server = new Server(
      {
        name: 'qft-lexicon',
        version: '1.0.0',
      },
      {
        capabilities: {
          tools: {},
        },
      }
    );

    this.qftRunner = new QFTRunner();
    this.index = null;
    this.lexicon = new Map();
    
    this.setupHandlers();
  }

  async initialize() {
    console.error('[QFT-Lexicon] Initializing...');
    
    // Load index if exists
    if (existsSync(this.config.indexPath)) {
      console.error(`[QFT-Lexicon] Loading index from ${this.config.indexPath}`);
      const indexData = JSON.parse(readFileSync(this.config.indexPath, 'utf-8'));
      this.index = indexData;
      this.buildLexiconMap(indexData);
    } else {
      console.error('[QFT-Lexicon] No index found. Run indexing first.');
    }
    
    console.error('[QFT-Lexicon] Ready');
  }

  buildLexiconMap(indexData) {
    // Build quick lookup map for terms
    for (const chunk of indexData.chunks || []) {
      if (chunk.metadata?.customTerms) {
        for (const term of chunk.metadata.customTerms) {
          if (!this.lexicon.has(term)) {
            this.lexicon.set(term, []);
          }
          this.lexicon.get(term).push(chunk.id);
        }
      }
    }
    console.error(`[QFT-Lexicon] Built lexicon with ${this.lexicon.size} terms`);
  }

  setupHandlers() {
    this.server.setRequestHandler(ListToolsRequestSchema, async () => ({
      tools: [
        {
          name: 'lexicon_lookup',
          description: 'Look up a term from your custom lexicon (e.g., "Wujudic Logic", "6GW", "Storyworlds"). Returns definition, usage history, and related concepts.',
          inputSchema: {
            type: 'object',
            properties: {
              term: {
                type: 'string',
                description: 'The term to look up from your lexicon',
              },
              includeHistory: {
                type: 'boolean',
                description: 'Include historical evolution across papers',
                default: true,
              },
              includeUsage: {
                type: 'boolean',
                description: 'Include usage examples',
                default: true,
              },
              maxResults: {
                type: 'number',
                description: 'Maximum number of results to return',
                default: 10,
              },
            },
            required: ['term'],
          },
        },
        {
          name: 'retrieve_paper',
          description: 'Retrieve one of your 17 AI papers by title or number. Returns summary, key points, and optionally full text.',
          inputSchema: {
            type: 'object',
            properties: {
              paperIdentifier: {
                type: 'string',
                description: 'Paper title or number (1-17)',
              },
              includeFullText: {
                type: 'boolean',
                description: 'Include full reconstructed text',
                default: false,
              },
              includeReferences: {
                type: 'boolean',
                description: 'Include cited papers and references',
                default: true,
              },
            },
            required: ['paperIdentifier'],
          },
        },
        {
          name: 'trace_concept_evolution',
          description: 'Trace how a concept evolved across your papers (e.g., how "narrative coherence" developed from Paper 2 to Paper 17).',
          inputSchema: {
            type: 'object',
            properties: {
              concept: {
                type: 'string',
                description: 'The concept to trace through your work',
              },
              startPaper: {
                type: 'number',
                description: 'Starting paper number',
                default: 1,
              },
              endPaper: {
                type: 'number',
                description: 'Ending paper number',
                default: 17,
              },
            },
            required: ['concept'],
          },
        },
        {
          name: 'semantic_neighbors',
          description: 'Find concepts semantically related to a given term in your lexicon.',
          inputSchema: {
            type: 'object',
            properties: {
              term: {
                type: 'string',
                description: 'The term to find neighbors for',
              },
              topK: {
                type: 'number',
                description: 'Number of neighbors to return',
                default: 10,
              },
              codec: {
                type: 'string',
                enum: ['lexical', 'genealogical', 'conceptual'],
                description: 'Which indexing metaphor to use',
                default: 'conceptual',
              },
            },
            required: ['term'],
          },
        },
        {
          name: 'cross_reference',
          description: 'Find where multiple concepts intersect across your papers.',
          inputSchema: {
            type: 'object',
            properties: {
              concepts: {
                type: 'array',
                items: { type: 'string' },
                description: 'List of concepts to cross-reference',
              },
              requireAll: {
                type: 'boolean',
                description: 'Require all concepts to appear together',
                default: false,
              },
              maxResults: {
                type: 'number',
                description: 'Maximum results to return',
                default: 10,
              },
            },
            required: ['concepts'],
          },
        },
        {
          name: 'list_papers',
          description: 'List all papers in your corpus with metadata.',
          inputSchema: {
            type: 'object',
            properties: {
              includeStats: {
                type: 'boolean',
                description: 'Include statistics about each paper',
                default: true,
              },
            },
          },
        },
        {
          name: 'list_lexicon_terms',
          description: 'List all custom terms in your lexicon.',
          inputSchema: {
            type: 'object',
            properties: {
              category: {
                type: 'string',
                description: 'Filter by category (optional)',
              },
              sortBy: {
                type: 'string',
                enum: ['alphabetical', 'frequency', 'recency'],
                default: 'frequency',
              },
            },
          },
        },
        {
          name: 'search_corpus',
          description: 'Search your entire 40M token corpus with QFT-enhanced retrieval.',
          inputSchema: {
            type: 'object',
            properties: {
              query: {
                type: 'string',
                description: 'Search query',
              },
              codec: {
                type: 'string',
                enum: ['temporal', 'lexical', 'genealogical', 'conceptual', 'auto'],
                description: 'Which indexing codec to use',
                default: 'auto',
              },
              topK: {
                type: 'number',
                description: 'Number of results',
                default: 10,
              },
              themeId: {
                type: 'number',
                description: 'QFT theme ID for interference pattern',
              },
            },
            required: ['query'],
          },
        },
      ],
    }));

    this.server.setRequestHandler(CallToolRequestSchema, async (request) => {
      try {
        const { name, arguments: args } = request.params;

        switch (name) {
          case 'lexicon_lookup':
            return await this.handleLexiconLookup(args);
          
          case 'retrieve_paper':
            return await this.handleRetrievePaper(args);
          
          case 'trace_concept_evolution':
            return await this.handleTraceEvolution(args);
          
          case 'semantic_neighbors':
            return await this.handleSemanticNeighbors(args);
          
          case 'cross_reference':
            return await this.handleCrossReference(args);
          
          case 'list_papers':
            return await this.handleListPapers(args);
          
          case 'list_lexicon_terms':
            return await this.handleListTerms(args);
          
          case 'search_corpus':
            return await this.handleSearchCorpus(args);
          
          default:
            throw new Error(`Unknown tool: ${name}`);
        }
      } catch (error) {
        return {
          content: [
            {
              type: 'text',
              text: `Error: ${error.message}`,
            },
          ],
          isError: true,
        };
      }
    });
  }

  async handleLexiconLookup(args) {
    const { term, includeHistory = true, includeUsage = true, maxResults = 10 } = args;
    
    console.error(`[QFT-Lexicon] Looking up term: ${term}`);
    
    if (!this.index) {
      return {
        content: [{
          type: 'text',
          text: 'Index not loaded. Please run indexing first.',
        }],
      };
    }

    // Find chunks containing this term
    const chunkIds = this.lexicon.get(term) || [];
    const chunks = chunkIds
      .map(id => this.index.chunks.find(c => c.id === id))
      .filter(Boolean)
      .slice(0, maxResults);

    if (chunks.length === 0) {
      return {
        content: [{
          type: 'text',
          text: `Term "${term}" not found in lexicon. Available terms:\n${
            Array.from(this.lexicon.keys()).slice(0, 20).join(', ')
          }${this.lexicon.size > 20 ? '...' : ''}`,
        }],
      };
    }

    // Build response
    const definitions = chunks
      .filter(c => c.metadata?.isDefinitional)
      .map(c => ({
        text: c.text,
        source: c.metadata?.paperTitle || 'Unknown',
        paperNumber: c.metadata?.paperNumber,
      }));

    const usage = includeUsage
      ? chunks
          .filter(c => !c.metadata?.isDefinitional)
          .map(c => ({
            text: c.text.substring(0, 300) + (c.text.length > 300 ? '...' : ''),
            source: c.metadata?.paperTitle || 'Unknown',
            paperNumber: c.metadata?.paperNumber,
          }))
      : [];

    const history = includeHistory
      ? chunks
          .sort((a, b) => 
            (a.metadata?.paperNumber || 0) - (b.metadata?.paperNumber || 0)
          )
          .map(c => ({
            paperNumber: c.metadata?.paperNumber,
            paperTitle: c.metadata?.paperTitle,
            timestamp: c.metadata?.timestamp,
            snippet: c.text.substring(0, 200) + '...',
          }))
      : [];

    const relatedTerms = this.findRelatedTerms(term, chunks);

    const response = {
      term,
      found: true,
      occurrences: chunks.length,
      definitions: definitions.length > 0 ? definitions : [
        {
          text: chunks[0].text,
          source: chunks[0].metadata?.paperTitle || 'Unknown',
          note: 'Primary occurrence (no explicit definition found)',
        },
      ],
      usage: usage.slice(0, 5),
      history: history.slice(0, 10),
      relatedTerms: relatedTerms.slice(0, 5),
      papers: [...new Set(chunks.map(c => c.metadata?.paperTitle))].filter(Boolean),
    };

    return {
      content: [{
        type: 'text',
        text: JSON.stringify(response, null, 2),
      }],
    };
  }

  async handleRetrievePaper(args) {
    const { paperIdentifier, includeFullText = false, includeReferences = true } = args;
    
    console.error(`[QFT-Lexicon] Retrieving paper: ${paperIdentifier}`);
    
    if (!this.index) {
      return {
        content: [{
          type: 'text',
          text: 'Index not loaded.',
        }],
      };
    }

    // Find chunks from this paper
    const paperNum = parseInt(paperIdentifier);
    const chunks = this.index.chunks.filter(c => {
      if (!isNaN(paperNum)) {
        return c.metadata?.paperNumber === paperNum;
      }
      return c.metadata?.paperTitle?.toLowerCase().includes(
        paperIdentifier.toLowerCase()
      );
    });

    if (chunks.length === 0) {
      return {
        content: [{
          type: 'text',
          text: `Paper "${paperIdentifier}" not found.`,
        }],
      };
    }

    // Sort by chunk number
    chunks.sort((a, b) => 
      (a.metadata?.chunkNumber || 0) - (b.metadata?.chunkNumber || 0)
    );

    const paper = {
      title: chunks[0].metadata?.paperTitle || paperIdentifier,
      paperNumber: chunks[0].metadata?.paperNumber,
      totalChunks: chunks.length,
      timestamp: chunks[0].metadata?.timestamp,
      summary: this.generateSummary(chunks),
      keyPoints: this.extractKeyPoints(chunks),
      introducedTerms: this.extractIntroducedTerms(chunks),
    };

    if (includeFullText) {
      paper.fullText = chunks.map(c => c.text).join('\n\n');
    }

    if (includeReferences) {
      paper.references = this.extractReferences(chunks);
    }

    return {
      content: [{
        type: 'text',
        text: JSON.stringify(paper, null, 2),
      }],
    };
  }

  async handleTraceEvolution(args) {
    const { concept, startPaper = 1, endPaper = 17 } = args;
    
    console.error(`[QFT-Lexicon] Tracing evolution: ${concept}`);
    
    if (!this.index) {
      return {
        content: [{
          type: 'text',
          text: 'Index not loaded.',
        }],
      };
    }

    // Find all chunks mentioning this concept
    const conceptLower = concept.toLowerCase();
    const relevantChunks = this.index.chunks.filter(c => {
      const paperNum = c.metadata?.paperNumber || 0;
      return (
        paperNum >= startPaper &&
        paperNum <= endPaper &&
        c.text.toLowerCase().includes(conceptLower)
      );
    });

    // Group by paper
    const byPaper = new Map();
    for (const chunk of relevantChunks) {
      const paperNum = chunk.metadata?.paperNumber || 0;
      if (!byPaper.has(paperNum)) {
        byPaper.set(paperNum, []);
      }
      byPaper.get(paperNum).push(chunk);
    }

    // Build timeline
    const timeline = [];
    for (const [paperNum, chunks] of [...byPaper.entries()].sort((a, b) => a - b)) {
      const summary = chunks
        .map(c => c.text.substring(0, 200))
        .join(' ... ');
      
      timeline.push({
        paperNumber: paperNum,
        paperTitle: chunks[0].metadata?.paperTitle || 'Unknown',
        occurrences: chunks.length,
        summary: summary.substring(0, 500) + (summary.length > 500 ? '...' : ''),
        introducedTerms: this.extractNewTermsFromChunks(chunks),
      });
    }

    return {
      content: [{
        type: 'text',
        text: JSON.stringify({
          concept,
          timeline,
          totalPapers: timeline.length,
          firstAppearance: timeline[0]?.paperNumber,
          lastAppearance: timeline[timeline.length - 1]?.paperNumber,
        }, null, 2),
      }],
    };
  }

  async handleSemanticNeighbors(args) {
    const { term, topK = 10, codec = 'conceptual' } = args;
    
    console.error(`[QFT-Lexicon] Finding neighbors for: ${term}`);
    
    // For now, use simple co-occurrence
    const termChunks = this.lexicon.get(term) || [];
    if (termChunks.length === 0) {
      return {
        content: [{
          type: 'text',
          text: `Term "${term}" not found.`,
        }],
      };
    }

    // Find terms that co-occur with this term
    const coOccurrence = new Map();
    for (const chunkId of termChunks) {
      const chunk = this.index.chunks.find(c => c.id === chunkId);
      if (chunk?.metadata?.customTerms) {
        for (const otherTerm of chunk.metadata.customTerms) {
          if (otherTerm !== term) {
            coOccurrence.set(
              otherTerm,
              (coOccurrence.get(otherTerm) || 0) + 1
            );
          }
        }
      }
    }

    const neighbors = [...coOccurrence.entries()]
      .sort((a, b) => b[1] - a[1])
      .slice(0, topK)
      .map(([term, count]) => ({ term, coOccurrences: count }));

    return {
      content: [{
        type: 'text',
        text: JSON.stringify({ term, neighbors, codec }, null, 2),
      }],
    };
  }

  async handleCrossReference(args) {
    const { concepts, requireAll = false, maxResults = 10 } = args;
    
    console.error(`[QFT-Lexicon] Cross-referencing: ${concepts.join(', ')}`);
    
    if (!this.index) {
      return {
        content: [{
          type: 'text',
          text: 'Index not loaded.',
        }],
      };
    }

    const conceptsLower = concepts.map(c => c.toLowerCase());
    const matches = this.index.chunks.filter(chunk => {
      const text = chunk.text.toLowerCase();
      if (requireAll) {
        return conceptsLower.every(c => text.includes(c));
      } else {
        return conceptsLower.some(c => text.includes(c));
      }
    });

    const results = matches.slice(0, maxResults).map(chunk => ({
      text: chunk.text.substring(0, 300) + '...',
      source: chunk.metadata?.paperTitle || 'Unknown',
      paperNumber: chunk.metadata?.paperNumber,
      matchedConcepts: conceptsLower.filter(c => 
        chunk.text.toLowerCase().includes(c)
      ),
    }));

    return {
      content: [{
        type: 'text',
        text: JSON.stringify({
          concepts,
          requireAll,
          matches: results.length,
          results,
        }, null, 2),
      }],
    };
  }

  async handleListPapers(args) {
    const { includeStats = true } = args;
    
    if (!this.index) {
      return {
        content: [{
          type: 'text',
          text: 'Index not loaded.',
        }],
      };
    }

    const papers = new Map();
    for (const chunk of this.index.chunks) {
      const paperNum = chunk.metadata?.paperNumber;
      const paperTitle = chunk.metadata?.paperTitle;
      if (paperNum && paperTitle) {
        if (!papers.has(paperNum)) {
          papers.set(paperNum, {
            number: paperNum,
            title: paperTitle,
            chunkCount: 0,
            customTerms: new Set(),
          });
        }
        const paper = papers.get(paperNum);
        paper.chunkCount++;
        if (chunk.metadata?.customTerms) {
          chunk.metadata.customTerms.forEach(t => paper.customTerms.add(t));
        }
      }
    }

    const paperList = [...papers.values()]
      .sort((a, b) => a.number - b.number)
      .map(p => ({
        number: p.number,
        title: p.title,
        ...(includeStats ? {
          chunks: p.chunkCount,
          customTerms: p.customTerms.size,
        } : {}),
      }));

    return {
      content: [{
        type: 'text',
        text: JSON.stringify({ papers: paperList, total: paperList.length }, null, 2),
      }],
    };
  }

  async handleListTerms(args) {
    const { category, sortBy = 'frequency' } = args;
    
    const terms = [...this.lexicon.entries()].map(([term, chunkIds]) => ({
      term,
      frequency: chunkIds.length,
    }));

    if (sortBy === 'alphabetical') {
      terms.sort((a, b) => a.term.localeCompare(b.term));
    } else if (sortBy === 'frequency') {
      terms.sort((a, b) => b.frequency - a.frequency);
    }

    return {
      content: [{
        type: 'text',
        text: JSON.stringify({
          total: terms.length,
          terms: terms.slice(0, 100),
          note: terms.length > 100 ? 'Showing top 100' : 'Complete list',
        }, null, 2),
      }],
    };
  }

  async handleSearchCorpus(args) {
    const { query, codec = 'auto', topK = 10, themeId } = args;
    
    console.error(`[QFT-Lexicon] Searching: ${query}`);
    
    // Simple keyword search for now
    const queryLower = query.toLowerCase();
    const matches = this.index.chunks
      .filter(c => c.text.toLowerCase().includes(queryLower))
      .slice(0, topK)
      .map(c => ({
        text: c.text.substring(0, 300) + '...',
        source: c.metadata?.paperTitle || 'Unknown',
        paperNumber: c.metadata?.paperNumber,
      }));

    return {
      content: [{
        type: 'text',
        text: JSON.stringify({
          query,
          codec,
          results: matches,
          total: matches.length,
        }, null, 2),
      }],
    };
  }

  // Helper methods
  findRelatedTerms(term, chunks) {
    const related = new Set();
    for (const chunk of chunks) {
      if (chunk.metadata?.customTerms) {
        chunk.metadata.customTerms.forEach(t => {
          if (t !== term) related.add(t);
        });
      }
    }
    return Array.from(related);
  }

  generateSummary(chunks) {
    // Take first few chunks as summary
    return chunks.slice(0, 3).map(c => c.text).join('\n\n').substring(0, 500) + '...';
  }

  extractKeyPoints(chunks) {
    // Look for chunks with definitional content
    return chunks
      .filter(c => c.metadata?.isDefinitional || c.text.match(/\b(define|refers to|means that)\b/i))
      .slice(0, 5)
      .map(c => c.text.substring(0, 200) + '...');
  }

  extractIntroducedTerms(chunks) {
    const terms = new Set();
    for (const chunk of chunks) {
      if (chunk.metadata?.customTerms) {
        chunk.metadata.customTerms.forEach(t => terms.add(t));
      }
    }
    return Array.from(terms);
  }

  extractNewTermsFromChunks(chunks) {
    return this.extractIntroducedTerms(chunks);
  }

  extractReferences(chunks) {
    const refs = new Set();
    for (const chunk of chunks) {
      if (chunk.metadata?.citedPapers) {
        chunk.metadata.citedPapers.forEach(r => refs.add(r));
      }
    }
    return Array.from(refs);
  }

  async start() {
    await this.initialize();
    
    const transport = new StdioServerTransport();
    await this.server.connect(transport);
    
    console.error('[QFT-Lexicon] MCP Server running on stdio');
  }
}

// Main
const config = {
  indexPath: process.env.LEXICON_INDEX || './lexicon_index.json',
  corpusPath: process.env.CORPUS_PATH || './corpus',
};

const server = new LexiconMCPServer(config);
server.start().catch(error => {
  console.error('[QFT-Lexicon] Fatal error:', error);
  process.exit(1);
});
