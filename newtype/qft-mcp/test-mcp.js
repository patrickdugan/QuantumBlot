#!/usr/bin/env node
/**
 * test-mcp.js - Test the QFT Lexicon MCP server locally
 */

import { spawn } from 'child_process';
import { readFileSync, writeFileSync } from 'fs';

class MCPTester {
  constructor() {
    this.serverProcess = null;
  }

  async startServer() {
    console.log('🚀 Starting MCP server...\n');
    
    this.serverProcess = spawn('node', ['qft-lexicon-mcp.js'], {
      env: {
        ...process.env,
        LEXICON_INDEX: './lexicon_index.json',
      },
      stdio: ['pipe', 'pipe', 'pipe'],
    });

    this.serverProcess.stderr.on('data', (data) => {
      console.error(`[Server] ${data.toString()}`);
    });

    // Wait for server to be ready
    await new Promise(resolve => setTimeout(resolve, 1000));
  }

  async sendRequest(method, params = {}) {
    const request = {
      jsonrpc: '2.0',
      id: Date.now(),
      method,
      params,
    };

    console.log(`📤 Sending: ${method}`);
    console.log(JSON.stringify(params, null, 2));

    this.serverProcess.stdin.write(JSON.stringify(request) + '\n');

    return new Promise((resolve) => {
      const handler = (data) => {
        try {
          const response = JSON.parse(data.toString());
          if (response.id === request.id) {
            this.serverProcess.stdout.off('data', handler);
            resolve(response);
          }
        } catch (e) {
          // Ignore parsing errors
        }
      };

      this.serverProcess.stdout.on('data', handler);
    });
  }

  async testListTools() {
    console.log('\n📋 Test 1: List Tools\n' + '='.repeat(50));
    
    const response = await this.sendRequest('tools/list');
    
    if (response.result?.tools) {
      console.log('✅ Available tools:');
      for (const tool of response.result.tools) {
        console.log(`   - ${tool.name}: ${tool.description.substring(0, 60)}...`);
      }
    } else {
      console.log('❌ Failed to list tools');
    }
  }

  async testLexiconLookup() {
    console.log('\n🔍 Test 2: Lexicon Lookup\n' + '='.repeat(50));
    
    const response = await this.sendRequest('tools/call', {
      name: 'lexicon_lookup',
      arguments: {
        term: 'Wujudic Logic',
        includeHistory: true,
        includeUsage: true,
      },
    });

    if (response.result?.content) {
      const data = JSON.parse(response.result.content[0].text);
      console.log('✅ Found term:');
      console.log(`   Term: ${data.term}`);
      console.log(`   Found: ${data.found}`);
      console.log(`   Occurrences: ${data.occurrences}`);
      console.log(`   Papers: ${data.papers?.join(', ')}`);
    } else {
      console.log('❌ Lookup failed');
    }
  }

  async testListPapers() {
    console.log('\n📚 Test 3: List Papers\n' + '='.repeat(50));
    
    const response = await this.sendRequest('tools/call', {
      name: 'list_papers',
      arguments: {
        includeStats: true,
      },
    });

    if (response.result?.content) {
      const data = JSON.parse(response.result.content[0].text);
      console.log('✅ Papers found:');
      console.log(`   Total: ${data.total}`);
      if (data.papers?.length > 0) {
        data.papers.slice(0, 5).forEach(p => {
          console.log(`   ${p.number}. ${p.title}`);
        });
      }
    } else {
      console.log('❌ List failed');
    }
  }

  async testSearchCorpus() {
    console.log('\n🔎 Test 4: Search Corpus\n' + '='.repeat(50));
    
    const response = await this.sendRequest('tools/call', {
      name: 'search_corpus',
      arguments: {
        query: 'quantum',
        topK: 5,
      },
    });

    if (response.result?.content) {
      const data = JSON.parse(response.result.content[0].text);
      console.log('✅ Search results:');
      console.log(`   Query: ${data.query}`);
      console.log(`   Results: ${data.total}`);
      if (data.results?.length > 0) {
        data.results.slice(0, 3).forEach((r, i) => {
          console.log(`   ${i + 1}. ${r.source}: ${r.text.substring(0, 80)}...`);
        });
      }
    } else {
      console.log('❌ Search failed');
    }
  }

  async testTraceEvolution() {
    console.log('\n📈 Test 5: Trace Concept Evolution\n' + '='.repeat(50));
    
    const response = await this.sendRequest('tools/call', {
      name: 'trace_concept_evolution',
      arguments: {
        concept: 'narrative',
        startPaper: 1,
        endPaper: 17,
      },
    });

    if (response.result?.content) {
      const data = JSON.parse(response.result.content[0].text);
      console.log('✅ Evolution traced:');
      console.log(`   Concept: ${data.concept}`);
      console.log(`   Papers: ${data.totalPapers}`);
      console.log(`   First: Paper ${data.firstAppearance}`);
      console.log(`   Last: Paper ${data.lastAppearance}`);
    } else {
      console.log('❌ Trace failed');
    }
  }

  async runAllTests() {
    try {
      await this.startServer();
      
      console.log('🧪 Running MCP Server Tests\n' + '='.repeat(50));
      
      await this.testListTools();
      await this.testListPapers();
      await this.testLexiconLookup();
      await this.testSearchCorpus();
      await this.testTraceEvolution();
      
      console.log('\n' + '='.repeat(50));
      console.log('✅ All tests completed!\n');
      
    } catch (error) {
      console.error('\n❌ Test failed:', error.message);
    } finally {
      if (this.serverProcess) {
        this.serverProcess.kill();
      }
    }
  }
}

// Run tests
const tester = new MCPTester();
tester.runAllTests();
