#!/usr/bin/env node
/**
 * example-usage.js - Programmatic usage examples for QFT Runner
 * 
 * This demonstrates how to use the QFTRunner class in your own Node.js code
 */

import { QFTRunner } from './qft-runner.js';

// ============================================
// Example 1: Basic Setup and Status Check
// ============================================
async function example1_status() {
  console.log('=== Example 1: Status Check ===\n');
  
  const runner = new QFTRunner();
  await runner.status();
}

// ============================================
// Example 2: Generate Embeddings
// ============================================
async function example2_embeddings() {
  console.log('\n=== Example 2: Generate Embeddings ===\n');
  
  const runner = new QFTRunner();
  
  // E5 embeddings (local)
  const vectors = await runner.embed('conversations.txt', 'e5');
  console.log(`Generated vectors: ${vectors}`);
  
  // Qwen embeddings (API - requires token)
  // const vectors = await runner.embed('data.jsonl', 'qwen', { 
  //   token: process.env.HF_TOKEN 
  // });
}

// ============================================
// Example 3: Run QFT with Custom Parameters
// ============================================
async function example3_qft() {
  console.log('\n=== Example 3: Run QFT ===\n');
  
  const runner = new QFTRunner();
  
  const results = await runner.runQFT({
    vectors: 'conversations_e5.npy',
    themeId: 2,
    backend: 'ibm_brisbane',
    shots: 8192,
    layered: true,
    row: 0,
    pos: 0,
  });
  
  console.log('Results:', results);
  
  // Read the counts
  if (results.counts) {
    const fs = await import('fs');
    const counts = JSON.parse(fs.readFileSync(results.counts, 'utf-8'));
    console.log('Top measurements:', Object.entries(counts)
      .sort(([,a], [,b]) => b - a)
      .slice(0, 10));
  }
}

// ============================================
// Example 4: Full Pipeline with Error Handling
// ============================================
async function example4_full_pipeline() {
  console.log('\n=== Example 4: Full Pipeline ===\n');
  
  const runner = new QFTRunner();
  
  try {
    await runner.full('conversations.txt', {
      themeId: 3,
      model: 'e5',
      backend: 'ibm_torino',
      shots: 4096,
      layered: true,
      force: false,
    });
    
    console.log('✅ Pipeline completed successfully!');
  } catch (error) {
    console.error('❌ Pipeline failed:', error.message);
    
    // Graceful degradation - try with fewer shots
    console.log('Retrying with fewer shots...');
    await runner.full('conversations.txt', {
      themeId: 3,
      model: 'e5',
      shots: 1024,
      layered: true,
    });
  }
}

// ============================================
// Example 5: Multi-Theme Analysis
// ============================================
async function example5_multi_theme() {
  console.log('\n=== Example 5: Multi-Theme Analysis ===\n');
  
  const runner = new QFTRunner();
  const fs = await import('fs');
  
  // First, generate embeddings once
  const vectors = await runner.embed('data.txt', 'e5');
  
  // Run QFT with different themes
  const themes = [0, 1, 2, 3, 4, 5];
  const results = {};
  
  for (const themeId of themes) {
    console.log(`\n🎨 Processing theme ${themeId}...`);
    
    const result = await runner.runQFT({
      vectors,
      themeId,
      shots: 4096,
      layered: true,
    });
    
    // Save theme-specific results
    if (result.counts) {
      const countsData = JSON.parse(fs.readFileSync(result.counts, 'utf-8'));
      results[themeId] = countsData;
      
      // Backup the counts
      fs.copyFileSync(result.counts, `theme_${themeId}_counts.json`);
    }
  }
  
  // Analyze theme differences
  console.log('\n📊 Theme Analysis:');
  for (const [theme, counts] of Object.entries(results)) {
    const total = Object.values(counts).reduce((a, b) => a + b, 0);
    const topBit = Object.entries(counts).sort(([,a], [,b]) => b - a)[0];
    console.log(`Theme ${theme}: ${Object.keys(counts).length} unique states, top: ${topBit[0]} (${topBit[1]}/${total})`);
  }
}

// ============================================
// Example 6: Batch Processing Files
// ============================================
async function example6_batch() {
  console.log('\n=== Example 6: Batch Processing ===\n');
  
  const runner = new QFTRunner();
  const fs = await import('fs');
  const path = await import('path');
  
  // Find all .txt files
  const files = fs.readdirSync('.')
    .filter(f => f.endsWith('.txt') && !f.startsWith('test'));
  
  console.log(`Found ${files.length} files to process`);
  
  for (const file of files) {
    console.log(`\n📄 Processing ${file}...`);
    
    try {
      await runner.full(file, {
        themeId: 2,
        model: 'e5',
        shots: 4096,
        layered: true,
      });
      
      // Move results to file-specific folder
      const basename = path.basename(file, '.txt');
      const resultDir = `results/${basename}`;
      fs.mkdirSync(resultDir, { recursive: true });
      
      if (fs.existsSync('qft_counts.json')) {
        fs.renameSync('qft_counts.json', `${resultDir}/counts.json`);
      }
      if (fs.existsSync('decoded_evidence.json')) {
        fs.renameSync('decoded_evidence.json', `${resultDir}/decoded.json`);
      }
      
      console.log(`✅ ${file} → ${resultDir}/`);
    } catch (error) {
      console.error(`❌ Failed to process ${file}:`, error.message);
    }
  }
}

// ============================================
// Example 7: Custom Configuration
// ============================================
async function example7_custom_config() {
  console.log('\n=== Example 7: Custom Configuration ===\n');
  
  // Load with custom config path
  const runner = new QFTRunner('./custom-config.env');
  
  // Override config programmatically
  runner.config.defaultBackend = 'ibm_osaka';
  runner.config.defaultShots = 16384;
  runner.config.targetDim = 512;  // Lower dimension for faster processing
  
  console.log('Custom config:', {
    backend: runner.config.defaultBackend,
    shots: runner.config.defaultShots,
    targetDim: runner.config.targetDim,
  });
  
  await runner.status();
}

// ============================================
// Example 8: Integration with TradeLayer
// ============================================
async function example8_tradelayer_integration() {
  console.log('\n=== Example 8: TradeLayer Integration ===\n');
  
  const runner = new QFTRunner();
  const fs = await import('fs');
  
  // Simulate TradeLayer data structure
  const marketData = {
    symbol: 'BTC-USD',
    timeframe: '1h',
    indicators: [
      'price action shows consolidation pattern',
      'volume declining on pullback',
      'RSI showing bullish divergence',
      'MACD histogram turning positive',
    ],
  };
  
  // Convert to text for embedding
  const text = marketData.indicators.join('\n');
  fs.writeFileSync('market_signal.txt', text);
  
  // Run QFT to find pattern interference
  await runner.full('market_signal.txt', {
    themeId: 5,  // Custom theme for trading signals
    shots: 8192,
    layered: true,
  });
  
  // Read results
  if (fs.existsSync('qft_counts.json')) {
    const counts = JSON.parse(fs.readFileSync('qft_counts.json', 'utf-8'));
    
    // Extract dominant pattern
    const pattern = Object.entries(counts)
      .sort(([,a], [,b]) => b - a)[0];
    
    console.log('\n📊 Dominant Pattern:', {
      bitstring: pattern[0],
      probability: pattern[1] / Object.values(counts).reduce((a,b) => a+b, 0),
      signal: parseInt(pattern[0], 2) % 100 > 50 ? 'BULLISH' : 'BEARISH',
    });
  }
}

// ============================================
// Example 9: Real-time Monitoring
// ============================================
async function example9_monitoring() {
  console.log('\n=== Example 9: Real-time Monitoring ===\n');
  
  const runner = new QFTRunner();
  
  // Monitor pipeline status
  setInterval(async () => {
    const timestamp = new Date().toISOString();
    console.log(`\n[${timestamp}] Status Check:`);
    await runner.status();
  }, 60000); // Every minute
  
  console.log('Monitoring started... (press Ctrl+C to stop)');
}

// ============================================
// Example 10: Testing & Validation
// ============================================
async function example10_testing() {
  console.log('\n=== Example 10: Testing & Validation ===\n');
  
  const runner = new QFTRunner();
  const fs = await import('fs');
  
  // Create test data
  const testData = [
    'quantum interference pattern alpha',
    'fourier transform signal beta',
    'embedding vector analysis gamma',
  ].join('\n');
  
  fs.writeFileSync('test_input.txt', testData);
  
  // Run with minimal shots for quick test
  console.log('🧪 Running test pipeline...');
  await runner.full('test_input.txt', {
    themeId: 1,
    shots: 1024,  // Fast test
    layered: true,
    force: true,
  });
  
  // Validate outputs exist
  const expectedFiles = [
    'qft_Z.npy',
    'vectors_pca_topk.npy',
    'qft_counts.json',
  ];
  
  console.log('\n✅ Validation:');
  for (const file of expectedFiles) {
    const exists = fs.existsSync(file);
    console.log(`   ${exists ? '✅' : '❌'} ${file}`);
  }
  
  // Cleanup
  fs.unlinkSync('test_input.txt');
  console.log('\n✅ Test complete!');
}

// ============================================
// Main Menu
// ============================================
async function main() {
  const examples = {
    '1': { name: 'Status Check', fn: example1_status },
    '2': { name: 'Generate Embeddings', fn: example2_embeddings },
    '3': { name: 'Run QFT', fn: example3_qft },
    '4': { name: 'Full Pipeline', fn: example4_full_pipeline },
    '5': { name: 'Multi-Theme Analysis', fn: example5_multi_theme },
    '6': { name: 'Batch Processing', fn: example6_batch },
    '7': { name: 'Custom Config', fn: example7_custom_config },
    '8': { name: 'TradeLayer Integration', fn: example8_tradelayer_integration },
    '9': { name: 'Real-time Monitoring', fn: example9_monitoring },
    '10': { name: 'Testing & Validation', fn: example10_testing },
  };
  
  const args = process.argv.slice(2);
  const exampleNum = args[0];
  
  if (!exampleNum || !examples[exampleNum]) {
    console.log('🌊 QFT Runner - Usage Examples\n');
    console.log('Usage: node example-usage.js <example_number>\n');
    console.log('Available examples:');
    for (const [num, { name }] of Object.entries(examples)) {
      console.log(`  ${num.padStart(2)}. ${name}`);
    }
    console.log('\nExample: node example-usage.js 1');
    return;
  }
  
  try {
    await examples[exampleNum].fn();
  } catch (error) {
    console.error('\n❌ Error:', error.message);
    process.exit(1);
  }
}

// Run if called directly
if (import.meta.url === `file://${process.argv[1]}`) {
  main();
}

export {
  example1_status,
  example2_embeddings,
  example3_qft,
  example4_full_pipeline,
  example5_multi_theme,
  example6_batch,
  example7_custom_config,
  example8_tradelayer_integration,
  example9_monitoring,
  example10_testing,
};
