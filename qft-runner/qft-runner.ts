#!/usr/bin/env node
/**
 * qft-runner.ts — TypeScript orchestrator for QFT pipeline
 * 
 * A clean Node.js interface to your Python QFT workflow.
 * Handles embedding generation, PCA, QFT execution, and result decoding.
 * 
 * Usage:
 *   npx tsx qft-runner.ts embed --input data.txt --model e5
 *   npx tsx qft-runner.ts run --vectors vectors.npy --theme-id 2
 *   npx tsx qft-runner.ts full --input conversations.txt --theme-id 3
 */

import { spawn } from 'child_process';
import { existsSync, readFileSync, writeFileSync } from 'fs';
import { resolve } from 'path';

interface Config {
  // IBM Quantum
  ibmApiKey?: string;
  ibmCrn?: string;
  defaultBackend: string;
  defaultShots: number;
  
  // Paths
  pythonScripts: string;
  outputDir: string;
  
  // QFT params
  targetDim: number;
  sparsity: number;
  nqubits: number;
  optimizationLevel: number;
}

interface RunOptions {
  input?: string;
  vectors?: string;
  themeId?: number;
  backend?: string;
  shots?: number;
  layered?: boolean;
  row?: number;
  pos?: number;
  force?: boolean;
  rope?: string;
}

class QFTRunner {
  private config: Config;

  constructor(configPath?: string) {
    this.config = this.loadConfig(configPath);
    this.validateEnvironment();
  }

  private loadConfig(configPath?: string): Config {
    // Load from qblot.env if exists
    const envPath = configPath || resolve(__dirname, 'qblot.env');
    
    const defaults: Config = {
      defaultBackend: 'ibm_torino',
      defaultShots: 8000,
      pythonScripts: '.',
      outputDir: './qft_output',
      targetDim: 768,
      sparsity: 0.7,
      nqubits: 17,
      optimizationLevel: 1,
    };

    if (existsSync(envPath)) {
      const envContent = readFileSync(envPath, 'utf-8');
      const lines = envContent.split('\n');
      
      for (const line of lines) {
        if (line.startsWith('export IBM_CLOUD_API_KEY=')) {
          defaults.ibmApiKey = line.split('=')[1].replace(/"/g, '');
          process.env.IBM_CLOUD_API_KEY = defaults.ibmApiKey;
        } else if (line.startsWith('export IBM_QUANTUM_CRN=')) {
          defaults.ibmCrn = line.split('=')[1].replace(/"/g, '');
          process.env.IBM_QUANTUM_CRN = defaults.ibmCrn;
        } else if (line.startsWith('DEFAULT_BACKEND=')) {
          defaults.defaultBackend = line.split('=')[1].trim();
        } else if (line.startsWith('DEFAULT_SHOTS=')) {
          defaults.defaultShots = parseInt(line.split('=')[1].trim());
        }
      }
    }

    return defaults;
  }

  private validateEnvironment(): void {
    if (!this.config.ibmApiKey) {
      console.warn('⚠️  IBM_CLOUD_API_KEY not set - QFT execution will fail');
    }
    if (!this.config.ibmCrn) {
      console.warn('⚠️  IBM_QUANTUM_CRN not set - QFT execution will fail');
    }
  }

  private async runPython(
    script: string, 
    args: string[]
  ): Promise<{ stdout: string; stderr: string; code: number }> {
    return new Promise((resolve, reject) => {
      const pythonPath = resolve(this.config.pythonScripts, script);
      
      if (!existsSync(pythonPath)) {
        reject(new Error(`Python script not found: ${pythonPath}`));
        return;
      }

      console.log(`\n🐍 Running: python3 ${script} ${args.join(' ')}`);
      
      const proc = spawn('python3', [pythonPath, ...args], {
        stdio: ['inherit', 'pipe', 'pipe'],
      });

      let stdout = '';
      let stderr = '';

      proc.stdout.on('data', (data) => {
        const text = data.toString();
        stdout += text;
        process.stdout.write(text);
      });

      proc.stderr.on('data', (data) => {
        const text = data.toString();
        stderr += text;
        process.stderr.write(text);
      });

      proc.on('close', (code) => {
        if (code !== 0) {
          reject(new Error(`Python script failed with code ${code}\n${stderr}`));
        } else {
          resolve({ stdout, stderr, code: code || 0 });
        }
      });
    });
  }

  /**
   * Step 1: Generate embeddings from text
   */
  async embed(input: string, model: 'e5' | 'qwen' = 'e5', options: { token?: string } = {}): Promise<string> {
    console.log(`\n📊 Embedding with ${model.toUpperCase()}...`);
    
    const outputJsonl = input.replace(/\.(txt|jsonl)$/, `_${model}.jsonl`);
    const outputNpy = outputJsonl.replace('.jsonl', '.npy');

    if (model === 'e5') {
      await this.runPython('embed_e5.py', [
        '--input', input,
        '--output-jsonl', outputJsonl,
        '--output-npy', outputNpy,
        '--batch-size', '32',
      ]);
    } else if (model === 'qwen') {
      if (!options.token) {
        throw new Error('--token required for Qwen embeddings');
      }
      await this.runPython('embed_qwen_api.py', [
        '--input', input,
        '--output-jsonl', outputJsonl,
        '--batch-size', '8',
        '--token', options.token,
      ]);
    }

    console.log(`✅ Embeddings saved: ${outputNpy}`);
    return outputNpy;
  }

  /**
   * Step 2: Clean chat history (if needed)
   */
  async cleanChat(conversationsJson: string): Promise<string> {
    console.log('\n🧹 Cleaning chat history...');
    
    await this.runPython('clean_chat.py', []);
    
    const output = 'conversations_bucketed.txt';
    console.log(`✅ Cleaned chat: ${output}`);
    return output;
  }

  /**
   * Step 3: Run full QFT pipeline
   */
  async runQFT(options: RunOptions): Promise<{
    counts: string;
    decoded?: string;
    payload?: string;
  }> {
    const {
      vectors,
      themeId = 0,
      backend = this.config.defaultBackend,
      shots = this.config.defaultShots,
      layered = true,
      row = 0,
      pos = 0,
      force = false,
      rope,
    } = options;

    if (!vectors) {
      throw new Error('--vectors path required');
    }

    console.log('\n🌊 Running QFT pipeline...');
    console.log(`   Theme: ${themeId} | Backend: ${backend} | Shots: ${shots}`);

    const args = [
      'all',
      '--src', vectors,
      '--target-dim', this.config.targetDim.toString(),
      '--sparsity', this.config.sparsity.toString(),
      '--nqubits', this.config.nqubits.toString(),
      '--backend', backend,
      '--shots', shots.toString(),
      '--row', row.toString(),
      '--pos', pos.toString(),
      '--optimization-level', this.config.optimizationLevel.toString(),
    ];

    if (layered) args.push('--layered');
    if (themeId > 0) args.push('--theme-id', themeId.toString());
    if (force) args.push('--force');
    if (rope) {
      args.push('--rope', rope);
      args.push('--vectors-jsonl', vectors.replace('.npy', '.jsonl'));
    }

    await this.runPython('qft_one.py', args);

    const results = {
      counts: 'qft_counts.json',
      decoded: existsSync('decoded_evidence.json') ? 'decoded_evidence.json' : undefined,
      payload: existsSync('request_skeleton.json') ? 'request_skeleton.json' : undefined,
    };

    console.log('\n✅ QFT Complete!');
    console.log(`   Counts: ${results.counts}`);
    if (results.decoded) console.log(`   Decoded: ${results.decoded}`);
    if (results.payload) console.log(`   Payload: ${results.payload}`);

    return results;
  }

  /**
   * Full workflow: embed → QFT → decode
   */
  async full(input: string, options: RunOptions & { model?: 'e5' | 'qwen'; token?: string }): Promise<void> {
    console.log('🚀 Starting full QFT workflow...\n');

    // 1. Clean if it's a conversations.json
    let textInput = input;
    if (input.includes('conversations.json')) {
      textInput = await this.cleanChat(input);
    }

    // 2. Generate embeddings
    const vectors = await this.embed(textInput, options.model || 'e5', {
      token: options.token,
    });

    // 3. Run QFT
    await this.runQFT({
      ...options,
      vectors,
    });

    console.log('\n✨ Full workflow complete!');
  }

  /**
   * Query status and job info
   */
  async status(): Promise<void> {
    console.log('\n📊 QFT Pipeline Status\n');
    console.log(`Backend: ${this.config.defaultBackend}`);
    console.log(`Shots: ${this.config.defaultShots}`);
    console.log(`Target Dim: ${this.config.targetDim}`);
    console.log(`Qubits: ${this.config.nqubits}`);
    console.log(`Sparsity: ${this.config.sparsity * 100}%`);
    
    const files = [
      'qft_Z.npy',
      'vectors_pca_topk.npy',
      'qft_counts.json',
      'decoded_evidence.json',
      'request_skeleton.json',
    ];

    console.log('\n📁 Pipeline Artifacts:');
    for (const file of files) {
      const exists = existsSync(file);
      console.log(`   ${exists ? '✅' : '❌'} ${file}`);
    }
  }
}

// CLI
async function main() {
  const args = process.argv.slice(2);
  const command = args[0];

  const runner = new QFTRunner();

  try {
    if (command === 'embed') {
      const inputIdx = args.indexOf('--input');
      const modelIdx = args.indexOf('--model');
      const tokenIdx = args.indexOf('--token');

      const input = inputIdx >= 0 ? args[inputIdx + 1] : undefined;
      const model = (modelIdx >= 0 ? args[modelIdx + 1] : 'e5') as 'e5' | 'qwen';
      const token = tokenIdx >= 0 ? args[tokenIdx + 1] : undefined;

      if (!input) throw new Error('--input required');
      await runner.embed(input, model, { token });
      
    } else if (command === 'run') {
      const vectorsIdx = args.indexOf('--vectors');
      const themeIdx = args.indexOf('--theme-id');
      const backendIdx = args.indexOf('--backend');
      const shotsIdx = args.indexOf('--shots');
      const ropeIdx = args.indexOf('--rope');

      const vectors = vectorsIdx >= 0 ? args[vectorsIdx + 1] : undefined;
      const themeId = themeIdx >= 0 ? parseInt(args[themeIdx + 1]) : undefined;
      const backend = backendIdx >= 0 ? args[backendIdx + 1] : undefined;
      const shots = shotsIdx >= 0 ? parseInt(args[shotsIdx + 1]) : undefined;
      const rope = ropeIdx >= 0 ? args[ropeIdx + 1] : undefined;
      const layered = args.includes('--layered');
      const force = args.includes('--force');

      if (!vectors) throw new Error('--vectors required');
      await runner.runQFT({ vectors, themeId, backend, shots, layered, force, rope });
      
    } else if (command === 'full') {
      const inputIdx = args.indexOf('--input');
      const themeIdx = args.indexOf('--theme-id');
      const modelIdx = args.indexOf('--model');
      const ropeIdx = args.indexOf('--rope');
      const tokenIdx = args.indexOf('--token');

      const input = inputIdx >= 0 ? args[inputIdx + 1] : undefined;
      const themeId = themeIdx >= 0 ? parseInt(args[themeIdx + 1]) : undefined;
      const model = (modelIdx >= 0 ? args[modelIdx + 1] : 'e5') as 'e5' | 'qwen';
      const rope = ropeIdx >= 0 ? args[ropeIdx + 1] : undefined;
      const token = tokenIdx >= 0 ? args[tokenIdx + 1] : undefined;
      const layered = args.includes('--layered');
      const force = args.includes('--force');

      if (!input) throw new Error('--input required');
      await runner.full(input, { themeId, model, layered, force, rope, token });
      
    } else if (command === 'status') {
      await runner.status();
      
    } else {
      console.log(`
🌊 QFT Runner - Quantum Fourier Transform Pipeline Orchestrator

Usage:
  npx tsx qft-runner.ts <command> [options]

Commands:
  embed              Generate embeddings from text
    --input <file>     Input .txt or .jsonl
    --model <e5|qwen>  Embedding model (default: e5)
    --token <token>    HuggingFace token (for qwen)

  run                Run QFT on existing vectors
    --vectors <file>   .npy vector file
    --theme-id <n>     Theme ID for interference
    --backend <name>   IBM backend (default: ${runner['config'].defaultBackend})
    --shots <n>        Number of shots (default: ${runner['config'].defaultShots})
    --rope <file>      RoPE hint JSON
    --layered          Use layered QFT (default: true)
    --force            Recompute existing outputs

  full               Run complete pipeline (embed → QFT → decode)
    --input <file>     Input text file
    --theme-id <n>     Theme ID
    --model <e5|qwen>  Embedding model
    --rope <file>      RoPE hint JSON
    --token <token>    HuggingFace token (for qwen)
    --layered          Use layered QFT
    --force            Recompute existing outputs

  status             Show pipeline status and artifacts

Examples:
  # Generate embeddings
  npx tsx qft-runner.ts embed --input data.txt --model e5

  # Run QFT on embeddings
  npx tsx qft-runner.ts run --vectors data_e5.npy --theme-id 2 --shots 8192

  # Full pipeline
  npx tsx qft-runner.ts full --input conversations.txt --theme-id 3 --layered

Environment:
  IBM_CLOUD_API_KEY   IBM Quantum API key
  IBM_QUANTUM_CRN     IBM Quantum instance CRN

Config loaded from: qblot.env
      `);
    }
  } catch (error) {
    console.error('\n❌ Error:', error instanceof Error ? error.message : error);
    process.exit(1);
  }
}

if (require.main === module) {
  main();
}

export { QFTRunner, RunOptions, Config };
