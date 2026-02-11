#!/usr/bin/env node

import { spawn } from 'child_process';
import { randomUUID } from 'crypto';
import { existsSync, readFileSync, readdirSync, statSync } from 'fs';
import { dirname, resolve } from 'path';
import { fileURLToPath } from 'url';

import {
  SC1028_VERSION,
  encodeSymbols,
  entropyAndTopkGap,
  stableHash,
  symbolsForChunk,
  toBase64Url,
} from './semantic_codex.js';
import {
  ChunkCycleTracker,
  JsonlTelemetryWriter,
  RSSWatchdog,
  RunLock,
} from './telemetry_runtime.js';

const __filename = fileURLToPath(import.meta.url);
const __dirname = dirname(__filename);

const DEFAULTS = {
  maxSteps: 32,
  maxEpisodes: 1,
  rssThreshold: 0.85,
  lockfile: 'run.lock',
  telemetryOut: 'sc1028_telemetry.jsonl',
  attestationMode: 'crypto',
  attestationKeyEnv: 'SEMANTIC_ATTESTATION_KEY',
  attestationComponents: 16,
  attestationFrequency: 3,
  attestationStrength: 0.08,
  attestationSeed: 7,
  inferenceOnly: true,
  evalApplied: true,
  noGradApplied: true,
};

class StepLimitError extends Error {}
class EpisodeLimitError extends Error {}
class RSSWatchdogError extends Error {}

class SemanticRuntime {
  constructor({
    runId,
    writer,
    lockfileAcquired,
    watchdog,
    maxSteps,
    maxEpisodes,
    provenanceTag,
    command,
    seed,
    inferenceOnly,
    evalApplied,
    noGradApplied,
  }) {
    this.runId = runId;
    this.writer = writer;
    this.lockfileAcquired = lockfileAcquired;
    this.watchdog = watchdog;
    this.maxSteps = maxSteps;
    this.maxEpisodes = maxEpisodes;
    this.provenanceTag = provenanceTag;
    this.command = command;
    this.seed = seed;
    this.inferenceOnly = inferenceOnly;
    this.evalApplied = evalApplied;
    this.noGradApplied = noGradApplied;

    this.chunkId = 0;
    this.episodeSeen = new Set();
    this.stepsByEpisode = new Map();
    this.cycleTracker = new ChunkCycleTracker(256);
  }

  ensureEpisode(episodeId) {
    if (!this.episodeSeen.has(episodeId)) {
      if (this.episodeSeen.size >= this.maxEpisodes) {
        throw new EpisodeLimitError(`max_episodes exceeded (${this.maxEpisodes})`);
      }
      this.episodeSeen.add(episodeId);
    }
  }

  emitChunk({
    episodeId,
    actionType,
    toolName,
    observation,
    action,
    termination = 'running',
    constraintPass = true,
    entropy = null,
    entropyNorm = null,
    topkGap = null,
    scalars = {},
    countStep = true,
    enforceWatchdog = true,
  }) {
    this.ensureEpisode(episodeId);

    if (countStep) {
      const used = this.stepsByEpisode.get(episodeId) || 0;
      if (used >= this.maxSteps) {
        throw new StepLimitError(`max_steps exceeded (${this.maxSteps}) in episode ${episodeId}`);
      }
      this.stepsByEpisode.set(episodeId, used + 1);
    }

    this.chunkId += 1;
    const obsHash = stableHash(observation);
    const actionHash = stableHash(action);
    const cycle = this.cycleTracker.update(obsHash, actionHash);

    const rss = this.watchdog.sample();
    const watchdogTripped = Boolean(enforceWatchdog && rss.tripped);
    const effectiveTermination = watchdogTripped ? 'watchdog_exit' : termination;

    const symbols = symbolsForChunk({
      actionType,
      toolName,
      termination: effectiveTermination,
      entropyNorm,
      topkGap,
      constraintPass,
      repeatObs: cycle.repeatObs,
      repeatAction: cycle.repeatAction,
      shortCycle: cycle.shortCycle,
      rssRatio: rss.ratio,
      watchdogTripped,
      lockfileAcquired: this.lockfileAcquired,
      provenanceTag: this.provenanceTag,
      seedPresent: this.seed !== undefined && this.seed !== null,
      command: this.command,
      inferenceOnly: this.inferenceOnly,
      evalApplied: this.evalApplied,
      noGradApplied: this.noGradApplied,
    });

    const bitset = encodeSymbols(symbols, { strict: true });
    const sc1028B64 = toBase64Url(bitset);

    const recordScalars = {
      rss_mb: rss.rssMb,
      rss_ratio: rss.ratio,
      total_ram_mb: rss.totalMb,
      entropy: entropy ?? null,
      entropy_norm: entropyNorm ?? null,
      topk_gap: topkGap ?? null,
      constraint_pass: constraintPass === null || constraintPass === undefined ? null : (constraintPass ? 1 : 0),
      repeat_obs: cycle.repeatObs ? 1 : 0,
      repeat_action: cycle.repeatAction ? 1 : 0,
      short_cycle: cycle.shortCycle ? 1 : 0,
      ...scalars,
    };

    const line = {
      run_id: this.runId,
      episode_id: episodeId,
      chunk_id: this.chunkId,
      obs_hash: obsHash,
      action_hash: actionHash,
      sc1028_b64: sc1028B64,
      sc1028_symbols: symbols,
      sc1028_version: SC1028_VERSION,
      scalars: recordScalars,
      termination: effectiveTermination,
      seed: this.seed ?? null,
      ts_unix: Date.now() / 1000,
    };

    this.writer.write(line);

    if (watchdogTripped) {
      throw new RSSWatchdogError(
        `rss watchdog triggered: ratio=${rss.ratio.toFixed(4)} threshold=${this.watchdog.threshold.toFixed(4)}`
      );
    }
  }
}

class QFTRunner {
  constructor(configPath, runtime) {
    this.runtime = runtime;
    this.config = this.loadConfig(configPath);
    this.validateEnvironment();
  }

  loadConfig(configPath) {
    const envPath = configPath || resolve(__dirname, 'qblot.env');

    const defaults = {
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
          defaults.defaultShots = parseInt(line.split('=')[1].trim(), 10);
        }
      }
    }

    return defaults;
  }

  validateEnvironment() {
    if (!this.config.ibmApiKey) {
      console.warn('WARNING: IBM_CLOUD_API_KEY not set; QFT execution may fail');
    }
    if (!this.config.ibmCrn) {
      console.warn('WARNING: IBM_QUANTUM_CRN not set; QFT execution may fail');
    }
  }

  async runPython(script, args) {
    const pythonPath = resolve(this.config.pythonScripts, script);
    if (!existsSync(pythonPath)) {
      throw new Error(`Python script not found: ${pythonPath}`);
    }

    const candidates = process.platform === 'win32'
      ? ['python', 'python3', 'py']
      : ['python3', 'python'];

    const isNotInstalled = (code, stderr) => {
      const msg = String(stderr || '').toLowerCase();
      return code === 9009 || msg.includes('python was not found') || msg.includes('command not found');
    };

    const runWith = (idx) => new Promise((resolvePromise, rejectPromise) => {
      if (idx >= candidates.length) {
        rejectPromise(new Error('No usable Python interpreter found (tried python/python3/py).'));
        return;
      }
      const bin = candidates[idx];
      const launchArgs = bin === 'py' ? ['-3', pythonPath, ...args] : [pythonPath, ...args];
      console.log(`\nRunning: ${bin} ${script} ${args.join(' ')}`);
      const proc = spawn(bin, launchArgs, {
        stdio: ['inherit', 'pipe', 'pipe'],
        env: {
          ...process.env,
          QFT_RUNNER_INFERENCE_ONLY: '1',
          QFT_RUNNER_NO_GRAD: '1',
          QFT_RUNNER_MODEL_EVAL: '1',
        },
      });

      let stdout = '';
      let stderr = '';
      let spawned = true;

      proc.on('error', (_err) => {
        spawned = false;
      });

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
        if (!spawned || isNotInstalled(code, stderr)) {
          resolvePromise(runWith(idx + 1));
          return;
        }
        if (code !== 0) {
          rejectPromise(new Error(`Python script failed with code ${code}\n${stderr}`));
          return;
        }
        resolvePromise({ stdout, stderr, code: code || 0 });
      });
    });

    return runWith(0);
  }

  async runSemanticAttestation({
    telemetryOut,
    mode,
    outPath,
    keyEnv,
    nComponents,
    watermarkFrequency,
    watermarkStrength,
    seed,
    spectralTraceOut,
  }) {
    if (mode === 'off') {
      return null;
    }
    const args = [
      'attest',
      '--telemetry', telemetryOut,
      '--mode', mode,
      '--out', outPath,
      '--key-env', keyEnv,
      '--n-components', String(nComponents),
      '--watermark-frequency', String(watermarkFrequency),
      '--watermark-strength', String(watermarkStrength),
      '--seed', String(seed),
    ];
    if (spectralTraceOut && (mode === 'spectral' || mode === 'both')) {
      args.push('--spectral-trace-out', spectralTraceOut);
    }
    const { stdout } = await this.runPython('semantic_proof.py', args);
    const lines = stdout
      .split(/\r?\n/)
      .map((line) => line.trim())
      .filter((line) => line.length > 0);
    let parsed = null;
    for (let i = lines.length - 1; i >= 0; i -= 1) {
      try {
        parsed = JSON.parse(lines[i]);
        break;
      } catch (_err) {
        continue;
      }
    }
    return {
      mode,
      out: outPath,
      spectralTraceOut: spectralTraceOut || null,
      summary: parsed,
    };
  }

  findCountsArtifact() {
    const candidates = [];
    if (existsSync('qft_counts.json')) {
      candidates.push('qft_counts.json');
    }
    const all = readdirSync(process.cwd());
    for (const name of all) {
      if (/^job_qft_counts_row.*\.json$/.test(name)) {
        candidates.push(name);
      }
    }
    if (!candidates.length) {
      return null;
    }
    candidates.sort((a, b) => statSync(b).mtimeMs - statSync(a).mtimeMs);
    return candidates[0];
  }

  readCountsMetrics(filePath) {
    if (!filePath || !existsSync(filePath)) {
      return { entropy: null, entropyNorm: null, topkGap: null };
    }
    try {
      const counts = JSON.parse(readFileSync(filePath, 'utf-8'));
      return entropyAndTopkGap(counts);
    } catch (_err) {
      return { entropy: null, entropyNorm: null, topkGap: null };
    }
  }

  async embed(input, model = 'e5', options = {}) {
    const outputJsonl = input.replace(/\.(txt|jsonl)$/, `_${model}.jsonl`);
    const outputNpy = outputJsonl.replace('.jsonl', '.npy');
    const constraintPass = existsSync(input);
    if (!constraintPass) {
      throw new Error(`Input file not found: ${input}`);
    }

    if (model === 'e5') {
      await this.runPython('embed_e5.py', [
        '--input', input,
        '--output-jsonl', outputJsonl,
        '--output-npy', outputNpy,
        '--batch-size', '32',
      ]);
      this.runtime.emitChunk({
        episodeId: options.episodeId ?? 0,
        actionType: 'embed',
        toolName: 'embed_e5.py',
        observation: { input, model },
        action: { output_jsonl: outputJsonl, output_npy: outputNpy },
        constraintPass,
      });
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
      this.runtime.emitChunk({
        episodeId: options.episodeId ?? 0,
        actionType: 'embed',
        toolName: 'embed_qwen_api.py',
        observation: { input, model },
        action: { output_jsonl: outputJsonl, output_npy: outputNpy },
        constraintPass,
      });
    } else {
      throw new Error(`Unsupported model: ${model}`);
    }

    console.log(`Embeddings saved: ${outputNpy}`);
    return outputNpy;
  }

  async cleanChat(conversationsJson, episodeId = 0) {
    await this.runPython('clean_chat.py', []);
    const output = 'conversations_bucketed.txt';
    this.runtime.emitChunk({
      episodeId,
      actionType: 'clean_chat',
      toolName: 'clean_chat.py',
      observation: { input: conversationsJson },
      action: { output },
      constraintPass: existsSync(conversationsJson),
    });
    return output;
  }

  async runQFT(options = {}) {
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
      episodeId = 0,
      maxSteps,
      maxEpisodes,
      rssThreshold,
    } = options;

    if (!vectors) {
      throw new Error('--vectors path required');
    }
    if (!existsSync(vectors)) {
      throw new Error(`Vectors file not found: ${vectors}`);
    }

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
      '--max-steps', String(maxSteps ?? DEFAULTS.maxSteps),
      '--max-episodes', String(maxEpisodes ?? DEFAULTS.maxEpisodes),
      '--rss-threshold', String(rssThreshold ?? DEFAULTS.rssThreshold),
    ];
    if (layered) args.push('--layered');
    if (themeId > 0) args.push('--theme-id', themeId.toString());
    if (force) args.push('--force');
    if (rope) {
      args.push('--rope', rope);
      args.push('--vectors-jsonl', vectors.replace('.npy', '.jsonl'));
    }

    await this.runPython('qft_one.py', args);
    const countsPath = this.findCountsArtifact();
    const metrics = this.readCountsMetrics(countsPath);

    this.runtime.emitChunk({
      episodeId,
      actionType: 'qft_run',
      toolName: 'qft_one.py',
      observation: { vectors, themeId, backend, shots, row, pos, layered },
      action: { counts_path: countsPath, rope: rope || null },
      constraintPass: true,
      entropy: metrics.entropy,
      entropyNorm: metrics.entropyNorm,
      topkGap: metrics.topkGap,
      scalars: {
        bins: countsPath && existsSync(countsPath) ? Object.keys(JSON.parse(readFileSync(countsPath, 'utf-8'))).length : null,
      },
    });

    return {
      counts: countsPath || 'qft_counts.json',
      decoded: existsSync('decoded_evidence.json') ? 'decoded_evidence.json' : undefined,
      payload: existsSync('request_skeleton.json') ? 'request_skeleton.json' : undefined,
    };
  }

  async full(input, options = {}) {
    const episodeId = options.episodeId ?? 0;
    let textInput = input;
    if (input.includes('conversations.json')) {
      textInput = await this.cleanChat(input, episodeId);
    }
    const vectors = await this.embed(textInput, options.model || 'e5', {
      token: options.token,
      episodeId,
    });
    const result = await this.runQFT({
      ...options,
      vectors,
      episodeId,
    });
    this.runtime.emitChunk({
      episodeId,
      actionType: 'qft_run',
      toolName: 'tool.none',
      observation: { input, model: options.model || 'e5' },
      action: { episode_done: true, counts: result.counts },
      termination: 'episode_done',
      constraintPass: true,
      countStep: false,
      enforceWatchdog: false,
    });
    return result;
  }

  async status() {
    const files = ['qft_Z.npy', 'vectors_pca_topk.npy', 'qft_counts.json', 'decoded_evidence.json', 'request_skeleton.json'];
    const artifacts = files.map((file) => ({ file, exists: existsSync(file) }));
    this.runtime.emitChunk({
      episodeId: 0,
      actionType: 'status',
      toolName: 'tool.none',
      observation: { command: 'status' },
      action: { artifacts },
      constraintPass: true,
      countStep: false,
      enforceWatchdog: false,
    });

    console.log('\nQFT Pipeline Status\n');
    console.log(`Backend: ${this.config.defaultBackend}`);
    console.log(`Shots: ${this.config.defaultShots}`);
    console.log(`Target Dim: ${this.config.targetDim}`);
    console.log(`Qubits: ${this.config.nqubits}`);
    console.log(`Sparsity: ${this.config.sparsity * 100}%`);
    console.log('\nPipeline Artifacts:');
    for (const row of artifacts) {
      console.log(`  ${row.exists ? '[ok]' : '[missing]'} ${row.file}`);
    }
  }
}

function getArgValue(args, flag, fallback = undefined) {
  const index = args.indexOf(flag);
  if (index >= 0 && index + 1 < args.length) {
    return args[index + 1];
  }
  return fallback;
}

function hasFlag(args, flag) {
  return args.includes(flag);
}

function parseIntArg(args, flag, fallback = undefined) {
  const raw = getArgValue(args, flag, undefined);
  if (raw === undefined) {
    return fallback;
  }
  const v = parseInt(raw, 10);
  return Number.isFinite(v) ? v : fallback;
}

function parseFloatArg(args, flag, fallback = undefined) {
  const raw = getArgValue(args, flag, undefined);
  if (raw === undefined) {
    return fallback;
  }
  const v = parseFloat(raw);
  return Number.isFinite(v) ? v : fallback;
}

function helpText() {
  return `
QFT Runner with SC1028 telemetry

Usage:
  node qft-runner.js <command> [options]

Commands:
  embed
  run
  full
  status

Runtime guardrails:
  --lockfile <path>           Lockfile path (default: ${DEFAULTS.lockfile})
  --telemetry_out <path>      SC1028 JSONL output (default: ${DEFAULTS.telemetryOut})
  --max_steps <n>             Max chunk steps per episode (default: ${DEFAULTS.maxSteps})
  --max_episodes <n>          Max episodes per run (default: ${DEFAULTS.maxEpisodes})
  --episodes <n>              Episodes to run for full command (default: 1)
  --rss_threshold <f>         RSS watchdog threshold [0.80-0.95] (default: ${DEFAULTS.rssThreshold})
  --seed <n>                  Seed tag for telemetry
  --attestation_mode <mode>   off|crypto|spectral|both (default: ${DEFAULTS.attestationMode})
  --attestation_out <path>    Attestation JSON output (default: <telemetry>.attestation.json)
  --attestation_key_env <env> Env var for crypto key (default: ${DEFAULTS.attestationKeyEnv})
  --attestation_components <n> PCA components for spectral mode (default: ${DEFAULTS.attestationComponents})
  --attestation_frequency <n> Spectral watermark frequency (default: ${DEFAULTS.attestationFrequency})
  --attestation_strength <f>  Spectral watermark strength (default: ${DEFAULTS.attestationStrength})
  --attestation_seed <n>      Spectral signature seed (default: ${DEFAULTS.attestationSeed})
  --spectral_trace_out <path> Explicit spectral trace artifact (.npy)
  --provenance_tag            Print explicit provenance block

Examples:
  node qft-runner.js full --input data.txt --theme-id 2 --provenance_tag
  node qft-runner.js run --vectors data_e5.npy --max_steps 16 --telemetry_out artifacts/sc1028.jsonl
  node qft-runner.js status --attestation_mode both --attestation_out artifacts/attest.json
  `;
}

async function main() {
  const args = process.argv.slice(2);
  const command = args[0];

  if (!command || command === '--help' || command === '-h') {
    console.log(helpText());
    return;
  }

  const runId = randomUUID().replace(/-/g, '');
  const lockfilePath = getArgValue(args, '--lockfile', DEFAULTS.lockfile);
  const telemetryOut = getArgValue(args, '--telemetry_out', DEFAULTS.telemetryOut);
  const maxSteps = parseIntArg(args, '--max_steps', DEFAULTS.maxSteps);
  const maxEpisodes = parseIntArg(args, '--max_episodes', DEFAULTS.maxEpisodes);
  const episodes = parseIntArg(args, '--episodes', 1);
  const rssThreshold = parseFloatArg(args, '--rss_threshold', DEFAULTS.rssThreshold);
  const seed = parseIntArg(args, '--seed', undefined);
  const attestationMode = getArgValue(args, '--attestation_mode', DEFAULTS.attestationMode);
  const attestationOut = getArgValue(args, '--attestation_out', `${telemetryOut}.attestation.json`);
  const attestationKeyEnv = getArgValue(args, '--attestation_key_env', DEFAULTS.attestationKeyEnv);
  const attestationComponents = parseIntArg(args, '--attestation_components', DEFAULTS.attestationComponents);
  const attestationFrequency = parseIntArg(args, '--attestation_frequency', DEFAULTS.attestationFrequency);
  const attestationStrength = parseFloatArg(args, '--attestation_strength', DEFAULTS.attestationStrength);
  const attestationSeed = parseIntArg(args, '--attestation_seed', DEFAULTS.attestationSeed);
  const spectralTraceOut = getArgValue(args, '--spectral_trace_out', `${attestationOut}.spectral.npy`);
  const provenanceTag = hasFlag(args, '--provenance_tag');

  if (!(maxSteps > 0)) throw new Error('--max_steps must be > 0');
  if (!(maxEpisodes > 0)) throw new Error('--max_episodes must be > 0');
  if (!(episodes > 0)) throw new Error('--episodes must be > 0');
  if (!(rssThreshold >= 0.8 && rssThreshold <= 0.95)) {
    throw new Error('--rss_threshold must be in [0.80, 0.95]');
  }
  if (!['off', 'crypto', 'spectral', 'both'].includes(attestationMode)) {
    throw new Error('--attestation_mode must be one of off|crypto|spectral|both');
  }
  if (!(attestationComponents > 0)) throw new Error('--attestation_components must be > 0');
  if (!(attestationFrequency > 0)) throw new Error('--attestation_frequency must be > 0');
  if (!(attestationStrength > 0)) throw new Error('--attestation_strength must be > 0');

  const writer = new JsonlTelemetryWriter(telemetryOut, { flushEvery: 1 });
  const lock = new RunLock(lockfilePath, runId);
  let runtime = null;

  try {
    lock.acquire();
    writer.open();

    runtime = new SemanticRuntime({
      runId,
      writer,
      lockfileAcquired: lock.lockHeld,
      watchdog: new RSSWatchdog(rssThreshold),
      maxSteps,
      maxEpisodes,
      provenanceTag,
      command,
      seed,
      inferenceOnly: DEFAULTS.inferenceOnly,
      evalApplied: DEFAULTS.evalApplied,
      noGradApplied: DEFAULTS.noGradApplied,
    });

    runtime.emitChunk({
      episodeId: 0,
      actionType: command === 'status' ? 'status' : command === 'run' ? 'qft_run' : command,
      toolName: 'tool.none',
      observation: { command },
      action: { run_start: true },
      termination: 'running',
      constraintPass: true,
      countStep: false,
      enforceWatchdog: false,
    });

    const runner = new QFTRunner(undefined, runtime);

    if (command === 'embed') {
      const input = getArgValue(args, '--input', undefined);
      const model = getArgValue(args, '--model', 'e5');
      const token = getArgValue(args, '--token', undefined);
      if (!input) throw new Error('--input required');
      await runner.embed(input, model, { token, episodeId: 0 });
      runtime.emitChunk({
        episodeId: 0,
        actionType: 'embed',
        toolName: 'tool.none',
        observation: { command: 'embed' },
        action: { run_done: true },
        termination: 'run_done',
        countStep: false,
        enforceWatchdog: false,
      });
    } else if (command === 'run') {
      const vectors = getArgValue(args, '--vectors', undefined);
      const themeId = parseIntArg(args, '--theme-id', 0);
      const backend = getArgValue(args, '--backend', undefined);
      const shots = parseIntArg(args, '--shots', undefined);
      const rope = getArgValue(args, '--rope', undefined);
      const row = parseIntArg(args, '--row', 0);
      const pos = parseIntArg(args, '--pos', 0);
      const layered = hasFlag(args, '--layered');
      const force = hasFlag(args, '--force');
      if (!vectors) throw new Error('--vectors required');
      await runner.runQFT({
        vectors,
        themeId,
        backend,
        shots,
        rope,
        row,
        pos,
        layered,
        force,
        episodeId: 0,
        maxSteps,
        maxEpisodes,
        rssThreshold,
      });
      runtime.emitChunk({
        episodeId: 0,
        actionType: 'qft_run',
        toolName: 'tool.none',
        observation: { command: 'run' },
        action: { run_done: true },
        termination: 'run_done',
        countStep: false,
        enforceWatchdog: false,
      });
    } else if (command === 'full') {
      const input = getArgValue(args, '--input', undefined);
      const themeId = parseIntArg(args, '--theme-id', 0);
      const model = getArgValue(args, '--model', 'e5');
      const rope = getArgValue(args, '--rope', undefined);
      const token = getArgValue(args, '--token', undefined);
      const backend = getArgValue(args, '--backend', undefined);
      const shots = parseIntArg(args, '--shots', undefined);
      const layered = hasFlag(args, '--layered');
      const force = hasFlag(args, '--force');
      if (!input) throw new Error('--input required');

      if (episodes > maxEpisodes) {
        runtime.emitChunk({
          episodeId: 0,
          actionType: 'full_pipeline',
          toolName: 'tool.none',
          observation: { requested_episodes: episodes, maxEpisodes },
          action: { blocked: true },
          termination: 'max_episodes_hit',
          constraintPass: false,
          countStep: false,
          enforceWatchdog: false,
        });
        throw new EpisodeLimitError(`episodes (${episodes}) exceeds --max_episodes (${maxEpisodes})`);
      }

      for (let episodeId = 0; episodeId < episodes; episodeId += 1) {
        await runner.full(input, {
          themeId,
          model,
          rope,
          token,
          backend,
          shots,
          layered,
          force,
          episodeId,
          maxSteps,
          maxEpisodes,
          rssThreshold,
        });
      }

      runtime.emitChunk({
        episodeId: 0,
        actionType: 'full_pipeline',
        toolName: 'tool.none',
        observation: { command: 'full', episodes },
        action: { run_done: true },
        termination: 'run_done',
        countStep: false,
        enforceWatchdog: false,
      });
    } else if (command === 'status') {
      await runner.status();
      runtime.emitChunk({
        episodeId: 0,
        actionType: 'status',
        toolName: 'tool.none',
        observation: { command: 'status' },
        action: { run_done: true },
        termination: 'run_done',
        countStep: false,
        enforceWatchdog: false,
      });
    } else {
      console.log(helpText());
    }

    let attestationResult = null;
    if (attestationMode !== 'off') {
      writer.flush();
      attestationResult = await runner.runSemanticAttestation({
        telemetryOut,
        mode: attestationMode,
        outPath: attestationOut,
        keyEnv: attestationKeyEnv,
        nComponents: attestationComponents,
        watermarkFrequency: attestationFrequency,
        watermarkStrength: attestationStrength,
        seed: attestationSeed,
        spectralTraceOut,
      });
      runtime.emitChunk({
        episodeId: 0,
        actionType: 'status',
        toolName: 'tool.none',
        observation: { attestation_mode: attestationMode },
        action: { attestation_out: attestationOut },
        constraintPass: true,
        countStep: false,
        enforceWatchdog: false,
      });
    }

    if (provenanceTag) {
      let block = `PROVENANCE: run_id=${runId} sc1028_sidecar=${telemetryOut}`;
      if (attestationResult) {
        block += ` attestation_mode=${attestationResult.mode} attestation_sidecar=${attestationResult.out}`;
      }
      console.log(block);
    }
  } catch (error) {
    if (runtime) {
      let termination = 'failure';
      if (error instanceof StepLimitError) termination = 'max_steps_hit';
      if (error instanceof EpisodeLimitError) termination = 'max_episodes_hit';
      if (error instanceof RSSWatchdogError) termination = 'watchdog_exit';
      runtime.emitChunk({
        episodeId: 0,
        actionType: command === 'status' ? 'status' : 'qft_run',
        toolName: 'tool.none',
        observation: { error: error?.message || String(error) },
        action: { exception: error?.name || 'Error' },
        termination,
        constraintPass: false,
        countStep: false,
        enforceWatchdog: false,
      });
    }
    const message = error instanceof Error ? error.message : String(error);
    console.error(`\nERROR: ${message}`);
    process.exit(1);
  } finally {
    writer.close();
    lock.release();
  }
}

if (process.argv[1] && resolve(process.argv[1]) === __filename) {
  main();
}

export { QFTRunner, SemanticRuntime, main };
