import { createHash } from 'crypto';

export const SC1028_VERSION = '1.0.0';
export const SC1028_BITS = 1028;
export const SC1028_STORAGE_BITS = 1032;
export const SC1028_STORAGE_BYTES = 129;

const PRIMITIVES = [
  ['meta.schema_v1', 0],
  ['meta.chunk_boundary', 1],
  ['meta.provenance_tag_enabled', 2],
  ['meta.seed_present', 3],
  ['meta.run_start', 4],
  ['meta.run_done', 5],
  ['action.embed', 16],
  ['action.clean_chat', 17],
  ['action.qft_run', 18],
  ['action.full_pipeline', 19],
  ['action.status', 20],
  ['action.tool_boundary', 21],
  ['tool.embed_e5', 64],
  ['tool.embed_qwen_api', 65],
  ['tool.clean_chat', 66],
  ['tool.qft_one', 67],
  ['tool.none', 68],
  ['term.chunk_ok', 128],
  ['term.chunk_failed', 129],
  ['term.chunk_skipped', 130],
  ['term.episode_done', 131],
  ['term.run_done', 132],
  ['term.max_steps_hit', 133],
  ['term.max_episodes_hit', 134],
  ['term.watchdog_exit', 135],
  ['uncert.entropy_available', 192],
  ['uncert.entropy_high', 193],
  ['uncert.entropy_low', 194],
  ['uncert.topk_gap_available', 195],
  ['uncert.topk_gap_high', 196],
  ['uncert.topk_gap_low', 197],
  ['constraint.check_available', 256],
  ['constraint.pass', 257],
  ['constraint.fail', 258],
  ['loop.repeat_obs_hash', 320],
  ['loop.repeat_action_hash', 321],
  ['loop.short_cycle', 322],
  ['mode.inference_only', 384],
  ['mode.eval_applied', 385],
  ['mode.no_grad_applied', 386],
  ['resource.rss_sampled', 448],
  ['resource.rss_watchdog_ok', 449],
  ['resource.rss_watchdog_tripped', 450],
  ['resource.lockfile_acquired', 451],
  ['resource.flush_per_chunk', 452],
];

export const SYMBOL_TO_BIT = Object.freeze(Object.fromEntries(PRIMITIVES));

const BIT_TO_SYMBOL = Object.freeze(
  PRIMITIVES.reduce((acc, [symbol, bit]) => {
    acc[bit] = symbol;
    return acc;
  }, {})
);

function assertBitset(bitset) {
  if (!(bitset instanceof Uint8Array) || bitset.length !== SC1028_STORAGE_BYTES) {
    throw new Error(`SC1028 bitset must be Uint8Array(${SC1028_STORAGE_BYTES})`);
  }
  if ((bitset[SC1028_STORAGE_BYTES - 1] & 0b11110000) !== 0) {
    throw new Error('SC1028 pad bits must be zero');
  }
}

export function encodeSymbols(symbols, { strict = true } = {}) {
  const bitset = new Uint8Array(SC1028_STORAGE_BYTES);
  for (const symbol of symbols) {
    const bit = SYMBOL_TO_BIT[symbol];
    if (bit === undefined) {
      if (strict) {
        throw new Error(`Unknown SC1028 symbol: ${symbol}`);
      }
      continue;
    }
    const byteIndex = Math.floor(bit / 8);
    const bitOffset = bit % 8;
    bitset[byteIndex] |= (1 << bitOffset);
  }
  assertBitset(bitset);
  return bitset;
}

export function decodeBitset(bitset) {
  assertBitset(bitset);
  const symbols = [];
  for (let bit = 0; bit < SC1028_BITS; bit += 1) {
    const byteIndex = Math.floor(bit / 8);
    const bitOffset = bit % 8;
    if ((bitset[byteIndex] & (1 << bitOffset)) !== 0) {
      const symbol = BIT_TO_SYMBOL[bit];
      if (symbol) {
        symbols.push(symbol);
      }
    }
  }
  return symbols;
}

export function toBase64Url(bitset) {
  assertBitset(bitset);
  return Buffer.from(bitset).toString('base64url');
}

export function fromBase64Url(token) {
  if (!token) {
    throw new Error('SC1028 token cannot be empty');
  }
  const buf = Buffer.from(token, 'base64url');
  const bitset = Uint8Array.from(buf);
  assertBitset(bitset);
  return bitset;
}

function stableStringify(value) {
  if (value === null || typeof value !== 'object') {
    return JSON.stringify(value);
  }
  if (Array.isArray(value)) {
    return `[${value.map((v) => stableStringify(v)).join(',')}]`;
  }
  const keys = Object.keys(value).sort();
  const items = keys.map((k) => `${JSON.stringify(k)}:${stableStringify(value[k])}`);
  return `{${items.join(',')}}`;
}

export function stableHash(payload) {
  const normalized = stableStringify(payload);
  return createHash('sha256').update(normalized).digest('hex').slice(0, 16);
}

export function entropyAndTopkGap(counts) {
  if (!counts || typeof counts !== 'object') {
    return { entropy: null, entropyNorm: null, topkGap: null };
  }
  const values = Object.values(counts).map((v) => Number(v)).filter((v) => Number.isFinite(v) && v >= 0);
  if (!values.length) {
    return { entropy: null, entropyNorm: null, topkGap: null };
  }
  const total = values.reduce((a, b) => a + b, 0);
  if (total <= 0) {
    return { entropy: 0, entropyNorm: 0, topkGap: 0 };
  }

  const probs = values.map((v) => v / total);
  const entropy = -probs.reduce((acc, p) => (p > 0 ? acc + (p * Math.log2(p)) : acc), 0);
  const maxEntropy = probs.length > 1 ? Math.log2(probs.length) : 1;
  const entropyNorm = maxEntropy > 0 ? entropy / maxEntropy : 0;
  const sorted = probs.slice().sort((a, b) => b - a);
  const topkGap = sorted.length > 1 ? sorted[0] - sorted[1] : sorted[0];
  return { entropy, entropyNorm, topkGap };
}

export function symbolsForChunk({
  actionType,
  toolName,
  termination,
  entropyNorm,
  topkGap,
  constraintPass,
  repeatObs,
  repeatAction,
  shortCycle,
  rssRatio,
  watchdogTripped,
  lockfileAcquired,
  provenanceTag,
  seedPresent,
  command,
  inferenceOnly = true,
  evalApplied = true,
  noGradApplied = true,
}) {
  const symbols = new Set();
  const add = (name) => {
    if (name) {
      symbols.add(name);
    }
  };

  add('meta.schema_v1');
  add('meta.chunk_boundary');
  if (provenanceTag) add('meta.provenance_tag_enabled');
  if (seedPresent) add('meta.seed_present');

  add(command === 'full' ? 'action.full_pipeline' : actionType === 'status' ? 'action.status' : null);
  if (actionType === 'embed') add('action.embed');
  if (actionType === 'clean_chat') add('action.clean_chat');
  if (actionType === 'qft_run') add('action.qft_run');
  add('action.tool_boundary');

  if (toolName === 'embed_e5.py') add('tool.embed_e5');
  else if (toolName === 'embed_qwen_api.py') add('tool.embed_qwen_api');
  else if (toolName === 'clean_chat.py') add('tool.clean_chat');
  else if (toolName === 'qft_one.py') add('tool.qft_one');
  else add('tool.none');

  if (termination === 'watchdog_exit') add('term.watchdog_exit');
  else if (termination === 'max_steps_hit') add('term.max_steps_hit');
  else if (termination === 'max_episodes_hit') add('term.max_episodes_hit');
  else if (termination === 'run_done') add('term.run_done');
  else if (termination === 'episode_done') add('term.episode_done');
  else if (termination === 'chunk_skipped') add('term.chunk_skipped');
  else if (termination === 'failure') add('term.chunk_failed');
  else add('term.chunk_ok');

  if (constraintPass !== null && constraintPass !== undefined) {
    add('constraint.check_available');
    add(constraintPass ? 'constraint.pass' : 'constraint.fail');
  }

  if (entropyNorm !== null && entropyNorm !== undefined && Number.isFinite(entropyNorm)) {
    add('uncert.entropy_available');
    add(entropyNorm >= 0.75 ? 'uncert.entropy_high' : 'uncert.entropy_low');
  }
  if (topkGap !== null && topkGap !== undefined && Number.isFinite(topkGap)) {
    add('uncert.topk_gap_available');
    add(topkGap >= 0.08 ? 'uncert.topk_gap_high' : 'uncert.topk_gap_low');
  }

  if (repeatObs) add('loop.repeat_obs_hash');
  if (repeatAction) add('loop.repeat_action_hash');
  if (shortCycle) add('loop.short_cycle');

  if (inferenceOnly) add('mode.inference_only');
  if (evalApplied) add('mode.eval_applied');
  if (noGradApplied) add('mode.no_grad_applied');

  if (rssRatio !== null && rssRatio !== undefined) {
    add('resource.rss_sampled');
    add(watchdogTripped ? 'resource.rss_watchdog_tripped' : 'resource.rss_watchdog_ok');
  }
  if (lockfileAcquired) add('resource.lockfile_acquired');
  add('resource.flush_per_chunk');

  return Array.from(symbols.values());
}
