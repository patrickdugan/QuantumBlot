"""SC1028 symbol registry (versioned).

Defines a stable mapping between symbolic reasoning primitives and bit indices.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List

SC1028_VERSION = "1.0.0"
SC1028_BITS = 1028
SC1028_STORAGE_BITS = 1032
SC1028_STORAGE_BYTES = 129


@dataclass(frozen=True)
class Primitive:
    name: str
    bit: int
    description: str


_PRIMITIVE_ROWS = [
    ("meta.schema_v1", 0, "Schema marker bit"),
    ("meta.chunk_boundary", 1, "Chunk/action boundary"),
    ("meta.provenance_tag_enabled", 2, "Explicit provenance tag is enabled"),
    ("meta.seed_present", 3, "Run seed provided"),
    ("meta.run_start", 4, "Run start marker"),
    ("meta.run_done", 5, "Run completion marker"),
    ("action.embed", 16, "Embedding action"),
    ("action.clean_chat", 17, "Chat cleaning action"),
    ("action.qft_run", 18, "QFT execution action"),
    ("action.full_pipeline", 19, "Full pipeline action"),
    ("action.status", 20, "Status command action"),
    ("action.tool_boundary", 21, "Tool/action boundary"),
    ("tool.embed_e5", 64, "embed_e5.py tool"),
    ("tool.embed_qwen_api", 65, "embed_qwen_api.py tool"),
    ("tool.clean_chat", 66, "clean_chat.py tool"),
    ("tool.qft_one", 67, "qft_one.py tool"),
    ("tool.none", 68, "No external tool"),
    ("term.chunk_ok", 128, "Chunk completed successfully"),
    ("term.chunk_failed", 129, "Chunk failed"),
    ("term.chunk_skipped", 130, "Chunk skipped"),
    ("term.episode_done", 131, "Episode done"),
    ("term.run_done", 132, "Run done"),
    ("term.max_steps_hit", 133, "Max steps reached"),
    ("term.max_episodes_hit", 134, "Max episodes reached"),
    ("term.watchdog_exit", 135, "RSS watchdog triggered"),
    ("uncert.entropy_available", 192, "Entropy metric available"),
    ("uncert.entropy_high", 193, "High entropy"),
    ("uncert.entropy_low", 194, "Low entropy"),
    ("uncert.topk_gap_available", 195, "Top-k gap metric available"),
    ("uncert.topk_gap_high", 196, "High confidence gap"),
    ("uncert.topk_gap_low", 197, "Low confidence gap"),
    ("constraint.check_available", 256, "Constraint checks performed"),
    ("constraint.pass", 257, "Constraint checks passed"),
    ("constraint.fail", 258, "Constraint checks failed"),
    ("loop.repeat_obs_hash", 320, "Repeated observation hash"),
    ("loop.repeat_action_hash", 321, "Repeated action hash"),
    ("loop.short_cycle", 322, "Detected short cycle"),
    ("mode.inference_only", 384, "Inference-only mode"),
    ("mode.eval_applied", 385, "eval() applied"),
    ("mode.no_grad_applied", 386, "no_grad applied"),
    ("resource.rss_sampled", 448, "RSS sampled"),
    ("resource.rss_watchdog_ok", 449, "RSS below threshold"),
    ("resource.rss_watchdog_tripped", 450, "RSS threshold crossed"),
    ("resource.lockfile_acquired", 451, "Run lockfile acquired"),
    ("resource.flush_per_chunk", 452, "Telemetry flushed per chunk"),
]


PRIMITIVES: List[Primitive] = [Primitive(name=n, bit=b, description=d) for (n, b, d) in _PRIMITIVE_ROWS]
SYMBOL_TO_BIT: Dict[str, int] = {p.name: p.bit for p in PRIMITIVES}
BIT_TO_SYMBOL: Dict[int, str] = {p.bit: p.name for p in PRIMITIVES}


def validate_registry() -> None:
    if len(SYMBOL_TO_BIT) != len(PRIMITIVES):
        raise ValueError("Duplicate SC1028 symbol names.")
    if len(BIT_TO_SYMBOL) != len(PRIMITIVES):
        raise ValueError("Duplicate SC1028 bit indices.")
    for primitive in PRIMITIVES:
        if primitive.bit < 0 or primitive.bit >= SC1028_BITS:
            raise ValueError(f"Bit index out of range for {primitive.name}: {primitive.bit}")


def known_symbols() -> List[str]:
    return sorted(SYMBOL_TO_BIT.keys())


def resolve_bits(symbols: Iterable[str], strict: bool = True) -> List[int]:
    bits: List[int] = []
    for symbol in symbols:
        bit = SYMBOL_TO_BIT.get(symbol)
        if bit is None:
            if strict:
                raise KeyError(f"Unknown SC1028 symbol: {symbol}")
            continue
        bits.append(bit)
    return bits


validate_registry()

