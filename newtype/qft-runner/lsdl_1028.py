"""LSDL-1028: Logic Symbolic DSL (v1).

This module defines a fixed-width 1028-bit schema for explicit semantic indexing.
Storage format is 129 bytes (1032 bits) in big-endian byte order with 4 top
padding bits that MUST be zero.

Bit offsets are defined over the 1028-bit payload (MSB -> LSB), i.e.:
- payload bit offset 0 is the most-significant semantic bit
- payload bit offset 1027 is the least-significant semantic bit

Schema (offset:width):

A) Header (64)
  0:8    version
  8:16   flags_global
  24:20  epoch_minutes_mod
  44:20  header_reserved (must be 0)

B) Topic/Domain (256)
  64:64   concept_hash0
  128:64  concept_hash1
  192:64  concept_hash2
  256:12  domain_id
  268:12  subdomain_id
  280:12  intent_id
  292:8   style_id
  300:20  topic_reserved (must be 0)

C) Logic/Math (256)
  320:64  logic_family_bits
  384:64  math_family_bits
  448:64  operator_bits
  512:8   proof_shape
  520:6   quantifier_depth
  526:6   nesting_depth
  532:6   recursion_depth
  538:38  logic_reserved (must be 0)

D) Meta/Pathology (256)
  576:128 pathology_bits
  704:64  self_model_bits
  768:32  safety_bits
  800:32  meta_reserved (must be 0)

E) Tail/Integrity (196)
  832:64  record_nonce
  896:64  checksum (first 64 bits of SHA-256 over canonical JSON with checksum=0)
  960:32  registry_hint
  992:36  tail_reserved (must be 0)
"""

from __future__ import annotations

import base64
import hashlib
import json
from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Tuple

TOTAL_BITS = 1028
STORAGE_BITS = 1032
STORAGE_BYTES = 129
VERSION_V1 = 1

TOPICAL_GLYPH_BASE = 0xE000
TOPICAL_GLYPH_SLOTS = 1024
LOGIC_GLYPH_BASE = 0xE400
LOGIC_GLYPH_SLOTS = 1024


def _check_uint(name: str, value: int, width: int) -> None:
    if not isinstance(value, int):
        raise TypeError(f"{name} must be int")
    if value < 0 or value >= (1 << width):
        raise ValueError(f"{name} out of range for {width} bits: {value}")


def set_bits(payload: int, offset: int, width: int, value: int) -> int:
    """Set payload bits at [offset, offset+width) in MSB-indexed payload space."""
    if width <= 0:
        raise ValueError("width must be > 0")
    if offset < 0 or offset + width > TOTAL_BITS:
        raise ValueError("bit range out of bounds")
    _check_uint("value", value, width)
    shift = TOTAL_BITS - (offset + width)
    mask = ((1 << width) - 1) << shift
    payload &= ~mask
    payload |= (value << shift)
    return payload


def get_bits(payload: int, offset: int, width: int) -> int:
    """Get payload bits at [offset, offset+width) in MSB-indexed payload space."""
    if width <= 0:
        raise ValueError("width must be > 0")
    if offset < 0 or offset + width > TOTAL_BITS:
        raise ValueError("bit range out of bounds")
    shift = TOTAL_BITS - (offset + width)
    return (payload >> shift) & ((1 << width) - 1)


def normalize_concept(text: str) -> str:
    return " ".join(text.strip().lower().split())


def concept_hash64(text: str) -> int:
    """SHA-256(text-normalized), first 64 bits interpreted as little-endian."""
    norm = normalize_concept(text)
    digest = hashlib.sha256(norm.encode("utf-8")).digest()
    return int.from_bytes(digest[:8], byteorder="little", signed=False)


def _first64_sha256(data: bytes) -> int:
    return int.from_bytes(hashlib.sha256(data).digest()[:8], byteorder="big", signed=False)


FIELD_SPECS: Tuple[Tuple[str, int, int], ...] = (
    ("version", 0, 8),
    ("flags_global", 8, 16),
    ("epoch_minutes_mod", 24, 20),
    ("header_reserved", 44, 20),
    ("concept_hash0", 64, 64),
    ("concept_hash1", 128, 64),
    ("concept_hash2", 192, 64),
    ("domain_id", 256, 12),
    ("subdomain_id", 268, 12),
    ("intent_id", 280, 12),
    ("style_id", 292, 8),
    ("topic_reserved", 300, 20),
    ("logic_family_bits", 320, 64),
    ("math_family_bits", 384, 64),
    ("operator_bits", 448, 64),
    ("proof_shape", 512, 8),
    ("quantifier_depth", 520, 6),
    ("nesting_depth", 526, 6),
    ("recursion_depth", 532, 6),
    ("logic_reserved", 538, 38),
    ("pathology_bits", 576, 128),
    ("self_model_bits", 704, 64),
    ("safety_bits", 768, 32),
    ("meta_reserved", 800, 32),
    ("record_nonce", 832, 64),
    ("checksum", 896, 64),
    ("registry_hint", 960, 32),
    ("tail_reserved", 992, 36),
)

WIDTH_BY_FIELD: Dict[str, int] = {name: width for name, _off, width in FIELD_SPECS}


# Bit names for deterministic glyph extraction.
LOGIC_FAMILY_INDEX: Dict[str, int] = {
    "propositional": 0,
    "first_order_logic": 1,
    "modal_logic": 2,
    "temporal_logic": 3,
    "deontic_logic": 4,
    "type_theory": 5,
    "category_theory": 6,
}

OPERATOR_INDEX: Dict[str, int] = {
    "and": 0,
    "or": 1,
    "not": 2,
    "implies": 3,
    "forall": 4,
    "exists": 5,
    "nec": 6,
    "poss": 7,
    "fix": 8,
    "mu": 9,
    "lambda": 10,
    "equal": 11,
    "in": 12,
    "subset": 13,
}

PATHOLOGY_INDEX: Dict[str, int] = {
    "contradiction": 0,
    "equivocation": 1,
    "circularity": 2,
    "goal_shift": 3,
    "hallucination_risk": 4,
    "missing_premise": 5,
    "overfit": 6,
    "adversarial_tone": 7,
    "speculation": 8,
    "uncertainty_high": 9,
}


@dataclass
class LSDL1028:
    # Header
    version: int = VERSION_V1
    flags_global: int = 0
    epoch_minutes_mod: int = 0
    header_reserved: int = 0

    # Topic block
    concept_hash0: int = 0
    concept_hash1: int = 0
    concept_hash2: int = 0
    domain_id: int = 0
    subdomain_id: int = 0
    intent_id: int = 0
    style_id: int = 0
    topic_reserved: int = 0

    # Logic/math block
    logic_family_bits: int = 0
    math_family_bits: int = 0
    operator_bits: int = 0
    proof_shape: int = 0
    quantifier_depth: int = 0
    nesting_depth: int = 0
    recursion_depth: int = 0
    logic_reserved: int = 0

    # Meta block
    pathology_bits: int = 0
    self_model_bits: int = 0
    safety_bits: int = 0
    meta_reserved: int = 0

    # Tail
    record_nonce: int = 0
    checksum: int = 0
    registry_hint: int = 0
    tail_reserved: int = 0

    _RESERVED_FIELDS: Tuple[str, ...] = field(
        default=("header_reserved", "topic_reserved", "logic_reserved", "meta_reserved", "tail_reserved"),
        init=False,
        repr=False,
    )

    def validate(self) -> None:
        for name, _offset, width in FIELD_SPECS:
            _check_uint(name, getattr(self, name), width)
        if self.version != VERSION_V1:
            raise ValueError(f"unsupported version {self.version}, expected {VERSION_V1}")
        for reserved_name in self._RESERVED_FIELDS:
            if getattr(self, reserved_name) != 0:
                raise ValueError(f"{reserved_name} must be zero")
        expected = self.compute_checksum()
        if self.checksum != expected:
            raise ValueError(f"checksum mismatch: got {self.checksum:#x}, expected {expected:#x}")

    def compute_checksum(self) -> int:
        payload = self.to_json(include_checksum=False)
        canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode("utf-8")
        return _first64_sha256(canonical)

    def update_checksum(self) -> None:
        self.checksum = self.compute_checksum()

    def to_bits(self) -> int:
        payload = 0
        for name, offset, width in FIELD_SPECS:
            payload = set_bits(payload, offset, width, getattr(self, name))
        if payload >= (1 << TOTAL_BITS):
            raise ValueError("payload overflow")
        return payload

    def to_bytes(self) -> bytes:
        payload = self.to_bits()
        data = payload.to_bytes(STORAGE_BYTES, byteorder="big", signed=False)
        if (data[0] & 0xF0) != 0:
            raise ValueError("top padding bits are not zero")
        return data

    @classmethod
    def from_bytes(cls, b: bytes) -> "LSDL1028":
        if len(b) != STORAGE_BYTES:
            raise ValueError(f"LSDL bytes must be exactly {STORAGE_BYTES} bytes")
        if (b[0] & 0xF0) != 0:
            raise ValueError("top padding bits must be zero")
        payload = int.from_bytes(b, byteorder="big", signed=False)
        kwargs: Dict[str, int] = {}
        for name, offset, width in FIELD_SPECS:
            kwargs[name] = get_bits(payload, offset, width)
        obj = cls(**kwargs)
        obj.validate()
        return obj

    def to_json(self, include_checksum: bool = True) -> Dict[str, int]:
        d: Dict[str, int] = {
            "version": self.version,
            "flags_global": self.flags_global,
            "epoch_minutes_mod": self.epoch_minutes_mod,
            "header_reserved": self.header_reserved,
            "concept_hash0": self.concept_hash0,
            "concept_hash1": self.concept_hash1,
            "concept_hash2": self.concept_hash2,
            "domain_id": self.domain_id,
            "subdomain_id": self.subdomain_id,
            "intent_id": self.intent_id,
            "style_id": self.style_id,
            "topic_reserved": self.topic_reserved,
            "logic_family_bits": self.logic_family_bits,
            "math_family_bits": self.math_family_bits,
            "operator_bits": self.operator_bits,
            "proof_shape": self.proof_shape,
            "quantifier_depth": self.quantifier_depth,
            "nesting_depth": self.nesting_depth,
            "recursion_depth": self.recursion_depth,
            "logic_reserved": self.logic_reserved,
            "pathology_bits": self.pathology_bits,
            "self_model_bits": self.self_model_bits,
            "safety_bits": self.safety_bits,
            "meta_reserved": self.meta_reserved,
            "record_nonce": self.record_nonce,
            "checksum": self.checksum if include_checksum else 0,
            "registry_hint": self.registry_hint,
            "tail_reserved": self.tail_reserved,
        }
        return d

    @classmethod
    def from_json(cls, d: Dict[str, int]) -> "LSDL1028":
        kwargs = {}
        for name, _off, _width in FIELD_SPECS:
            if name not in d:
                raise ValueError(f"missing field in JSON: {name}")
            kwargs[name] = int(d[name])
        obj = cls(**kwargs)
        obj.validate()
        return obj

    def stable_bytes(self) -> bytes:
        """Stable serialization suitable for hashing/signatures."""
        return self.to_bytes()

    def stable_json_bytes(self, include_checksum: bool = True) -> bytes:
        return json.dumps(self.to_json(include_checksum=include_checksum), sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode("utf-8")

    def hex(self) -> str:
        return self.to_bytes().hex()

    def b64(self) -> str:
        return base64.urlsafe_b64encode(self.to_bytes()).decode("ascii").rstrip("=")

    @classmethod
    def from_concepts(
        cls,
        *,
        domain_id: int,
        subdomain_id: int,
        intent_id: int,
        style_id: int,
        concepts: Iterable[str],
        logic_family_bits: int = 0,
        math_family_bits: int = 0,
        operator_bits: int = 0,
        proof_shape: int = 0,
        quantifier_depth: int = 0,
        nesting_depth: int = 0,
        recursion_depth: int = 0,
        pathology_bits: int = 0,
        self_model_bits: int = 0,
        safety_bits: int = 0,
        record_nonce: int = 0,
        registry_hint: int = 0,
        flags_global: int = 0,
        epoch_minutes_mod: int = 0,
    ) -> "LSDL1028":
        concept_list = list(concepts)[:3]
        while len(concept_list) < 3:
            concept_list.append("")
        obj = cls(
            version=VERSION_V1,
            flags_global=flags_global,
            epoch_minutes_mod=epoch_minutes_mod,
            concept_hash0=concept_hash64(concept_list[0]),
            concept_hash1=concept_hash64(concept_list[1]),
            concept_hash2=concept_hash64(concept_list[2]),
            domain_id=domain_id,
            subdomain_id=subdomain_id,
            intent_id=intent_id,
            style_id=style_id,
            logic_family_bits=logic_family_bits,
            math_family_bits=math_family_bits,
            operator_bits=operator_bits,
            proof_shape=proof_shape,
            quantifier_depth=quantifier_depth,
            nesting_depth=nesting_depth,
            recursion_depth=recursion_depth,
            pathology_bits=pathology_bits,
            self_model_bits=self_model_bits,
            safety_bits=safety_bits,
            record_nonce=record_nonce,
            registry_hint=registry_hint,
        )
        obj.update_checksum()
        obj.validate()
        return obj

    def glyph_string(self) -> str:
        topical = [
            chr(TOPICAL_GLYPH_BASE + (self.domain_id % TOPICAL_GLYPH_SLOTS)),
            chr(TOPICAL_GLYPH_BASE + (self.subdomain_id % TOPICAL_GLYPH_SLOTS)),
            chr(TOPICAL_GLYPH_BASE + (self.intent_id % TOPICAL_GLYPH_SLOTS)),
            chr(TOPICAL_GLYPH_BASE + (self.style_id % TOPICAL_GLYPH_SLOTS)),
        ]

        logic_idxs = _top_set_bit_indices(self.logic_family_bits, limit=4)
        op_idxs = _top_set_bit_indices(self.operator_bits, limit=8)

        modal: List[str] = []
        for idx in logic_idxs + op_idxs:
            modal.append(chr(LOGIC_GLYPH_BASE + (idx % LOGIC_GLYPH_SLOTS)))

        return "".join(topical + modal)


def _top_set_bit_indices(mask: int, limit: int) -> List[int]:
    out: List[int] = []
    for i in range(64):
        if (mask >> i) & 1:
            out.append(i)
            if len(out) >= limit:
                break
    return out


def pack_lsdl(record: LSDL1028) -> bytes:
    return record.to_bytes()


def unpack_lsdl(data: bytes) -> LSDL1028:
    return LSDL1028.from_bytes(data)


def set_named_bits(names: Iterable[str], mapping: Dict[str, int]) -> int:
    value = 0
    for name in names:
        idx = mapping.get(name)
        if idx is None:
            raise KeyError(f"unknown bit name: {name}")
        value |= (1 << idx)
    return value


__all__ = [
    "LSDL1028",
    "TOTAL_BITS",
    "STORAGE_BITS",
    "STORAGE_BYTES",
    "VERSION_V1",
    "LOGIC_FAMILY_INDEX",
    "OPERATOR_INDEX",
    "PATHOLOGY_INDEX",
    "concept_hash64",
    "normalize_concept",
    "set_bits",
    "get_bits",
    "pack_lsdl",
    "unpack_lsdl",
    "set_named_bits",
]

