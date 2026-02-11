"""SC1028 encoder/decoder.

Binary layout:
- Semantic bits: 1028
- Storage bits: 1032 (129 bytes)
- High nibble in final byte is reserved padding and must be zero.
"""

from __future__ import annotations

import base64
from typing import Iterable, List, Sequence

from sc1028_symbols import (
    BIT_TO_SYMBOL,
    SC1028_BITS,
    SC1028_STORAGE_BITS,
    SC1028_STORAGE_BYTES,
    resolve_bits,
)


def _assert_valid_bit_index(bit_index: int) -> None:
    if bit_index < 0 or bit_index >= SC1028_BITS:
        raise ValueError(f"Bit index out of range: {bit_index}")


def _assert_padding_zero(bitset: bytes) -> None:
    if len(bitset) != SC1028_STORAGE_BYTES:
        raise ValueError(
            f"SC1028 bitset must be exactly {SC1028_STORAGE_BYTES} bytes, got {len(bitset)}"
        )
    if bitset[-1] & 0b11110000:
        raise ValueError("SC1028 reserved pad bits must be zero.")


def normalize_bitset(bitset: bytes | bytearray | memoryview | Sequence[int]) -> bytes:
    if isinstance(bitset, memoryview):
        data = bitset.tobytes()
    elif isinstance(bitset, (bytes, bytearray)):
        data = bytes(bitset)
    else:
        data = bytes(bitset)
    _assert_padding_zero(data)
    return data


def encode(symbols: Iterable[str], strict: bool = True) -> bytes:
    payload = bytearray(SC1028_STORAGE_BYTES)
    for bit_index in resolve_bits(symbols, strict=strict):
        _assert_valid_bit_index(bit_index)
        byte_index = bit_index // 8
        bit_offset = bit_index % 8
        payload[byte_index] |= 1 << bit_offset
    data = bytes(payload)
    _assert_padding_zero(data)
    return data


def decode(bitset: bytes) -> List[str]:
    data = normalize_bitset(bitset)
    symbols: List[str] = []
    for bit_index in range(SC1028_BITS):
        byte_index = bit_index // 8
        bit_offset = bit_index % 8
        if data[byte_index] & (1 << bit_offset):
            symbol = BIT_TO_SYMBOL.get(bit_index)
            if symbol:
                symbols.append(symbol)
    return symbols


def bit_count(bitset: bytes) -> int:
    data = normalize_bitset(bitset)
    return sum(b.bit_count() for b in data)


def to_base64url(bitset: bytes) -> str:
    data = normalize_bitset(bitset)
    return base64.urlsafe_b64encode(data).decode("ascii").rstrip("=")


def from_base64url(token: str) -> bytes:
    if not token:
        raise ValueError("SC1028 base64url token cannot be empty.")
    padded = token + ("=" * ((-len(token)) % 4))
    data = base64.urlsafe_b64decode(padded.encode("ascii"))
    return normalize_bitset(data)


def encode_to_base64url(symbols: Iterable[str], strict: bool = True) -> str:
    return to_base64url(encode(symbols, strict=strict))


def decode_from_base64url(token: str) -> List[str]:
    return decode(from_base64url(token))


def storage_layout() -> dict:
    return {
        "semantic_bits": SC1028_BITS,
        "storage_bits": SC1028_STORAGE_BITS,
        "storage_bytes": SC1028_STORAGE_BYTES,
        "pad_bits": SC1028_STORAGE_BITS - SC1028_BITS,
        "pad_location": "high nibble of final byte",
        "bit_order": "little-endian within byte",
    }


__all__ = [
    "encode",
    "decode",
    "bit_count",
    "normalize_bitset",
    "to_base64url",
    "from_base64url",
    "encode_to_base64url",
    "decode_from_base64url",
    "storage_layout",
]

