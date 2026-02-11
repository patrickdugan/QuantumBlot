#!/usr/bin/env python3
"""Semantic proof infrastructure with explicit attestation modes.

Modes:
- crypto: Merkle-root + optional HMAC signature over chunk telemetry records.
- spectral: PCA -> FFT watermark injection over explicit sidecar trace artifacts.
"""

from __future__ import annotations

import argparse
import hashlib
import hmac
import json
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Literal, Optional, Sequence, Tuple

import numpy as np

try:
    from sc1028 import decode_from_base64url
except Exception:
    # Allow standalone module execution from other cwd.
    _THIS_DIR = Path(__file__).resolve().parent
    if str(_THIS_DIR) not in sys.path:
        sys.path.insert(0, str(_THIS_DIR))
    from sc1028 import decode_from_base64url  # type: ignore


BIT_WIDTH: int = 1028
AttestationMode = Literal["crypto", "spectral", "both"]


def _canonical_json(obj: dict) -> str:
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=True)


def _hash_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _hash_bytes(blob: bytes) -> str:
    return hashlib.sha256(blob).hexdigest()


@dataclass(frozen=True)
class ReasoningPrimitive:
    """Atomic reasoning unit represented as a fixed-width 1028-dimensional vector."""

    bits: np.ndarray

    def __post_init__(self) -> None:
        if self.bits.shape != (BIT_WIDTH,):
            raise ValueError(f"ReasoningPrimitive must have shape ({BIT_WIDTH},), got {self.bits.shape}")
        if self.bits.dtype != np.float64:
            object.__setattr__(self, "bits", self.bits.astype(np.float64))


class SpectralWatermarker:
    """PCA -> FFT watermarking/verification pipeline for explicit attestation artifacts."""

    def __init__(
        self,
        n_components: int = 16,
        watermark_frequency: int = 3,
        watermark_strength: float = 0.08,
        seed: int = 7,
    ) -> None:
        if n_components <= 0:
            raise ValueError("n_components must be > 0")
        if watermark_frequency <= 0:
            raise ValueError("watermark_frequency must be > 0")
        if watermark_strength <= 0:
            raise ValueError("watermark_strength must be > 0")
        self.n_components = n_components
        self.watermark_frequency = watermark_frequency
        self.watermark_strength = watermark_strength
        self.seed = seed

    def compress_trace(
        self,
        primitives: Iterable[ReasoningPrimitive],
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Reduce sequence of primitives to principal latent components.

        Returns:
        - latent_components: shape (T, k)
        - basis: shape (1028, k)
        - mean: shape (1028,)
        """
        rows = [p.bits for p in primitives]
        if not rows:
            raise ValueError("compress_trace requires at least one primitive")
        X = np.stack(rows, axis=0)  # (T, 1028)
        mean = X.mean(axis=0)
        Xc = X - mean

        u, s, vt = np.linalg.svd(Xc, full_matrices=False)
        k = min(self.n_components, vt.shape[0])
        basis = vt[:k].T
        latent = Xc @ basis
        _ = u, s
        return latent, basis, mean

    def _signature_vector(self, width: int) -> np.ndarray:
        rng = np.random.default_rng(self.seed)
        phases = rng.uniform(0.0, 2.0 * np.pi, size=(width,))
        signature = np.exp(1j * phases) * self.watermark_strength
        return signature

    def embed_spectral_signature(self, latent_components: np.ndarray) -> np.ndarray:
        """Apply FFT, inject deterministic signature, apply inverse FFT."""
        if latent_components.ndim != 2:
            raise ValueError("latent_components must have shape (T, k)")
        T, K = latent_components.shape
        if T <= 2:
            raise ValueError("latent trace too short for spectral injection")

        freq = self.watermark_frequency
        if freq >= T:
            freq = max(1, T // 2 - 1)

        spectrum = np.fft.fft(latent_components, axis=0)
        signature = self._signature_vector(K)
        spectrum[freq, :] = spectrum[freq, :] + signature
        mirror = (-freq) % T
        spectrum[mirror, :] = spectrum[mirror, :] + np.conjugate(signature)

        reconstructed = np.fft.ifft(spectrum, axis=0).real
        return reconstructed

    def reconstruct_trace(
        self,
        latent_components: np.ndarray,
        basis: np.ndarray,
        mean: np.ndarray,
    ) -> np.ndarray:
        if latent_components.ndim != 2:
            raise ValueError("latent_components must have shape (T, k)")
        if basis.ndim != 2:
            raise ValueError("basis must have shape (1028, k)")
        if mean.shape != (BIT_WIDTH,):
            raise ValueError(f"mean must have shape ({BIT_WIDTH},)")
        return latent_components @ basis.T + mean

    def verify_signature(
        self,
        trace: Iterable[ReasoningPrimitive],
        correlation_threshold: float = 0.12,
        amplitude_threshold: float = 0.001,
    ) -> bool:
        """Detect spectral signature presence in a trace by PCA + FFT."""
        latent, _, _ = self.compress_trace(trace)
        return self.verify_latent_signature(
            latent,
            correlation_threshold=correlation_threshold,
            amplitude_threshold=amplitude_threshold,
        )

    def _signature_stats_from_latent(self, latent: np.ndarray) -> Tuple[float, float]:
        if latent.ndim != 2:
            return 0.0, 0.0

        T, K = latent.shape
        if T <= 2:
            return 0.0, 0.0
        freq = self.watermark_frequency
        if freq >= T:
            freq = max(1, T // 2 - 1)

        spectrum = np.fft.fft(latent, axis=0)
        coeff = spectrum[freq, :]
        target = self._signature_vector(K)

        coeff_norm = np.linalg.norm(coeff) + 1e-12
        target_norm = np.linalg.norm(target) + 1e-12
        corr = float(np.abs(np.vdot(coeff, target)) / (coeff_norm * target_norm))
        amp = float(np.mean(np.abs(coeff)))
        return corr, amp

    def verify_latent_signature(
        self,
        latent_components: np.ndarray,
        correlation_threshold: float = 0.12,
        amplitude_threshold: float = 0.001,
    ) -> bool:
        corr, amp = self._signature_stats_from_latent(latent_components)
        return bool(corr >= correlation_threshold and amp >= amplitude_threshold)


class CryptoReasoningAttestor:
    """Explicit cryptographic attestation over chunk telemetry records."""

    def __init__(self, key: Optional[str] = None) -> None:
        self.key = key.encode("utf-8") if key else None

    def record_hash(self, record: dict) -> str:
        return _hash_text(_canonical_json(record))

    def merkle_root(self, hashes: Sequence[str]) -> str:
        if not hashes:
            return _hash_text("")
        level = [bytes.fromhex(h) for h in hashes]
        while len(level) > 1:
            nxt: List[bytes] = []
            for i in range(0, len(level), 2):
                left = level[i]
                right = level[i + 1] if (i + 1) < len(level) else left
                nxt.append(hashlib.sha256(left + right).digest())
            level = nxt
        return level[0].hex()

    def sign_root(self, root_hex: str) -> Optional[str]:
        if self.key is None:
            return None
        return hmac.new(self.key, root_hex.encode("utf-8"), hashlib.sha256).hexdigest()

    def verify_signature(self, root_hex: str, signature_hex: str) -> bool:
        if self.key is None:
            return False
        expected = self.sign_root(root_hex)
        if expected is None:
            return False
        return hmac.compare_digest(expected, signature_hex)


def _decode_sc1028_primitive(sc1028_b64: str) -> ReasoningPrimitive:
    symbols = decode_from_base64url(sc1028_b64)
    # Build vector via symbol hashes for deterministic dense representation.
    bits = np.zeros((BIT_WIDTH,), dtype=np.float64)
    for sym in symbols:
        idx = int(hashlib.sha256(sym.encode("utf-8")).hexdigest(), 16) % BIT_WIDTH
        bits[idx] = 1.0
    return ReasoningPrimitive(bits=bits)


def load_telemetry_records(path: str) -> List[dict]:
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Telemetry file not found: {path}")
    records: List[dict] = []
    with p.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(obj, dict):
                records.append(obj)
    return records


def primitives_from_records(records: Sequence[dict]) -> List[ReasoningPrimitive]:
    out: List[ReasoningPrimitive] = []
    for record in records:
        token = record.get("sc1028_b64")
        if isinstance(token, str) and token:
            try:
                out.append(_decode_sc1028_primitive(token))
            except Exception:
                continue
    return out


def run_crypto_attestation(records: Sequence[dict], key: Optional[str]) -> dict:
    attestor = CryptoReasoningAttestor(key=key)
    record_hashes = [attestor.record_hash(r) for r in records]
    root = attestor.merkle_root(record_hashes)
    signature = attestor.sign_root(root)
    return {
        "mode": "crypto",
        "record_count": len(records),
        "record_hashes_sha256": record_hashes,
        "merkle_root_sha256": root,
        "signature_algorithm": "hmac-sha256" if signature is not None else "none",
        "signature": signature,
        "signed": signature is not None,
    }


def run_spectral_attestation(
    records: Sequence[dict],
    n_components: int,
    watermark_frequency: int,
    watermark_strength: float,
    seed: int,
    spectral_trace_path: Optional[str],
) -> dict:
    primitives = primitives_from_records(records)
    if len(primitives) < 3:
        return {
            "mode": "spectral",
            "record_count": len(records),
            "primitive_count": len(primitives),
            "status": "insufficient_trace",
            "verified": False,
        }

    watermarker = SpectralWatermarker(
        n_components=n_components,
        watermark_frequency=watermark_frequency,
        watermark_strength=watermark_strength,
        seed=seed,
    )
    latent, basis, mean = watermarker.compress_trace(primitives)
    marked_latent = watermarker.embed_spectral_signature(latent)
    reconstructed_trace = watermarker.reconstruct_trace(marked_latent, basis, mean)

    marked_primitives = [ReasoningPrimitive(bits=row.astype(np.float64)) for row in reconstructed_trace]
    verified_latent = watermarker.verify_latent_signature(
        marked_latent,
        correlation_threshold=0.1,
        amplitude_threshold=0.001,
    )
    verified_trace = watermarker.verify_signature(
        marked_primitives,
        correlation_threshold=0.1,
        amplitude_threshold=0.001,
    )
    verified = bool(verified_latent or verified_trace)

    trace_hash = _hash_bytes(reconstructed_trace.astype(np.float64).tobytes())
    trace_out = None
    if spectral_trace_path:
        trace_file = Path(spectral_trace_path)
        trace_file.parent.mkdir(parents=True, exist_ok=True)
        np.save(trace_file, reconstructed_trace.astype(np.float32))
        trace_out = str(trace_file)

    return {
        "mode": "spectral",
        "record_count": len(records),
        "primitive_count": len(primitives),
        "status": "ok",
        "verified": bool(verified),
        "verified_latent": bool(verified_latent),
        "verified_recompressed_trace": bool(verified_trace),
        "n_components": int(min(n_components, latent.shape[1])),
        "watermark_frequency": int(min(watermark_frequency, max(1, latent.shape[0] // 2 - 1))),
        "watermark_strength": float(watermark_strength),
        "seed": int(seed),
        "latent_shape": [int(latent.shape[0]), int(latent.shape[1])],
        "spectral_trace_sha256": trace_hash,
        "spectral_trace_npy": trace_out,
    }


def attest(
    telemetry_path: str,
    mode: AttestationMode,
    out_path: str,
    key_env: str = "SEMANTIC_ATTESTATION_KEY",
    n_components: int = 16,
    watermark_frequency: int = 3,
    watermark_strength: float = 0.08,
    seed: int = 7,
    spectral_trace_path: Optional[str] = None,
) -> dict:
    records = load_telemetry_records(telemetry_path)
    key = os.environ.get(key_env)

    payload: Dict[str, object] = {
        "telemetry_path": telemetry_path,
        "mode": mode,
        "record_count": len(records),
        "key_env": key_env,
        "key_present": bool(key),
        "results": {},
    }

    if mode in ("crypto", "both"):
        payload["results"]["crypto"] = run_crypto_attestation(records, key=key)  # type: ignore[index]
    if mode in ("spectral", "both"):
        payload["results"]["spectral"] = run_spectral_attestation(
            records,
            n_components=n_components,
            watermark_frequency=watermark_frequency,
            watermark_strength=watermark_strength,
            seed=seed,
            spectral_trace_path=spectral_trace_path,
        )  # type: ignore[index]

    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return payload


def verify(attestation_path: str, key_env: str = "SEMANTIC_ATTESTATION_KEY") -> dict:
    p = Path(attestation_path)
    if not p.exists():
        raise FileNotFoundError(f"Attestation file not found: {attestation_path}")
    payload = json.loads(p.read_text(encoding="utf-8"))
    results = payload.get("results", {})

    out = {"attestation_path": attestation_path, "verified": True, "details": {}}

    crypto = results.get("crypto")
    if isinstance(crypto, dict):
        key = os.environ.get(key_env)
        attestor = CryptoReasoningAttestor(key=key)
        root = crypto.get("merkle_root_sha256")
        signature = crypto.get("signature")
        if isinstance(root, str) and isinstance(signature, str):
            ok = attestor.verify_signature(root, signature)
            out["details"]["crypto"] = {"verified": ok}
            out["verified"] = bool(out["verified"] and ok)
        else:
            out["details"]["crypto"] = {"verified": False, "reason": "unsigned_or_missing"}
            out["verified"] = False

    spectral = results.get("spectral")
    if isinstance(spectral, dict):
        # Spectral mode is verified at attestation time and recorded explicitly.
        ok = bool(spectral.get("verified", False))
        out["details"]["spectral"] = {"verified": ok}
        out["verified"] = bool(out["verified"] and ok)

    return out


def _parse_mode(value: str) -> AttestationMode:
    allowed = {"crypto", "spectral", "both"}
    if value not in allowed:
        raise argparse.ArgumentTypeError(f"--mode must be one of {sorted(allowed)}")
    return value  # type: ignore[return-value]


def main() -> None:
    parser = argparse.ArgumentParser(description="Semantic proof attestation tooling.")
    sub = parser.add_subparsers(dest="cmd", required=True)

    p_att = sub.add_parser("attest", help="Generate attestation from telemetry JSONL.")
    p_att.add_argument("--telemetry", required=True, type=str)
    p_att.add_argument("--mode", type=_parse_mode, default="crypto")
    p_att.add_argument("--out", required=True, type=str)
    p_att.add_argument("--key-env", default="SEMANTIC_ATTESTATION_KEY", type=str)
    p_att.add_argument("--n-components", default=16, type=int)
    p_att.add_argument("--watermark-frequency", default=3, type=int)
    p_att.add_argument("--watermark-strength", default=0.08, type=float)
    p_att.add_argument("--seed", default=7, type=int)
    p_att.add_argument("--spectral-trace-out", default=None, type=str)

    p_ver = sub.add_parser("verify", help="Verify an attestation file.")
    p_ver.add_argument("--attestation", required=True, type=str)
    p_ver.add_argument("--key-env", default="SEMANTIC_ATTESTATION_KEY", type=str)

    args = parser.parse_args()

    if args.cmd == "attest":
        payload = attest(
            telemetry_path=args.telemetry,
            mode=args.mode,
            out_path=args.out,
            key_env=args.key_env,
            n_components=args.n_components,
            watermark_frequency=args.watermark_frequency,
            watermark_strength=args.watermark_strength,
            seed=args.seed,
            spectral_trace_path=args.spectral_trace_out,
        )
        print(json.dumps({"ok": True, "out": args.out, "mode": args.mode, "record_count": payload["record_count"]}))
        return

    if args.cmd == "verify":
        result = verify(attestation_path=args.attestation, key_env=args.key_env)
        print(json.dumps(result))
        if not result["verified"]:
            raise SystemExit(2)


if __name__ == "__main__":
    main()
