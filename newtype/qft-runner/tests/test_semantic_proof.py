import json
import os
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(THIS_DIR)
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from semantic_proof import (  # noqa: E402
    BIT_WIDTH,
    ReasoningPrimitive,
    SpectralWatermarker,
    attest,
    run_crypto_attestation,
)
from sc1028 import encode_to_base64url  # noqa: E402


class TestSemanticProof(unittest.TestCase):
    def test_reasoning_primitive_shape(self) -> None:
        row = np.zeros((BIT_WIDTH,), dtype=np.float64)
        prim = ReasoningPrimitive(bits=row)
        self.assertEqual(prim.bits.shape, (BIT_WIDTH,))

    def test_spectral_embed_and_verify(self) -> None:
        rng = np.random.default_rng(7)
        primitives = [
            ReasoningPrimitive(bits=rng.integers(0, 2, size=(BIT_WIDTH,)).astype(np.float64))
            for _ in range(40)
        ]
        wm = SpectralWatermarker(n_components=12, watermark_frequency=4, watermark_strength=0.12, seed=9)
        latent, basis, mean = wm.compress_trace(primitives)
        marked_latent = wm.embed_spectral_signature(latent)
        reconstructed = wm.reconstruct_trace(marked_latent, basis, mean)
        marked_primitives = [ReasoningPrimitive(bits=row.astype(np.float64)) for row in reconstructed]
        self.assertTrue(wm.verify_signature(marked_primitives, correlation_threshold=0.1, amplitude_threshold=0.001))

    def test_crypto_attestation_signing(self) -> None:
        records = [
            {"run_id": "r1", "chunk_id": 1, "value": 10},
            {"run_id": "r1", "chunk_id": 2, "value": 11},
        ]
        out = run_crypto_attestation(records, key="k-test")
        self.assertEqual(out["mode"], "crypto")
        self.assertTrue(out["signed"])
        self.assertIsInstance(out["signature"], str)
        self.assertGreater(len(out["signature"]), 10)

    def test_attest_both_modes_from_jsonl(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            telemetry_path = Path(td) / "telemetry.jsonl"
            attestation_path = Path(td) / "attestation.json"
            spectral_out = Path(td) / "spectral.npy"

            token = encode_to_base64url(["meta.schema_v1"])
            rows = [
                {
                    "run_id": "r1",
                    "episode_id": 0,
                    "chunk_id": i + 1,
                    "sc1028_b64": token,
                    "sc1028_symbols": ["meta.schema_v1"],
                }
                for i in range(8)
            ]
            with telemetry_path.open("w", encoding="utf-8") as f:
                for row in rows:
                    f.write(json.dumps(row) + "\n")

            os.environ["SEMANTIC_ATTESTATION_KEY"] = "secret-test"
            payload = attest(
                telemetry_path=str(telemetry_path),
                mode="both",
                out_path=str(attestation_path),
                spectral_trace_path=str(spectral_out),
            )
            self.assertEqual(payload["mode"], "both")
            self.assertTrue(attestation_path.exists())
            self.assertTrue(spectral_out.exists())


if __name__ == "__main__":
    unittest.main()
