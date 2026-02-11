import os
import sys
import unittest

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(THIS_DIR)
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from sc1028 import decode, encode, from_base64url, to_base64url  # noqa: E402
from sc1028_symbols import SC1028_STORAGE_BYTES  # noqa: E402


class TestSC1028(unittest.TestCase):
    def test_roundtrip_encode_decode(self) -> None:
        symbols = [
            "meta.schema_v1",
            "action.qft_run",
            "tool.qft_one",
            "term.chunk_ok",
            "resource.flush_per_chunk",
        ]
        bitset = encode(symbols)
        decoded = decode(bitset)
        for symbol in symbols:
            self.assertIn(symbol, decoded)

    def test_fixed_length_storage(self) -> None:
        bitset = encode(["meta.schema_v1"])
        self.assertEqual(len(bitset), SC1028_STORAGE_BYTES)

    def test_pad_bits_must_be_zero(self) -> None:
        bitset = bytearray(encode(["meta.schema_v1"]))
        bitset[-1] |= 0b11110000
        with self.assertRaises(ValueError):
            decode(bytes(bitset))

    def test_base64url_roundtrip(self) -> None:
        bitset = encode(["meta.schema_v1", "action.embed"])
        token = to_base64url(bitset)
        back = from_base64url(token)
        self.assertEqual(bitset, back)


if __name__ == "__main__":
    unittest.main()

