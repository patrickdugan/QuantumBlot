import unittest

from lsdl_1028 import (
    LSDL1028,
    LOGIC_FAMILY_INDEX,
    OPERATOR_INDEX,
    PATHOLOGY_INDEX,
    STORAGE_BYTES,
    concept_hash64,
    set_named_bits,
)


class TestLSDL1028(unittest.TestCase):
    def _sample(self) -> LSDL1028:
        return LSDL1028.from_concepts(
            domain_id=33,
            subdomain_id=12,
            intent_id=5,
            style_id=2,
            concepts=["proof sketch", "modal fol", "safety audit"],
            logic_family_bits=set_named_bits(["modal_logic", "first_order_logic"], LOGIC_FAMILY_INDEX),
            operator_bits=set_named_bits(["forall", "exists", "nec", "fix"], OPERATOR_INDEX),
            pathology_bits=set_named_bits(["speculation"], PATHOLOGY_INDEX),
            self_model_bits=0b1010,
            safety_bits=0b100,
            record_nonce=0xABCDEF,
            registry_hint=9,
            flags_global=1,
            epoch_minutes_mod=17,
        )

    def test_round_trip_bytes(self) -> None:
        record = self._sample()
        data = record.to_bytes()
        back = LSDL1028.from_bytes(data)
        self.assertEqual(record.to_json(), back.to_json())

    def test_padding_bits_zero(self) -> None:
        record = self._sample()
        data = record.to_bytes()
        self.assertEqual(len(data), STORAGE_BYTES)
        self.assertEqual(data[0] & 0xF0, 0)

    def test_checksum_matches(self) -> None:
        record = self._sample()
        expected = record.compute_checksum()
        self.assertEqual(record.checksum, expected)
        # Tamper and ensure checksum detection fails.
        tampered = record.to_json()
        tampered["domain_id"] += 1
        tampered_record = LSDL1028(**tampered)
        with self.assertRaises(ValueError):
            tampered_record.validate()

    def test_deterministic_glyph_string(self) -> None:
        a = self._sample()
        b = self._sample()
        self.assertEqual(a.glyph_string(), b.glyph_string())

    def test_concept_hash_stable(self) -> None:
        self.assertEqual(concept_hash64("  Hello   World  "), concept_hash64("hello world"))


if __name__ == "__main__":
    unittest.main()

