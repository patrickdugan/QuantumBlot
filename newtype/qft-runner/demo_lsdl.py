#!/usr/bin/env python3

import json

from lsdl_1028 import (
    LSDL1028,
    LOGIC_FAMILY_INDEX,
    OPERATOR_INDEX,
    PATHOLOGY_INDEX,
    pack_lsdl,
    set_named_bits,
    unpack_lsdl,
)


def main() -> None:
    sample_prompt = (
        "Design a compact symbolic DSL for explicit semantic indexing with "
        "logical structure and pathology audit flags."
    )

    sample_trace_summary = [
        "specification planning",
        "modal first-order constraints",
        "safety and pathology audit",
    ]

    logic_bits = set_named_bits(["modal_logic", "first_order_logic"], LOGIC_FAMILY_INDEX)
    operator_bits = set_named_bits(["forall", "exists", "nec", "fix"], OPERATOR_INDEX)
    pathology_bits = set_named_bits(["speculation"], PATHOLOGY_INDEX)

    lsdl = LSDL1028.from_concepts(
        domain_id=42,
        subdomain_id=7,
        intent_id=3,  # e.g., spec/plan
        style_id=1,
        concepts=[sample_prompt, *sample_trace_summary],
        logic_family_bits=logic_bits,
        operator_bits=operator_bits,
        proof_shape=2,
        quantifier_depth=3,
        nesting_depth=4,
        recursion_depth=2,
        pathology_bits=pathology_bits,
        self_model_bits=0b101,
        safety_bits=0b10010,
        record_nonce=0x1122334455667788,
        registry_hint=1,
        flags_global=0b11,
        epoch_minutes_mod=0,
    )

    data = pack_lsdl(lsdl)
    roundtrip = unpack_lsdl(data)

    print("=== LSDL-1028 Demo ===")
    print(f"Round-trip OK: {lsdl.to_json() == roundtrip.to_json()}")
    print(f"Bytes: {len(data)}")
    print(f"Hex: {lsdl.hex()}")
    print(f"Base64url: {lsdl.b64()}")
    glyphs = lsdl.glyph_string()
    try:
        print(f"Glyphs: {glyphs}")
    except UnicodeEncodeError:
        print(f"Glyphs (escaped): {glyphs.encode('unicode_escape').decode('ascii')}")
    print("JSON:")
    print(json.dumps(lsdl.to_json(), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
