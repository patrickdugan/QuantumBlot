#!/usr/bin/env python3
"""
build_rope.py — PCA basis builder for QFT decoder
Loads vectors.jsonl (~1.2 GB), runs PCA, saves rope_hint.json.
"""

import json, numpy as np
from sklearn.decomposition import PCA

# ---- config ----
VECTORS_PATH = r"C:\projects\QuantumBlot\qctx\vectors.jsonl"
OUT_PATH     = r"C:\projects\QuantumBlot\qctx\rope_hint.json"
N_COMPONENTS = 256   # match qubits: 8 qubits → 256 components
# ----------------

def main():
    print(f"[info] loading {VECTORS_PATH} …")
    vecs = []
    with open(VECTORS_PATH, "r", encoding="utf-8") as f:
        for i, line in enumerate(f, 1):
            obj = json.loads(line)
            vecs.append(obj["vector"])
            if i % 10000 == 0:
                print(f"  loaded {i:,} vectors …")
    X = np.array(vecs, dtype=np.float32)   # saves RAM
    print(f"[info] finished load: {X.shape[0]} vectors × {X.shape[1]} dims")
    print(f"[mem] array size ~{X.nbytes/1e9:.2f} GB in RAM")

    print(f"[info] running PCA with {N_COMPONENTS} components …")
    pca = PCA(n_components=N_COMPONENTS, svd_solver="randomized")
    pca.fit(X)

    rope = {
        "raw": {
            "pca": {
                "components": pca.components_.tolist(),
                "mean": pca.mean_.tolist(),
            }
        }
    }
    with open(OUT_PATH, "w", encoding="utf-8") as f:
        json.dump(rope, f, ensure_ascii=False, indent=2)
    print(f"[ok] wrote PCA basis to {OUT_PATH}")

if __name__ == "__main__":
    main()
