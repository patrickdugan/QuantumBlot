#!/usr/bin/env python3
import json, numpy as np
from sklearn.decomposition import PCA

# --- paths ---
VECTORS_PATH = r"C:\projects\QuantumBlot\qctx\vectors.jsonl"
OUT_PATH     = r"C:\projects\QuantumBlot\qctx\rope_hint.json"
# Fit once at max C, then slice to smaller views
C_MAX = 384
VIEWS = [64, 256, 384]  # add 32/384 if you want

def main():
    # load vectors.jsonl -> X
    vecs = []
    with open(VECTORS_PATH, "r", encoding="utf-8") as f:
        for line in f:
            obj = json.loads(line)
            vecs.append(obj["vector"])
    X = np.array(vecs, dtype=np.float32)
    print(f"[load] {X.shape[0]} vectors × {X.shape[1]} dims")

    # PCA once at C_MAX
    print(f"[pca] fitting randomized PCA with {C_MAX} comps…")
    pca = PCA(n_components=C_MAX, svd_solver="randomized")
    pca.fit(X)

    views = {}
    for C in VIEWS:
        views[f"C{C}"] = {
            "components": pca.components_[:C].tolist(),
            "mean": pca.mean_.tolist(),
            "explained_variance_ratio_sum": float(pca.explained_variance_ratio_[:C].sum()),
            "C": C
        }

    rope = {
        "schema": 1,
        "raw": {
            "pca": {
                "views": views,
                "fitted_C": C_MAX,
                "input_dim": int(X.shape[1]),
            }
        }
    }
    with open(OUT_PATH, "w", encoding="utf-8") as f:
        json.dump(rope, f, ensure_ascii=False, indent=2)
    print(f"[ok] wrote {OUT_PATH}")

if __name__ == "__main__":
    main()
