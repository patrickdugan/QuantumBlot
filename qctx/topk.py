#!/usr/bin/env python3
"""
topk.py — PCA reduction + sparsify for QFT prep.
- Runs PCA on vectors.jsonl
- Pads to nearest power-of-two (for qubits)
- Sparsifies top-k (absolute values)
- Saves .npy with normalized vectors
"""

import json
import numpy as np
from sklearn.decomposition import PCA
from tqdm import tqdm

# ---- config ----
VECTORS_PATH = "./vectors.jsonl"
OUT_PATH     = "./vectors_pca_topk.npy"

N_COMPONENTS = 384  # ≤ 384 for MiniLM (try 128, 256, 384)
SPARSE_FRAC  = 0.5   # fraction of dims to keep (e.g. 0.2 → 20%)
# ----------------


def load_vectors(jsonl_path):
    texts, vecs = [], []
    with open(jsonl_path, "r", encoding="utf-8") as f:
        for line in tqdm(f, desc="Loading vectors"):
            obj = json.loads(line)
            texts.append(obj.get("text", ""))
            vecs.append(obj["vector"])
    return texts, np.array(vecs, dtype=np.float32)


def reduce_pca(vectors, n_components=128):
    print(f"[pca] running PCA → {n_components} comps")
    pca = PCA(n_components=n_components, svd_solver="randomized")
    reduced = pca.fit_transform(vectors)
    print(f"[pca] explained variance sum = {pca.explained_variance_ratio_.sum():.4f}")
    return reduced, pca


def pad_to_pow2(vectors: np.ndarray) -> np.ndarray:
    """Pad each row with zeros up to the next power-of-two length."""
    n, d = vectors.shape
    target = 1 << (d - 1).bit_length()
    if d == target:
        return vectors
    pad = target - d
    print(f"[pad] {d} → {target} (adding {pad} zeros)")
    return np.hstack([vectors, np.zeros((n, pad), dtype=vectors.dtype)])


def sparsify_topk(vectors, frac=0.2):
    """Keep top-|frac| dims by abs value per row."""
    n, d = vectors.shape
    k = max(1, int(d * frac))
    print(f"[sparse] keeping top-{k} of {d} dims per vector (~{frac*100:.1f}%)")
    out = np.zeros_like(vectors)
    for i in tqdm(range(n), desc="Sparsifying"):
        row = vectors[i]
        idx = np.argpartition(np.abs(row), -k)[-k:]
        out[i, idx] = row[idx]
    return out


def save_sparse(path, arr, texts=None):
    np.save(path, arr)
    print(f"[ok] saved {arr.shape} to {path}")
    if texts is not None:
        with open(path + ".txt", "w", encoding="utf-8") as f:
            for t in texts:
                f.write(t.replace("\n", " ") + "\n")


if __name__ == "__main__":
    texts, vecs = load_vectors(VECTORS_PATH)

    # Step 1: PCA
    reduced, pca = reduce_pca(vecs, n_components=N_COMPONENTS)

    # Step 2: pad to power-of-two for QFT
    padded = pad_to_pow2(reduced)
    n_qubits = int(np.log2(padded.shape[1]))
    print(f"[info] PCA {reduced.shape} → padded {padded.shape} (for {n_qubits} qubits)")

    # Step 3: sparsify + normalize
    sparse = sparsify_topk(padded, frac=SPARSE_FRAC)
    norms = np.linalg.norm(sparse, axis=1, keepdims=True) + 1e-9
    sparse = sparse / norms

    # Step 4: save
    save_sparse(OUT_PATH, sparse, texts=texts)
