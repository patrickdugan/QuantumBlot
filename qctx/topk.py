import json
import numpy as np
from sklearn.decomposition import PCA
from tqdm import tqdm

def load_vectors(jsonl_path):
    texts, vecs = [], []
    with open(jsonl_path, "r", encoding="utf-8") as f:
        for line in tqdm(f, desc="Loading vectors"):
            obj = json.loads(line)
            texts.append(obj["text"])
            vecs.append(np.array(obj["vector"], dtype=np.float32))
    return texts, np.vstack(vecs)

def reduce_pca(vectors, n_components=64):
    print(f"Running PCA → {n_components} dims...")
    pca = PCA(n_components=n_components, random_state=0)
    reduced = pca.fit_transform(vectors)
    return reduced, pca

def sparsify_topk(vectors, k=16):
    print(f"Applying top-{k} sparsification...")
    sparse = np.zeros_like(vectors)
    for i, row in enumerate(vectors):
        # pick top-k absolute values
        idx = np.argpartition(np.abs(row), -k)[-k:]
        sparse[i, idx] = row[idx]
    return sparse

def save_sparse(out_path, sparse_vecs, texts=None):
    np.save(out_path, sparse_vecs)
    print(f"Wrote sparse array: {out_path} ({sparse_vecs.shape})")
    if texts is not None:
        with open(out_path + ".txt", "w", encoding="utf-8") as f:
            for t in texts:
                f.write(t.replace("\n"," ") + "\n")

if __name__ == "__main__":
    # paths
    jsonl_path = "./vectors.jsonl"
    out_path = "./vectors_pca_topk.npy"

    # pipeline
    texts, vecs = load_vectors(jsonl_path)
    reduced, pca = reduce_pca(vecs, n_components=64)
    sparse = sparsify_topk(reduced, k=16)

    # normalize for amplitude encoding (optional)
    norms = np.linalg.norm(sparse, axis=1, keepdims=True) + 1e-9
    sparse = sparse / norms

    save_sparse(out_path, sparse, texts=texts)
