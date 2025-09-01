import numpy as np
from collections import defaultdict

# ---------- TRAIN ----------
def fit_pca(X, d):
    # X: [N, D] dense/sparse -> assume dense here
    mu = X.mean(0, keepdims=True)
    Xc = X - mu
    U, S, Vt = np.linalg.svd(Xc, full_matrices=False)
    W = Vt[:d].T  # [D, d]
    return mu, W

def project_normalize(X, mu, W):
    Z = (X - mu) @ W                # [N, d]
    Z /= (np.linalg.norm(Z, axis=1, keepdims=True) + 1e-12)
    return Z

def carriers(key_seed, d, K):
    rng = np.random.default_rng(key_seed)
    C = rng.normal(size=(K, d))
    C /= np.linalg.norm(C, axis=1, keepdims=True)
    return C  # [K, d]

def bit_signature(Z, C, tau=None):
    # Z: [N, d], C: [K, d]
    A = Z @ C.T                     # [N, K] amplitudes
    if tau is None:
        tau = np.zeros((1, A.shape[1]))
    B = (A >= tau).astype(np.uint8) # [N, K] bits {0,1}
    return A, B

def band_index(B, L=16):
    N, K = B.shape
    r = K // L
    bands = []
    for j in range(L):
        slice_bits = B[:, j*r:(j+1)*r]
        # pack r bits -> int
        vals = np.packbits(slice_bits, axis=1, bitorder='little')
        # If r not multiple of 8, vals will be >1 byte; convert to tuple-of-bytes
        bands.append(vals)
    return bands  # list length L, each [N, bytes_per_band]

def build_inverted_index(B, ids, L=16):
    idx = [defaultdict(list) for _ in range(L)]
    bands = band_index(B, L=L)
    for j in range(L):
        band = bands[j]  # [N, nbytes]
        for i in range(band.shape[0]):
            key = tuple(band[i].tolist())
            idx[j][key].append(ids[i])
    return idx  # list of band dicts

# ---------- QUERY ----------
def hamming_pre_rank(Bq, Bcands):
    # Bq: [K], Bcands: [M, K]
    return (Bq == Bcands).sum(axis=1)

def sieve_candidates(zq, C, idx, corpus_bits, ids, L=16):
    Aq = zq @ C.T                     # [K]
    Bq = (Aq >= 0).astype(np.uint8)   # simple threshold
    # bucket lookups
    r = Bq.shape[0] // L
    keys = []
    for j in range(L):
        band_bits = Bq[j*r:(j+1)*r][None, :]
        band_key = tuple(np.packbits(band_bits, bitorder='little').tolist())
        keys.append(band_key)
    # union of buckets
    cand = set()
    for j in range(L):
        cand.update(idx[j].get(keys[j], []))
    cand = list(cand)
    if not cand: return [], Bq, Aq
    # pre-rank by Hamming sim on full K bits
    Bc = corpus_bits[cand]              # [M, K]
    sim = (Bc == Bq).sum(axis=1)
    order = np.argsort(-sim)
    return [cand[i] for i in order], Bq, Aq

# ---------- FINAL RERANK ----------
def cosine(a, b): return float(a @ b)
