#!/usr/bin/env python3
"""
Hardcoded QFT decoder → evidence pack
"""

import json, math, heapq
from pathlib import Path
import numpy as np

# ----------- HARD CODE YOUR PATHS HERE ------------
COUNTS_PATH  = r"C:\projects\QuantumBlot\qctx\job_d2pr065poa4c73c94i9g_pub0_counts.json"
ROPE_PATH    = r"C:\projects\QuantumBlot\qctx\rope_hint.json"
VECTORS_PATH = r"C:\projects\QuantumBlot\qctx\vectors.jsonl"
OUT_JSON     = r"C:\projects\QuantumBlot\qctx\evidence.json"
OUT_MD       = r"C:\projects\QuantumBlot\qctx\evidence.md"
TOPK         = 200
USE_PHASE    = False   # set True to enable Gerchberg–Saxton phase retrieval
# --------------------------------------------------

def bitstr_to_index(b: str, reverse_bits: bool) -> int:
    return int(b[::-1] if reverse_bits else b, 2)

def counts_to_spectrum(counts, N, reverse_bits=False, alpha=1.0):
    S = np.zeros(N, dtype=complex)
    total = sum(counts.values())
    if total <= 0: return S
    for b, c in counts.items():
        idx = bitstr_to_index(b, reverse_bits)
        if idx < N:
            p = (c + alpha/N) / (total + alpha)
            S[idx] = math.sqrt(p)
    return S

def ifft_reconstruct(mag):
    x = np.fft.ifft(mag).real
    nrm = np.linalg.norm(x)
    return x / (nrm if nrm else 1.0)

def pca_to_embedding(pca_basis, pca_coords):
    comps = np.asarray(pca_basis["components"], dtype=float)
    mean  = np.asarray(pca_basis["mean"], dtype=float)
    C = comps.shape[0]
    v = pca_coords
    if len(v) < C:
        vv = np.zeros(C); vv[:len(v)] = v; v = vv
    elif len(v) > C:
        v = v[:C]
    emb = v @ comps + mean
    return emb / (np.linalg.norm(emb) or 1.0)

def stream_topk_neighbors(vectors_jsonl, query_vec, k=20):
    H = []
    with open(vectors_jsonl, "r", encoding="utf-8") as f:
        for line in f:
            obj = json.loads(line)
            vid = int(obj.get("id", -1))
            vec = np.asarray(obj["vector"], dtype=float)
            v = vec / (np.linalg.norm(vec) or 1.0)
            score = float(np.dot(query_vec, v))
            snippet = obj.get("text","")[:240].replace("\n"," ")
            if len(H) < k:
                heapq.heappush(H, (score, vid, snippet))
            elif score > H[0][0]:
                heapq.heapreplace(H, (score, vid, snippet))
    return sorted(H, key=lambda t: -t[0])

def main():
    counts = json.load(open(COUNTS_PATH, "r", encoding="utf-8"))
    rope   = json.load(open(ROPE_PATH, "r", encoding="utf-8"))

    # Figure out dimension C_max from rope
    pca_root = rope["raw"]["pca"]

    if "views" in pca_root:
        # Get the max available components across views
        available = [int(k[1:]) for k in pca_root["views"].keys()]
        C_max = max(available)
    else:
        C_max = len(pca_root["components"])

    # Compute N = nearest power of two ≥ C_max
    N = 1 << int(math.ceil(math.log2(C_max)))

    # Now select PCA view (wide or simple)
    if "views" in pca_root:
        views = pca_root["views"]
        key = f"C{N}"
        if key not in views:
            # fallback: closest available
            candidates = sorted(available, key=lambda c: abs(c - N))
            key = f"C{candidates[0]}"
        print(f"[info] using PCA view {key}")
        pca = views[key]
    else:
        pca = pca_root

    comps = pca["components"]
    C = len(comps)
    # ------------------------------------------

    N = 1 << int(math.ceil(math.log2(C)))

    # Try both bit orders
    best = None
    for reverse in (False, True):
        mag = counts_to_spectrum(counts, N, reverse)
        x   = ifft_reconstruct(mag)
        emb = pca_to_embedding(pca, x)
        top = stream_topk_neighbors(VECTORS_PATH, emb, k=TOPK)
        score = top[0][0] if top else -1
        if best is None or score > best[0]:
            best = (score, reverse, x, emb, top)

    score, reverse, pca_coords, emb, neighbors = best
    evidence = {
        "meta": {"decoder":"ifft","reverse_bits":reverse,"top1_score":round(score,4),
                 "N":N,"components":C,"topk":TOPK},
        "neighbors":[
            {"rank":i+1,"score":float(s),"id":int(vid),"snippet":txt}
            for i,(s,vid,txt) in enumerate(neighbors)
        ],
        "embedding": emb.tolist(),
        "pca_coords": pca_coords.tolist(),
    }
    json.dump(evidence, open(OUT_JSON,"w",encoding="utf-8"), indent=2)
    with open(OUT_MD,"w",encoding="utf-8") as f:
        f.write("# QFT Evidence Pack\n")
        f.write(f"- counts: {COUNTS_PATH}\n- rope: {ROPE_PATH}\n- vectors: {VECTORS_PATH}\n")
        f.write(f"- top1 cosine: {score:.4f}\n\n## Top neighbors\n")
        for n in evidence["neighbors"][:12]:
            f.write(f"- **{n['score']:.4f}** — id {n['id']} — {n['snippet']}\n")
    print(f"[ok] wrote {OUT_JSON} and {OUT_MD}")

if __name__ == "__main__":
    main()
