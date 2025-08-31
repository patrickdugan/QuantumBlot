#!/usr/bin/env python3
"""
qft_decoder_evidence.py
-----------------------
Builds an "evidence pack" from a QFT histogram:
- Autoloads env (qblot.env / .env)
- Finds latest job_*_counts.json (or uses RECOVER_JOB_ID / QFT_COUNTS)
- Decodes counts -> PCA coords (IFFT, optional Gerchberg–Saxton phase retrieval)
- Maps to embedding space via rope PCA
- Retrieves nearest text chunks from vectors.jsonl
- Writes evidence.json and evidence.md for LLM consumption

ENV overrides (optional):
  QFT_COUNTS         path to counts JSON
  RECOVER_JOB_ID     job id to prefer (e.g., d2pr065poa4c73c94i9g)
  QFT_ROPE           path to rope_hint.json
  QFT_VECTORS        path to vectors.jsonl
  QFT_OUT_JSON       output evidence JSON (default: evidence.json)
  QFT_OUT_MD         output Markdown summary (default: evidence.md)
  QFT_REVERSE_BITS   "1"/"true" to force bit reversal; otherwise auto-choose
  QFT_TOPK_NEIGHBORS integer (default 24)
  QFT_USE_PHASE      "1"/"true" to enable Gerchberg–Saxton refinement
"""

import os, re, glob, json, math, heapq
from pathlib import Path
from typing import Dict, Tuple, List
import numpy as np

# ---------------- Env helpers ----------------
def load_env_file(path: str):
    if not Path(path).exists():
        return
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            if line.startswith("export "):
                line = line[len("export "):]
            m = re.match(r'^([A-Za-z_][A-Za-z0-9_]*)\s*=\s*(.*)$', line)
            if m:
                k, v = m.group(1), m.group(2)
                if len(v) >= 2 and v[0] == v[-1] and v[0] in {"'", '"'}:
                    v = v[1:-1]
                os.environ[k] = v

def env_bool(name: str, default=False) -> bool:
    val = os.environ.get(name, "")
    if not val:
        return default
    return val.strip().lower() in ("1","true","yes","y","on")

def env_int(name: str, default: int) -> int:
    v = os.environ.get(name, "")
    try:
        return int(v) if v else default
    except Exception:
        return default

load_env_file("qblot.env")
load_env_file(".env")

# ---------------- Path resolution ----------------
def resolve_counts_path() -> str:
    p = os.environ.get("QFT_COUNTS")
    if p and Path(p).exists():
        return p
    jid = os.environ.get("RECOVER_JOB_ID")
    if jid:
        candidates = sorted(Path(".").glob(f"job_{jid}_pub*_counts.json"))
        if candidates:
            return str(max(candidates, key=lambda x: x.stat().st_mtime))
    # latest job_*_counts.json
    matches = sorted(glob.glob("job_*_counts.json"), key=lambda s: Path(s).stat().st_mtime if Path(s).exists() else 0)
    if matches:
        return matches[-1]
    raise FileNotFoundError("No counts JSON found. Set QFT_COUNTS or RECOVER_JOB_ID, or place job_*_counts.json in CWD.")

def resolve_path(env_key: str, default_path: str) -> str:
    p = os.environ.get(env_key, default_path)
    if not Path(p).exists():
        raise FileNotFoundError(f"Required file not found: {p} (set {env_key})")
    return p

# ---------------- Decode helpers ----------------
def bitstr_to_index(b: str, reverse_bits: bool) -> int:
    s = b.strip()
    if reverse_bits:
        s = s[::-1]
    return int(s, 2)

def counts_to_spectrum(counts: Dict[str,int], N:int, reverse_bits: bool, alpha: float=1.0) -> np.ndarray:
    """Dirichlet-smoothed probs -> magnitude spectrum sqrt(p)."""
    S = np.zeros(N, dtype=complex)
    total = sum(counts.values())
    if total <= 0:
        return S
    # smoothing
    base = alpha / N
    for k in range(N):
        # prefill smoothing
        S[k] = math.sqrt(base / (total + alpha))
    # add data
    for b, c in counts.items():
        idx = bitstr_to_index(b, reverse_bits)
        if idx < N:
            p = (c + alpha/N) / (total + alpha)
            S[idx] = math.sqrt(p)
    return S

def ifft_reconstruct(mag: np.ndarray) -> np.ndarray:
    x = np.fft.ifft(mag)
    x = x.real
    nrm = np.linalg.norm(x)
    return x / (nrm if nrm else 1.0)

def gs_phase_retrieval(mag: np.ndarray, iters:int=30, seed:int=0) -> np.ndarray:
    """Gerchberg–Saxton: find x (real) s.t. |FFT(x)| ≈ mag. Returns x normalized."""
    rng = np.random.default_rng(seed)
    phase = rng.uniform(-np.pi, np.pi, size=mag.shape[0])
    y = mag * np.exp(1j*phase)
    x = None
    for _ in range(iters):
        x = np.fft.ifft(y)
        # enforce real signal (PCA coords are real)
        x = x.real
        # forward
        Y = np.fft.fft(x)
        y = mag * np.exp(1j * np.angle(Y + 1e-12))
    if x is None:
        x = np.zeros_like(mag, dtype=float)
    nrm = np.linalg.norm(x)
    return (x / nrm) if nrm else x

def pca_to_embedding(pca_basis: dict, pca_coords: np.ndarray) -> np.ndarray:
    # expects keys: components (2D list), mean (1D list)
    comps = np.asarray(pca_basis["components"], dtype=float)   # (C, D)
    mean  = np.asarray(pca_basis["mean"], dtype=float)         # (D,)
    C = comps.shape[0]
    v = pca_coords
    if len(v) < C:
        vv = np.zeros(C, dtype=float); vv[:len(v)] = v; v = vv
    elif len(v) > C:
        v = v[:C]
    emb = v @ comps + mean
    nrm = np.linalg.norm(emb)
    return emb / (nrm if nrm else 1.0)

def stream_topk_neighbors(vectors_jsonl: str, query_vec: np.ndarray, k: int = 24) -> List[Tuple[float,int,str]]:
    H: List[Tuple[float,int,str]] = []
    q = query_vec
    with open(vectors_jsonl, "r", encoding="utf-8") as f:
        for line in f:
            obj = json.loads(line)
            vid = int(obj.get("id", -1))
            vec = np.asarray(obj["vector"], dtype=float)
            v = vec / (np.linalg.norm(vec) if np.linalg.norm(vec) else 1.0)
            score = float(np.dot(q, v))
            snippet = obj.get("text","")[:240].replace("\n"," ")
            if len(H) < k:
                heapq.heappush(H, (score, vid, snippet))
            else:
                if score > H[0][0]:
                    heapq.heapreplace(H, (score, vid, snippet))
    return sorted(H, key=lambda t: -t[0])

def choose_bit_order_and_decode(counts: Dict[str,int], N:int, pca:dict, vectors_path:str, use_phase:bool, k:int):
    """Try normal vs reversed bit order; pick whichever yields higher top-1 NN score."""
    best = None
    for reverse in (False, True):
        mag = counts_to_spectrum(counts, N, reverse_bits=reverse, alpha=1.0)
        x1  = ifft_reconstruct(mag)
        x2  = gs_phase_retrieval(mag, iters=25, seed=0) if use_phase else None

        cand = []
        for label, x in (("ifft", x1), ("gs", x2)):
            if x is None: continue
            emb = pca_to_embedding(pca, x)
            top = stream_topk_neighbors(vectors_path, emb, k=k)
            best_score = top[0][0] if top else -1.0
            cand.append((best_score, reverse, label, x, emb, top))

        if cand:
            cand.sort(key=lambda t: t[0], reverse=True)
            topcand = cand[0]
            if (best is None) or (topcand[0] > best[0]):
                best = topcand

    if best is None:
        return None
    # Unpack
    best_score, reverse_bits, decoder_label, pca_coords, emb, neighbors = best
    return {
        "reverse_bits": reverse_bits,
        "decoder": decoder_label,
        "top1_score": float(best_score),
        "pca_coords": pca_coords.tolist(),
        "embedding": emb.tolist(),
        "neighbors": [
            {"rank": i+1, "score": float(s), "id": int(vid), "snippet": text}
            for i, (s, vid, text) in enumerate(neighbors)
        ]
    }

def main():
    # Resolve paths
    counts_path  = resolve_counts_path()
    rope_path    = resolve_path("QFT_ROPE",    "rope_hint.json")
    vectors_path = resolve_path("QFT_VECTORS", "vectors.jsonl")
    out_json     = os.environ.get("QFT_OUT_JSON", "evidence.json")
    out_md       = os.environ.get("QFT_OUT_MD",   "evidence.md")
    topk         = env_int("QFT_TOPK_NEIGHBORS", 24)
    use_phase    = env_bool("QFT_USE_PHASE", False)

    print(f"[info] counts:  {counts_path}")
    print(f"[info] rope:    {rope_path}")
    print(f"[info] vectors: {vectors_path}")
    print(f"[info] out:     {out_json} / {out_md}")
    print(f"[info] phase_retrieval={use_phase}")

    # Load inputs
    with open(counts_path,"r",encoding="utf-8") as f:
        counts = json.load(f)
    with open(rope_path,"r",encoding="utf-8") as f:
        rope = json.load(f)

    pca = rope["raw"]["pca"]
    comps = pca.get("components")
    if isinstance(comps, list):
        C = len(comps)  # rows = components
    else:
        C = int(comps)
    N = 1 << int(math.ceil(math.log2(C)))

    result = choose_bit_order_and_decode(counts, N, pca, vectors_path, use_phase=use_phase, k=topk)
    if result is None:
        raise RuntimeError("Decoding failed (no neighbors found).")

    # Build evidence pack
    evidence = {
        "meta": {
            "counts_path": counts_path,
            "rope_path": rope_path,
            "vectors_path": vectors_path,
            "decoder": result["decoder"],
            "reverse_bits": result["reverse_bits"],
            "top1_score": round(result["top1_score"], 4),
            "N": N,
            "components": C,
            "topk": topk,
            "phase_retrieval": use_phase,
        },
        "neighbors": result["neighbors"],
        "embedding": result["embedding"],     # normalized
        "pca_coords": result["pca_coords"],   # for debugging
    }

    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(evidence, f, ensure_ascii=False, indent=2)
    print(f"[ok] wrote {out_json}")

    # Markdown summary
    lines = []
    lines.append(f"# QFT Evidence Pack\n")
    lines.append(f"- counts: `{counts_path}`")
    lines.append(f"- rope: `{rope_path}`")
    lines.append(f"- vectors: `{vectors_path}`")
    lines.append(f"- decoder: `{evidence['meta']['decoder']}`  reverse_bits: `{evidence['meta']['reverse_bits']}`  N: `{N}`  C: `{C}`")
    lines.append(f"- top1 cosine: `{evidence['meta']['top1_score']}`  neighbors: `{len(evidence['neighbors'])}`\n")
    lines.append("## Top neighbors\n")
    for n in evidence["neighbors"][:12]:
        lines.append(f"- **{n['score']:.4f}** — id `{n['id']}` — {n['snippet']}")
    with open(out_md, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    print(f"[ok] wrote {out_md}")

if __name__ == "__main__":
    main()
