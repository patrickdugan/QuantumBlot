#!/usr/bin/env python3
"""
decode_qft_hist_noflags.py
--------------------------
No-CLI version. Configure via the constants below or env vars.
It will auto-pick the latest counts JSON if none is provided.

ENV OVERRIDES (optional):
  QFT_COUNTS        = path to job_*_counts.json
  RECOVER_JOB_ID    = prefer counts file for this job id (e.g., d2pr065poa4c73c94i9g)
  QFT_ROPE          = path to rope_hint.json
  QFT_VECTORS       = path to vectors.jsonl
  QFT_OUT           = output JSON path (decoded_cards.json by default)
  QFT_REVERSE_BITS  = "1"/"true" to reverse bit order before int()

If you keep a qblot.env with exports, we'll load it automatically.
"""

import os, re, glob, json, math, heapq
from pathlib import Path
from typing import Dict, Tuple, List
import numpy as np

# ---------- user-configurable defaults ----------
COUNTS_GLOB   = "job_*_counts.json"   # fallback glob to auto-pick latest
ROPE_PATH     = "rope_hint.json"
VECTORS_PATH  = "vectors.jsonl"
OUTPUT_PATH   = "decoded_cards.json"
REVERSE_BITS  = False                 # set True if your bit order is reversed
TOP_NEIGHBORS = 24                    # how many cards to emit

# ---------- load your env (qblot.env / .env) ----------
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

load_env_file("qblot.env")
load_env_file(".env")

# ---------- small helpers ----------
def env_bool(name: str, default: bool=False) -> bool:
    val = os.environ.get(name, "")
    if not val:
        return default
    return val.strip().lower() in ("1", "true", "yes", "y", "on")

def resolve_counts_path() -> str:
    # Highest precedence: explicit env path
    p = os.environ.get("QFT_COUNTS")
    if p and Path(p).exists():
        return p

    # Next: RECOVER_JOB_ID matches job_<id>_pub*_counts.json
    jid = os.environ.get("RECOVER_JOB_ID")
    if jid:
        candidates = sorted(Path(".").glob(f"job_{jid}_pub*_counts.json"))
        if candidates:
            # pick the newest pub file
            return str(max(candidates, key=lambda x: x.stat().st_mtime))

    # Else: pick latest matching COUNTS_GLOB
    matches = sorted(glob.glob(COUNTS_GLOB), key=lambda s: Path(s).stat().st_mtime if Path(s).exists() else 0)
    if matches:
        return matches[-1]

    # Final attempt: any *_counts.json
    matches = sorted(glob.glob("*_counts.json"), key=lambda s: Path(s).stat().st_mtime if Path(s).exists() else 0)
    if matches:
        return matches[-1]

    raise FileNotFoundError("Could not locate a counts JSON (set QFT_COUNTS or RECOVER_JOB_ID, or place job_*_counts.json in CWD).")

def resolve_path(env_key: str, default_path: str) -> str:
    p = os.environ.get(env_key, default_path)
    if not Path(p).exists():
        raise FileNotFoundError(f"Required file not found: {p} (set {env_key})")
    return p

def bitstr_to_index(b: str, reverse_bits: bool) -> int:
    s = b.strip()
    if reverse_bits:
        s = s[::-1]
    return int(s, 2)

def spectrum_from_counts(counts: Dict[str,int], n: int, reverse_bits: bool) -> np.ndarray:
    S = np.zeros(n, dtype=complex)
    total = max(1, sum(counts.values()))
    # amplitude estimator ~ sqrt(prob)
    for b, c in counts.items():
        try:
            idx = bitstr_to_index(b, reverse_bits)
            if idx < n:
                S[idx] += math.sqrt(c / total)
        except Exception:
            # ignore malformed keys
            pass
    return S

def ifft_reconstruct(S: np.ndarray) -> np.ndarray:
    x = np.fft.ifft(S)
    x = x.real
    nrm = np.linalg.norm(x)
    return x / (nrm if nrm else 1.0)

def pca_to_embedding(pca_basis: dict, pca_coords: np.ndarray) -> np.ndarray:
    # expects keys: components (2D), mean (1D)
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

def stream_topk_neighbors(vectors_jsonl: str, query_vec: np.ndarray, k: int = 20) -> List[Tuple[float,int,str]]:
    H: List[Tuple[float,int,str]] = []
    q = query_vec
    with open(vectors_jsonl, "r", encoding="utf-8") as f:
        for line in f:
            obj = json.loads(line)
            vid = int(obj.get("id", -1))
            vec = np.asarray(obj["vector"], dtype=float)
            v = vec / (np.linalg.norm(vec) if np.linalg.norm(vec) else 1.0)
            score = float(np.dot(q, v))
            snippet = obj.get("text","")[:240].replace("\\n"," ")
            if len(H) < k:
                heapq.heappush(H, (score, vid, snippet))
            else:
                if score > H[0][0]:
                    heapq.heapreplace(H, (score, vid, snippet))
    return sorted(H, key=lambda t: -t[0])

def main():
    # Resolve inputs/outputs
    counts_path  = resolve_counts_path()
    rope_path    = resolve_path("QFT_ROPE",    ROPE_PATH)
    vectors_path = resolve_path("QFT_VECTORS", VECTORS_PATH)
    out_path     = os.environ.get("QFT_OUT", OUTPUT_PATH)
    reverse_bits = env_bool("QFT_REVERSE_BITS", REVERSE_BITS)

    print(f"[info] counts:  {counts_path}")
    print(f"[info] rope:    {rope_path}")
    print(f"[info] vectors: {vectors_path}")
    print(f"[info] out:     {out_path}")
    print(f"[info] reverse_bits={reverse_bits}")

    with open(counts_path,"r",encoding="utf-8") as f:
        counts = json.load(f)
    with open(rope_path,"r",encoding="utf-8") as f:
        rope = json.load(f)

    # Infer N from PCA components
    pca = rope["raw"]["pca"]
    # components can be a full matrix (list[list]) or a count. Try both.
    comps = pca.get("components")
    if isinstance(comps, list):
        C = len(comps)  # rows = components
    else:
        C = int(comps)
    N = 1 << int(math.ceil(math.log2(C)))

    # 1) counts → spectrum
    S = spectrum_from_counts(counts, N, reverse_bits=reverse_bits)

    # 2) IFFT → PCA coords
    pca_coords = ifft_reconstruct(S)

    # 3) PCA coords → embedding
    emb = pca_to_embedding(pca, pca_coords)

    # 4) nearest neighbors
    top = stream_topk_neighbors(vectors_path, emb, k=TOP_NEIGHBORS)

    out = [
        {"rank": i+1, "score": round(float(s), 4), "id": int(vid), "snippet": text}
        for i, (s, vid, text) in enumerate(top)
    ]
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)

    print(f"[ok] Saved {len(out)} cards → {out_path}")
    for row in out[:8]:
        print(f"[{row['rank']:02}] score={row['score']:.4f} id={row['id']}  {row['snippet'][:100]}")

if __name__ == "__main__":
    main()
