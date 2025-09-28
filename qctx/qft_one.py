
#!/usr/bin/env python3
"""
qft_one.py — single-entrypoint CLI for your QFT pipeline

Stages:
  1) pca        : Fit PCA (or load existing), project & L2-normalize
  2) prep       : Sparsify (top-k by magnitude), pad to pow2 (for qubits), save .npy
  3) prerank    : (Optional) Theme-gated preselection using bit-signatures (PCA space)
  4) qft        : Build & run QFT circuit on IBM Runtime (basic or layered)
  5) decode     : Convert counts -> spectrum -> nearest neighbors (uses your decoders)
  6) emit       : Emit compact system/payload using rope/evidence
  7) all        : Run 1–6 in sequence (skips steps whose outputs already exist unless --force)

Notes:
- Embeddings: this script assumes you already have vectors (dense) as either:
    a) a .npy matrix (N x D), or
    b) a .jsonl with {"id":..., "text":..., "vector":[...]}
  If you want *larger* embeddings than what you've got, you must re-embed upstream.
  PCA cannot invent information that isn't in the current vectors.

Env for IBM Runtime:
  export IBM_CLOUD_API_KEY=...   # API key
  export IBM_QUANTUM_CRN=...     # instance CRN (e.g., "crn:v1:bluemix:public:quantum-computing:...")

Examples:
  # Fit PCA to 768 dims, sparsify to 70%, prep for 17 qubits, then run and decode:
  python qft_one.py all --src vectors.npy --target-dim 768 --sparsity 0.7 \
    --nqubits 17 --backend ibm_brisbane --shots 4096 \
    --rope rope_hint.json --vectors-jsonl vectors.jsonl --theme-id 2

  # Run only QFT on an existing prepped .npy row 0:
  python qft_one.py qft --prepped vectors_pca_topk.npy --row 0 --nqubits 17 --backend ibm_brisbane --shots 8192
"""

import os, sys, json, math, argparse
from pathlib import Path
from typing import Optional, List, Tuple

import numpy as np

# --- Optional imports from user's helpers (loaded if present) ---
def _try_import():
    mods = {}
    try:
        import qft_pca  # has PCA helpers + fourier_fingerprint_runtime
        mods['qft_pca'] = qft_pca
    except Exception as e:
        mods['qft_pca_err'] = e

    try:
        import prerank_wedge  # theme pre-ranker (optional)
        mods['prerank_wedge'] = prerank_wedge
    except Exception as e:
        mods['prerank_wedge_err'] = e

    try:
        import layered_qft  # build_interference_circuit, counts_to_band_signature
        mods['layered_qft'] = layered_qft
    except Exception as e:
        mods['layered_qft_err'] = e

    try:
        import decode_qft_hist  # env-driven decoder (expects env vars)
        mods['decode_qft_hist'] = decode_qft_hist
    except Exception as e:
        mods['decode_qft_hist_err'] = e

    try:
        import emit_prompt  # emits compact prompt/payload
        mods['emit_prompt'] = emit_prompt
    except Exception as e:
        mods['emit_prompt_err'] = e

    return mods

MODS = _try_import()

# ---------- IO helpers ----------

def load_vectors(src: str) -> Tuple[np.ndarray, Optional[List[dict]]]:
    """Load vectors from .npy or .jsonl.
       Returns (X, meta_rows) where meta_rows includes dicts with at least 'id' and 'text' if jsonl was used.
    """
    p = Path(src)
    if not p.exists():
        raise FileNotFoundError(f"Source not found: {src}")

    if p.suffix.lower() == ".npy":
        X = np.load(str(p))
        return X, None

    if p.suffix.lower() == ".jsonl":
        rows = []
        with p.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line: continue
                obj = json.loads(line)
                rows.append(obj)
        # Build matrix
        if not rows or 'vector' not in rows[0]:
            raise ValueError("JSONL rows must include a 'vector' field.")
        X = np.array([r['vector'] for r in rows], dtype=float)
        return X, rows

    raise ValueError("Unsupported source format. Use .npy or .jsonl")

def save_jsonl(rows: List[dict], path: str):
    with open(path, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

# ---------- Stage 1: PCA ----------

def run_pca(src: str, target_dim: int, out_prefix: str, force: bool = False) -> str:
    out_mu = f"{out_prefix}_mu.npy"
    out_W  = f"{out_prefix}_W.npy"
    out_Z  = f"{out_prefix}_Z.npy"

    if (not force) and Path(out_Z).exists() and Path(out_mu).exists() and Path(out_W).exists():
        print(f"[pca] Re-using existing {out_Z}")
        return out_Z

    X, _ = load_vectors(src)
    print(f"[pca] Loaded {X.shape} from {src}")

    if 'qft_pca' not in MODS:
        raise RuntimeError("qft_pca.py not available; cannot run PCA here. (Place it beside this script.)")

    mu, W = MODS['qft_pca'].fit_pca(X, target_dim)  # SVD-based PCA
    Z = MODS['qft_pca'].project_normalize(X, mu, W) # L2 normalize per row

    np.save(out_mu, mu)
    np.save(out_W,  W)
    np.save(out_Z,  Z)
    print(f"[pca] Wrote {out_Z} (and {out_mu}, {out_W})")
    return out_Z

# ---------- Stage 2: prep (sparsify + pad pow2) ----------

def sparsify_pad(Z: np.ndarray, sparsity: float) -> np.ndarray:
    """Zero-out smallest components per row to achieve given sparsity fraction (0..1), then return Z."""
    if not (0.0 < sparsity < 1.0):
        raise ValueError("--sparsity must be in (0,1)")

    Z2 = Z.copy()
    N, D = Z2.shape
    k = int(round((1.0 - sparsity) * D))
    k = max(1, min(k, D))
    # keep top-k by absolute value per row
    idxs = np.argpartition(np.abs(Z2), -k, axis=1)[:, :-k]
    Z2[np.arange(N)[:, None], idxs] = 0.0
    # renormalize rows
    norms = np.linalg.norm(Z2, axis=1, keepdims=True) + 1e-12
    Z2 = Z2 / norms
    return Z2

def pad_pow2(X: np.ndarray) -> np.ndarray:
    """Right-pad columns with zeros to next power-of-two column count."""
    N, D = X.shape
    n = int(math.ceil(math.log2(max(1, D))))
    size = 1 << n
    if size == D:
        return X
    pad = np.zeros((N, size), dtype=X.dtype)
    pad[:, :D] = X
    return pad

def run_prep(pca_Z_path: str, sparsity: float, out_path: str, force: bool = False) -> str:
    if (not force) and Path(out_path).exists():
        print(f"[prep] Re-using existing {out_path}")
        return out_path

    Z = np.load(pca_Z_path)
    print(f"[prep] Loaded PCA-projected matrix: {Z.shape}")
    Zs = sparsify_pad(Z, sparsity=sparsity)
    Zp = pad_pow2(Zs)
    np.save(out_path, Zp)
    print(f"[prep] Wrote {out_path} (shape={Zp.shape}, qubits={int(math.log2(Zp.shape[1]))})")
    return out_path

# ---------- Stage 3: prerank (optional) ----------

def run_prerank(prepped_path: str, theme_id: int, bands: int = 64, gate: float = 0.35, sample: int = 0) -> List[int]:
    if 'prerank_wedge' not in MODS:
        print("[prerank] prerank_wedge.py not available; skipping.")
        return []
    X = np.load(prepped_path)
    pr = MODS['prerank_wedge'].PreRanker(d=X.shape[1], bands=bands, mixer_layers=2, gate=gate)
    cache = [pr.build_cache_entry(x, theme_id=theme_id, pos=i) for i, x in enumerate(X)]
    # Simple ranking by 'weight' then return indexes (descending)
    scored = sorted([(i, c['weight']) for i, c in enumerate(cache)], key=lambda t: -t[1])
    order = [i for i, _ in scored]
    if sample > 0:
        order = order[:sample]
    print(f"[prerank] Selected {len(order)} rows (theme_id={theme_id})")
    return order

# ---------- Stage 4: QFT run ----------

def run_qft(prepped_path: str, row: int, nqubits: Optional[int], backend: str, shots: int,
            layered: bool = False, theme_id: int = 0, pos: int = 0, optimization_level: int = 1) -> str:
    vecs = np.load(prepped_path)
    if row < 0 or row >= vecs.shape[0]:
        raise IndexError(f"row {row} out of range 0..{vecs.shape[0]-1}")
    v = vecs[row]

    if nqubits is None:
        nqubits = int(math.ceil(math.log2(len(v))))

    # Dispatch to layered or plain runtime
    if layered:
        if 'layered_qft' not in MODS:
            raise RuntimeError("layered_qft.py not available for --layered run.")
        # Build a backend via qiskit_ibm_runtime
        from qiskit_ibm_runtime import QiskitRuntimeService, SamplerV2 as Sampler
        service = QiskitRuntimeService(
            channel="ibm_cloud",
            token=os.environ.get("IBM_CLOUD_API_KEY"),
            instance=os.environ.get("IBM_QUANTUM_CRN"),
        )
        bk = service.backend(backend)
        qc = MODS['layered_qft'].build_interference_circuit(v, n_qubits=nqubits, theme_id=theme_id, pos=pos, backend=bk)
        sampler = Sampler(bk)
        job = sampler.run([qc], shots=shots)
        result = job.result()
        record = result[0]
        qd = record.data
        counts = {str(k): int(float(val) * shots) for k, val in qd.items()}
    else:
        # Use qft_pca's fourier_fingerprint_runtime (initialize + QFT + measure)
        if 'qft_pca' not in MODS:
            raise RuntimeError("qft_pca.py not available for basic QFT run.")
        counts = MODS['qft_pca'].fourier_fingerprint_runtime(
            amplitudes=v, shots=shots, backend_name=backend, optimization_level=optimization_level
        )

    out_path = f"job_qft_counts_row{row}_{backend}_{shots}.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(counts, f, indent=2)
    print(f"[qft] Saved counts → {out_path}")
    return out_path

# ---------- Stage 5: decode ----------

def run_decode(counts_path: str, rope_path: Optional[str], vectors_jsonl: Optional[str], out_path: str) -> str:
    # We prefer the user's env-driven decoder if import succeeded.
    if 'decode_qft_hist' in MODS:
        # Set env for the module's main() expectations
        os.environ["QFT_COUNTS"] = counts_path
        if rope_path: os.environ["QFT_ROPE"] = rope_path
        if vectors_jsonl: os.environ["QFT_VECTORS"] = vectors_jsonl
        os.environ["QFT_OUT"] = out_path
        # Try to call module-level main() if exists, else fallback to import-time side effects
        if hasattr(MODS['decode_qft_hist'], "main"):
            MODS['decode_qft_hist'].main()
        else:
            # If the module decodes on import using env, re-import
            import importlib
            importlib.reload(MODS['decode_qft_hist'])
        print(f"[decode] Wrote {out_path}")
        return out_path
    else:
        # Minimal fallback: just dump normalized histogram & leave spectrum/NN for later
        with open(counts_path, "r", encoding="utf-8") as f:
            counts = json.load(f)
        total = sum(counts.values()) or 1
        probs = {k: v/total for k,v in counts.items()}
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump({"probs": probs, "note": "fallback decoder (install decode_qft_hist.py)"}, f, indent=2)
        print(f"[decode] Fallback wrote {out_path}")
        return out_path

# ---------- Stage 6: emit ----------

def run_emit(rope_path: str, background_json: str, payload_out: str, counts_top: int = 32, basis_top: int = 24, clusters: int = 16):
    if 'emit_prompt' not in MODS:
        raise RuntimeError("emit_prompt.py not available for --emit stage.")
    # Use emit_prompt module's CLI-style entry if available; otherwise call function if exposed
    # Here we just shell out via os.system to keep compatibility with user's existing script arguments.
    cmd = f'python -m emit_prompt --rope "{rope_path}" --background "{background_json}" --emit-payload "{payload_out}" --counts-top {counts_top} --basis-top {basis_top} --clusters {clusters}'
    print("[emit] Running:", cmd)
    rc = os.system(cmd)
    if rc != 0:
        raise RuntimeError(f"emit_prompt failed with exit code {rc}")
    print(f"[emit] Wrote {payload_out}")

# ---------- CLI ----------

def main():
    ap = argparse.ArgumentParser(description="Single-file QFT pipeline orchestration.")
    sub = ap.add_subparsers(dest="cmd", required=True)

    # Shared options
    def add_common_io(a):
        a.add_argument("--src", type=str, help="Vectors source: .npy or .jsonl with 'vector' fields.")
        a.add_argument("--prepped", type=str, help="Prepped .npy (sparse, pow2) if skipping PCA/prep.")
        a.add_argument("--out-prefix", type=str, default="qft", help="Output prefix for PCA artifacts.")
        a.add_argument("--force", action="store_true", help="Recompute even if outputs exist.")

    # pca
    p_pca = sub.add_parser("pca", help="Fit PCA & project.")
    add_common_io(p_pca)
    p_pca.add_argument("--target-dim", type=int, required=True, help="Dimension after PCA.")
    p_pca.add_argument("--out-Z", type=str, default="qft_Z.npy", help="Projected matrix output path.")

    # prep
    p_prep = sub.add_parser("prep", help="Sparsify & pad pow2.")
    p_prep.add_argument("--pca-Z", type=str, required=True, help="Input PCA-projected matrix (.npy).")
    p_prep.add_argument("--sparsity", type=float, required=True, help="Fraction of zeros to impose per row (0..1).")
    p_prep.add_argument("--out", type=str, default="vectors_pca_topk.npy", help="Output prepped .npy")

    # prerank
    p_rank = sub.add_parser("prerank", help="Theme-gated preselection (optional).")
    p_rank.add_argument("--prepped", type=str, required=True, help="Prepped .npy")
    p_rank.add_argument("--theme-id", type=int, required=True)
    p_rank.add_argument("--bands", type=int, default=64)
    p_rank.add_argument("--gate", type=float, default=0.35)
    p_rank.add_argument("--sample", type=int, default=0, help="Return only top N indexes (0=all)")

    # qft
    p_qft = sub.add_parser("qft", help="Run QFT on IBM.")
    p_qft.add_argument("--prepped", type=str, required=True, help="Prepped .npy")
    p_qft.add_argument("--row", type=int, default=0)
    p_qft.add_argument("--nqubits", type=int, default=None)
    p_qft.add_argument("--backend", type=str, default="ibm_brisbane")
    p_qft.add_argument("--shots", type=int, default=4096)
    p_qft.add_argument("--layered", action="store_true", help="Use layered_qft interference circuit.")
    p_qft.add_argument("--theme-id", type=int, default=0)
    p_qft.add_argument("--pos", type=int, default=0)
    p_qft.add_argument("--optimization-level", type=int, default=1)

    # decode
    p_dec = sub.add_parser("decode", help="Decode counts -> spectrum/NN (uses decode_qft_hist if available).")
    p_dec.add_argument("--counts", type=str, required=True)
    p_dec.add_argument("--rope", type=str, default=None)
    p_dec.add_argument("--vectors-jsonl", type=str, default=None)
    p_dec.add_argument("--out", type=str, default="decoded_evidence.json")

    # emit
    p_emit = sub.add_parser("emit", help="Emit compact system/payload using emit_prompt.py")
    p_emit.add_argument("--rope", type=str, required=True)
    p_emit.add_argument("--background", type=str, required=True, help="evidence.json or decoded JSON")
    p_emit.add_argument("--payload-out", type=str, default="request_skeleton.json")
    p_emit.add_argument("--counts-top", type=int, default=32)
    p_emit.add_argument("--basis-top", type=int, default=24)
    p_emit.add_argument("--clusters", type=int, default=16)

    # all
    p_all = sub.add_parser("all", help="Run pca->prep->(prerank)->qft->decode->emit in sequence.")
    add_common_io(p_all)
    p_all.add_argument("--target-dim", type=int, required=True)
    p_all.add_argument("--sparsity", type=float, required=True)
    p_all.add_argument("--row", type=int, default=0)
    p_all.add_argument("--nqubits", type=int, default=None)
    p_all.add_argument("--backend", type=str, default="ibm_brisbane")
    p_all.add_argument("--shots", type=int, default=4096)
    p_all.add_argument("--layered", action="store_true")
    p_all.add_argument("--theme-id", type=int, default=None, help="If set, run prerank and pick first index as --row.")
    p_all.add_argument("--pos", type=int, default=0)
    p_all.add_argument("--optimization-level", type=int, default=1)
    p_all.add_argument("--rope", type=str, default=None)
    p_all.add_argument("--vectors-jsonl", type=str, default=None)
    p_all.add_argument("--payload-out", type=str, default="request_skeleton.json")

    args = ap.parse_args()

    if args.cmd == "pca":
        if not args.src:
            ap.error("pca requires --src")
        run_pca(args.src, args.target_dim, args.out_prefix, force=args.force)
        sys.exit(0)

    if args.cmd == "prep":
        run_prep(args.pca_Z, args.sparsity, args.out, force=args.force)
        sys.exit(0)

    if args.cmd == "prerank":
        idxs = run_prerank(args.prepped, args.theme_id, bands=args.bands, gate=args.gate, sample=args.sample)
        print(json.dumps({"order": idxs[:args.sample or len(idxs)]}))
        sys.exit(0)

    if args.cmd == "qft":
        run_qft(args.prepped, args.row, args.nqubits, args.backend, args.shots, layered=args.layered,
                theme_id=args.theme_id, pos=args.pos, optimization_level=args.optimization_level)
        sys.exit(0)

    if args.cmd == "decode":
        run_decode(args.counts, args.rope, args.vectors_jsonl, args.out)
        sys.exit(0)

    if args.cmd == "emit":
        run_emit(args.rope, args.background, args.payload_out, args.counts_top, args.basis_top, args.clusters)
        sys.exit(0)

    if args.cmd == "all":
        if not args.src and not args.prepped:
            ap.error("all requires either --src (.npy/.jsonl) or --prepped (.npy)")

        # 1) PCA (if src provided and not skipping)
        if args.src:
            Z_path = run_pca(args.src, args.target_dim, args.out_prefix, force=args.force)
        else:
            # Skip PCA: assume user already has PCA-projected file specified in --prepped; we still want sparsify/pad
            Z_path = args.prepped

        # 2) prep
        prepped_path = run_prep(Z_path, args.sparsity, "vectors_pca_topk.npy", force=args.force)

        # 3) prerank (optional)
        row = args.row
        if args.theme_id is not None:
            order = run_prerank(prepped_path, args.theme_id, sample=1)
            if order:
                row = order[0]
                print(f"[all] Using preranked row={row}")

        # 4) qft
        counts_path = run_qft(prepped_path, row, args.nqubits, args.backend, args.shots,
                              layered=args.layered, theme_id=(args.theme_id or 0), pos=args.pos,
                              optimization_level=args.optimization_level)

        # 5) decode (if decoder available & inputs provided)
        decoded_path = "decoded_evidence.json"
        if args.rope or args.vectors_jsonl:
            run_decode(counts_path, args.rope, args.vectors_jsonl, decoded_path)
        else:
            print("[all] Skipping decode (no --rope/--vectors-jsonl).")

        # 6) emit (if rope + background available)
        if args.rope and Path(decoded_path).exists():
            run_emit(args.rope, decoded_path, args.payload_out)
        else:
            print("[all] Skipping emit (missing --rope or decoded file).")

        print("[all] DONE.")
        sys.exit(0)

if __name__ == "__main__":
    main()
