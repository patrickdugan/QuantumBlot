# demo_prerank_qft_gate.py
import json, os, math, numpy as np

THEMATICS_PATH = "thematics_summary.json"
EVIDENCE_PATH  = "evidence.json"
OUTPUT_PATH    = "product_of_query.json"

# ---- FFT helpers ----
def project_pow2(x: np.ndarray, d: int = 1024):
    x = x[:d] if x.size >= d else np.pad(x, (0, d - x.size))
    return x / (np.linalg.norm(x) + 1e-12)

def fft_psd_bands(x: np.ndarray, bands: int = 48):
    spec = np.fft.fft(x)
    psd  = (spec.conj() * spec).real
    edges = np.linspace(0, psd.size, bands + 1, dtype=int)
    return np.array([psd[edges[i]:edges[i+1]].sum() for i in range(bands)], dtype=np.float32)

def cosine(a, b):
    return float(np.dot(a, b) / ((np.linalg.norm(a)*np.linalg.norm(b)) + 1e-12))

def norm_dot(a, b):
    return float(np.dot(a, b) / ((np.linalg.norm(a)*np.linalg.norm(b)) + 1e-12))

def top_peaks(sig, k=8):
    sig = np.array(sig)
    idx = np.argpartition(sig, -k)[-k:]
    return idx[np.argsort(sig[idx])[::-1]].tolist()

# ---- counts → signature ----
def bit_reverse_indices(n):
    d = 1 << n
    return np.array([int(bin(i)[2:].zfill(n)[::-1], 2) for i in range(d)])

def counts_to_band_signature(counts: dict, n_qubits: int, bands: int, reverse_bits=False):
    d = 1 << n_qubits
    p = np.zeros(d, dtype=np.float64)
    total = 0
    for bstr, c in counts.items():
        try:
            i = int(bstr, 2)
            p[i] += c; total += c
        except: pass
    p /= (total + 1e-12)
    if reverse_bits:
        p = p[bit_reverse_indices(n_qubits)]
    spec = np.fft.fft(p)
    psd  = (spec.conj()*spec).real
    edges = np.linspace(0, psd.size, bands+1, dtype=int)
    return np.array([psd[edges[i]:edges[i+1]].sum() for i in range(bands)], dtype=np.float32)

# ---- PreRanker with QFT gate ----
class PreRanker:
    def __init__(self, d=1024, bands=48, gate_overlap=0.25, w_cos=0.6, w_sig=0.4):
        self.d = d; self.bands = bands
        self.gate_overlap = gate_overlap
        self.w_cos = w_cos; self.w_sig = w_sig

    def build_fft_cache(self, emb):
        x = project_pow2(emb.astype(np.float64), d=self.d)
        sig = fft_psd_bands(x, bands=self.bands)
        return {"sig": sig.tolist()}

    def build_qft_cache(self, counts, n_qubits, reverse_bits=False):
        sig = counts_to_band_signature(counts, n_qubits, bands=self.bands, reverse_bits=reverse_bits)
        return {"sig": sig.tolist()}

    def fused_score(self, q_emb, d_emb, q_sig, d_sig):
        return self.w_cos * cosine(q_emb, d_emb) + self.w_sig * norm_dot(q_sig, d_sig)

    def rerank(self, query, docs, qft_peaks):
        q_emb = query["emb"]; q_sig = np.array(query["fft"]["sig"], dtype=np.float32)
        survivors = []
        for d in docs:
            dsig = np.array(d["fft"]["sig"], dtype=np.float32)
            # ----- HARD GATE: must overlap QFT peaks -----
            overlap = len(set(qft_peaks) & set(top_peaks(dsig, 8))) / 8.0
            if overlap >= self.gate_overlap:
                s = self.fused_score(q_emb, d["emb"], q_sig, dsig)
                survivors.append((s, d, overlap))
        survivors.sort(key=lambda t: t[0], reverse=True)
        return survivors

# ----------------- run -----------------
thematics = json.load(open(THEMATICS_PATH, "r"))["thematics"]
theme_names = list(thematics.keys())
default_theme = next((n for n in theme_names if "AI Research" in n), theme_names[0])

ev = json.load(open(EVIDENCE_PATH, "r"))
q_vec = np.array(ev.get("pca_coords") or ev.get("embedding") or np.random.randn(1024), dtype=float)

# load counts if present
counts_path = ev.get("counts_path")
q_counts = json.load(open(counts_path)) if counts_path and os.path.exists(counts_path) else None
meta = ev.get("meta", {})
n_qubits = int(round(math.log2(meta.get("N", 512))))
reverse_bits = bool(meta.get("reverse_bits", False))

pr = PreRanker(d=1024, bands=48, gate_overlap=0.25)

# Build caches
q_fft = pr.build_fft_cache(q_vec)
if q_counts: q_qft = pr.build_qft_cache(q_counts, n_qubits, reverse_bits)
else: q_qft = {"sig": q_fft["sig"]}

q = {"emb": q_vec, "fft": q_fft, "qft": q_qft, "theme": default_theme}

# toy docs (replace with real docs)
rng = np.random.default_rng(42)
docs = []
for i in range(6):
    emb = q_vec + rng.normal(0, (0.06 if i<3 else 0.3), size=q_vec.shape)
    docs.append({"id": f"doc-{i}", "emb": emb, "fft": pr.build_fft_cache(emb)})

# Gate by QFT spikes
qft_peaks = top_peaks(q_qft["sig"], 8)
top = pr.rerank(q, docs, qft_peaks)

# ---- artifact ----
product = {
    "query": {"theme": q["theme"], "signature_peaks": qft_peaks, "source":"qft_gate"},
    "params": {"bands": pr.bands, "gate_overlap": pr.gate_overlap},
    "results": [
        {
            "id": d["id"],
            "score": round(s,4),
            "overlap": round(overlap,2),
            "cosine": round(cosine(q["emb"], d["emb"]),4),
            "sig_sim": round(norm_dot(np.array(q["fft"]["sig"]), np.array(d["fft"]["sig"])),4),
            "fft_peaks": top_peaks(d["fft"]["sig"], 8)
        }
        for (s,d,overlap) in top
    ]
}
json.dump(product, open(OUTPUT_PATH,"w"), indent=2)

# console print
print(f"\nQuery theme: {q['theme']} | QFT spike peaks: {qft_peaks}")
print("---- Survivors ----")
for r in product["results"]:
    print(f"{r['id']} | fused={r['score']} | overlap={r['overlap']} "
          f"| cosine={r['cosine']} | sig_sim={r['sig_sim']} | peaks={r['fft_peaks']}")
print(f"\nArtifact written to {OUTPUT_PATH}")
