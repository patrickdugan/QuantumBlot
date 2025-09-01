import json, numpy as np
import matplotlib.pyplot as plt

# ---- paths ----
EVIDENCE_PATH  = "evidence.json"
COUNTS_PATH = "job_d2q45r5t2ras73bdnk40_pub0_counts.json"

# ---- load embedding ----
ev = json.load(open(EVIDENCE_PATH, "r"))
emb = np.array(ev.get("pca_coords") or ev.get("embedding"), dtype=float)

# ---- FFT from embedding ----
def fft_psd_bands(x: np.ndarray, bands: int = 48):
    spec = np.fft.fft(x)
    psd  = (spec.conj()*spec).real
    edges = np.linspace(0, psd.size, bands + 1, dtype=int)
    return np.array([psd[edges[i]:edges[i+1]].sum() for i in range(bands)], dtype=np.float32)

emb_bands = fft_psd_bands(emb, bands=48)

# ---- counts → signature ----
def bit_reverse_indices(n):
    d = 1 << n
    idx = np.arange(d)
    rev = np.array([int(bin(i)[2:].zfill(n)[::-1], 2) for i in idx])
    return rev

def counts_to_band_signature(counts: dict, n_qubits: int, bands: int, reverse_bits: bool=False):
    d = 1 << n_qubits
    p = np.zeros(d, dtype=np.float64)
    total = 0
    for bstr, c in counts.items():
        p[int(bstr, 2)] += c
        total += c
    p /= (total + 1e-12)
    if reverse_bits:
        p = p[bit_reverse_indices(n_qubits)]
    spec = np.fft.fft(p)
    psd  = (spec.conj()*spec).real
    edges = np.linspace(0, psd.size, bands+1, dtype=int)
    return np.array([psd[edges[i]:edges[i+1]].sum() for i in range(bands)], dtype=np.float32)

counts = json.load(open(COUNTS_PATH, "r"))
counts_bands = counts_to_band_signature(counts, n_qubits=9, bands=48, reverse_bits=False)

# ---- plot ----
plt.figure(figsize=(12,5))
plt.subplot(1,2,1)
plt.bar(range(len(emb_bands)), emb_bands)
plt.title("Embedding → FFT bands")
plt.subplot(1,2,2)
plt.bar(range(len(counts_bands)), counts_bands, color='orange')
plt.title("QFT counts → FFT bands")
plt.tight_layout()
plt.show()

# quick numeric diff
print("Cosine similarity between band vectors:",
      float(np.dot(emb_bands, counts_bands) /
            (np.linalg.norm(emb_bands)*np.linalg.norm(counts_bands) + 1e-12)))
