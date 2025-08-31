import numpy as np

# ---------- Core helpers ----------
def project_pow2(x, d=1024):
    x = x[:d] if x.size >= d else np.pad(x, (0, d - x.size))
    n = np.linalg.norm(x) + 1e-12
    return x / n

def fft_psd_bands(x, bands=48):
    spec = np.fft.fft(x)
    psd = (spec.conj() * spec).real
    edges = np.linspace(0, psd.size, bands + 1, dtype=int)
    return np.array([psd[edges[i]:edges[i+1]].sum() for i in range(bands)], dtype=np.float32)

# Quantum-inspired mixer: y = F^-1( e^{iθ} ⊙ F x ), stacked L times, with a light permutation
def qi_qft_signature(x, theme_id: int, layers=2, bands=48):
    d = x.size
    # deterministic phase schedule per theme_id
    rng = np.random.default_rng(theme_id * 9973 + 11)
    theta = rng.uniform(-np.pi, np.pi, size=d)
    phase = np.exp(1j * theta)

    y = x.astype(np.complex128, copy=False)
    for _ in range(layers):
        y = np.fft.ifft(np.fft.fft(y) * phase)
        # light permutation to reduce leakage (even/odd interleave + half swap)
        y = np.concatenate([y[::2], y[1::2]])[:d]
        half = d // 2
        y[:half], y[half:] = y[half:].copy(), y[:half].copy()

    spec = np.fft.fft(y)
    psd = (spec.conj() * spec).real
    edges = np.linspace(0, d, bands + 1, dtype=int)
    return np.array([psd[edges[i]:edges[i+1]].sum() for i in range(bands)], dtype=np.float32)

def cosine(a, b):
    return float(np.dot(a, b) / ((np.linalg.norm(a)*np.linalg.norm(b)) + 1e-12))

def norm_dot(a, b):
    return float(np.dot(a, b) / ((np.linalg.norm(a)*np.linalg.norm(b)) + 1e-12))

# ---------- Public API ----------
class PreRanker:
    def __init__(self, d=1024, bands=48, mixer_layers=2, gate=0.35, w_cos=0.6, w_sig=0.4):
        self.d = d
        self.bands = bands
        self.mixer_layers = mixer_layers
        self.gate = gate
        self.w_cos = w_cos
        self.w_sig = w_sig

    def build_cache_entry(self, emb: np.ndarray, theme_id: int, pos: int = 0):
        # 1) project
        x = project_pow2(emb.astype(np.float64), d=self.d)
        # 2) optional positional binding (RoPE-lite): phase the odd indices
        if pos != 0:
            d = x.size
            idx = np.arange(d//2)
            ang = pos * (10000.0 ** (-idx/(d//2)))
            z = x.astype(np.complex128)
            for k, a in enumerate(ang):
                i1 = 2*k + 1
                if i1 < d:
                    z[i1] *= (np.cos(a) + 1j*np.sin(a))
            x = (z / (np.linalg.norm(z) + 1e-12)).astype(np.complex128)
        # 3) fft bands (base)
        psd_band = fft_psd_bands(x if np.isrealobj(x) else np.real(x), bands=self.bands)
        # 4) mixer signature (small, theme-keyed)
        sig = qi_qft_signature(np.real(x), theme_id=theme_id, layers=self.mixer_layers, bands=self.bands)
        return {"proj": x, "psd": psd_band, "sig": sig}

    def score(self, q_emb, d_emb, q_sig, d_sig):
        c = cosine(q_emb, d_emb)
        s = norm_dot(q_sig, d_sig)
        return self.w_cos * c + self.w_sig * s

    def rerank(self, query, docs, theme_id: int, topk=20):
        """
        query: {"emb": np.array, "cache": {"psd","sig"}}
        docs:  list of {"id":..., "emb": np.array, "cache": {"psd","sig"}}
        """
        q_emb = query["emb"]; q_sig = query["cache"]["sig"]
        # hard gate by signature similarity
        survivors = []
        for d in docs:
            sim_sig = norm_dot(q_sig, d["cache"]["sig"])
            if sim_sig >= self.gate:
                s = self.score(q_emb, d["emb"], q_sig, d["cache"]["sig"])
                survivors.append((s, d))
        survivors.sort(key=lambda t: t[0], reverse=True)
        return survivors[:topk]
