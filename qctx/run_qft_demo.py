# run_qft_demo.py
import os, json, numpy as np
from qft_pca import fourier_fingerprint_runtime, _normalize_pow2
# --- load env from qblot.env ---
from pathlib import Path
import re, os

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


def main():
    # --- load one of your prepped sparse vectors ---
    vecs = np.load("vectors_pca_topk.npy")
    print(f"Loaded sparse PCA vectors: {vecs.shape}")

    # pick the first one (id=0) just to test
    amplitudes = vecs[0]

    # --- check qubit count ---
    _, n_qubits = _normalize_pow2(amplitudes)
    print(f"Prepared state → {len(amplitudes)} components → {n_qubits} qubits")

    # --- run the QFT fingerprint ---
    counts = fourier_fingerprint_runtime(
        amplitudes,
        shots=1024,                    # bump to 2048 if budget allows
        backend_name="ibm_brisbane",     # or any available backend
        token=os.environ.get("IBM_CLOUD_API_KEY"),
        instance=os.environ.get("IBM_QUANTUM_CRN"),
    )

    # --- save to JSON for post-processing ---
    out_path = "qft_counts_demo.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(counts, f, indent=2)
    print(f"Saved QFT counts → {out_path}")

    # quick peek at top 20
    for bitstr, cnt in sorted(counts.items(), key=lambda kv: kv[1], reverse=True)[:20]:
        print(f"{bitstr}: {cnt}")

if __name__ == "__main__":
    main()
