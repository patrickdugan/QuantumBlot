# run_layered_qft.py
import os, json, numpy as np
from layered_qft import build_interference_circuit
from pathlib import Path
import re

# --- load IBM Quantum env ---
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

# --- IBM Runtime imports ---
from qiskit_ibm_runtime import QiskitRuntimeService, SamplerV2 as Sampler

def main():
    # --- load one of your prepped sparse vectors ---
    vecs = np.load("vectors_pca_topk.npy")
    print(f"Loaded sparse PCA vectors: {vecs.shape}")

    # pick the first one
    amplitudes = vecs[0]
    n_qubits = int(np.ceil(np.log2(len(amplitudes))))
    print(f"Prepared {len(amplitudes)} components → {n_qubits} qubits")

    # --- build layered QFT circuit ---
    qc = build_interference_circuit(amplitudes, n_qubits=n_qubits,
                                    theme_id=2, pos=0)

    # --- connect to IBM Brisbane ---
    service = QiskitRuntimeService(
        channel="ibm_quantum",
        token=os.environ.get("IBM_CLOUD_API_KEY"),
        instance=os.environ.get("IBM_QUANTUM_CRN"),
    )
    backend = service.backend("ibm_brisbane")

    sampler = Sampler(backend)
    job = sampler.run([qc], shots=8192)
    result = job.result()
    counts = result[0].data.meas.get_counts()

    # --- save to JSON ---
    out_path = "layered_qft_counts.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(counts, f, indent=2)
    print(f"Saved layered QFT counts → {out_path}")

    # peek at top 20
    for bitstr, cnt in sorted(counts.items(), key=lambda kv: kv[1], reverse=True)[:20]:
        print(f"{bitstr}: {cnt}")

if __name__ == "__main__":
    main()
