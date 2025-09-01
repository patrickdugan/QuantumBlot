# run_layered_qft.py
import os, json, numpy as np, math, re
from pathlib import Path
from layered_qft import build_interference_circuit, counts_to_band_signature

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

    # --- connect to IBM Brisbane ---
    service = QiskitRuntimeService(
        channel="ibm_cloud",
        token=os.environ.get("IBM_CLOUD_API_KEY"),
        instance=os.environ.get("IBM_QUANTUM_CRN"),
    )
    backend = service.backend("ibm_brisbane")

    # --- build layered QFT circuit (auto-transpiled for backend) ---
    qc = build_interference_circuit(amplitudes,
                                    n_qubits=n_qubits,
                                    theme_id=2,
                                    pos=0,
                                    backend=backend)   # hardware-safe

    # --- run job ---
    sampler = Sampler(backend)
    job = sampler.run([qc], shots=8192)
    result = job.result()

    # SamplerV2 result is indexable
    record = result[0]
    qd = record.data   # QuasiDistribution mapping {BitArray: probability}

    shots = 8192
    counts = {str(k): int(float(v) * shots) for k, v in qd.items()}

    # --- save to JSON ---
    
    for idx, rec in enumerate(result):
        qd = rec.data
        counts = {str(k): int(float(v) * shots) for k, v in qd.items()}
        out_path = f"job_{job.job_id()}_pub{idx}_counts.json"
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(counts, f, indent=2)
        print(f"Saved counts for pub {idx} → {out_path}")

if __name__ == "__main__":
    main()
