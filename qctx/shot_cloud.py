# shot_cloud.py
import os, re, argparse
import numpy as np
from qiskit import QuantumCircuit, transpile
from qiskit.circuit.library import QFTGate
from qiskit_ibm_runtime import QiskitRuntimeService, SamplerV2 as Sampler

# ---------- env loading ----------
def load_env_file(path: str) -> None:
    if not os.path.exists(path):
        return
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            if line.startswith("export "):
                line = line[len("export "):]
            m = re.match(r'^([A-Za-z_][A-Za-z0-9_]*)\s*=\s*(.*)$', line)
            if not m:
                continue
            k, v = m.group(1), m.group(2)
            if len(v) >= 2 and v[0] == v[-1] and v[0] in {"'", '"'}:
                v = v[1:-1]
            os.environ.setdefault(k, v)

load_env_file("qblot.env")
load_env_file(".env")

# ---------- helpers ----------
def vector_to_amplitudes(vec: np.ndarray, n_qubits: int) -> np.ndarray:
    """Truncate/pad to 2^n_qubits and renormalize (float64)."""
    target_len = 1 << n_qubits
    v = np.asarray(vec, dtype=np.float64)
    if v.size >= target_len:
        v = v[:target_len]
    else:
        v = np.pad(v, (0, target_len - v.size), mode="constant")

    norm2 = float(np.dot(v, v))
    if norm2 <= 0:
        v[:] = 0.0
        v[0] = 1.0
    else:
        v /= np.sqrt(norm2)
        v /= np.sqrt(float(np.dot(v, v)))
    return v

def build_qft_circuit(vec: np.ndarray, n_qubits: int, mode: str) -> QuantumCircuit:
    qc = QuantumCircuit(n_qubits)

    if mode == "amplitude":
        amps = vector_to_amplitudes(vec, n_qubits)
        qc.initialize(amps, range(n_qubits))

    elif mode == "angles":
        # truncate to n_qubits features, map to RY rotations
        v = np.asarray(vec, dtype=np.float64)
        if v.size < n_qubits:
            v = np.pad(v, (0, n_qubits - v.size), mode="constant")
        else:
            v = v[:n_qubits]

        # normalize to [-pi, pi]
        v = np.pi * (v / np.max(np.abs(v))) if np.max(np.abs(v)) > 0 else v
        for i, angle in enumerate(v):
            qc.ry(angle, i)
    else:
        raise ValueError(f"Unknown encoding mode {mode}")

    qc.append(QFTGate(n_qubits), range(n_qubits))
    qc.measure_all()
    return qc


# ---------- args ----------
ap = argparse.ArgumentParser()
ap.add_argument("--vec", default="vectors_pca_topk.npy")
ap.add_argument("--row", type=int, default=0)
ap.add_argument("--nqubits", type=int, default=17)
ap.add_argument("--backend", default=os.environ.get("DEFAULT_BACKEND", "ibm_brisbane"))
ap.add_argument("--shots", type=int, default=int(os.environ.get("DEFAULT_SHOTS", "500")))
ap.add_argument("--encode", choices=["amplitude", "angles"], default="amplitude")
ap.add_argument("--env", default=None)
args = ap.parse_args()

if args.env:
    load_env_file(args.env)

# ---------- credentials ----------
API_KEY = os.environ.get("IBM_CLOUD_API_KEY")
INSTANCE = os.environ.get("IBM_QUANTUM_CRN")
if not API_KEY or not INSTANCE:
    raise RuntimeError("Set IBM_CLOUD_API_KEY and IBM_QUANTUM_CRN via qblot.env/.env.")

service = QiskitRuntimeService(channel="ibm_cloud", token=API_KEY, instance=INSTANCE)

# ---------- data prep ----------
vecs = np.load(args.vec)
if not (0 <= args.row < len(vecs)):
    raise IndexError(f"--row {args.row} out of range (len={len(vecs)})")
vec = vecs[args.row]

# ---------- backend + circuit ----------
backend = service.backend(args.backend)
max_q = backend.configuration().num_qubits
if args.nqubits > max_q:
    raise ValueError(f"Requested {args.nqubits} qubits; {args.backend} supports {max_q}.")

qc = build_qft_circuit(vec, args.nqubits, args.encode)
tqc = transpile(qc, backend=backend, optimization_level=1)

# ---------- Runtime primitives ----------
sampler = Sampler(mode=backend)
job = sampler.run([tqc], shots=args.shots)
print("Submitted job:", job.job_id())
res = job.result()

# Convert quasi-probabilities to counts
quasi = res[0].data.meas.get("quasi_dist", res[0].data)  # fallback if API changes
counts = {k: int(round(v * args.shots)) for k, v in quasi.items()}

for bitstr, cnt in sorted(counts.items(), key=lambda kv: kv[1], reverse=True)[:20]:
    print(f"{bitstr}: {cnt}")
