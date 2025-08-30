import os
import numpy as np
from dotenv import load_dotenv
from qiskit_ibm_runtime import QiskitRuntimeService
from qiskit import QuantumCircuit, transpile
from qiskit_aer import Aer
from qiskit_aer.primitives import Sampler

def fourier_fingerprint_runtime(amplitudes, shots=1024, backend=None, backend_name=None):
    """Run a QFT on the input amplitudes and return measurement counts."""
    from qiskit import QuantumCircuit
    from qiskit.circuit.library import QFT

    n = int(np.log2(len(amplitudes)))
    qc = QuantumCircuit(n)
    qc.initialize(amplitudes, range(n))
    qc.append(QFT(num_qubits=n, do_swaps=True).to_gate(), range(n))
    qc.measure_all()

    if backend is None and backend_name is None:
        backend = Aer.get_backend("aer_simulator")
    elif backend is None and backend_name is not None:
        from qiskit_ibm_runtime import QiskitRuntimeService
        service = QiskitRuntimeService()
        backend = service.backend(backend_name)

    tqc = transpile(qc, backend)
    job = backend.run(tqc, shots=shots)
    result = job.result()
    return result.get_counts()


# Load IBM Cloud credentials from qblot.env
load_dotenv("qblot.env")

service = QiskitRuntimeService(
    channel="ibm_cloud",
    token=os.environ.get("IBM_CLOUD_API_KEY"),
    instance=os.environ.get("IBM_QUANTUM_CRN"),
)

# --- Load reduced PCA vector ---
vec = np.load("vectors_pca_topk.npy")   # <-- replace with your PCA output
n = len(vec)

# Ensure power-of-two length (pad with zeros if needed)
pow2 = 1 << (n - 1).bit_length()
if vec.ndim == 1:
    vec = np.pad(vec, (0, pow2 - len(vec)), mode="constant")
elif vec.ndim == 2:
    # Pad only the first axis (rows), not columns
    pad_len = pow2 - vec.shape[0]
    vec = np.pad(vec, ((0, pad_len), (0, 0)), mode="constant")


# Normalize amplitudes
amplitudes = vec / np.linalg.norm(vec)

# --- Run on IBM backend ---
counts = fourier_fingerprint_runtime(
    amplitudes,
    shots=2048,
    backend_name="ibm_brisbane",   # or any available backend in your account
)

print(counts)
