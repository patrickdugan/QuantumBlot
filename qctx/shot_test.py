
from qiskit_aer import Aer
import numpy as np
from qiskit import QuantumCircuit
from qiskit import transpile

# --- Normalization helper ---
def normalize(vec: np.ndarray) -> np.ndarray:
    vec = np.array(vec, dtype=np.complex128)
    norm = np.linalg.norm(vec)
    if norm == 0:
        raise ValueError("Amplitude vector is zero; cannot normalize.")
    # re-scale and force exact normalization
    vec = vec / norm
    return vec / np.linalg.norm(vec)

# --- Example Fourier fingerprint runtime ---
def fourier_fingerprint_runtime(amplitudes: np.ndarray, shots: int = 1024):
    n = int(np.ceil(np.log2(len(amplitudes))))  # number of qubits needed
    pow2 = 2 ** n

    # pad or truncate to power of 2
    if len(amplitudes) < pow2:
        amplitudes = np.pad(amplitudes, (0, pow2 - len(amplitudes)), mode="constant")
    elif len(amplitudes) > pow2:
        amplitudes = amplitudes[:pow2]

    # normalize amplitudes
    amplitudes = normalize(amplitudes)

    # build circuit
    qc = QuantumCircuit(n)
    qc.initialize(amplitudes, range(n))
    qc.h(range(n))
    qc.measure_all()

    # run locally
    backend = Aer.get_backend("qasm_simulator")
    tqc = transpile(qc, backend)
    result = backend.run(tqc, shots=shots).result()
    counts = result.get_counts()
    return counts

if __name__ == "__main__":
    # example small vector
    vec = np.random.rand(20).astype(np.float32)
    counts = fourier_fingerprint_runtime(vec, shots=1024)
    print(counts)
