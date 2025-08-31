import os, math, numpy as np
from qiskit import QuantumCircuit
from qiskit.circuit.library import QFT
from qiskit.transpiler.preset_passmanagers import generate_preset_pass_manager
from qiskit_ibm_runtime import QiskitRuntimeService, SamplerV2 as Sampler

def _normalize_pow2(vec: np.ndarray):
    v = np.asarray(vec, dtype=complex).ravel()
    if np.linalg.norm(v) == 0:
        raise ValueError("zero vector not allowed")
    v = v / np.linalg.norm(v)
    L = v.shape[0]
    n = int(math.ceil(math.log2(L)))
    size = 1 << n
    if L < size:
        w = np.zeros(size, dtype=complex)
        w[:L] = v
        v = w
    elif L > size:
        v = v[:size]
        v = v / np.linalg.norm(v)
    return v, n

def build_qft_measured(state: np.ndarray) -> QuantumCircuit:
    state_vec, n = _normalize_pow2(state)
    qc = QuantumCircuit(n, n)
    qc.initialize(state_vec.tolist(), range(n))
    qc.append(QFT(n), range(n))
    qc.measure(range(n), range(n))
    return qc

def fourier_fingerprint_runtime(
    amplitudes: np.ndarray,
    shots: int = 2048,
    backend_name: str = "ibm_brisbane",
    token: str | None = None,
    instance: str | None = None,
    optimization_level: int = 1,
):
    """
    Run QFT on an IBM backend (Sampler) and return counts.
    Requires a valid IBM Cloud API key + instance string.
    """
    # --- Auth + pick backend ---
    service = QiskitRuntimeService(
        channel="ibm_cloud",
        token=token or os.environ.get("IBM_CLOUD_API_KEY"),
        instance=instance or os.environ.get("IBM_QUANTUM_CRN"),
    )
    backend = service.backend(backend_name)
    print(f"Using IBM backend: {backend_name}")

    # --- Build & transpile ---
    qc = build_qft_measured(np.asarray(amplitudes))
    pm = generate_preset_pass_manager(backend=backend, optimization_level=optimization_level)
    isa_qc = pm.run(qc)

    # --- Run with Sampler ---
    sampler = Sampler(backend)
    job = sampler.run([isa_qc], shots=shots)
    res = job.result()
    pub = res[0]  # first circuit’s result

    # Try to get counts if available
    counts = None
    if hasattr(pub.data, "get_counts"):
        counts = pub.data.get_counts()
    elif hasattr(pub.data, "meas") and hasattr(pub.data.meas, "get_counts"):
        counts = pub.data.meas.get_counts()
    elif hasattr(pub.data, "meas"):
        # Fallback: build histogram manually
        bitstrings = [str(b) for b in pub.data.meas]
        counts = {}
        for b in bitstrings:
            counts[b] = counts.get(b, 0) + 1
    else:
        raise RuntimeError("Sampler result format not recognized")

    return counts
