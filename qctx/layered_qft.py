from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister
from qiskit.circuit.library import QFT, StatePreparation
import numpy as np

def build_interference_circuit(vec, n_qubits, theme_id=0, pos=0, base=10000.0):
    """
    vec: real/complex vector to load (len must be <= 2**n_qubits)
    n_qubits: size of register (e.g. 9 for 512 amps)
    theme_id: int, used to seed a deterministic phase schedule
    pos: chunk position (for RoPE)
    """
    qr = QuantumRegister(n_qubits, "q")
    cr = ClassicalRegister(n_qubits, "c")
    qc = QuantumCircuit(qr, cr)

    # -- 0. Project to length 2^n
    d = 1 << n_qubits
    x = np.zeros(d, dtype=np.complex128)
    v = np.array(vec, dtype=np.complex128)
    L = min(len(v), d)
    x[:L] = v
    x /= np.linalg.norm(x) + 1e-12

    # -- 1. RoPE-style phase on odd indices (position binding)
    idx = np.arange(d//2)
    theta = pos * (base ** (-idx/(d//2)))
    for k, ang in enumerate(theta):
        i1 = 2*k+1
        if i1 < L:
            x[i1] *= np.cos(ang) + 1j*np.sin(ang)

    # -- 2. Load amplitudes
    qc.append(StatePreparation(x), qr)

    # -- 3. Spread with Hadamards
    qc.h(qr)

    # -- 4. QFT (no swaps, so decoding is consistent)
    qft = QFT(num_qubits=n_qubits, do_swaps=False, approximation_degree=0)
    qc.append(qft, qr)

    # -- 5. Theme-gated phases
    rng = np.random.default_rng(theme_id*991)
    for i in range(n_qubits):
        angle = float(rng.uniform(-np.pi, np.pi)) * 0.25
        qc.rz(angle, qr[i])

    # -- 6. Inverse QFT
    iqft = QFT(num_qubits=n_qubits, inverse=True, do_swaps=False, approximation_degree=0)
    qc.append(iqft, qr)

    # -- 7. Measure
    qc.measure(qr, cr)
    return qc
