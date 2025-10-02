import numpy as np
from qiskit import QuantumCircuit, transpile
from qiskit.circuit.library import QFT, StatePreparation

def build_interference_circuit(vec,
                               n_qubits,
                               theme_id: int = 0,
                               pos: int = 0,
                               base: float = 10000.0,
                               backend=None,
                               use_stateprep: bool = False):
    """
    Build a layered QFT circuit, Qiskit 1.x style.
    """
    # Create circuit with n_qubits and matching classical bits
    qc = QuantumCircuit(n_qubits, n_qubits)

    # 0. Pad/project input
    d = 1 << n_qubits
    x = np.zeros(d, dtype=np.complex128)
    v = np.array(vec, dtype=np.complex128)
    L = min(len(v), d)
    x[:L] = v
    x /= np.linalg.norm(x) + 1e-12

    # 1. RoPE phase on odd indices
    idx = np.arange(d // 2)
    theta = pos * (base ** (-idx / (d // 2)))
    for k, ang in enumerate(theta):
        i1 = 2 * k + 1
        if i1 < L:
            x[i1] *= np.cos(ang) + 1j * np.sin(ang)

    # 2. Encode input
    if use_stateprep and backend is None:
        qc.append(StatePreparation(x), range(n_qubits))
    else:
        for i, amp in enumerate(x[:n_qubits]):
            qc.ry(float(np.real(amp)) * np.pi, i)

    # 3. Spread with Hadamards
    qc.h(range(n_qubits))

    # 4. QFT
    qft = QFT(num_qubits=n_qubits, do_swaps=False, approximation_degree=0)
    qc.append(qft, range(n_qubits))

    # 5. Theme-gated RZ phases
    rng = np.random.default_rng(theme_id * 991)
    for i in range(n_qubits):
        angle = float(rng.uniform(-np.pi, np.pi)) * 0.25
        qc.rz(angle, i)

    # 6. Inverse QFT
    iqft = QFT(num_qubits=n_qubits, inverse=True, do_swaps=False, approximation_degree=0)
    qc.append(iqft, range(n_qubits))

    # 7. Measure
    qc.measure(range(n_qubits), range(n_qubits))

    # --- Transpile if backend provided ---
    if backend is not None:
        qc = transpile(qc, backend=backend, optimization_level=1)

    return qc
