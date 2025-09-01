"""
Layered QFT circuit builder + helpers.

Pipeline:
  1. Pad input to 2^n
  2. RoPE-style positional phase (odd indices)
  3. Encode input (StatePreparation on simulators, angle encoding on hardware)
  4. Hadamards (spread amplitudes)
  5. QFT (no swaps)
  6. Theme-gated RZ rotations
  7. Inverse QFT (no swaps)
  8. Measure
"""

import numpy as np
from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister, transpile
from qiskit.circuit.library import QFT, StatePreparation

# ---------------------------------------------------------------------
# Circuit builder
# ---------------------------------------------------------------------
def build_interference_circuit(vec,
                               n_qubits,
                               theme_id: int = 0,
                               pos: int = 0,
                               base: float = 10000.0,
                               backend=None,
                               use_stateprep: bool = False):
    """
    Build a layered QFT circuit.

    Args:
        vec: Input vector (list/ndarray), will be padded/truncated to 2^n.
        n_qubits: Number of qubits.
        theme_id: Int seed for theme-specific RZ phase schedule.
        pos: Position index (for RoPE binding).
        base: RoPE base constant.
        backend: Optional Qiskit backend — if provided, transpile for hardware.
        use_stateprep: Force StatePreparation (only works on simulators).

    Returns:
        QuantumCircuit (transpiled if backend is provided).
    """
    qr = QuantumRegister(n_qubits, "q")
    cr = ClassicalRegister(n_qubits, "c")
    qc = QuantumCircuit(qr, cr)

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
        # Simulator path — exact amplitude encoding
        qc.append(StatePreparation(x), qr)
    else:
        # Hardware path — angle encoding (safe)
        for i, amp in enumerate(x[:n_qubits]):
            qc.ry(float(np.real(amp)) * np.pi, qr[i])

    # 3. Spread with Hadamards
    qc.h(qr)

    # 4. QFT
    qft = QFT(num_qubits=n_qubits, do_swaps=False, approximation_degree=0)
    qc.append(qft, qr)

    # 5. Theme-gated RZ phases
    rng = np.random.default_rng(theme_id * 991)
    for i in range(n_qubits):
        angle = float(rng.uniform(-np.pi, np.pi)) * 0.25
        qc.rz(angle, qr[i])

    # 6. Inverse QFT
    iqft = QFT(num_qubits=n_qubits, inverse=True, do_swaps=False, approximation_degree=0)
    qc.append(iqft, qr)

    # 7. Measure
    qc.measure(qr, cr)

    # --- Transpile if backend provided ---
    if backend is not None:
        qc = transpile(qc, backend=backend, optimization_level=1)

    return qc

# ---------------------------------------------------------------------
# Decoders
# ---------------------------------------------------------------------
def bit_reverse_indices(n: int):
    """Return indices for bit reversal on n-bit strings."""
    d = 1 << n
    return np.array([int(bin(i)[2:].zfill(n)[::-1], 2) for i in range(d)])

def counts_to_band_signature(counts: dict, n_qubits: int, bands: int = 48, reverse_bits: bool = False):
    """
    Convert counts dict → probability vector → FFT PSD bands.

    Args:
        counts: dict {bitstring: shots}
        n_qubits: number of qubits
        bands: number of frequency bands to reduce into
        reverse_bits: whether to bit-reverse order (matches QFT do_swaps=False)

    Returns:
        np.ndarray of length = bands
    """
    d = 1 << n_qubits
    p = np.zeros(d, dtype=np.float64)
    total = 0
    for bstr, c in counts.items():
        try:
            i = int(bstr, 2)
            p[i] += c; total += c
        except:
            continue
    p /= (total + 1e-12)

    if reverse_bits:
        p = p[bit_reverse_indices(n_qubits)]

    spec = np.fft.fft(p)
    psd = (spec.conj() * spec).real
    edges = np.linspace(0, psd.size, bands + 1, dtype=int)
    return np.array([psd[edges[i]:edges[i+1]].sum() for i in range(bands)], dtype=np.float32)
