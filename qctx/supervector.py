import numpy as np
from qiskit import QuantumCircuit
from qiskit.circuit.library import QFT

# ----- make supervector -----
def random_project(x, out_dim, seed=0):
    rng = np.random.default_rng(seed)
    R = rng.normal(size=(x.shape[-1], out_dim)).astype(np.float32)
    R /= np.linalg.norm(R, axis=0, keepdims=True) + 1e-8
    y = x @ R
    return y

def to_pow2(v):
    D = len(v)
    n = int(np.ceil(np.log2(D)))
    pad = (1<<n) - D
    if pad: v = np.concatenate([v, np.zeros(pad, dtype=np.float64)])
    v = v / (np.linalg.norm(v) + 1e-12)
    return v, n

# ----- amplitude-encode + QFT pass -----
def amp_encode_qft(state_vec):
    # NOTE: in practice use a state-preparation routine; this is conceptual
    v, n = to_pow2(state_vec)
    qc = QuantumCircuit(n)
    # Replace with synthesis for Initialize(v) if available in your toolchain.
    # For brevity, pretend we have an 'Initialize' (older qiskit had it):
    # from qiskit.circuit.library import Initialize
    # qc.append(Initialize(v), qc.qubits)
    # Then apply QFT or your analysis circuit:
    qc.append(QFT(num_qubits=n, do_swaps=False), qc.qubits)
    return qc, n
