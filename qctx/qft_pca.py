from __future__ import annotations
import numpy as np
from qiskit import QuantumCircuit, transpile
from qiskit.circuit.library import QFT
try:
    from qiskit_aer import Aer
    _backend=Aer.get_backend('qasm_simulator')
except Exception:
    from qiskit import Aer
    _backend=Aer.get_backend('qasm_simulator')

def _normalize_pow2(vec: np.ndarray):
    v=np.asarray(vec,dtype=complex).ravel()
    if np.linalg.norm(v)==0: raise ValueError('zero vector not allowed')
    v=v/np.linalg.norm(v); L=v.shape[0]
    n=int(np.ceil(np.log2(L))); size=1<<n
    if L<size:
        w=np.zeros(size,dtype=complex); w[:L]=v; v=w
    elif L>size:
        v=v[:size]; v=v/np.linalg.norm(v)
    return v,n

def build_qft_circuit(state: np.ndarray)->QuantumCircuit:
    state_vec,n=_normalize_pow2(state)
    qc=QuantumCircuit(n,n)
    qc.initialize(state_vec.tolist(),list(range(n)))
    qc.append(QFT(n),list(range(n)))
    qc.measure(range(n),range(n))
    return qc

def fourier_fingerprint(vector: np.ndarray, shots:int=1024):
    qc=build_qft_circuit(vector)
    compiled=transpile(qc,_backend)
    job=_backend.run(compiled,shots=shots)
    res=job.result()
    return res.get_counts(qc)
