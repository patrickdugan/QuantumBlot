"""Exact measured-output reduction for the legacy uniform-input BifocalQFT.

Only valid for the existing fixed circuit, computational-basis readout, and
unobserved shadow register. It does not load text or preserve shadow coherence.
"""
from __future__ import annotations
import numpy as np


def masks(n_coarse, n_fine, codec):
    codec = getattr(codec, 'value', codec)
    if codec not in ('temporal', 'conceptual', 'genealogical', 'narrative'):
        raise ValueError('unknown codec')
    if not isinstance(n_coarse, int) or not isinstance(n_fine, int) or min(n_coarse, n_fine) < 1:
        raise ValueError('positive integer widths required')
    fine = ((1 << min(n_fine, 4))-1) << n_coarse
    return ((1 << n_coarse)-1, fine) if codec == 'temporal' else (fine,)


def distribution(n_coarse, n_fine, codec):
    states = [0]
    for m in masks(n_coarse, n_fine, codec):
        states += [x ^ m for x in states]
    width = n_coarse+n_fine
    return {format(x, f'0{width}b'): 1/len(states) for x in states}


def sample_counts(n_coarse, n_fine, codec, shots=1024, *, seed=None):
    if not isinstance(shots, int) or shots < 1:
        raise ValueError('positive integer shots required')
    p = distribution(n_coarse, n_fine, codec)
    draws = np.random.default_rng(seed).multinomial(shots, list(p.values()))
    return {k:int(v) for k,v in zip(p, draws) if v}


def build_compact_circuit(n_coarse, n_fine, codec):
    """No shadow qubits, QFTs or controlled phase gates. Same ideal counts law."""
    from qiskit import QuantumCircuit
    ms = masks(n_coarse, n_fine, codec); n = n_coarse+n_fine
    qc = QuantumCircuit(n, n)
    for mask in ms:
        bits = [i for i in range(n) if mask >> i & 1]
        qc.h(bits[0])
        for i in bits[1:]:
            qc.cx(bits[0], i)
    qc.measure(range(n), range(n))
    return qc
