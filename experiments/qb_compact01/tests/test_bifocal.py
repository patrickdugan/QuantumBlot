from pathlib import Path
import sys
import numpy as np
import pytest
sys.path.insert(0,str(Path(__file__).resolve().parents[3]/'newtype/bifocal-QFT'))
from bifocal_compact import distribution, sample_counts


def original_reference(c,f,codec):
    # Independent statevector simulation of the pinned original gate sequence.
    # Basis bit i is Qiskit wire i. QFT acts separately on coarse/fine blocks.
    n=c+f+4; d=1<<n; ids=np.arange(d); a=np.ones(d,dtype=complex)/np.sqrt(d)
    def qft(a,offset,width):
        z=a.reshape([2]*n); axes=list(range(n-offset-width,n-offset)); other=[i for i in range(n) if i not in axes]; perm=other+axes
        zz=z.transpose(perm).reshape(-1,1<<width); zz=np.fft.ifft(zz,axis=-1,norm='ortho')
        return zz.reshape([2]*n).transpose(np.argsort(perm)).ravel()
    def phase(a,wire,theta,ctrl=None):
        hit=(ids>>wire&1)!=0
        if ctrl is not None:hit &= (ids>>ctrl&1)!=0
        a[hit]*=np.exp(1j*theta);return a
    def cx(a,ctrl,target):return a[ids^(((ids>>ctrl)&1)<<target)]
    a=qft(a,0,c)
    if codec in ('temporal','conceptual'):
        for i in range(c):a=phase(a,i,(-1 if codec=='conceptual' else 1)*np.pi*i/c)
    elif codec=='genealogical':
        for i in range(c-1):a=phase(a,i+1,np.pi/4,i)
    elif codec=='narrative':
        for i in range(0,c-1,2):a=cx(a,i,i+1)
    a=qft(a,c,f)
    if codec=='temporal':
        for i in range(c):a=cx(a,c+f,i)
    for i in range(min(f,4)):a=cx(a,c+f+1,c+i)
    if codec=='genealogical':
        for i in range(c):a=phase(a,i,np.pi,c+f+2)
    if codec=='conceptual':
        for i in range(f):a=phase(a,c+i,np.pi*i/f,c+f+3)
    return np.sum(abs(a.reshape(16,1<<(c+f)))**2,axis=0)


@pytest.mark.parametrize('codec',['temporal','conceptual','genealogical','narrative'])
@pytest.mark.parametrize('c,f',[(1,1),(2,3),(3,2),(3,5),(8,8)])
def test_original_equivalence(c,f,codec):
    r=original_reference(c,f,codec); p=distribution(c,f,codec); q=np.zeros(len(r))
    for k,v in p.items():q[int(k,2)]=v
    assert np.max(abs(r-q))<1e-12
    s=sample_counts(c,f,codec,shots=1024,seed=19)
    assert set(s)<=set(p) and sum(s.values())==1024


def test_qiskit_optional():
    pytest.importorskip('qiskit');pytest.importorskip('qiskit_aer')
    from qiskit.quantum_info import Statevector
    from bifocal_qft import BifocalQFT,BifocalConfig,CodecType
    from bifocal_compact import build_compact_circuit
    for codec in CodecType:
        cfg=BifocalConfig(n_coarse=2,n_fine=3,n_shadow=4,codec=codec)
        original=BifocalQFT(cfg).build_circuit().remove_final_measurements(inplace=False)
        p=Statevector.from_instruction(original).probabilities(qargs=list(range(5)))
        compact=build_compact_circuit(2,3,codec).remove_final_measurements(inplace=False)
        q=Statevector.from_instruction(compact).probabilities()
        assert np.allclose(p,q,atol=1e-12)
