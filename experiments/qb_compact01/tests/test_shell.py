from pathlib import Path
import sys
import numpy as np
import pytest
sys.path.insert(0, str(Path(__file__).resolve().parents[3]/'qctx'))
from shell_spectrum import shell_probabilities, shell_coordinates, shell_transform, counts_shell_probabilities, ShellIndex


def labels(n):
    b = n.bit_length()-1
    return np.array([b-(k & -k).bit_length()+1 if k else 0 for k in range(n)])


def fft_ref(x):
    f = np.fft.fft(x, norm='ortho'); z = abs(f)**2
    return np.stack([np.sum(z[..., labels(x.shape[-1])==j], axis=-1) for j in range(x.shape[-1].bit_length())], axis=-1)/np.sum(z, axis=-1, keepdims=True)


@pytest.mark.parametrize('b', range(0,13))
@pytest.mark.parametrize('complex_input', [False, True])
def test_shell_fft(b, complex_input):
    r = np.random.default_rng(1900+b); n=1<<b
    x = r.normal(size=(7,n)); x = x+1j*r.normal(size=x.shape) if complex_input else x
    assert np.allclose(shell_probabilities(x),fft_ref(x),atol=2e-14,rtol=2e-14)
    assert np.allclose(shell_probabilities(x).sum(axis=-1),1)
    for x1 in (np.ones(n),np.eye(1,n)[0]):
        assert np.allclose(shell_probabilities(x1),fft_ref(x1))


@pytest.mark.parametrize('b',range(0,11))
def test_shell_projection(b):
    n=1<<b; l=labels(n); w=np.bincount(l); rng=np.random.default_rng(2200+b)
    x=rng.normal(size=(4,n))+1j*rng.normal(size=(4,n)); x/=np.linalg.norm(x,axis=-1,keepdims=True)
    c,res=shell_coordinates(x); y=c[...,l]/np.sqrt(w[l]); f=np.fft.fft(y,norm='ortho'); out=shell_transform(c)
    assert np.allclose(out[...,l]/np.sqrt(w[l]),f,atol=2e-14)
    assert np.allclose(res,np.sum(abs(x-y)**2,axis=-1))
    assert np.allclose(np.sum(abs(c)**2,axis=-1)+res,1)
    assert np.allclose(shell_transform(shell_transform(c)),c)
    for xx,yy,ee in zip(x,y,res):
        if np.linalg.norm(yy)>1e-10:
            tv=0.5*np.sum(abs(abs(np.fft.fft(xx,norm='ortho'))**2-abs(np.fft.fft(yy/np.linalg.norm(yy),norm='ortho'))**2))
            assert tv<=np.sqrt(ee)+1e-12


def test_counts_and_validation():
    r=np.random.default_rng(3)
    for b in range(1,12):
        d=1<<b; c=r.integers(0,50,size=d); labels_=labels(d)
        for reverse in (False,True):
            cs={format(i,f'0{b}b')[::-1] if reverse else format(i,f'0{b}b'):int(v) for i,v in enumerate(c) if v}
            assert np.allclose(counts_shell_probabilities(cs,b,reverse_bits=reverse),np.bincount(labels_,weights=c)/sum(c))
    for x in ([0,0],[1,2,3],[1,float('nan')],[],[1,float('inf')]):
        with pytest.raises(ValueError):shell_probabilities(x)
    for c in ({'00':-1},{'0':1},{'xx':1},{'00':0},{'00':1.5}):
        with pytest.raises(ValueError):counts_shell_probabilities(c,2)
    assert np.allclose(shell_probabilities([1e300,1e-300]),[.5,.5])


@pytest.mark.parametrize('structured',[False,True])
@pytest.mark.parametrize('complex_input',[False,True])
def test_search(structured,complex_input):
    r=np.random.default_rng(882); n=256; x=r.normal(size=(300,n)); q=r.normal(size=(12,n))
    if structured:
        l=labels(n); f=np.fft.fft(x,axis=-1); f*= (l[None,:]==r.integers(1,9,size=(300,1))); x=np.fft.ifft(f,axis=-1).real
        q=x[r.choice(len(x),12,replace=False)]+r.normal(scale=.0001,size=(12,n))
    if complex_input:x=x+1j*x[:,::-1]*.1; q=q+1j*q[:,::-1]*.1
    x[1]=x[0]; ix=ShellIndex(x)
    for z in [*q,x[0]]:
        for k in (1,7,20):
            a=ix.search(z,k,batch=8); b=ix.search(z,k,batch=8,screen=False)
            assert a['indices']==b['indices'] and np.allclose(a['scores'],b['scores'])
            assert a['evaluated']<=len(x)


def test_qiskit_optional():
    qiskit=pytest.importorskip('qiskit')
    from qiskit.quantum_info import Statevector
    from qiskit.circuit.library import QFTGate
    for b in range(1,6):
        n=1<<b; x=np.random.default_rng(b).normal(size=n); x/=np.linalg.norm(x)
        q=qiskit.QuantumCircuit(b);q.initialize(x,range(b));q.append(QFTGate(b),range(b))
        f=Statevector.from_instruction(q).probabilities()
        assert np.allclose(np.bincount(labels(n),weights=f),shell_probabilities(x))
