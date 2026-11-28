from pathlib import Path
import sys
import numpy as np
import pytest
sys.path.insert(0,str(Path(__file__).resolve().parents[3]/'newtype/qft-runner'))
from spectral_bin import inject_conjugate_pair, selected_bin


@pytest.mark.parametrize('T',[3,4,5,16,31,128,1024])
@pytest.mark.parametrize('complex_input',[False,True])
def test_equivalent(T,complex_input):
    r=np.random.default_rng(15+T);x=r.normal(size=(T,16));s=r.normal(size=16)+1j*r.normal(size=16)
    if complex_input:x=x+1j*r.normal(size=x.shape)
    for f in sorted({0,1,T//2,T-1}):
        a=np.fft.fft(x,axis=0);assert np.allclose(selected_bin(x,f),a[f],atol=2e-12,rtol=2e-12)
        a[f]+=s;a[-f%T]+=s.conj();want=np.fft.ifft(a,axis=0).real
        got=inject_conjugate_pair(x,f,s)
        assert np.allclose(got,want,atol=1e-12,rtol=1e-12)
