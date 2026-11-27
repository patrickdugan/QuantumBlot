import numpy as np
import pytest
from qctx.rank_policy import fit_policy, retained_energy


def test_stump_and_guard():
    e=np.linspace(0,1,40)
    p=fit_policy(e,{'low_rank':np.where(e>.5,1.,8.),'high_rank':np.where(e>.5,6.,2.),'full':np.full(40,5.)},gate_ms=.1,min_leaf=8)
    assert p.threshold==e[20] and p.choose(.3)=='high_rank' and p.choose(.7)=='low_rank'
    assert p.development_mean_ms<p.static_mean_ms
    q=fit_policy(e,{'full':np.ones(40),'rank':np.full(40,1.01)},gate_ms=.1)
    assert q.threshold is None and q.choose(.5)=='full'
    assert retained_energy([1,0],np.eye(2)[:,:1])==1
    with pytest.raises(ValueError):fit_policy(e,{'full':np.full(40,-1.)})


def test_adjacent_float_split():
    a=1.; b=np.nextafter(a,np.inf)
    e=np.array([a]*8+[b]*8)
    p=fit_policy(e,{'a':np.array([1.]*8+[5.]*8),'b':np.array([5.]*8+[1.]*8)},min_leaf=8)
    assert p.threshold==b and p.choose(a)=='a' and p.choose(b)=='b'
