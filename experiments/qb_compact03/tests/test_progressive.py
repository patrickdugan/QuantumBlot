import numpy as np
import pytest
from qctx.learned_screen import fit_basis, normalize
from qctx.progressive_screen import ProgressiveScreen


@pytest.mark.parametrize('d,ranks',[(1,(1,)),(8,(1,2,4,8)),(64,(4,16,32)),(128,(16,32,64,128))])
@pytest.mark.parametrize('structured',[False,True])
def test_bound_and_rank(d,ranks,structured):
    rng=np.random.default_rng(3010+d); tr=rng.normal(size=(180,d)); x=rng.normal(size=(240,d))
    if structured:
        u=np.linalg.qr(rng.normal(size=(d,d)))[0]
        x=(x*np.geomspace(1,.002,d))@u.T
        tr=(tr*np.geomspace(1,.002,d))@u.T
    u=fit_basis(tr,ranks[-1]);x[1]=x[0];x[2]=-x[0];x[3]=u[:,0];x[4]=-u[:,0]
    index=ProgressiveScreen(x,u,ranks)
    for q in [*rng.normal(size=(9,d)),x[0],x[2],x[3]]:
        actual=index.x@normalize(q)
        bounds=index.prefix_bounds(q)
        for lo,hi in bounds:
            assert np.min(actual-lo)>=-1e-11 and np.min(hi-actual)>=-1e-11
        for (a,b),(c,e) in zip(bounds,bounds[1:]):
            assert np.max(e-b)<1e-10 and np.max(a-c)<1e-10
        for k in (1,7,20,240):
            truth=index.search(q,k,screen=False)
            for early in (0,.05):
                for lower in (False,True):
                    z=index.search(q,k,pilot=16,early_fraction=early,lower_bound=lower)
                    assert z['indices']==truth['indices']
                    assert np.allclose(z['scores'],truth['scores'],atol=1e-14)
                    assert z['unique_evaluated']<=len(x)
                    assert z['coordinate_work']==z['projected_coordinates']+z['query_coordinates']+z['evaluated']*d
                    assert sum(s['width']*s['block_rows'] for s in z['stages'])==z['projected_coordinates']
                    assert all(b['active_before']==a['active_after'] for a,b in zip(z['stages'],z['stages'][1:]))


def test_validation_and_extremes():
    for ranks in ((),(0,),(2,2),(3,1),(5,)):
        with pytest.raises(ValueError): ProgressiveScreen(np.eye(4),np.eye(4),ranks)
    with pytest.raises(ValueError): ProgressiveScreen(np.eye(4),np.ones((4,2)),(2,))
    with pytest.raises(ValueError): ProgressiveScreen(np.eye(4),np.eye(4)[:,:2]+1j,(2,))
    x=np.array([[1e300,1e-300],[1e-300,1e300],[-1e300,0.]])
    a=ProgressiveScreen(x,np.eye(2),(1,2))
    assert a.search(x[0],3)['indices']==[0,1,2]
    for q in ([0.,0.],[np.nan,0.],[1.],[1+1j,0.]):
        with pytest.raises(ValueError): a.search(q)
    for k in (0,4):
        with pytest.raises(ValueError): a.search(x[0],k)
