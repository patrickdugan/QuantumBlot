import numpy as np
import pytest
from qctx.learned_screen import LearnedScreen, fit_basis, normalize

@pytest.mark.parametrize('d,r',[(8,1),(16,4),(64,8),(64,64),(128,32)])
@pytest.mark.parametrize('method',['random','pca','class_energy'])
def test_bounds_and_exact(d,r,method):
    rng=np.random.default_rng(d*100+r); train=rng.normal(size=(160,d)); labels=np.arange(len(train))%2
    if method=='class_energy' and r<2: pytest.skip('two classes require rank >=2')
    u=fit_basis(train,r,method,labels=labels)
    assert np.max(abs(u.T@u-np.eye(r)))<1e-12
    x=rng.normal(size=(180,d)); x[1]=x[0]; x[2]=-x[0]; x[3]=u[:,0]; x[4]=-u[:,0]
    sig=LearnedScreen(x,u,mode='signed'); eng=LearnedScreen(x,u,mode='energy'); raw=LearnedScreen(x,mode='shell')
    for q in [*rng.normal(size=(8,d)),x[0],x[3],x[4]]:
        nq=normalize(q); exact=sig.x@nq; a,_=sig.bounds(nq); b,_=eng.bounds(nq); c,_=raw.bounds(nq)
        assert np.min(a-exact)>=-1e-12 and np.min(b-exact)>=-1e-12 and np.min(c-exact)>=-1e-12
        assert np.max(a-b)<1e-12
        for k in (1,7,20):
            baseline=sig.search(q,k,screen=False)
            for ix in (sig,eng,raw):
                z=ix.search(q,k,pilot=16)
                assert z['indices']==baseline['indices']
                assert np.allclose(z['scores'],baseline['scores'],atol=1e-14)


def test_projection_decomposition():
    rng=np.random.default_rng(18); x=normalize(rng.normal(size=(32,64))); u=fit_basis(x,16)
    # Complete orthonormal basis and prove score/energy identities by direct transform.
    v=np.linalg.qr(u,mode='complete')[0]; y=x@v
    assert np.allclose(y@y.T,x@x.T,atol=1e-13)
    # Periodic folding is an orthogonal transform. Inverse in low-to-high block order.
    z=y[:,:1].copy()
    for j in range(6):
        w=1<<j; diff=y[:,w:2*w]; z=np.concatenate([z+diff,z-diff],axis=1)/np.sqrt(2)
    from qctx.shell_spectrum import shell_probabilities
    mass=shell_probabilities(z)
    blocks=[y[:,0]**2]+[np.sum(y[:,1<<j:2<<j]**2,axis=1) for j in range(6)]
    assert np.allclose(mass,np.stack(blocks,axis=1),atol=1e-13)


def test_validation():
    for a in ([0,0],[1,np.nan], [1+1j,2]):
        with pytest.raises(ValueError): normalize(a)
    with pytest.raises(ValueError): LearnedScreen(np.eye(4),np.ones((4,2)))
    with pytest.raises(ValueError): fit_basis(np.eye(4),2,'class_energy')
    a=LearnedScreen(np.eye(4),np.eye(4)[:,:2])
    with pytest.raises(ValueError): a.search([1,2,3])
