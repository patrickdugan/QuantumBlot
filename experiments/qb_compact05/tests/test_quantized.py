import numpy as np
import pytest
from qctx.learned_screen import fit_basis,normalize
from qctx.batch_screen import BatchScreen
from qctx.quantized_screen import QuantizedBatchScreen,up32,gamma


@pytest.mark.parametrize('storage',['float32','int16','int8'])
@pytest.mark.parametrize('d,ranks',[(8,(2,4,8)),(64,(8,16,32)),(128,(16,32,64,128))])
def test_enclosure_and_search(storage,d,ranks):
    rng=np.random.default_rng(5983+d);x=rng.normal(size=(177,d));tr=rng.normal(size=(200,d))
    # Strongly varying singular values; include exact duplicates, negatives,
    # nearly all projected energy, and an exactly zero projection direction.
    x[:80]*=np.geomspace(1,.00001,d);tr*=np.geomspace(1,.00001,d)
    u=fit_basis(tr,ranks[-1]);x[1]=x[0];x[2]=-x[0];x[3]=u[:,0];x[4]=-x[3]
    q=np.vstack([x[0],x[2],x[3],x[4],rng.normal(size=(5,d))]);ref=BatchScreen(x,u,ranks)
    ix=QuantizedBatchScreen(x,u,ranks,storage=storage,build_rows=31)
    actual=ix.x@normalize(q).T
    for r in ranks:
        for a,l,h in ix.bounds(q,r,tile_rows=37):
            assert np.all(actual[a:a+len(l)]>=l) and np.all(actual[a:a+len(l)]<=h)
        # Verify the narrower quantization-only enclosure against FP64 projected products.
        pp=ix._query_projection(normalize(q),r);s=ix._projection_panel(0,len(x),r,pp)
        rad=ix._uncertainty(0,len(x),ranks.index(r),pp)-ix.res[:,ranks.index(r),None].astype(float)*pp[1][None,:]
        direct=(ix.x@ix.u[:,:r])@(normalize(q)@ix.u[:,:r]).T
        assert np.all(abs(s-direct)<=rad)
        for k in (1,7,20,len(x)):
            truth=ref.search(q,k,mode='full',rank=r,tile_rows=177)
            for mode in ('auto','lower','gather','union'):
                z=ix.search(q,k,rank=r,mode=mode,tile_rows=43,query_block=4,group=3,pilot=10)
                assert z['indices']==truth['indices']
                assert np.allclose(z['scores'],truth['scores'],atol=1e-13,rtol=1e-13)
                assert z['stats']['max_score_cells']<=43*4
    assert ix.index_bytes<ref.index_bytes


@pytest.mark.parametrize('storage',['float32','int16','int8'])
def test_zero_projection(storage):
    x=np.array([[1.,0,0,0],[0,0,1.,0],[0,0,-1.,0],[1e-300,0,1e300,0],[-1,0,0,0]])
    u=np.eye(4)[:,:2];ix=QuantizedBatchScreen(x,u,(1,2),storage=storage)
    for r in (1,2):
        for a,l,h in ix.bounds(x,r):
            dot=ix.x@normalize(x).T
            assert np.all(l<=dot) and np.all(dot<=h)
        a=ix.search(x,3,rank=r,mode='lower',tile_rows=2)
        b=ix.search(x,3,rank=r,mode='full')
        assert a['indices']==b['indices']


def test_radius_rounding_and_validation():
    x=np.array([0,np.nextafter(1.,np.inf),1e-50,1e-10,1e-2,1.])
    assert np.all(up32(x).astype(float)>=x)
    assert gamma(128,np.float32)>128*2**-24
    for u,r in ((np.ones((4,2)),(2,)),(np.eye(4),(4,2)),(np.eye(4)+1j,(4,))):
        with pytest.raises(ValueError):QuantizedBatchScreen(np.eye(4),u,r)
    with pytest.raises(ValueError):QuantizedBatchScreen(np.eye(4),np.eye(4),(4,),storage='uint8')


def test_native():
    import os
    from qctx.quantized_screen import NativeInt8Screen
    lib=os.environ.get('QB_INT8_LIBRARY')
    if not lib:pytest.skip('set QB_INT8_LIBRARY to the compiled local kernel')
    rng=np.random.default_rng(94);d=128;rs=(16,32,64,128);x=rng.normal(size=(521,d));q=rng.normal(size=(13,d));x[1]=x[0];q[0]=x[0]
    u=fit_basis(rng.normal(size=(200,d)),128);ix=NativeInt8Screen(x,u,lib,rs)
    nq=normalize(q);actual=ix.x@nq.T
    for r in rs:
        p=ix._query_projection(nq,r);a=ix._projection_panel(0,len(x),r,p)
        b=(ix.c[:,:r].astype(np.int64)@p[0].astype(np.int64).T).astype(float)*ix.scales[:,None]*p[5][None,:]
        assert np.array_equal(a,b)
        for start,lo,hi in ix.bounds(q,r,tile_rows=55):
            assert np.all(lo<=actual[start:start+len(lo)]) and np.all(actual[start:start+len(lo)]<=hi)
        for mode in ('auto','lower'):
            z=ix.search(q,20,rank=r,mode=mode,tile_rows=77,query_block=4)
            assert z['indices']==ix.search(q,20,rank=r,mode='full')['indices']
