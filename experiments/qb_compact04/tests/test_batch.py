import numpy as np
import pytest
from qctx.learned_screen import fit_basis,normalize
from qctx.batch_screen import BatchScreen

@pytest.mark.parametrize('d,ranks',[(8,(2,4,8)),(64,(8,16,32)),(128,(16,32,64))])
@pytest.mark.parametrize('structured',[False,True])
def test_search(d,ranks,structured):
    rng=np.random.default_rng(319+d);x=rng.normal(size=(257,d));tr=rng.normal(size=(180,d))
    if structured:
        a=np.geomspace(1,.001,d);x*=a;tr*=a
    x[1]=x[0];x[2]=-x[0];u=fit_basis(tr,ranks[-1]);q=np.vstack([x[0],-x[0],rng.normal(size=(9,d))]);ix=BatchScreen(x,u,ranks,build_rows=43)
    for k in (1,7,20,257):
        base=ix.search(q,k,mode='full',rank=ranks[0],tile_rows=257)
        for r in ranks:
            for mode in ('full','gather','union','auto','lower'):
                a=ix.search(q,k,rank=r,mode=mode,tile_rows=53,query_block=4,group=3,pilot=10)
                assert a['indices']==base['indices']
                assert np.allclose(a['scores'],base['scores'],atol=1e-13,rtol=1e-13)
                assert a['stats']['max_score_cells']<=53*4
                assert a['stats']['coordinate_work']==a['stats']['exact_pairs']*d+a['stats']['projection_pairs']*r+a['stats']['query_coordinates']

def test_single_tile_and_dispatch():
    rng=np.random.default_rng(120);u=np.eye(16)[:,:4];x=rng.normal(size=(101,16));x[:,4:]*=.001;q=x[:9]
    ix=BatchScreen(x,u,(4,));a=ix.search(q,5,rank=4,mode='auto',tile_rows=101,query_block=5,group=3)
    b=ix.search(q,5,rank=4,mode='full');assert a['indices']==b['indices'];assert a['stats']['projection_pairs']==len(x)*len(q)
    assert a['stats']['candidate_pairs']<=len(x)*len(q)
    c=ix.search(q,5,rank=4,mode='union',tile_rows=17,query_block=4)
    assert c['stats']['projection_pairs']==2*len(x)*len(q)
    e=ix.search(q,5,rank=4,mode='lower',tile_rows=17,query_block=4)
    assert e['indices']==b['indices']
    assert e['stats']['projection_pairs']==len(x)*len(q)
    assert e['stats']['max_score_cells']==17*4

def test_validation():
    x=np.eye(4)
    for u,r in ((np.ones((4,2)),(2,)),(x,(4,2)),(x,(0,4)),(x+1j,(4,))):
        with pytest.raises(ValueError):BatchScreen(x,u,r)
    ix=BatchScreen(x,x,(4,))
    for q in ([0,0,0,0],[1,2],[np.nan,0,0,1]):
        with pytest.raises(ValueError):ix.search(q,1,rank=4)
