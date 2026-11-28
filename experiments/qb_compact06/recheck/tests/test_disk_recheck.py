from pathlib import Path
import numpy as np
import pytest
from experiments.qb_compact06.recheck.reference_disk import DiskScreen, build_index
from qctx.learned_screen import fit_basis, normalize, rank_top

@pytest.mark.parametrize('d,r',[(8,4),(65,16),(128,32)])
@pytest.mark.parametrize('structured',[False,True])
def test_modes(tmp_path,d,r,structured):
    rng=np.random.default_rng(942+d);x=rng.normal(size=(193,d));tr=rng.normal(size=(180,d))
    if structured:
        a=np.geomspace(1,.001,d);x*=a;tr*=a
    x[1]=x[0];x[2]=-x[0];u=fit_basis(tr,r)
    x[3]=u[:,0];x[4]=-x[3];q=np.vstack([x[:5],rng.normal(size=(8,d))]);p=tmp_path/'idx'
    meta=build_index(p,x,u,block=31);idx=DiskScreen(p);norm=normalize(x)
    assert idx.index_bytes==(d+r+20)*len(x)+d*r*8
    assert (p/'rows.bin').stat().st_size==norm.nbytes
    for k in (1,7,20,193):
        truth=[rank_top(np.einsum('ij,j->i',norm,z,optimize=False),np.arange(len(x)),k) for z in normalize(q)]
        for mode in ('mmap','read','projection','codes','cascade'):
            for gap in (0,3):
                got=idx.search(q,k,mode=mode,tile=37,query_block=4,gap=gap)
                assert got['indices']==[a.tolist() for a,b in truth]
                assert np.allclose(got['scores'],[b.tolist() for a,b in truth],atol=1e-14)
                s=got['stats'];assert s['max_read_rows']<=37 and s['max_upper_cells']<=len(x)*4
                assert s['backing_bytes']<=norm.nbytes*4
    # Independent comparison of enclosure endpoints.
    qq=normalize(q);p0=qq@idx.basis
    for v,projected in ((qq,False),(p0,True)):
        v32=v.astype(np.float32);err=np.linalg.norm(v-v32.astype(float),axis=1)+idx.slack;qn=np.sqrt(np.sum(qq*qq,axis=1)+idx.slack)
        lo,hi=idx._bounds(0,len(x),v,v32,err,qn,projected=projected);dot=norm@qq.T
        assert np.all(lo<=dot) and np.all(dot<=hi)


def test_integrity(tmp_path):
    p=tmp_path/'idx';x=np.eye(4);build_index(p,x,x[:,:2]);idx=DiskScreen(p)
    assert idx.search(x,1)['indices']==[[0],[1],[2],[3]]
    with pytest.raises(FileExistsError):build_index(p,x,x)
    with open(p/'rows.bin','r+b') as f:f.write(b'\xff')
    with pytest.raises(RuntimeError):idx.search(x,1)
    with pytest.raises(ValueError):DiskScreen(p)


def test_validation(tmp_path):
    p=tmp_path/'idx';x=np.array([[1e300,0,0,0],[0,1e-300,0,0],[-1e300,0,0,0]])
    build_index(p,x,np.eye(4)[:,:2],block=1);idx=DiskScreen(p)
    for mode in ('mmap','read','projection','codes','cascade'):
        assert idx.search(x[0],3,mode=mode,tile=1)['indices']==[[0,1,2]]
    with pytest.raises(ValueError):idx.search([0,0,0,0],1)
    with pytest.raises(ValueError):idx.search(x,4)
    with pytest.raises(ValueError):build_index(tmp_path/'bad',x,np.ones((4,2)))
    with pytest.raises(ValueError):build_index(tmp_path/'zero',np.zeros((4,4)),np.eye(4))
    assert not (tmp_path/'zero').exists()
