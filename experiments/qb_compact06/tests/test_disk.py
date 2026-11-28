import json
from pathlib import Path
import numpy as np
import pytest
from qctx.disk_screen import DiskScreen,build_store,pack,_query,_panel
from qctx.learned_screen import normalize,fit_basis,rank_top


@pytest.mark.parametrize('storage',['int8','int16'])
@pytest.mark.parametrize('structured',[False,True])
def test_search(tmp_path,storage,structured):
    rng=np.random.default_rng(608);d=64;x=rng.normal(size=(259,d));tr=rng.normal(size=(180,d))
    if structured:x*=np.geomspace(1,.0001,d);tr*=np.geomspace(1,.0001,d)
    x[1]=x[0];x[2]=-x[0];u=fit_basis(tr,16);q=np.vstack([x[0],-x[0],rng.normal(size=(7,d))]);store=tmp_path/'index'
    build_store(store,x,u,storage=storage,build_rows=37)
    nx=normalize(x);nq=normalize(q);ids=np.arange(len(x))
    with DiskScreen(store) as ix:
        assert not hasattr(ix,'x') and ix.backing_bytes==nx.nbytes
        assert ix.index_bytes<ix.backing_bytes
        for k in (1,7,20,len(x)):
            truth=[rank_top(np.sum(nx*z,axis=1),ids,k) for z in nq]
            for mode in ('mmap','full','code','projection','cascade'):
                for gap in (0,3):
                    z=ix.search(q,k,mode=mode,tile_rows=53,query_block=4,gap_rows=gap)
                    assert z['indices']==[a.tolist() for a,b in truth]
                    assert np.array_equal(z['scores'],[b.tolist() for a,b in truth])
                    t=z['stats'];assert t['max_panel_cells']<=53*4
                    assert t['backing_logical_bytes']<=3*ix.backing_bytes
                    assert t['read_bytes']==t['rows_read']*8*d
                    assert t['candidate_pairs']<=len(x)*len(q)
                    if mode=='mmap':assert t['mapped_bytes']==3*ix.backing_bytes and t['read_calls']==0
                    if mode=='full':assert t['read_bytes']==3*ix.backing_bytes


def test_intervals_and_zero_projection(tmp_path):
    x=np.array([[1.,0,0,0],[0,0,1.,0],[0,0,-1.,0],[1e-300,0,1e300,0],[-1,0,0,0]])
    u=np.eye(4)[:,:2];root=tmp_path/'index';build_store(root,x,u)
    nx=normalize(x)
    with DiskScreen(root) as ix:
        for z,code,scale,error in [(nx,ix.code,ix.scale,ix.error),(nx@ix.basis,ix.projection,ix.p_scale,ix.p_error)]:
            v,e=_panel(code,scale,error,_query(z,ix.slack),ix.norm,ix.slack)
            assert np.all(abs(v-z@z.T)<=e)
        for mode in ('code','projection','cascade'):
            assert ix.search(x,3,mode=mode,tile_rows=2)['indices']==ix.search(x,3,mode='full')['indices']


def test_memmap_build_and_integrity(tmp_path):
    x=np.random.default_rng(1).normal(size=(40,16));path=tmp_path/'src.npy';np.save(path,x)
    mm=np.load(path,mmap_mode='r',allow_pickle=False);ids=np.arange(39,-1,-2)
    root=tmp_path/'index';meta=build_store(root,mm,row_ids=ids,build_rows=3)
    assert np.array_equal(np.load(root/'source_rows.npy'),ids)
    with DiskScreen(root) as ix:
        assert ix.search(x[:4],3,mode='code')['indices']==ix.search(x[:4],3,mode='mmap')['indices']
        with pytest.raises(ValueError):ix.search(x[:4],3,mode='projection')
        with open(root/'rows.f64','ab') as f:f.write(b'x')
        with pytest.raises(ValueError):ix.search(x[:4],3,mode='code')
    with pytest.raises(ValueError):DiskScreen(root)
    with pytest.raises(FileExistsError):build_store(root,x)


def test_bad_store_and_inputs(tmp_path):
    x=np.eye(4)
    for v,u in [(np.zeros((3,4)),None),(x,np.ones((4,2))),(x+1j,None)]:
        with pytest.raises(ValueError):build_store(tmp_path/str(len(list(tmp_path.iterdir()))),v,u)
    p=tmp_path/'ok';build_store(p,x,x[:,:2]);f=p/'code.npy'
    with open(f,'r+b') as z:
        z.seek(-1,2);v=z.read(1);z.seek(-1,2);z.write(bytes([v[0]^1]))
    with pytest.raises(ValueError):DiskScreen(p)
    # Incomplete build must not open as a completed store.
    p=tmp_path/'partial';p.mkdir()
    with pytest.raises(FileNotFoundError):DiskScreen(p)


def test_pack_ranges():
    x=np.array([[0.,0.],[-1.,1.],[1e-300,-1e-300]])
    for dt in (np.dtype('int8'),np.dtype('int16')):
        c,s,e=pack(x,dt,1e-14);lim=np.iinfo(dt).max
        assert np.min(c)>=-lim and np.max(c)<=lim
        assert np.all(np.linalg.norm(x-c.astype(float)*s[:,None],axis=1)<=e)


def test_parallel_positional_reads(tmp_path):
    import os
    if not hasattr(os,'pread'):pytest.skip('positional reads unavailable')
    rng=np.random.default_rng(13);x=rng.normal(size=(400,64));q=rng.normal(size=(13,64));u=fit_basis(x[:120],16)
    root=tmp_path/'index';build_store(root,x,u)
    with DiskScreen(root) as ix:
        truth=ix.search(q,20,mode='mmap')
        for mode in ('code','projection','cascade'):
            for gap in (0,4):
                a=ix.search(q,20,mode=mode,gap_rows=gap,tile_rows=113,query_block=5)
                b=ix.search(q,20,mode=mode,gap_rows=gap,tile_rows=113,query_block=5,read_workers=4)
                assert a['indices']==b['indices']==truth['indices']
                assert a['scores']==b['scores']==truth['scores']
                assert all(a['stats'][k]==v for k,v in b['stats'].items() if k!='read_workers')
