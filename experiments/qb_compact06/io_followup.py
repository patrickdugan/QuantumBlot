"""Fresh cold-advice read-scheduling pilot; fit decisions on development only."""
from pathlib import Path
import argparse,json,tempfile
import numpy as np
from qctx.disk_screen import DiskScreen,build_store
from qctx.learned_screen import fit_basis
from .benchmark import timed
from .synthetic_reference import synthetic


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--seed',type=int,default=58914);p.add_argument('--family',choices=['centroid','mixed','dense'],default='centroid');p.add_argument('--rank',type=int,default=32);p.add_argument('--out',type=Path,required=True);a=p.parse_args()
    tr,x,dev,test,*_=synthetic(a.seed,a.family,m=16384);dev=dev[:8];test=test[:24]
    u=fit_basis(tr,a.rank);rng=np.random.default_rng(a.seed+87);out=[];checks=0
    with tempfile.TemporaryDirectory() as t:
        root=Path(t)/'index';build_store(root,x,u)
        with DiskScreen(root) as ix:
            specs={'mmap':dict(mode='mmap')}
            for workers in (1,4):
                for gap in (0,16,64):specs[f'cascade_w{workers}_g{gap}']=dict(mode='cascade',gap_rows=gap,read_workers=workers)
            for batch in (1,8,24):
                kw=dict(k=20,query_block=batch,tile_rows=4096)
                def run(q):
                    nonlocal checks
                    truth=ix.search(q,**kw,mode='mmap');rows={}
                    for name in rng.permutation(list(specs)):
                        r,z=timed(ix,q,{**kw,**specs[name]},True,1)
                        assert z['indices']==truth['indices'] and z['scores']==truth['scores']
                        if name!='mmap':checks+=len(q)
                        rows[name]=r
                    return rows
                dv=run(dev);best=min(specs,key=lambda n:dv[n]['ms']);chosen=best if dv[best]['ms']<.95*dv['mmap']['ms'] else 'mmap'
                te=run(test)
                for z in te.values():z['speedup']=te['mmap']['ms']/z['ms'];z['read_ratio']=z['backing_logical_bytes']/te['mmap']['backing_logical_bytes']
                out.append(dict(batch=batch,chosen=chosen,dev=dv,test=te));print(a.seed,batch,chosen,[(n,round(z['ms'],1),z['read_calls'],z['os_delta'].get('read_bytes')) for n,z in te.items()],flush=True)
    a.out.write_text(json.dumps(dict(scope='SYNTHETIC_BEST_EFFORT_COLD_ADVICE',seed=a.seed,family=a.family,rank=a.rank,dev=len(dev),test=len(test),checks=checks,rows=out),indent=2)+'\n')

if __name__=='__main__':main()
