"""Disk backing benchmark. Actual read requests and optional OS I/O counters.
No network, no model calls. Cold advice is best effort, not a physical SSD test.
"""
from __future__ import annotations
import argparse,json,os,platform,statistics,tempfile,time
from pathlib import Path
import numpy as np
from qctx.disk_screen import DiskScreen,build_store
from qctx.learned_screen import fit_basis
from .synthetic_reference import synthetic


def counters():
    out={}
    try:
        with open('/proc/self/io') as f:
            for l in f:
                k,v=l.split(':');out[k]=int(v)
    except OSError:pass
    try:
        import resource
        r=resource.getrusage(resource.RUSAGE_SELF);out.update(minflt=r.ru_minflt,majflt=r.ru_majflt)
    except ImportError:pass
    return out


def warm(ix):
    ix.file.seek(0)
    while ix.file.read(1<<20):pass


def timed(ix,q,kw,cold=False,repeats=2):
    rows=[];got=None
    for _ in range(repeats):
        if cold:advised=ix.advise(True)
        else:warm(ix);advised=False
        a=counters();t=time.perf_counter_ns();z=ix.search(q,**kw);elapsed=(time.perf_counter_ns()-t)/1e6;b=counters()
        rows.append(dict(ms=elapsed,cold_advice=advised,os_delta={k:b[k]-v for k,v in a.items() if k in b},**z['stats']))
        if got is not None:assert z['indices']==got['indices'] and z['scores']==got['scores']
        got=z
    med={**rows[0],'ms':statistics.median(x['ms'] for x in rows),'trials':rows}
    return med,got


def evaluate(tr,x,dev,test,*,root,seed,storage,rank,k,tile,batches,repeats,source_ids=None):
    t=time.perf_counter();u=fit_basis(tr,rank);fit=time.perf_counter()-t
    t=time.perf_counter();meta=build_store(root,x,u,storage=storage,row_ids=source_ids,build_rows=1024);build=time.perf_counter()-t
    t=time.perf_counter();ix=DiskScreen(root);opening=time.perf_counter()-t
    try:
        specs={'mmap':dict(mode='mmap'),'full':dict(mode='full'),'projection':dict(mode='projection'),'code':dict(mode='code'),'cascade':dict(mode='cascade'),'cascade_gap2':dict(mode='cascade',gap_rows=2)}
        # Fixed roster: identical data, original vectors, stable score arithmetic.
        rng=np.random.default_rng(seed+811);rows=[];matches=0;maxerr=0.
        for batch in batches:
            kw=dict(k=k,tile_rows=tile,query_block=batch)
            def run(q,cold=False,names=None):
                nonlocal matches,maxerr
                truth=ix.search(q,**kw,mode='mmap');out={}
                for name in rng.permutation(list(specs) if names is None else names):
                    rec,got=timed(ix,q,{**kw,**specs[name]},cold,repeats if not cold else 1)
                    assert got['indices']==truth['indices'],(seed,batch,name,'rank')
                    err=float(np.max(abs(np.asarray(got['scores'])-truth['scores'])));assert err<1e-13
                    rec['max_score_error']=err;maxerr=max(maxerr,err);out[name]=rec
                    if name!='mmap':matches+=len(q)
                return out
            development=run(dev);best=min(specs,key=lambda n:(development[n]['ms'],n));chosen=best if development[best]['ms']<.95*development['mmap']['ms'] else 'mmap'
            testing=run(test);base=testing['mmap']
            for v in testing.values():v['speedup']=base['ms']/v['ms'];v['read_ratio']=v['backing_logical_bytes']/base['backing_logical_bytes']
            # Cold-advice check uses same held-out queries and a fixed roster.
            cold=run(test,True,['mmap','full','projection','code','cascade','cascade_gap2'])
            for v in cold.values():v['speedup']=cold['mmap']['ms']/v['ms'];v['read_ratio']=v['backing_logical_bytes']/cold['mmap']['backing_logical_bytes']
            rows.append(dict(batch=batch,chosen=chosen,dev=development,warm=testing,cold_advised=cold))
            print(seed,'batch',batch,'choice',chosen,'warm',[(n,round(v['ms'],1),round(v['read_ratio'],3),v['read_calls']) for n,v in testing.items()],'cold',[(n,round(v['ms'],1),v['os_delta'].get('read_bytes')) for n,v in cold.items()],flush=True)
        return dict(seed=seed,rows=ix.m,dimension=ix.d,rank=rank,storage=storage,train=len(tr),dev=len(dev),test=len(test),index_bytes=ix.index_bytes,backing_bytes=ix.backing_bytes,stored_bytes=sum(v['bytes'] for v in meta['files'].values()),fit_s=fit,build_s=build,verified_open_s=opening,k=k,tile=tile,match_checks=matches,max_score_error=maxerr,results=rows,manifest=meta)
    finally:ix.close()


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--seed',type=int,default=58901);p.add_argument('--family',choices=['centroid','mixed','dense'],default='centroid');p.add_argument('--corpus-size',type=int,default=16384);p.add_argument('--rank',type=int,default=32);p.add_argument('--tile',type=int,default=4096);p.add_argument('--batches',type=int,nargs='+',default=[1,8,24]);p.add_argument('--storage',choices=['int8','int16'],default='int8');p.add_argument('--dev-size',type=int,default=8);p.add_argument('--test-size',type=int,default=24);p.add_argument('--repeats',type=int,default=2);p.add_argument('--k',type=int,default=20);p.add_argument('--split-file');p.add_argument('--workdir',type=Path);p.add_argument('--out',required=True,type=Path);a=p.parse_args()
    ids=None
    if a.split_file:
        x=np.load(a.split_file,mmap_mode='r',allow_pickle=False);rng=np.random.default_rng(a.seed)
        if x.ndim!=2 or len(x)<a.rank+a.k+a.dev_size+a.test_size:p.error('insufficient source rows')
        ix=rng.permutation(len(x));n=min(1536,max(a.rank,len(x)//5));m=min(a.corpus_size,len(x)-n-a.dev_size-a.test_size)
        if m<a.k:p.error('insufficient corpus')
        tr=x[ix[:n]];ids=ix[n:n+m];dev=x[ix[n+m:n+m+a.dev_size]];test=x[ix[n+m+a.dev_size:n+m+a.dev_size+a.test_size]];scope='LOCAL_DISJOINT_ROW_SPLIT'
    else:
        tr,x,dev,test,*_=synthetic(a.seed,a.family,m=a.corpus_size);dev=dev[:a.dev_size];test=test[:a.test_size];scope='SYNTHETIC_NO_LANGUAGE_MODEL'
    if a.workdir:a.workdir.mkdir(parents=True,exist_ok=True)
    with tempfile.TemporaryDirectory(dir=a.workdir) as t:
        r=evaluate(tr,x,dev,test,root=Path(t)/'index',seed=a.seed,storage=a.storage,rank=a.rank,k=a.k,tile=a.tile,batches=a.batches,repeats=a.repeats,source_ids=ids)
    a.out.write_text(json.dumps(dict(scope=scope,family=a.family if scope.startswith('SYNTHETIC') else None,python=platform.python_version(),numpy=np.__version__,machine=platform.machine(),**r),indent=2)+'\n')

if __name__=='__main__':main()
