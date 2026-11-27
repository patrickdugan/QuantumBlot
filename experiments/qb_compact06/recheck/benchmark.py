"""Disk-backed retrieval pilot. Local files only, synthetic or disjoint row split."""
from __future__ import annotations
import argparse, json, os, platform, statistics, tempfile, time
from pathlib import Path
import numpy as np
from experiments.qb_compact06.recheck.reference_disk import DiskScreen, build_index
from qctx.learned_screen import fit_basis, normalize
from experiments.qb_compact04.benchmark import synthetic


def read_bytes():
    try:
        for line in Path('/proc/self/io').read_text().splitlines():
            if line.startswith('read_bytes:'):return int(line.split()[1])
    except OSError:pass
    return None


def prepare_cache(path, cold):
    with open(path,'rb',buffering=0) as f:
        if cold:
            if not hasattr(os,'posix_fadvise') or not hasattr(os,'POSIX_FADV_DONTNEED'):return False
            try:os.posix_fadvise(f.fileno(),0,0,os.POSIX_FADV_DONTNEED)
            except OSError:return False
        else:
            while f.read(1<<20):pass
    return True


def evaluate(tr,x,test,*,seed,rank,tile,batches,repeats,k,root,cold):
    start=time.perf_counter();u=fit_basis(tr,rank);fit=time.perf_counter()-start
    start=time.perf_counter();build_index(root,x,u);build=time.perf_counter()-start
    start=time.perf_counter();ix=DiskScreen(root);opened=time.perf_counter()-start
    q=normalize(test);checks=0;min_margin=float('inf')
    # Audit bounds directly against the stored normalized float64 rows.
    norm=np.sqrt(np.sum(q*q,axis=1)+ix.slack)
    with open(ix.backing,'rb') as f:
        for a in range(0,ix.m,tile):
            end=min(ix.m,a+tile);rows=np.frombuffer(f.read((end-a)*ix.d*8),dtype='<f8').reshape(end-a,ix.d);actual=rows@q.T
            for projected in (False,True):
                v=q@ix.basis if projected else q;v32=v.astype(np.float32);err=np.linalg.norm(v-v32.astype(float),axis=1)+ix.slack
                lo,hi=ix._bounds(a,end,v,v32,err,norm,projected=projected)
                checks+=2*actual.size;min_margin=min(min_margin,float(np.min(actual-lo)),float(np.min(hi-actual)))
                assert np.all(lo<=actual) and np.all(actual<=hi)
    truth=ix.search(test,k,mode='mmap',tile=tile,query_block=24)
    rng=np.random.default_rng(seed+821);results=[];matches=0;maxerr=0.
    modes={'mmap':('mmap',0),'read':('read',0),'projection':('projection',0),'codes':('codes',0),'cascade':('cascade',0),'cascade_gap2':('cascade',2)}
    for b in batches:
        for condition in (('warm','cold_advice') if cold else ('warm',)):
            samples={n:[] for n in modes}
            for rep in range(repeats):
                for arm in rng.permutation(list(modes)):
                    mode,gap=modes[arm];advice=prepare_cache(ix.backing,condition=='cold_advice')
                    before=read_bytes();t=time.perf_counter_ns();v=ix.search(test,k,mode=mode,tile=tile,query_block=b,gap=gap);ms=(time.perf_counter_ns()-t)/1e6;after=read_bytes()
                    assert v['indices']==truth['indices'],(seed,b,condition,arm)
                    err=float(np.max(abs(np.asarray(v['scores'])-truth['scores'])));assert err<1e-12;maxerr=max(maxerr,err);matches+=len(test)
                    samples[arm].append(dict(ms=ms,kernel_read_bytes=after-before if before is not None and after is not None else None,cache_preparation_succeeded=advice,max_score_error=err,stats=v['stats']))
            med={}
            for arm,z in samples.items():
                a=[v['kernel_read_bytes'] for v in z if v['kernel_read_bytes'] is not None]
                med[arm]=dict(ms=statistics.median(v['ms'] for v in z),kernel_read_bytes=statistics.median(a) if a else None,backing_bytes=statistics.median(v['stats']['backing_bytes'] for v in z),read_calls=statistics.median(v['stats']['read_calls'] for v in z),logical_page_visits=statistics.median(v['stats']['logical_page_visits'] for v in z),exact_pairs=statistics.median(v['stats']['exact_pairs'] for v in z))
            for arm,v in med.items():
                v['speedup_vs_mmap']=med['mmap']['ms']/v['ms'];v['byte_ratio_vs_mmap']=v['backing_bytes']/med['mmap']['backing_bytes']
                b0=med['mmap']['kernel_read_bytes'];v['kernel_ratio_vs_mmap']=v['kernel_read_bytes']/b0 if b0 and v['kernel_read_bytes'] is not None else None
            results.append(dict(batch=b,condition=condition,medians=med,samples=samples))
            print(seed,b,condition,[(a,round(v['ms'],1),round(v['byte_ratio_vs_mmap'],4),v['kernel_read_bytes']) for a,v in med.items()],flush=True)
    return dict(seed=seed,rank=rank,dimension=ix.d,corpus=ix.m,train=len(tr),test=len(test),k=k,tile=tile,batches=batches,repeats=repeats,fit_s=fit,build_s=build,verified_open_s=opened,index_bytes=ix.index_bytes,backing_bytes=ix.backing.stat().st_size,bound_checks=checks,min_bound_margin=min_margin,ranking_checks=matches,max_score_error=maxerr,results=results,numpy=np.__version__,platform=platform.platform(),note='Logical bytes requested/touched by application; kernel read_bytes may include readahead. Cold advice is not guaranteed cache eviction. No physical SSD claim.')


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--family',choices=['centroid','mixed','dense'],default='centroid');p.add_argument('--seed',type=int,default=58901);p.add_argument('--corpus-size',type=int,default=16384);p.add_argument('--rank',type=int,default=32);p.add_argument('--tile',type=int,default=1024);p.add_argument('--batches',type=int,nargs='+',default=[1,8,24]);p.add_argument('--repeats',type=int,default=2);p.add_argument('--test-size',type=int,default=24);p.add_argument('--k',type=int,default=20);p.add_argument('--warm-only',action='store_true');p.add_argument('--split-file');p.add_argument('--workdir',type=Path);p.add_argument('--out',type=Path,required=True);a=p.parse_args()
    if a.split_file:
        x=np.load(a.split_file,allow_pickle=False,mmap_mode='r');ids=np.random.default_rng(a.seed).permutation(len(x));n=min(1536,max(a.rank,len(x)//5));m=min(a.corpus_size,len(x)-n-a.test_size)
        if x.ndim!=2 or m<a.k or a.rank>x.shape[1]:p.error('invalid corpus or rank')
        tr,x,test=x[ids[:n]],x[ids[n:n+m]],x[ids[n+m:n+m+a.test_size]];scope='LOCAL_DISJOINT_ROW_SPLIT'
    else:
        tr,x,_,test,*_=synthetic(a.seed,a.family,m=a.corpus_size);test=test[:a.test_size];scope='SYNTHETIC_NO_LANGUAGE_MODEL'
    with tempfile.TemporaryDirectory(dir=a.workdir) as td:
        v=evaluate(tr,x,test,seed=a.seed,rank=a.rank,tile=a.tile,batches=a.batches,repeats=a.repeats,k=a.k,root=Path(td)/'index',cold=not a.warm_only)
    a.out.write_text(json.dumps(dict(scope=scope,family=a.family if scope.startswith('SYNTHETIC') else None,**v),indent=2)+'\n')

if __name__=='__main__':main()
