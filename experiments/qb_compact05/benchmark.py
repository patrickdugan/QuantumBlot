"""Matched CPU quantized-panel benchmarks. Synthetic or local row split."""
import argparse,json,statistics,time
from pathlib import Path
import numpy as np
from qctx.learned_screen import fit_basis,normalize
from qctx.batch_screen import BatchScreen
from qctx.quantized_screen import QuantizedBatchScreen,NativeInt8Screen
from experiments.qb_compact04.benchmark import synthetic


def timed(fn,repeats):
    z=fn();ts=[]
    for _ in range(repeats):
        t=time.perf_counter_ns();a=fn();ts.append((time.perf_counter_ns()-t)/1e6)
        assert a['indices']==z['indices']
    return statistics.median(ts),z


def evaluate(tr,x,dev,test,*,seed,rank,ranks,tile,batches,repeats,k,native_library=None):
    t=time.perf_counter();u=fit_basis(tr,ranks[-1]);fit=time.perf_counter()-t
    t=time.perf_counter();ref=BatchScreen(x,u,ranks);build={'float64':time.perf_counter()-t};ixs={'float64':ref}
    for kind in ('float32','int16','int8'):
        t=time.perf_counter();ixs[kind]=QuantizedBatchScreen(x,u,ranks,storage=kind);build[kind]=time.perf_counter()-t
    if native_library:
        t=time.perf_counter();ixs['native_int8']=NativeInt8Screen(x,u,native_library,ranks);build['native_int8']=time.perf_counter()-t
    m,d=ref.x.shape;rng=np.random.default_rng(seed+41);checks=0;maxerr=0.;radius={};rows=[]
    q=normalize(test);actual=ref.x@q.T
    for kind,ix in ixs.items():
        if kind=='float64':continue
        p=ix._query_projection(q,rank);ri=ranks.index(rank);errs=[];rad=[]
        for a,lo,hi in ix.bounds(test,rank,tile_rows=tile):
            end=a+len(lo);s=actual[a:end];checks+=2*s.size
            assert np.all(lo<=s) and np.all(s<=hi),(kind,'interval')
            v=ix._projection_panel(a,end,rank,p);ex=(ix.x[a:end]@ix.u[:,:rank])@(q@ix.u[:,:rank]).T
            e=ix._uncertainty(a,end,ri,p)-ix.res[a:end,ri,None].astype(float)*p[1][None,:]
            checks+=2*s.size;err=abs(v-ex);assert np.all(err<=e),(kind,'quantization interval');errs.append(err.max());rad.append(e.ravel())
        maxerr=max(maxerr,float(max(errs)));r=np.concatenate(rad)
        radius[kind]=dict(max_dot_error=float(max(errs)),median_extra_radius=float(np.median(r)),max_extra_radius=float(r.max()))
    specs={'full':(ref,'full')}
    for kind,ix in ixs.items():
        for mode in ('auto','lower'):specs[kind+'_'+mode]=(ix,mode)
    for batch in batches:
        def run(q):
            truth=ref.search(q,k,rank=rank,mode='full',tile_rows=tile,query_block=batch);out={}
            for name in rng.permutation(list(specs)):
                ix,mode=specs[name];ms,z=timed(lambda:ix.search(q,k,rank=rank,mode=mode,tile_rows=tile,query_block=batch),repeats)
                assert z['indices']==truth['indices'],(seed,batch,name,'ranking')
                err=float(np.max(abs(np.asarray(z['scores'])-truth['scores'])));assert err<1e-12
                out[name]=dict(ms=ms,score_error=err,**z['stats'])
            return out
        a,b=run(dev),run(test);best=min(specs,key=lambda n:(a[n]['ms'],n));chosen=best if a[best]['ms']<.95*a['full']['ms'] else 'full'
        b64=min([n for n in specs if n.startswith('float64_')],key=lambda n:(a[n]['ms'],n));bq=min([n for n in specs if n.startswith(('float32_','int16_','int8_'))],key=lambda n:(a[n]['ms'],n))
        for v in b.values():v['speedup_vs_full']=b['full']['ms']/v['ms'];v['exact_pair_ratio']=v['exact_pairs']/(m*len(test));v['coordinate_ratio']=v['coordinate_work']/(m*d*len(test))
        rows.append(dict(batch=batch,chosen=chosen,best64=b64,best_quantized=bq,dev=a,test=b))
        print(seed,batch,chosen,b64,bq,[(n,round(v['ms'],3)) for n,v in b.items()],flush=True)
    return dict(seed=seed,dimension=d,corpus=m,train=len(tr),dev=len(dev),test=len(test),rank=rank,ranks=list(ranks),tile_rows=tile,k=k,fit_s=fit,build_s=build,index_bytes={n:ix.index_bytes for n,ix in ixs.items()},corpus_bytes=ref.x.nbytes,bound_checks=checks,max_dot_error=maxerr,radius=radius,rows=rows)


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--seed',type=int,default=58801);p.add_argument('--family',choices=['centroid','dense','mixed'],default='centroid');p.add_argument('--corpus-size',type=int,default=16384);p.add_argument('--rank',type=int,default=32);p.add_argument('--ranks',type=int,nargs='+',default=[16,32,64,128]);p.add_argument('--tile',type=int,default=16384);p.add_argument('--batches',type=int,nargs='+',default=[1,8,32]);p.add_argument('--repeats',type=int,default=2);p.add_argument('--dev-size',type=int,default=16);p.add_argument('--test-size',type=int,default=32);p.add_argument('--k',type=int,default=20);p.add_argument('--split-file');p.add_argument('--native-library');p.add_argument('--out',type=Path,required=True);a=p.parse_args()
    if a.split_file:
        x=np.load(a.split_file,allow_pickle=False,mmap_mode='r');rng=np.random.default_rng(a.seed)
        if x.ndim!=2 or len(x)<max(a.ranks)+a.k+a.dev_size+a.test_size:p.error('insufficient rows')
        ids=rng.permutation(len(x));n=min(1536,max(max(a.ranks),len(x)//5));m=min(a.corpus_size,len(x)-n-a.dev_size-a.test_size);s=n+m;t=s+a.dev_size
        if m<a.k:p.error('insufficient corpus')
        tr,co,dev,test=x[ids[:n]],x[ids[n:s]],x[ids[s:t]],x[ids[t:t+a.test_size]];scope='LOCAL_DISJOINT_ROW_SPLIT'
    else:
        tr,co,dev,test,*_=synthetic(a.seed,a.family,m=a.corpus_size);dev=dev[:a.dev_size];test=test[:a.test_size];scope='SYNTHETIC_NO_LANGUAGE_MODEL'
    r=evaluate(tr,co,dev,test,seed=a.seed,rank=a.rank,ranks=tuple(a.ranks),tile=a.tile,batches=a.batches,repeats=a.repeats,k=a.k,native_library=a.native_library)
    a.out.write_text(json.dumps(dict(scope=scope,family=a.family if scope.startswith('SYNTHETIC') else None,numpy=np.__version__,**r),indent=2)+'\n')

if __name__=='__main__':main()
