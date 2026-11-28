"""Batch retrieval pilot; held-out synthetic or local embeddings, no network."""
from __future__ import annotations
import argparse,json,statistics,time
from pathlib import Path
import numpy as np
from qctx.learned_screen import fit_basis,normalize
from qctx.batch_screen import BatchScreen


def synthetic(seed,mode,m=16384,d=512):
    rng=np.random.default_rng(seed); widths=[4]*8 if mode!='mixed' else [2,2,4,4,8,16,32,64]
    off=np.cumsum([0]+widths);r=int(off[-1]);u=np.linalg.qr(rng.normal(size=(d,r+8)),mode='reduced')[0]
    def sample(n):
        labels=rng.integers(8,size=n);noise=normalize(rng.normal(size=(n,d)))
        if mode=='dense':return noise,labels
        a=np.zeros((n,r))
        for g,w in enumerate(widths):
            sel=np.flatnonzero(labels==g);z=rng.normal(size=(len(sel),w))
            if mode=='centroid' or (mode=='mixed' and g<4):z*=.22;z[:,0]+=1
            a[sel,off[g]:off[g+1]]=normalize(z)
        return normalize(.80*a@u[:,:r].T+.50*normalize(rng.normal(size=(n,8)))@u[:,r:].T+.25*noise),labels
    tr,_=sample(1536);x,_=sample(m);dev,dl=sample(32);test,tl=sample(64)
    return tr,x,dev,test,dl,tl


def timed(fn,repeats=3):
    got=fn();ts=[]
    for _ in range(repeats):
        t=time.perf_counter();z=fn();ts.append((time.perf_counter()-t)*1000)
        assert z['indices']==got['indices']
    return statistics.median(ts),got


def evaluate(tr,x,dev,test,dl,tl,*,seed=58701,k=20,ranks=(16,32,64,128),tile=16384,batches=(1,8,32),repeats=3,quick=False):
    t=time.perf_counter();u=fit_basis(tr,ranks[-1]);fit=time.perf_counter()-t
    t=time.perf_counter();ix=BatchScreen(x,u,ranks);build=time.perf_counter()-t
    m,d=ix.x.shape;out=[];rng=np.random.default_rng(seed+1)
    for batch in batches:
        arms={'full':dict(mode='full',rank=ranks[0])}
        for r in ranks[:-1]:
            arms[f'auto{r}']=dict(mode='auto',rank=r)
            arms[f'lower{r}']=dict(mode='lower',rank=r)
        r=ranks[min(1,len(ranks)-1)]
        arms[f'gather{r}']=dict(mode='gather',rank=r)
        arms[f'union{r}']=dict(mode='union',rank=r)
        if quick:
            arms={a:v for a,v in arms.items() if a in ('full',f'auto{r}',f'lower{r}',f'auto{ranks[-2]}')}
        for order in (('mixed',) if quick else ('mixed','clustered')):
            if order=='clustered' and (dl is None or tl is None):continue
            dd,tt=(dev,test) if order=='mixed' else (dev[np.argsort(dl,kind='stable')],test[np.argsort(tl,kind='stable')])
            def run(qq):
                truth=ix.search(qq,k,mode='full',rank=ranks[0],tile_rows=tile,query_block=batch);results={}
                for arm in rng.permutation(list(arms)):
                    lat,z=timed(lambda:ix.search(qq,k,tile_rows=tile,query_block=batch,**arms[arm]),repeats)
                    assert z['indices']==truth['indices'],(batch,order,arm)
                    err=float(np.max(abs(np.asarray(z['scores'])-truth['scores'])));assert err<1e-12
                    results[arm]=dict(ms=lat,max_score_error=err,**z['stats'])
                return results
            development=run(dd);best=min(arms,key=lambda a:(development[a]['ms'],a));chosen=best if development[best]['ms']<.95*development['full']['ms'] else 'full'
            testing=run(tt);full=testing['full']['ms']
            for a,z in testing.items():z['speedup']=full/z['ms'];z['coordinate_ratio']=z['coordinate_work']/(m*d*len(tt));z['exact_pair_ratio']=z['exact_pairs']/(m*len(tt))
            out.append(dict(batch=batch,order=order,chosen=chosen,dev=development,test=testing))
            print(seed,batch,order,'selected',chosen,'full ms',round(full,2),'selected speedup',round(testing[chosen]['speedup'],3),'best screen',round(max(testing[a]['speedup'] for a in arms if a!='full'),3),flush=True)
    return dict(seed=seed,train=len(tr),corpus=m,dev=len(dev),test=len(test),dimension=d,ranks=ranks,k=k,tile_rows=tile,fit_s=fit,build_s=build,index_bytes=ix.index_bytes,corpus_bytes=ix.x.nbytes,rows=out)


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--seed',type=int,default=58701);p.add_argument('--family',choices=['dense','centroid','mixed'],default='mixed');p.add_argument('--corpus-size',type=int,default=16384);p.add_argument('--tile',type=int,default=16384);p.add_argument('--batches',type=int,nargs='+',default=[1,8,32]);p.add_argument('--repeats',type=int,default=3);p.add_argument('--k',type=int,default=20);p.add_argument('--ranks',type=int,nargs='+',default=[16,32,64,128]);p.add_argument('--train');p.add_argument('--corpus');p.add_argument('--dev');p.add_argument('--test');p.add_argument('--split-file');p.add_argument('--out',type=Path,required=True);p.add_argument('--quick',action='store_true');a=p.parse_args()
    if a.split_file:
        if any((a.train,a.corpus,a.dev,a.test)):p.error('choose one split method')
        x=np.load(a.split_file,allow_pickle=False,mmap_mode='r');rng=np.random.default_rng(a.seed)
        if x.ndim!=2 or len(x)<sum([max(a.ranks),a.k,96]):p.error('not enough rows')
        order=rng.permutation(len(x));ntrain=min(1536,max(max(a.ranks),len(x)//5));ncorpus=min(a.corpus_size,len(x)-ntrain-96)
        if ncorpus<a.k:p.error('not enough corpus rows')
        data=[x[order[:ntrain]],x[order[ntrain:ntrain+ncorpus]],x[order[ntrain+ncorpus:ntrain+ncorpus+32]],x[order[ntrain+ncorpus+32:ntrain+ncorpus+96]],None,None];scope='LOCAL_DISJOINT_ROW_SPLIT'
    elif any((a.train,a.corpus,a.dev,a.test)):
        paths=[a.train,a.corpus,a.dev,a.test]
        if not all(paths) or len({Path(x).resolve() for x in paths})!=4:p.error('provide four separate arrays')
        data=[*[np.load(x,allow_pickle=False,mmap_mode='r') for x in paths],None,None];scope='LOCAL_USER_SUPPLIED_SPLITS'
    else:
        data=synthetic(a.seed,a.family,a.corpus_size);scope='SYNTHETIC_NO_LANGUAGE_MODEL'
    if a.quick:
        data=list(data);data[2]=data[2][:8];data[3]=data[3][:16]
        if data[4] is not None:data[4]=data[4][:8];data[5]=data[5][:16]
    res=evaluate(*data,seed=a.seed,k=a.k,ranks=tuple(a.ranks),tile=a.tile,batches=a.batches,repeats=a.repeats,quick=a.quick)
    a.out.write_text(json.dumps(dict(scope=scope,family=a.family if scope.startswith('SYNTHETIC') else None,numpy=np.__version__,**res),indent=2)+'\n')

if __name__=='__main__':main()
