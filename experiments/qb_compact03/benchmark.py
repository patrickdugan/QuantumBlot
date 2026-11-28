"""Held-out nested-vs-fixed cosine screening. CPU-only, no model or network calls."""
from __future__ import annotations
import argparse, hashlib, json, platform, statistics, time
from pathlib import Path
import numpy as np
from qctx.learned_screen import LearnedScreen, fit_basis, normalize
from qctx.progressive_screen import ProgressiveScreen

HERE = Path(__file__).resolve().parent
FAMILIES = ('dense', 'centroid', 'subspace', 'heterogeneous')


def synthetic(seed, mode, d=512):
    rng = np.random.default_rng(seed)
    widths = [2,2,4,4,8,16,32,64] if mode=='heterogeneous' else [4]*8
    offsets = np.cumsum([0]+widths); r = int(offsets[-1]); shared=8
    latent = np.linalg.qr(rng.normal(size=(d,r+shared)),mode='reduced')[0]
    def sample(n):
        lab = rng.integers(8,size=n); noise=normalize(rng.normal(size=(n,d)))
        if mode=='dense': return noise, lab
        common=normalize(rng.normal(size=(n,shared)))@latent[:,-shared:].T
        co=np.zeros((n,r))
        for g,w in enumerate(widths):
            ii=np.flatnonzero(lab==g); a=rng.normal(size=(len(ii),w))
            if mode=='centroid' or (mode=='heterogeneous' and g<4):
                a*=.22; a[:,0]+=1
            co[ii,offsets[g]:offsets[g+1]]=normalize(a)
        signal=co@latent[:,:r].T
        return normalize(.80*signal+.50*common+.25*noise), lab
    tr,_=sample(1536); corpus,_=sample(8192); dev,dl=sample(24); test,tl=sample(48)
    return tr,corpus,dev,test,dl,tl,(16,32,64,128)


def digits():
    from sklearn.datasets import load_digits
    data=load_digits();x,y=data.data,data.target;rng=np.random.default_rng(58413)
    tr=[];co=[];dev=[];te=[]
    for lab in np.unique(y):
        ids=rng.permutation(np.flatnonzero(y==lab));n=len(ids);a=n*4//10;b=n*9//10;c=b+(n-b)//2
        tr.extend(ids[:a]);co.extend(ids[a:b]);dev.extend(ids[b:c]);te.extend(ids[c:])
    return x[tr],x[co],x[dev],x[te],y[dev],y[te],(8,16,32,64)


def timed(fn):
    t=time.perf_counter_ns();result=fn();return (time.perf_counter_ns()-t)/1e6,result


def evaluate(data, family, root, k=20, repeats=3, native_library=None):
    tr,corpus,dev,test,devlab,testlab,ranks=data
    d=corpus.shape[1];m=len(corpus);k=min(k,m)
    fit_ms,u=timed(lambda:fit_basis(tr,ranks[-1],'pca'))
    build_ms,prog=timed(lambda:ProgressiveScreen(corpus,u,ranks))
    fixed={};fixed_build={}
    for r in ranks:
        fixed_build[str(r)],fixed[r]=timed(lambda:LearnedScreen(corpus,u[:,:r],mode='signed'))
    arms=['full']+[f'fixed{r}' for r in ranks]+['progressive','progressive_early','progressive_no_lower']
    native=None;native_build_ms=None
    if native_library:
        from qctx.native_screen import NativeProgressiveScreen
        native_build_ms,native=timed(lambda:NativeProgressiveScreen(corpus,u,native_library,ranks))
        arms += [f'native_fixed{r}' for r in ranks]+['native_progressive','native_from_second','native_no_lower']
    def search(arm,q):
        if arm=='full':return prog.search(q,k,screen=False)
        if arm.startswith('native_fixed'):
            r=int(arm[len('native_fixed'):]);return native.search(q,k,start_rank=r,end_rank=r)
        if arm.startswith('native_'):
            return native.search(q,k,start_rank=ranks[1] if arm=='native_from_second' else ranks[0],lower_bound=arm!='native_no_lower')
        if arm.startswith('fixed'):
            r=int(arm[5:]);z=fixed[r].search(q,k)
            return {**z,'coordinate_work':m*r+d*r+z['evaluated']*d,
                    'projected_coordinates':m*r,'query_coordinates':d*r,'stages':[],
                    'stop_reason':'full_fallback' if z['fallback'] else 'fixed'}
        return prog.search(q,k,early_fraction=.015 if arm=='progressive_early' else 0.,lower_bound=arm!='progressive_no_lower')
    rng=np.random.default_rng(root+101)
    def split(queries,labs,audit=False):
        rows=[];checks=0;worst_lower=0.;worst_upper=0.;worst_monotone=0.;score_error=0.
        for i,(q,lab) in enumerate(zip(queries,labs)):
            truth=search('full',q); rec={}
            for arm in rng.permutation(arms):
                got=search(arm,q);ts=[]
                for _ in range(repeats):
                    t,v=timed(lambda:search(arm,q));ts.append(t)
                    assert v['indices']==got['indices'],(family,root,i,arm,'nondeterministic')
                assert got['indices']==truth['indices'],(family,root,i,arm,got['indices'],truth['indices'])
                err=float(np.max(abs(np.asarray(got['scores'])-truth['scores'])));score_error=max(score_error,err)
                assert err<1e-12
                rec[arm]=dict(ms=statistics.median(ts),evaluated=got['evaluated'],unique=got['unique_evaluated'],
                    coordinate_work=got['coordinate_work'],projected_coordinates=got['projected_coordinates'],
                    query_coordinates=got['query_coordinates'],fallback=got['fallback'],stages=got['stages'],stop_reason=got['stop_reason'],max_score_error=err)
            if audit:
                exact=prog.x@normalize(q); bounds=prog.prefix_bounds(q)
                for lo,hi in bounds:
                    checks+=2*m;worst_lower=max(worst_lower,float(np.max(lo-exact)));worst_upper=max(worst_upper,float(np.max(exact-hi)))
                for (l,h),(l2,h2) in zip(bounds,bounds[1:]):
                    worst_monotone=max(worst_monotone,float(np.max(l-l2)),float(np.max(h2-h)))
                assert max(worst_lower,worst_upper,worst_monotone)<1e-10
            rows.append(dict(query=i,group=int(lab),topk_sha256=hashlib.sha256(np.asarray(truth['indices'],dtype='<i8').tobytes()).hexdigest(),arms=rec))
        return rows,dict(checks=checks,max_lower_violation=worst_lower,max_upper_violation=worst_upper,max_monotonicity_violation=worst_monotone,max_score_error=score_error)
    devrows,da=split(dev,devlab)
    med={arm:statistics.median(x['arms'][arm]['ms'] for x in devrows) for arm in arms}
    best=min(arms,key=lambda a:(med[a],a)); chosen=best if med[best]<=.95*med['full'] else 'full'
    testrows,ta=split(test,testlab,True)
    def metrics(rows,arm):
        z=[x['arms'][arm] for x in rows]
        return dict(queries=len(z),median_ms=statistics.median(v['ms'] for v in z),p90_ms=float(np.percentile([v['ms'] for v in z],90)),
            median_evaluated=statistics.median(v['evaluated'] for v in z),median_unique=statistics.median(v['unique'] for v in z),
            median_coordinate_work=statistics.median(v['coordinate_work'] for v in z),median_projected_coordinates=statistics.median(v['projected_coordinates'] for v in z),
            fallbacks=sum(v['fallback'] for v in z),ranks_reached={str(r):sum(any(s['rank']==r for s in v['stages']) for v in z) for r in ranks},
            stop_reasons={reason:sum(v['stop_reason']==reason for v in z) for reason in sorted({v['stop_reason'] for v in z})})
    testmet={arm:metrics(testrows,arm) for arm in arms};testmet['dev_selected']={**testmet[chosen],'chosen':chosen}
    for v in testmet.values():
        v['speedup_vs_full']=testmet['full']['median_ms']/v['median_ms'];v['coordinate_ratio_vs_full']=v['median_coordinate_work']/(m*d)
    meta=dict(family=family,root=root,d=d,corpus=m,train=len(tr),dev=len(dev),test=len(test),ranks=list(ranks),k=k,fit_ms=fit_ms,progressive_build_ms=build_ms,fixed_build_ms=fixed_build,
              progressive_extra_bytes=prog.index_bytes,native_extra_bytes=native.index_bytes if native else None,native_build_ms=native_build_ms,full_rows_bytes=prog.x.nbytes,dev_median_ms=med,dev_selected=chosen,metrics=testmet,dev_audit=da,test_audit=ta)
    print(family,root,'selected',chosen,'test',[(a,round(v['speedup_vs_full'],2),v['median_unique'],round(v['coordinate_ratio_vs_full'],3)) for a,v in testmet.items()],flush=True)
    return dict(summary=meta,dev_rows=devrows,test_rows=testrows)


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--root',type=int,default=58401);p.add_argument('--family',choices=FAMILIES);p.add_argument('--digits',action='store_true');p.add_argument('--train');p.add_argument('--corpus');p.add_argument('--dev');p.add_argument('--test');p.add_argument('--ranks',type=int,nargs='+',default=[16,32,64,128]);p.add_argument('--out',type=Path);p.add_argument('--k',type=int,default=20);p.add_argument('--native-library');a=p.parse_args()
    if any((a.train,a.corpus,a.dev,a.test)):
        files=[a.train,a.corpus,a.dev,a.test]
        if not all(files) or len({Path(f).resolve() for f in files})!=4: p.error('provide four different local .npy paths')
        tr,co,dv,te=[np.load(f,allow_pickle=False) for f in files]
        if any(x.ndim!=2 or len(x)==0 for x in (tr,co,dv,te)) or len({x.shape[1] for x in (tr,co,dv,te)})!=1:p.error('incompatible local arrays')
        rows=[evaluate((tr,co,dv,te,np.full(len(dv),-1),np.full(len(te),-1),tuple(a.ranks)),'local',a.root,a.k,native_library=a.native_library)]
        out=a.out or HERE/'AUDIT_local.json'
    elif a.digits:
        rows=[evaluate(digits(),'digits',58413,a.k,native_library=a.native_library)];out=a.out or HERE/'AUDIT_digits.json'
    else:
        rows=[evaluate(synthetic(a.root,f),f,a.root,a.k,native_library=a.native_library) for f in ([a.family] if a.family else FAMILIES)]
        out=a.out or HERE/f'AUDIT_{a.root}{"_"+a.family if a.family else ""}.json'
    if a.native_library and a.out is None:out=out.with_name(out.stem+'_native.json')
    result=dict(scope='CPU_PROGRESSIVE_SCREEN_EXACT_RANKING_CONTROLS',numpy=np.__version__,machine=platform.machine(),rows=rows)
    out.write_text(json.dumps(result,indent=2)+'\n');print('wrote',out,flush=True)


if __name__=='__main__':main()
