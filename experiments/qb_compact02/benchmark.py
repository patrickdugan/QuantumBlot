"""Held-out CPU screening benchmark. Training/corpus/query splits are disjoint."""
from pathlib import Path
import argparse, json, time, statistics
import numpy as np
from qctx.learned_screen import LearnedScreen, fit_basis, normalize, rank_top

HERE=Path(__file__).resolve().parent
ARMS=('full','shell','random_signed','pca_energy','pca_signed','class_energy','class_signed')


def synthetic(seed,mode,d=512):
    rng=np.random.default_rng(seed); cls=8; width=4
    latent=np.linalg.qr(rng.normal(size=(d,cls*width+8)),mode='reduced')[0]
    def samples(n):
        lab=rng.integers(cls,size=n); noise=normalize(rng.normal(size=(n,d)))
        if mode=='dense': return noise,lab
        common=normalize(rng.normal(size=(n,8)))@latent[:,-8:].T
        coef=np.zeros((n,cls*width)); a=rng.normal(size=(n,width))
        if mode=='centroid':
            a*=.22;a[:,0]+=1
        a=normalize(a)
        coef[np.arange(n)[:,None],lab[:,None]*width+np.arange(width)]=a
        signal=coef@latent[:,:cls*width].T
        return normalize(.80*signal+.50*common+.25*noise),lab
    tr,y=samples(1536); docs,dy=samples(8192); q,qy=samples(48)
    return tr,y,docs,dy,q,qy,32


def digits():
    from sklearn.datasets import load_digits
    data=load_digits();x=data.data;y=data.target;rng=np.random.default_rng(58313)
    a=[];b=[];c=[]
    for lab in np.unique(y):
        i=rng.permutation(np.flatnonzero(y==lab));n=len(i);a.extend(i[:n*4//10]);b.extend(i[n*4//10:n*9//10]);c.extend(i[n*9//10:])
    return x[a],y[a],x[b],y[b],x[c],y[c],16


def ms(fn):
    t=time.perf_counter_ns();out=fn();return (time.perf_counter_ns()-t)/1e6,out


def evaluate(data,label,root):
    tr,ty,docs,dy,q,qy,rank=data;d=docs.shape[1];k=min(20,len(docs));x=normalize(docs)
    bases={};fit={}
    for method in ('random','pca','class_energy'):
        elapsed,u=ms(lambda:fit_basis(tr,rank,method,labels=ty,seed=root+41));bases[method]=u;fit[method]=elapsed
    specs={'shell':(None,'shell'),'random_signed':(bases['random'],'signed'),'pca_energy':(bases['pca'],'energy'),'pca_signed':(bases['pca'],'signed'),'class_energy':(bases['class_energy'],'energy'),'class_signed':(bases['class_energy'],'signed')}
    indexes={};build={}
    for arm,(u,mode) in specs.items():
        build[arm],indexes[arm]=ms(lambda:LearnedScreen(docs,u,mode=mode))
    ids=np.arange(len(x))
    def full(z):
        i,s=rank_top(x@normalize(z),ids,k)
        return dict(indices=i.tolist(),scores=s.tolist(),evaluated=len(x),unique_evaluated=len(x),fallback=False,survivors=len(x))
    records={a:[] for a in ARMS};err={a:0. for a in ARMS};relevance=[];checks=0;bound_min={a:0. for a in specs}
    rng=np.random.default_rng(root+91)
    for z,y in zip(q,qy):
        truth=full(z);relevance.append(float(np.mean(dy[truth['indices']]==y)))
        nq=normalize(z);dots=x@nq
        for arm,ix in indexes.items():
            upper,_=ix.bounds(nq);bound_min[arm]=min(bound_min[arm],float(np.min(upper-dots)));checks+=len(dots)
        for arm in rng.permutation(ARMS):
            fn=(lambda:full(z)) if arm=='full' else (lambda:indexes[arm].search(z,k))
            got=fn();ts=[]
            for _ in range(3):
                t,res=ms(fn);ts.append(t);assert res['indices']==got['indices']
            assert got['indices']==truth['indices'],(label,root,arm,got['indices'],truth['indices'])
            err[arm]=max(err[arm],float(np.max(abs(np.array(got['scores'])-truth['scores']))))
            records[arm].append(dict(ms=statistics.median(ts),evaluated=got['evaluated'],unique=got['unique_evaluated'],fallback=int(got['fallback']),survivors=got['survivors']))
    metrics={}
    for arm,rows in records.items():
        metrics[arm]=dict(queries=len(rows),matched=len(rows),median_ms=statistics.median(r['ms'] for r in rows),median_evaluated=statistics.median(r['evaluated'] for r in rows),median_unique=statistics.median(r['unique'] for r in rows),median_survivors=statistics.median(r['survivors'] for r in rows),fallbacks=sum(r['fallback'] for r in rows),max_score_error=err[arm],max_unique=max(r['unique'] for r in rows))
        if arm in specs: metrics[arm]['min_bound_minus_exact']=bound_min[arm]
    for a in metrics: metrics[a]['speedup_vs_full']=metrics['full']['median_ms']/metrics[a]['median_ms']
    result=dict(dataset=label,root=root,d=d,rank=rank,train=len(tr),corpus=len(docs),queries=len(q),k=k,fit_ms=fit,build_ms=build,metrics=metrics,bound_checks=checks,baseline_label_precision_at_k=float(np.mean(relevance)))
    print(label,root,'full',round(metrics['full']['median_ms'],3),'exact evaluations',[(a,metrics[a]['median_unique'],round(metrics[a]['speedup_vs_full'],2)) for a in ARMS[1:]],flush=True)
    return result


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--root',type=int,default=58301);ap.add_argument('--digits',action='store_true');args=ap.parse_args()
    rows=[evaluate(digits(),'digits',58313)] if args.digits else [evaluate(synthetic(args.root,m),m,args.root) for m in ('dense','centroid','subspace')]
    import sklearn
    out=dict(scope='CPU_SYNTHETIC_AND_BUNDLED_DIGITS_NO_SEMANTIC_MODEL',numpy=np.__version__,sklearn=sklearn.__version__,rows=rows)
    (HERE/f'AUDIT_{"digits" if args.digits else args.root}.json').write_text(json.dumps(out,indent=2)+'\n')

if __name__=='__main__':main()
