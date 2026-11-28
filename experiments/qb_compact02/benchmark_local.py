"""Evaluate the optional PCA screen on three disjoint local .npy matrices.
No model calls or external services. Output contains aggregate metrics only.
"""
import argparse,json,time,statistics
import numpy as np
from qctx.learned_screen import fit_basis,LearnedScreen


def run(train,corpus,queries,rank=32,k=20):
    if any(x.ndim!=2 or len(x)==0 for x in (train,corpus,queries)):
        raise ValueError('expected three nonempty row matrices')
    if len({x.shape[1] for x in (train,corpus,queries)})!=1:
        raise ValueError('embedding dimensions differ')
    start=time.perf_counter();u=fit_basis(train,rank,'pca');fit=time.perf_counter()-start
    start=time.perf_counter();idx=LearnedScreen(corpus,u);build=time.perf_counter()-start
    times={'full':[],'screen':[]};work=[];fb=0
    for i,q in enumerate(queries):
        results={}
        for arm in (('full','screen') if i%2==0 else ('screen','full')):
            idx.search(q,k,screen=arm=='screen');vals=[]
            for _ in range(3):
                t=time.perf_counter();out=idx.search(q,k,screen=arm=='screen');vals.append((time.perf_counter()-t)*1000)
            times[arm].append(statistics.median(vals));results[arm]=out
        a,b=results['full'],results['screen']
        if a['indices']!=b['indices'] or not np.allclose(a['scores'],b['scores'],atol=1e-13):
            raise RuntimeError('screen diverged from full scan; disable it')
        work.append(b['evaluated']);fb+=b['fallback']
    med={a:statistics.median(v) for a,v in times.items()}
    return dict(scope='LOCAL_EMBEDDING_BENCHMARK',rank=rank,dimension=corpus.shape[1],training_rows=len(train),corpus_rows=len(corpus),queries=len(queries),matched_queries=len(queries),k=k,fit_s=fit,build_s=build,median_ms=med,speedup=med['full']/med['screen'],median_exact_scores=statistics.median(work),fallbacks=fb)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--train',required=True);p.add_argument('--corpus',required=True);p.add_argument('--queries',required=True);p.add_argument('--rank',type=int,default=32);p.add_argument('--k',type=int,default=20);a=p.parse_args()
    arrays=[np.load(f,allow_pickle=False) for f in (a.train,a.corpus,a.queries)]
    print(json.dumps(run(*arrays,rank=a.rank,k=a.k),indent=2))
