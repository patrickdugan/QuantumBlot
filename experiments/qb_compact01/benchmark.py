"""Seeded CPU-only benchmark. Synthetic retrieval; no models, network or IBM."""
import json, sys, time, statistics, platform
from pathlib import Path
import numpy as np
P=Path(__file__).parent
sys.path.insert(0,str(P.parents[1]/'qctx'))
from shell_spectrum import shell_probabilities, shell_coordinates, ShellIndex


def rfft_shell(x):
    n=x.shape[-1]; b=n.bit_length()-1
    s=abs(np.fft.rfft(x,axis=-1))**2
    if n>1:s[...,1:-1]*=2
    out=[s[...,0]]
    for j in range(1,b+1):
        step=n//(1<<j);out.append(s[...,step::2*step].sum(axis=-1))
    p=np.stack(out,axis=-1);return p/p.sum(axis=-1,keepdims=True)


def median_ms(f,reps=15):
    f();t=[]
    for _ in range(reps):
        a=time.perf_counter_ns();f();t.append((time.perf_counter_ns()-a)/1e6)
    return statistics.median(t)


def run():
    rng=np.random.default_rng(62701);arith=[]
    for n in (256,1024,4096,16384,65536):
        x=rng.standard_normal((64,n));p=shell_probabilities(x);f=rfft_shell(x)
        err=float(np.max(abs(p-f)));assert err<1e-12
        arith.append(dict(n=n,batch=64,profiles=p.size,fold_ms=median_ms(lambda:shell_probabilities(x)),rfft_ms=median_ms(lambda:rfft_shell(x)),max_abs_error=err))
    retrieval=[];d=1024;m=4096;k=20
    l=np.array([d.bit_length()-1-(i & -i).bit_length()+1 if i else 0 for i in range(d)])
    for mode in ('dense','single_shell'):
        x=rng.standard_normal((m,d));
        if mode=='single_shell':
            f=np.fft.fft(x,axis=-1);f*=l[None,:]==rng.integers(1,d.bit_length(),size=(m,1));x=np.fft.ifft(f,axis=-1).real.copy()
        q=x[rng.choice(m,32,replace=False)]+rng.normal(scale=.01,size=(32,d))
        ix=ShellIndex(x);evals=[];times=[];direct=[];errors=[];residual=shell_coordinates(q)[1]
        for z in q:
            a=time.perf_counter_ns();r=ix.search(z,k);times.append((time.perf_counter_ns()-a)/1e6)
            z=z/np.linalg.norm(z);a=time.perf_counter_ns();v=ix.vectors@z;ids=np.lexsort((np.arange(m),-v))[:k];direct.append((time.perf_counter_ns()-a)/1e6)
            assert r['indices']==ids.tolist(),(mode,r['indices'],ids.tolist())
            errors.append(float(np.max(abs(np.array(r['scores'])-v[ids]))));evals.append(r['evaluated'])
        retrieval.append(dict(mode=mode,n=d,corpus=m,queries=len(q),k=k,matched_queries=len(q),median_evaluated=statistics.median(evals),min_evaluated=min(evals),max_evaluated=max(evals),median_screen_ms=statistics.median(times),median_full_blas_ms=statistics.median(direct),max_score_error=max(errors),median_invariant_projection_loss=float(np.median(residual))))
    result=dict(scope='SYNTHETIC_CPU_ONLY',python=sys.version,numpy=np.__version__,machine=platform.machine(),arithmetic=arith,retrieval=retrieval)
    (P/'AUDIT.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2))


if __name__=='__main__':run()
