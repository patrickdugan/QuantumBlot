"""Fresh held-out latency-stump pilot. No real corpus, QPU, or model calls."""
import argparse, dataclasses, json, statistics
from pathlib import Path
import numpy as np
from qctx.learned_screen import LearnedScreen, fit_basis
from qctx.rank_policy import fit_policy, retained_energy
from .benchmark import synthetic, timed

HERE=Path(__file__).resolve().parent


def run(root,family):
    # Development and test are sampled independently after training and corpus.
    tr,x,dev,test,dl,tl,ranks=synthetic(root,family)
    u=fit_basis(tr,ranks[-1]);frame=u[:,:ranks[0]].copy()
    indexes={f'r{r}':LearnedScreen(x,u[:,:r]) for r in ranks}
    arms=['full',*indexes];ref=indexes['r16'];rng=np.random.default_rng(root+801)
    def call(a,q):return ref.search(q,20,screen=False) if a=='full' else indexes[a].search(q,20)
    def measure(fn):
        got=fn();times=[]
        for _ in range(3):t,z=timed(fn);times.append(t);assert got['indices']==z['indices']
        return statistics.median(times),got
    def dev_run():
        e=[];gate=[];lat={a:[] for a in arms}
        for q in dev:
            vals=[]
            for _ in range(5):t,z=timed(lambda:retained_energy(q,frame));vals.append(t)
            e.append(z);gate.append(statistics.median(vals));truth=call('full',q)
            for a in rng.permutation(arms):
                t,z=measure(lambda:call(a,q));assert z['indices']==truth['indices'];lat[a].append(t)
        return e,lat,statistics.mean(gate)
    e,lat,gate=dev_run();policy=fit_policy(e,lat,gate_ms=gate,min_leaf=6,improvement=.10)
    def select(q):
        return policy.static if policy.threshold is None else policy.choose(retained_energy(q,frame))
    results=[]
    for q in test:
        truth=call('full',q);row={}
        def adaptive():return call(select(q),q)
        fs={'full':lambda:call('full',q),'static':lambda:call(policy.static,q),'adaptive':adaptive}
        for a in rng.permutation(list(fs)):
            t,z=measure(fs[a]);assert z['indices']==truth['indices'];row[a]={'ms':t,'unique':z['unique_evaluated'],'evaluated':z['evaluated']}
        row['choice']=select(q);row['energy']=retained_energy(q,frame);results.append(row)
    metrics={a:{'mean_ms':statistics.mean(r[a]['ms'] for r in results),'median_ms':statistics.median(r[a]['ms'] for r in results),'p90_ms':float(np.percentile([r[a]['ms'] for r in results],90)), 'median_unique':statistics.median(r[a]['unique'] for r in results)} for a in fs}
    for v in metrics.values():v['mean_speedup_vs_full']=metrics['full']['mean_ms']/v['mean_ms']
    info=dict(root=root,family=family,train=len(tr),corpus=len(x),dev=len(dev),test=len(test),policy=dataclasses.asdict(policy),metrics=metrics,choices={a:sum(r['choice']==a for r in results) for a in arms},development_energy=e,development_latency_ms=lat,gate_ms=gate,test_rows=results)
    print(family,root,policy,[(a,round(v['mean_speedup_vs_full'],2)) for a,v in metrics.items()],flush=True)
    return info


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--root',type=int,required=True);a=p.parse_args()
    r=[run(a.root,f) for f in ('dense','heterogeneous')]
    (HERE/f'POLICY_{a.root}.json').write_text(json.dumps(r,indent=2)+'\n')
