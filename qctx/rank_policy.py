"""Development-calibrated query rank policy; screening correctness is separate.

Fit a small decision stump on query retained energy and measured latency.
Choices are explicit prepared retrieval paths. Never use test queries to fit.
"""
from dataclasses import dataclass
import numpy as np
from .learned_screen import normalize


@dataclass(frozen=True)
class RankPolicy:
    static: str
    low: str
    high: str
    threshold: float | None
    development_mean_ms: float
    static_mean_ms: float

    def choose(self, retained_energy):
        if not np.isfinite(retained_energy):raise ValueError('nonfinite query energy')
        if self.threshold is None:return self.static
        return self.low if retained_energy<self.threshold else self.high


def retained_energy(query, frame):
    q=normalize(query);u=np.asarray(frame,dtype=np.float64)
    if q.ndim!=1 or u.ndim!=2 or q.size!=u.shape[0]:raise ValueError('query/frame mismatch')
    z=q@u;return float(z@z)


def fit_policy(energy, latency_ms, *, gate_ms=0., min_leaf=8, improvement=.10):
    """Fit at most one split by mean development latency, including gating cost.

    A split needs both leaves large enough and a specified improvement over
    the best static path. No inference-time latency feedback or labels.
    """
    e=np.asarray(energy,dtype=np.float64);names=tuple(sorted(latency_ms))
    if e.ndim!=1 or not len(e) or not np.all(np.isfinite(e)) or not names or min_leaf<1 or not 0<=improvement<1 or not np.isfinite(gate_ms) or gate_ms<0:
        raise ValueError('invalid development data')
    x=np.stack([np.asarray(latency_ms[a],dtype=np.float64) for a in names],axis=1)
    if x.shape!=(len(e),len(names)) or not np.all(np.isfinite(x)) or np.any(x<0):raise ValueError('invalid latency matrix')
    means=x.mean(axis=0);j=int(np.argmin(means));static=names[j];best_mean=float(means[j])
    best=None
    order=np.argsort(e,kind='stable');ee=e[order];cum=np.cumsum(x[order],axis=0)
    for k in range(min_leaf,len(e)-min_leaf+1):
        if ee[k-1]>=ee[k]:continue
        a=int(np.argmin(cum[k-1]));b=int(np.argmin(cum[-1]-cum[k-1]))
        if a==b:continue
        cost=float((cum[k-1,a]+cum[-1,b]-cum[k-1,b])/len(e)+gate_ms)
        threshold=float(ee[k])  # right endpoint preserves the split even for adjacent floats
        key=(cost,threshold,names[a],names[b])
        if best is None or key<best:best=key
    if best is None or best[0]>(1-improvement)*best_mean:
        return RankPolicy(static,static,static,None,best_mean,best_mean)
    cost,t,a,b=best;return RankPolicy(static,a,b,t,cost,best_mean)
