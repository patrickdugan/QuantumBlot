"""Shared orthogonal index with tiled exact-score cosine search (NumPy, CPU).

Fixed prefixes share one corpus and one projected matrix. Bounds hold in
exact arithmetic; the float64 slack is an engineering safeguard.
"""
from __future__ import annotations
import math
import numpy as np
from .learned_screen import normalize, rank_top


class BatchScreen:
    def __init__(self, vectors, basis, ranks=(16,32,64,128), *, build_rows=4096):
        self.x=normalize(vectors)
        if self.x.ndim!=2 or not len(self.x) or build_rows<1:
            raise ValueError('expected nonempty corpus and positive build_rows')
        self.m,self.d=self.x.shape
        self.ranks=tuple(ranks)
        if (not self.ranks or any(not isinstance(r,int) or isinstance(r,bool) for r in self.ranks)
                or not 1<=self.ranks[0] or self.ranks[-1]>self.d
                or any(a>=b for a,b in zip(self.ranks,self.ranks[1:]))):
            raise ValueError('invalid nested ranks')
        u=np.asarray(basis)
        if np.iscomplexobj(u):raise ValueError('real frame required')
        u=np.asarray(u,dtype=np.float64)
        if u.shape!=(self.d,self.ranks[-1]) or not np.all(np.isfinite(u)):
            raise ValueError('invalid basis shape/values')
        self.slack=256*np.finfo(float).eps*self.d
        defect=float(np.linalg.norm(u.T@u-np.eye(u.shape[1]),ord=2))
        if defect>1e-8:raise ValueError('frame is not orthonormal')
        self.u=np.ascontiguousarray(u/np.sqrt(1+defect+self.slack))
        self.c=np.empty((self.m,self.ranks[-1]));self.res=np.empty((self.m,len(self.ranks)))
        for a in range(0,self.m,build_rows):
            z=self.x[a:a+build_rows]@self.u
            self.c[a:a+len(z)]=z
            self.res[a:a+len(z)]=np.sqrt(np.maximum(0,1-np.cumsum(z*z,axis=1)[:,np.array(self.ranks)-1]+self.slack))

    @property
    def index_bytes(self):return self.u.nbytes+self.c.nbytes+self.res.nbytes

    def search(self, queries, k=20, *, rank=32, mode='auto', tile_rows=16384,
               query_block=32, group=8, pilot=64):
        """Full, per-query gather, or coalesced union scoring with exact reranking.

        Each tile computes at most tile_rows*query_block pair scores per
        matrix. For multi-tile screening, seed selection is a first pass;
        projected scores are recomputed on a second pass. Full mode has no
        projection work. Auto chooses dense tile GEMM or a union GEMM when
        the number of extra products is modest, and otherwise gathers rows
        for each query. Lower mode obtains a tile-local kth lower bound
        in one projection pass, without seeds. Dispatch affects cost only.
        """
        q=normalize(queries)
        if q.ndim==1:q=q[None,:]
        if (q.ndim!=2 or q.shape[1]!=self.d or not len(q) or not 1<=k<=self.m
                or mode not in ('full','gather','union','auto','lower') or rank not in self.ranks
                or min(tile_rows,query_block,group,pilot)<1):
            raise ValueError('invalid search arguments')
        ids=[];scores=[]
        stats=dict(queries=len(q),projection_pairs=0,seed_pairs=0,exact_pairs=0,
                   candidate_pairs=0,union_calls=0,gather_calls=0,dense_calls=0,tie_rescores=0,
                   max_score_cells=0,query_coordinates=0)
        ri=self.ranks.index(rank); ns=min(self.m,max(k,pilot)); m=self.m; d=self.d
        for qb in range(0,len(q),query_block):
            z=q[qb:qb+query_block]; b=len(z); best=[(np.empty(0,dtype=np.int64),np.empty(0)) for _ in z]
            def update(j,ii,ss):
                oi,os=best[j];ii=np.concatenate((oi,ii));ss=np.concatenate((os,ss))
                # Different GEMM/GEMV shapes can round mathematically equal scores
                # differently. Retain the numerical boundary band, then use one
                # deterministic summation for all potentially tied contenders.
                if len(ss)>k:
                    kth=np.partition(ss,len(ss)-k)[len(ss)-k]
                    keep=ss>=kth-4*self.slack;ii,ss=ii[keep],ss[keep]
                order=np.argsort(-ss,kind='stable');v=ss[order]
                near=np.flatnonzero(v[:-1]-v[1:]<=4*self.slack)
                if len(near):
                    ix=order[np.unique(np.concatenate((near,near+1)))];ss=ss.copy()
                    for t in ix:ss[t]=math.fsum(self.x[ii[t]]*z[j])
                    stats['tie_rescores']+=len(ix);stats['exact_pairs']+=len(ix)
                best[j]=rank_top(ss,ii,k)
            if mode=='full':
                for a in range(0,m,tile_rows):
                    v=self.x[a:a+tile_rows]@z.T; ii=np.arange(a,a+len(v))
                    stats['exact_pairs']+=v.size;stats['dense_calls']+=1
                    stats['max_score_cells']=max(stats['max_score_cells'],v.size)
                    for j in range(b):update(j,ii,v[:,j])
                stats['candidate_pairs']+=m*b
            else:
                p=z@self.u[:,:rank];qr=np.sqrt(np.maximum(0,1-np.sum(p*p,axis=1)+self.slack));stats['query_coordinates']+=b*d*rank
                cache=None;thresholds=np.full(b,-np.inf)
                seeds=[(np.empty(0,dtype=np.int64),np.empty(0)) for _ in z]
                if mode!='lower':
                    seeds=[(np.empty(0,dtype=np.int64),np.empty(0)) for _ in z];cache=None
                    for a in range(0,m,tile_rows):
                        v=self.c[a:a+tile_rows,:rank]@p.T;ii=np.arange(a,a+len(v))
                        stats['projection_pairs']+=v.size;stats['max_score_cells']=max(stats['max_score_cells'],v.size)
                        if tile_rows>=m:cache=v
                        for j in range(b):
                            oi,os=seeds[j];seeds[j]=rank_top(np.concatenate((os,v[:,j])),np.concatenate((oi,ii)),ns)
                    thresholds=np.empty(b)
                    for j,(ii,_) in enumerate(seeds):
                        ss=self.x[ii]@z[j];update(j,ii,ss);thresholds[j]=best[j][1][-1]
                        stats['seed_pairs']+=len(ii);stats['exact_pairs']+=len(ii)
                for a in range(0,m,tile_rows):
                    end=min(m,a+tile_rows);l=end-a
                    v=cache if cache is not None else self.c[a:end,:rank]@p.T
                    if cache is None:stats['projection_pairs']+=v.size
                    stats['max_score_cells']=max(stats['max_score_cells'],v.size)
                    e=self.res[a:end,ri,None]*qr[None,:]
                    if mode=='lower' and l>=k:
                        lower=np.partition(v-e,l-k,axis=0)[l-k]-2*self.slack
                        thresholds=np.maximum(thresholds,lower)
                    mask=(v+e+2*self.slack)>=thresholds
                    for j,(ii,_) in enumerate(seeds):
                        hit=ii[(ii>=a)&(ii<end)]-a;mask[hit,j]=False
                    stats['candidate_pairs']+=int(mask.sum())
                    for g in range(0,b,group):
                        h=min(b,g+group); sub=mask[:,g:h]; count=int(sub.sum())
                        if not count:continue
                        union=np.flatnonzero(np.any(sub,axis=1));width=h-g
                        dense=(mode in ('auto','lower') and count>=.45*l*width)
                        together=(mode=='union' or (mode in ('auto','lower') and len(union)*width<=2*count))
                        if dense or together:
                            rr=np.arange(l) if dense else union
                            mat=(self.x[a:end] if dense else self.x[a+rr])@z[g:h].T
                            stats['exact_pairs']+=mat.size;stats['max_score_cells']=max(stats['max_score_cells'],mat.size)
                            stats['dense_calls' if dense else 'union_calls']+=1
                            for j in range(g,h):
                                take=sub[rr,j-g];update(j,a+rr[take],mat[take,j-g])
                        else:
                            for j in range(g,h):
                                rr=np.flatnonzero(mask[:,j]);
                                if not len(rr):continue
                                ss=self.x[a+rr]@z[j];update(j,a+rr,ss)
                                stats['exact_pairs']+=len(rr);stats['gather_calls']+=1
                    thresholds[:]=[ss[-1] if len(ss)>=k else -np.inf for _,ss in best]
                if mode!='lower':stats['candidate_pairs']+=b*ns
            for ii,ss in best:ids.append(ii.tolist());scores.append(ss.tolist())
        stats['coordinate_work']=stats['exact_pairs']*d+stats['projection_pairs']*rank+stats['query_coordinates']
        return dict(indices=ids,scores=scores,stats=stats)
