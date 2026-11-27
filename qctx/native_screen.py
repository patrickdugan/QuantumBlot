"""Optional C++ streaming bound filter. Load an explicitly built local library.

The kernel reads packed projections in one pass, refines only unresolved rows,
maintains a heap of valid lower bounds, and returns survivors for NumPy scoring.
No automatic compilation, external service, or binary dependency in qctx.
"""
from __future__ import annotations
import ctypes as ct
from pathlib import Path
import numpy as np
from .learned_screen import normalize, rank_top
from .progressive_screen import ProgressiveScreen


class NativeProgressiveScreen:
    def __init__(self, vectors, basis, library, ranks=(16,32,64,128)):
        base=ProgressiveScreen(vectors,basis,ranks)
        self.x=base.x;self.ids=base.ids;self.m=base.m;self.d=base.d
        self.slack=base.slack;self.ranks=base.ranks
        self.u=np.ascontiguousarray(np.concatenate(base.basis_blocks,axis=1))
        self.c=np.ascontiguousarray(np.concatenate(base.blocks,axis=1))
        self.res=np.ascontiguousarray(np.stack(base.residuals,axis=1))
        self.rr=np.array(self.ranks,dtype=np.int32)
        # Caller must supply the path of the locally compiled, trusted library.
        self.lib=ct.CDLL(str(Path(library).resolve()))
        self.scan=self.lib.qb_scan_v1
        dp=np.ctypeslib.ndpointer(dtype=np.float64,flags='C_CONTIGUOUS')
        i32=np.ctypeslib.ndpointer(dtype=np.int32,flags='C_CONTIGUOUS')
        i64=np.ctypeslib.ndpointer(dtype=np.int64,flags='C_CONTIGUOUS')
        u8=np.ctypeslib.ndpointer(dtype=np.uint8,flags='C_CONTIGUOUS')
        self.scan.argtypes=[ct.c_int64,ct.c_int32,ct.c_int32,ct.c_int32,ct.c_int32,
            i32,dp,dp,dp,dp,dp,u8,ct.c_int32,dp,ct.c_double,ct.c_int32,i64,dp,i64,dp]
        self.scan.restype=ct.c_int64

    @property
    def index_bytes(self):return self.u.nbytes+self.c.nbytes+self.res.nbytes+self.rr.nbytes

    def search(self,query,k=20,*,pilot=64,start_rank=None,end_rank=None,lower_bound=True,fallback_fraction=.5):
        q=normalize(query)
        if q.shape!=(self.d,) or not isinstance(k,int) or not 1<=k<=self.m or not isinstance(pilot,int) or pilot<1 or not 0<fallback_fraction<=1:
            raise ValueError('invalid query parameters')
        start=self.ranks.index(start_rank) if start_rank is not None else 0
        end=self.ranks.index(end_rank) if end_rank is not None else len(self.ranks)-1
        if start>end:raise ValueError('start rank exceeds end rank')
        p=np.ascontiguousarray(q@self.u[:,:self.ranks[end]])
        mass=np.cumsum(p*p);residual=np.zeros(len(self.ranks))
        residual[:end+1]=np.sqrt(np.maximum(0.,1-mass[self.rr[:end+1]-1]+self.slack))
        r=self.ranks[start]
        partial=np.ascontiguousarray(self.c[:,:r]@p[:r])
        nseed=min(self.m,max(k,pilot));seeds,_=rank_top(partial,self.ids,nseed)
        exact=self.x[seeds]@q;_,best=rank_top(exact,seeds,k)
        mask=np.zeros(self.m,dtype=np.uint8);mask[seeds]=1
        out=np.empty(self.m,dtype=np.int64);upper=np.empty(self.m)
        visited=np.zeros(len(self.ranks),dtype=np.int64);threshold=np.zeros(1)
        n=int(self.scan(self.m,self.ranks[-1],len(self.ranks),start,end,self.rr,
            self.c,self.res,p,residual,partial,mask,k,np.ascontiguousarray(best-self.slack),
            self.slack,int(lower_bound),out,upper,visited,threshold))
        if not 0<=n<=self.m-nseed:raise RuntimeError('native survivor count invalid')
        active=out[:n];survivors=nseed+n
        fallback=survivors>=fallback_fraction*self.m
        if fallback:
            ids,scores=rank_top(self.x@q,self.ids,k);evaluated=self.m+nseed;unique=self.m
        else:
            ids,scores=rank_top(np.concatenate((exact,self.x[active]@q)),np.concatenate((seeds,active)),k)
            evaluated=unique=survivors
        work=self.m*r
        trace=[]
        for j in range(start,end+1):
            width=self.ranks[j] if j==start else self.ranks[j]-self.ranks[j-1]
            # Initial projection is a full BLAS sweep and already includes seeds.
            rows=self.m if j==start else int(visited[j])
            if j!=start:work+=rows*width
            trace.append(dict(rank=self.ranks[j],width=width,block_rows=rows,visited_unseeded=int(visited[j])))
        query_work=self.d*self.ranks[end]
        return dict(indices=ids.tolist(),scores=scores.tolist(),evaluated=int(evaluated),unique_evaluated=int(unique),
            fallback=bool(fallback),survivors=int(survivors),stages=trace,projected_coordinates=int(work),
            query_coordinates=int(query_work),coordinate_work=int(work+query_work+evaluated*self.d),
            stop_reason='full_fallback' if fallback else 'native',final_threshold=float(threshold[0]))
