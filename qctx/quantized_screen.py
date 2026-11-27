"""Compressed projection panels; exact-score reranking via BatchScreen.

The quantization enclosure follows from Cauchy-Schwarz. FP32 dot error
uses a standard gamma bound (ordinary IEEE round-to-nearest accumulation,
no reduced-precision GEMM or overflow). FP64 margins are conservative
engineering guards, not a machine-checked interval certificate.
"""
from __future__ import annotations
import numpy as np
from .batch_screen import BatchScreen
from .learned_screen import normalize


def gamma(n, dtype):
    u=np.finfo(dtype).eps/2
    if n*u>=1:raise ValueError('accumulation bound undefined')
    return n*u/(1-n*u)


def up32(x):
    """Round a nonnegative FP64 radius outwards to binary32."""
    a=np.asarray(x,dtype=np.float64)
    b=a.astype(np.float32)
    return np.where(b.astype(np.float64)<a,np.nextafter(b,np.float32(np.inf)),b)


def prefix_norms(z,ranks):
    return np.sqrt(np.cumsum(z*z,axis=1)[:,np.asarray(ranks)-1])


class QuantizedBatchScreen(BatchScreen):
    """Single packed corpus projection, per-prefix outward residual/error radii.

    FP32 or symmetric int16/int8, one FP32 scale per row for integer storage.
    Query projections are rounded to FP32. Integer coefficients are unpacked
    to FP32 per tile, GEMM is FP32, and integer scales multiply in FP64.
    Full normalized vectors remain FP64. No approximate scores are returned.
    """
    def __init__(self,vectors,basis,ranks=(16,32,64,128),*,storage='int8',build_rows=4096):
        self.x=normalize(vectors)
        if self.x.ndim!=2 or not len(self.x) or build_rows<1:raise ValueError('invalid corpus')
        self.m,self.d=self.x.shape;self.ranks=tuple(ranks)
        if (not self.ranks or any(not isinstance(r,int) or isinstance(r,bool) for r in self.ranks)
                or not 1<=self.ranks[0] or self.ranks[-1]>self.d
                or any(a>=b for a,b in zip(self.ranks,self.ranks[1:]))):raise ValueError('invalid ranks')
        u=np.asarray(basis)
        if np.iscomplexobj(u):raise ValueError('real basis required')
        u=np.asarray(u,dtype=np.float64);r=self.ranks[-1]
        if u.shape!=(self.d,r) or not np.all(np.isfinite(u)):raise ValueError('invalid basis')
        self.slack=256*np.finfo(float).eps*self.d
        defect=float(np.linalg.norm(u.T@u-np.eye(r),ord=2))
        if defect>1e-8:raise ValueError('basis is not orthonormal')
        self.u=np.ascontiguousarray(u/np.sqrt(1+defect+self.slack))
        if storage not in ('float32','int16','int8'):raise ValueError('unknown storage')
        self.storage=storage;dt=np.dtype(storage)
        self.c=np.empty((self.m,r),dtype=dt)
        self.res=np.empty((self.m,len(self.ranks)),dtype=np.float32)
        self.err=np.empty_like(self.res)
        self.scales=None if storage=='float32' else np.empty(self.m,dtype=np.float32)
        # Global upper norm bound on the actual normalized stored rows. The
        # additional FP64 slack covers norm and Gram/projection computations
        # under conventional accumulation; certification of all FP64 steps
        # is deliberately outside this experiment.
        self.xnorm=float(np.sqrt(np.max(np.einsum('ij,ij->i',self.x,self.x))+self.slack))
        for a in range(0,self.m,build_rows):
            x=self.x[a:a+build_rows];z=x@self.u;n=len(z)
            if self.scales is None:
                code=z.astype(dt);reconstructed=code.astype(np.float64)
            else:
                lim=np.iinfo(dt).max
                s=np.maximum(np.max(abs(z),axis=1)/lim,np.finfo(np.float32).tiny).astype(np.float32)
                code=np.clip(np.rint(z/s[:,None]),-lim,lim).astype(dt)
                self.scales[a:a+n]=s
                # Binary32 scale times an integer of <=15 magnitude bits is
                # represented exactly by binary64 (unless under/overflow).
                reconstructed=code.astype(np.float64)*s[:,None]
            self.c[a:a+n]=code
            self.err[a:a+n]=up32(prefix_norms(z-reconstructed,self.ranks)+self.slack)
            mass=np.cumsum(z*z,axis=1)[:,np.array(self.ranks)-1]
            self.res[a:a+n]=up32(np.sqrt(np.maximum(0,1-mass+self.slack)))

    @property
    def index_bytes(self):
        return self.u.nbytes+self.c.nbytes+self.res.nbytes+self.err.nbytes+(0 if self.scales is None else self.scales.nbytes)

    def _query_projection(self,z,rank):
        p=z@self.u[:,:rank];p32=p.astype(np.float32)
        qerr=np.linalg.norm(p-p32.astype(np.float64),axis=1)+self.slack
        qr=np.sqrt(np.maximum(0,1-np.sum(p*p,axis=1)+self.slack))
        qnorm=np.sqrt(np.sum(z*z,axis=1)+self.slack)
        return p32,qr,qerr,qnorm,rank

    def _projection_panel(self,a,end,rank,query):
        # Int unpacking is included in timed search. No persistent decoded cache.
        v=(self.c[a:end,:rank].astype(np.float32,copy=False)@query[0].T).astype(np.float64)
        if self.scales is not None:v*=self.scales[a:end,None]
        return v

    def _uncertainty(self,a,end,ri,query):
        _,qr,eq,yn,rank=query
        ex=self.err[a:end,ri,None].astype(np.float64)
        ax=self.xnorm+ex;ay=yn+eq
        # |c.p - c_hat.p_hat| <= ex*||p|| + ||c_hat||*eq.
        # gamma bounds FP32 dot accumulation; scaled integer dot has the
        # same relative absolute-product error bound after applying scale.
        # 2r+8 covers non-fused multiply/add and final small operations.
        quant=ex*yn[None,:]+ax*eq[None,:]
        roundoff=(gamma(2*rank+8,np.float32)+self.slack)*ax*ay[None,:]
        return self.res[a:end,ri,None].astype(np.float64)*qr[None,:]+quant+roundoff+self.slack

    def bounds(self,queries,rank,*,tile_rows=4096):
        """Audit iterator: (row offset, lower, upper) for normalized queries."""
        q=normalize(queries)
        if q.ndim==1:q=q[None,:]
        if q.ndim!=2 or q.shape[1]!=self.d or rank not in self.ranks or tile_rows<1:raise ValueError('invalid query')
        p=self._query_projection(q,rank);ri=self.ranks.index(rank)
        for a in range(0,self.m,tile_rows):
            end=min(a+tile_rows,self.m);s=self._projection_panel(a,end,rank,p);e=self._uncertainty(a,end,ri,p)+2*self.slack
            yield a,s-e,s+e


class NativeInt8Screen(QuantizedBatchScreen):
    """Optional exact integer int8 x int16 projection panel, rank <= 512.

    Query scale is binary32, and all integer sums fit signed int32 because
    the encoder uses symmetric ranges [-127,127], [-32767,32767]. The only
    non-integer panel arithmetic is two final FP64 scale multiplications.
    Load an explicitly built, trusted local shared library.
    """
    def __init__(self,vectors,basis,library,ranks=(16,32,64,128),*,build_rows=4096):
        import ctypes as ct
        from pathlib import Path
        if not ranks or max(ranks)>512:raise ValueError('native rank must be <= 512')
        super().__init__(vectors,basis,ranks,storage='int8',build_rows=build_rows)
        self.lib=ct.CDLL(str(Path(library).resolve()));self.kernel=self.lib.qb_int8_i16_panel
        ptr=lambda dt:np.ctypeslib.ndpointer(dtype=dt,flags='C_CONTIGUOUS')
        self.kernel.argtypes=[ct.c_int64,ct.c_int32,ct.c_int32,ct.c_int32,ptr(np.int8),ptr(np.int16),ptr(np.int32)]
        self.kernel.restype=ct.c_int

    def _query_projection(self,z,rank):
        p=z@self.u[:,:rank]
        scale=np.maximum(np.max(abs(p),axis=1)/32767,np.finfo(np.float32).tiny).astype(np.float32)
        code=np.ascontiguousarray(np.clip(np.rint(p/scale[:,None]),-32767,32767).astype(np.int16))
        err=np.linalg.norm(p-code.astype(float)*scale[:,None],axis=1)+self.slack
        qr=np.sqrt(np.maximum(0,1-np.sum(p*p,axis=1)+self.slack))
        qnorm=np.sqrt(np.sum(z*z,axis=1)+self.slack)
        return code,qr,err,qnorm,rank,scale

    def _projection_panel(self,a,end,rank,query):
        code=query[0];out=np.empty((end-a,len(code)),dtype=np.int32)
        rc=self.kernel(end-a,len(code),self.c.shape[1],rank,self.c[a:end],code,out)
        if rc!=0:raise RuntimeError('native kernel rejected dimensions')
        return out.astype(float)*self.scales[a:end,None]*query[5][None,:]

    def _uncertainty(self,a,end,ri,query):
        _,qr,eq,yn,rank,_=query
        ex=self.err[a:end,ri,None].astype(float);ax=self.xnorm+ex;ay=yn+eq
        return (self.res[a:end,ri,None].astype(float)*qr[None,:]+ex*yn[None,:]
                +ax*eq[None,:]+(gamma(4,np.float64)+self.slack)*ax*ay[None,:]+self.slack)
