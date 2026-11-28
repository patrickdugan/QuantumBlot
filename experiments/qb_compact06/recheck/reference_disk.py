"""Immutable local backing store and bounded-error int8 cosine screening.

Exact-arithmetic enclosures; conventional floating point with engineering slack.
Returned scores use stored normalized float64 rows, never compressed scores.
"""
from __future__ import annotations
import hashlib
import json
import os
from pathlib import Path
import shutil
import tempfile
import numpy as np
from qctx.learned_screen import normalize, rank_top
from qctx.quantized_screen import gamma, up32


def digest(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for b in iter(lambda: f.read(1 << 20), b''):
            h.update(b)
    return h.hexdigest()


def stamp(st):
    return st.st_dev, st.st_ino, st.st_size, st.st_mtime_ns, st.st_ctime_ns


def encode(z, slack):
    scale = np.maximum(np.max(abs(z), axis=1)/127, np.finfo(np.float32).tiny).astype(np.float32)
    code = np.clip(np.rint(z/scale[:, None]), -127, 127).astype(np.int8)
    err = up32(np.linalg.norm(z-code.astype(float)*scale[:, None], axis=1)+slack)
    return code, scale, err


def build_index(path, vectors, basis, *, block=1024):
    """Stream rows into a NEW directory; no existing index is overwritten."""
    path = Path(path).resolve()
    if path.exists():
        raise FileExistsError(path)
    x = np.asarray(vectors); u = np.asarray(basis)
    if x.ndim != 2 or min(x.shape) < 1 or np.iscomplexobj(x) or block < 1:
        raise ValueError('invalid corpus or block size')
    m,d = x.shape
    if u.ndim != 2 or u.shape[0] != d or not 1 <= u.shape[1] <= d or np.iscomplexobj(u) or not np.all(np.isfinite(u)):
        raise ValueError('invalid frame')
    u = np.asarray(u, dtype=np.float64); r = u.shape[1]
    slack = 256*np.finfo(float).eps*d
    defect = float(np.linalg.norm(u.T@u-np.eye(r), ord=2))
    if defect > 1e-8:
        raise ValueError('frame must be orthonormal')
    u = u/np.sqrt(1+defect+slack)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = Path(tempfile.mkdtemp(prefix='.qb-build-', dir=path.parent))
    specs = {'codes': ((m,d),'i1'), 'scale': ((m,),'<f4'), 'error': ((m,),'<f4'),
             'proj': ((m,r),'i1'), 'pscale': ((m,),'<f4'), 'perror': ((m,),'<f4'),
             'residual': ((m,),'<f4')}
    try:
        arrays = {k: np.lib.format.open_memmap(tmp/(k+'.npy'), mode='w+', dtype=dt, shape=sh)
                  for k,(sh,dt) in specs.items()}
        normmax = 0.
        with open(tmp/'rows.bin', 'xb') as f:
            for a in range(0,m,block):
                z = normalize(x[a:a+block]); end = a+len(z); p = z@u
                z2 = np.einsum('ij,ij->i',z,z); normmax = max(normmax,float(z2.max()))
                for names,v in ((('codes','scale','error'),z),(('proj','pscale','perror'),p)):
                    for key,value in zip(names,encode(v,slack)):
                        arrays[key][a:end] = value
                arrays['residual'][a:end] = up32(np.sqrt(np.maximum(0,z2-np.sum(p*p,axis=1)+slack)))
                f.write(np.asarray(z,dtype='<f8',order='C').tobytes())
            f.flush(); os.fsync(f.fileno())
        for arr in arrays.values(): arr.flush()
        del arr, arrays
        np.save(tmp/'basis.npy',u,allow_pickle=False)
        names = ['rows.bin','basis.npy',*(k+'.npy' for k in specs)]
        meta = dict(schema='qb-disk-v1',m=m,d=d,r=r,slack=slack,xnorm=float(np.sqrt(normmax+slack)),
                    files={n:{'bytes':(tmp/n).stat().st_size,'sha256':digest(tmp/n)} for n in names})
        (tmp/'manifest.json').write_text(json.dumps(meta,indent=2)+'\n',encoding='utf-8')
        if path.exists(): raise FileExistsError(path)
        tmp.rename(path)
    finally:
        if tmp.exists(): shutil.rmtree(tmp)
    return meta


class DiskScreen:
    """Open a trusted, immutable index. Verify all files once on open by default.

    Resident bytes count explicit NumPy arrays only, not OS cache or temporaries.
    Search holds at most M*query_block upper scores and bounded tile panels.
    """
    def __init__(self,path,*,verify=True):
        self.path=Path(path).resolve(); meta=json.loads((self.path/'manifest.json').read_text())
        if meta['schema']!='qb-disk-v1':raise ValueError('unsupported index')
        self.m,self.d,self.r=[int(meta[k]) for k in ('m','d','r')]
        m,d,r=self.m,self.d,self.r
        if min(m,d,r)<1 or r>d:raise ValueError('invalid dimensions')
        self.slack=float(meta['slack']);self.xnorm=float(meta['xnorm'])
        if not np.isfinite(self.xnorm) or self.xnorm<=0 or self.slack!=256*np.finfo(float).eps*d:raise ValueError('invalid numerical metadata')
        specs={'codes':((m,d),'i1'),'scale':((m,),'<f4'),'error':((m,),'<f4'),
               'proj':((m,r),'i1'),'pscale':((m,),'<f4'),'perror':((m,),'<f4'),
               'residual':((m,),'<f4'),'basis':((d,r),'<f8')}
        expected={'rows.bin',*(k+'.npy' for k in specs)}
        if set(meta['files'])!=expected:raise ValueError('invalid file roster')
        for n,f in meta['files'].items():
            p=self.path/n
            if p.stat().st_size!=f['bytes'] or (verify and digest(p)!=f['sha256']):raise ValueError('index integrity mismatch: '+n)
        self.backing=self.path/'rows.bin';self._stamp=stamp(self.backing.stat())
        if self._stamp[2]!=m*d*8:raise ValueError('backing size mismatch')
        for k,(shape,dt) in specs.items():
            a=np.load(self.path/(k+'.npy'),allow_pickle=False)
            if a.shape!=shape or a.dtype!=np.dtype(dt) or not np.all(np.isfinite(a)):raise ValueError('invalid array: '+k)
            if k in ('scale','pscale') and np.any(a<=0):raise ValueError('invalid scale')
            if k in ('error','perror','residual') and np.any(a<0):raise ValueError('invalid radius')
            if k in ('codes','proj') and np.any(a==-128):raise ValueError('invalid code')
            a.flags.writeable=False;setattr(self,k,a)
        self.index_bytes=sum(getattr(self,k).nbytes for k in specs)

    def _check(self,f):
        if stamp(os.fstat(f.fileno()))!=self._stamp or stamp(self.backing.stat())!=self._stamp:
            raise RuntimeError('backing changed; reopen verified index')

    def _bounds(self,a,end,q,q32,qe,qn,*,projected=False,selected=None):
        # selected is relative to the tile; avoid decoding non-survivor rows.
        rows=slice(a,end) if selected is None else a+selected
        if projected:
            c,s,e=self.proj[rows],self.pscale[rows],self.perror[rows]
        else:
            c,s,e=self.codes[rows],self.scale[rows],self.error[rows]
        p=(c.astype(np.float32)@q32.T).astype(float)*s[:,None]
        e=e.astype(float)[:,None];ax=self.xnorm+e
        radius=e*qn+ax*qe+(gamma(2*q.shape[1]+8,np.float32)+self.slack)*ax*(qn+qe)+self.slack
        if projected:
            radius+=self.residual[rows,None]*np.sqrt(np.maximum(0,qn*qn-np.sum(q*q,axis=1)+self.slack))
        return p-radius,p+radius

    def search(self,queries,k=20,*,mode='mmap',tile=1024,query_block=24,gap=0):
        """Modes full mmap, full file read, codes, projection, cascade.

        gap is the maximum number of noncandidate rows to bridge in one read.
        Full and screened paths share exact-score reranking and tie handling.
        """
        q=normalize(queries)
        if q.ndim==1:q=q[None,:]
        if q.ndim!=2 or q.shape[1]!=self.d or not len(q) or not 1<=k<=self.m or min(tile,query_block)<1 or gap<0 or mode not in ('mmap','read','codes','projection','cascade'):
            raise ValueError('invalid search arguments')
        m,d=self.m,self.d;out=[];scores=[]
        stats=dict(backing_bytes=0,read_calls=0,mmap_tiles=0,logical_page_visits=0,exact_pairs=0,
                   projection_pairs=0,code_pairs=0,candidate_pairs=0,union_rows=0,tie_rescores=0,max_upper_cells=0,max_read_rows=0)
        with open(self.backing,'rb',buffering=0) as f:
            self._check(f)
            mm=np.memmap(f,dtype='<f8',mode='r',shape=(m,d)) if mode=='mmap' else None
            try:
                for start in range(0,len(q),query_block):
                    z=q[start:start+query_block];b=len(z); best=[(np.empty(0,dtype=np.int64),np.empty(0)) for _ in z]
                    norm=np.sqrt(np.sum(z*z,axis=1)+self.slack)
                    def prep(p):
                        fp=p.astype(np.float32);err=np.linalg.norm(p-fp.astype(float),axis=1)+self.slack
                        return p,fp,err,norm
                    upper=None
                    if mode not in ('mmap','read'):
                        upper=np.empty((m,b));stats['max_upper_cells']=max(stats['max_upper_cells'],upper.size)
                        lows=np.full((k,b),-np.inf)
                        projected=mode!='codes'; pq=prep(z@self.basis if projected else z)
                        for a in range(0,m,tile):
                            end=min(m,a+tile);lo,hi=self._bounds(a,end,*pq,projected=projected)
                            upper[a:end]=hi;combined=np.concatenate((lows,lo),axis=0)
                            lows=np.partition(combined,len(combined)-k,axis=0)[-k:]
                            stats['projection_pairs' if projected else 'code_pairs']+=hi.size
                        thresh=lows.min(axis=0)
                        if mode=='cascade':
                            lows.fill(-np.inf);cq=prep(z)
                            for a in range(0,m,tile):
                                end=min(m,a+tile);mask=upper[a:end]>=thresh-self.slack
                                rr=np.flatnonzero(mask.any(axis=1));upper[a:end]=-np.inf
                                if not len(rr):continue
                                lo,hi=self._bounds(a,end,*cq,selected=rr);stats['code_pairs']+=hi.size
                                lo=np.where(mask[rr],lo,-np.inf);hi=np.where(mask[rr],hi,-np.inf)
                                upper[a+rr]=hi;combined=np.concatenate((lows,lo),axis=0)
                                lows=np.partition(combined,len(combined)-k,axis=0)[-k:]
                            thresh=np.maximum(thresh,lows.min(axis=0))
                    last_page=-1
                    for a in range(0,m,tile):
                        end=min(m,a+tile);mask=np.ones((end-a,b),bool) if upper is None else upper[a:end]>=thresh-self.slack
                        rr=np.flatnonzero(mask.any(axis=1));stats['candidate_pairs']+=int(mask.sum());stats['union_rows']+=len(rr)
                        if not len(rr):continue
                        # Split sorted row IDs into coalesced ranges, capped by tile.
                        ranges=np.split(rr,np.flatnonzero(np.diff(rr)>gap+1)+1)
                        for group in ranges:
                            lo=int(group[0])+a;hi=int(group[-1])+a+1; n=hi-lo;size=n*d*8;offset=lo*d*8
                            if mm is not None:
                                rows=mm[lo:hi];stats['mmap_tiles']+=1
                            else:
                                f.seek(offset);raw=f.read(size)
                                if len(raw)!=size:raise RuntimeError('short backing read')
                                rows=np.frombuffer(raw,dtype='<f8').reshape(n,d);stats['read_calls']+=1
                            stats['backing_bytes']+=size;stats['max_read_rows']=max(stats['max_read_rows'],n)
                            first=offset//4096;last=(offset+size-1)//4096
                            stats['logical_page_visits']+=max(0,last-max(last_page,first-1));last_page=max(last_page,last)
                            v=rows@z.T;stats['exact_pairs']+=v.size
                            # Exact scores use one fixed einsum reduction on the
                            # contenders, independent of GEMM shape and range.
                            for j in range(b):
                                ix=np.flatnonzero(mask[lo-a:hi-a,j]);ss=v[ix,j]
                                if not len(ix):continue
                                if len(ix)>k:
                                    cut=np.partition(ss,len(ss)-k)[len(ss)-k]
                                    ix=ix[ss>=cut-4*self.slack]
                                ss=np.einsum('ij,j->i',rows[ix],z[j],optimize=False)
                                stats['tie_rescores']+=len(ix);stats['exact_pairs']+=len(ix)
                                oi,oscore=best[j];best[j]=rank_top(np.concatenate((oscore,ss)),np.concatenate((oi,lo+ix)),k)
                        if upper is not None:
                            thresh=np.maximum(thresh,np.array([s[-1]-self.slack if len(s)==k else -np.inf for _,s in best]))
                    for i,s in best:
                        if len(i)!=k:raise RuntimeError('insufficient survivors; disable screen')
                        out.append(i.tolist());scores.append(s.tolist())
                self._check(f)
            finally:
                if mm is not None:del mm
        return dict(indices=out,scores=scores,stats=stats)
