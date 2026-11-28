"""Disk-backed exact-score cosine screening with packed RAM indexes.

Create immutable stores in a new directory. Originals are normalized binary64
rows in a little-endian flat file. Resident codes are symmetric signed int8 or
int16, one binary32 scale and outward error radius per row. All screening
bounds are exact-arithmetic results with conventional FP32 GEMM assumptions
and engineering FP64 slack, not a machine-certified interval implementation.
"""
from __future__ import annotations
import hashlib
import json
import os
from pathlib import Path
import numpy as np
from .learned_screen import normalize, rank_top
from .quantized_screen import gamma, up32


def digest(path):
    h=hashlib.sha256()
    with open(path,'rb') as f:
        for b in iter(lambda:f.read(1<<20),b''):h.update(b)
    return h.hexdigest()


def pack(x, dtype, slack):
    lim=np.iinfo(dtype).max
    scale=np.maximum(np.max(abs(x),axis=1)/lim,np.finfo(np.float32).tiny).astype(np.float32)
    code=np.clip(np.rint(x/scale[:,None]),-lim,lim).astype(dtype)
    restored=code.astype(np.float64)*scale[:,None]
    err=up32(np.linalg.norm(x-restored,axis=1)+slack)
    return code,scale,err


def _query(x,slack):
    f=x.astype(np.float32)
    err=np.linalg.norm(x-f.astype(np.float64),axis=1)+slack
    return f,err,np.sqrt(np.sum(x*x,axis=1)+slack)


def _panel(code,scale,error,q,norm,slack):
    p,ep,yn=q; e=error.astype(float)[:,None];xn=norm+e
    v=(code.astype(np.float32)@p.T).astype(float)*scale[:,None]
    radius=e*yn[None,:]+xn*ep[None,:]+(gamma(2*code.shape[1]+8,np.float32)+slack)*xn*(yn+ep)[None,:]+slack
    return v,radius


def _top_merge(old, values, k):
    t=np.concatenate((old,values),axis=0)
    return np.partition(t,len(t)-k,axis=0)[-k:]


def build_store(root,vectors,basis=None,*,storage='int8',projection_storage='int16',build_rows=2048,row_ids=None):
    """Stream normalized originals and packed descriptors to an immutable store.

    Input may be a read-only npy memmap. Optional row_ids defines store order;
    search returns store row numbers, not source IDs. Never overwrite a store.
    """
    if storage not in ('int8','int16') or projection_storage not in ('int8','int16') or not isinstance(build_rows,int) or build_rows<1:
        raise ValueError('invalid storage or build_rows')
    x=np.asarray(vectors)
    if x.ndim!=2 or not len(x) or not x.shape[1] or np.iscomplexobj(x):raise ValueError('expected nonempty real corpus')
    if row_ids is not None:
        row_ids=np.asarray(row_ids)
        if row_ids.ndim!=1 or not len(row_ids) or row_ids.dtype.kind not in 'iu' or np.any(row_ids<0) or np.any(row_ids>=len(x)) or len(np.unique(row_ids))!=len(row_ids):raise ValueError('invalid source rows')
    m=len(x) if row_ids is None else len(row_ids);d=x.shape[1];slack=256*np.finfo(float).eps*d
    u=None;r=0
    if basis is not None:
        u=np.asarray(basis)
        if np.iscomplexobj(u):raise ValueError('real basis required')
        u=np.asarray(u,dtype=float)
        if u.ndim!=2 or u.shape[0]!=d or not 1<=u.shape[1]<=d or not np.all(np.isfinite(u)):raise ValueError('invalid basis')
        r=u.shape[1];defect=float(np.linalg.norm(u.T@u-np.eye(r),ord=2))
        if defect>1e-8:raise ValueError('frame must be orthonormal')
        u=np.ascontiguousarray(u/np.sqrt(1+defect+slack))
    # Check the accumulation model before creating any artifacts.
    gamma(2*d+8,np.float32)
    root=Path(root);root.mkdir(parents=True,exist_ok=False)
    fields={'code':((m,d),storage),'scale':((m,),'float32'),'error':((m,),'float32')}
    if r:fields.update({'projection':((m,r),projection_storage),'p_scale':((m,),'float32'),'p_error':((m,),'float32'),'residual':((m,),'float32')})
    arrays={name:np.lib.format.open_memmap(root/(name+'.npy'),mode='w+',shape=shape,dtype=dt) for name,(shape,dt) in fields.items()}
    norm=0.
    with open(root/'rows.f64','xb') as f:
        for a in range(0,m,build_rows):
            end=min(m,a+build_rows);z=normalize(x[a:end] if row_ids is None else x[row_ids[a:end]])
            f.write(np.ascontiguousarray(z,dtype='<f8').tobytes());norm=max(norm,float(np.max(np.linalg.norm(z,axis=1))))
            c,s,e=pack(z,np.dtype(storage),slack)
            for name,v in zip(('code','scale','error'),(c,s,e)):arrays[name][a:end]=v
            if r:
                p=z@u;c,s,e=pack(p,np.dtype(projection_storage),slack)
                res=up32(np.sqrt(np.maximum(0,1-np.sum(p*p,axis=1)+slack)))
                for name,v in zip(('projection','p_scale','p_error','residual'),(c,s,e,res)):arrays[name][a:end]=v
        f.flush();os.fsync(f.fileno())
    for v in arrays.values():v.flush()
    del arrays,v
    if r:np.save(root/'basis.npy',u,allow_pickle=False)
    if row_ids is not None:np.save(root/'source_rows.npy',row_ids.astype(np.int64),allow_pickle=False)
    files={p.name:{'bytes':p.stat().st_size,'sha256':digest(p)} for p in sorted(root.iterdir())}
    meta=dict(schema='qb.disk_screen.v1',rows=m,dimension=d,rank=r,storage=storage,projection_storage=projection_storage,slack=slack,xnorm=norm+slack,files=files)
    # Manifest last; a failed build cannot masquerade as a complete store.
    (root/'manifest.json').write_text(json.dumps(meta,indent=2)+'\n',encoding='utf8')
    return meta


class DiskScreen:
    def __init__(self,root,*,verify=True):
        self.root=Path(root);meta=json.loads((self.root/'manifest.json').read_text(encoding='utf8'))
        if meta.get('schema')!='qb.disk_screen.v1':raise ValueError('unknown store format')
        self.m,self.d,self.r=(int(meta[k]) for k in ('rows','dimension','rank'))
        if min(self.m,self.d)<1 or not 0<=self.r<=self.d:raise ValueError('invalid manifest dimensions')
        self.slack=256*np.finfo(float).eps*self.d;self.norm=float(meta['xnorm'])
        if not np.isfinite(self.norm) or not 1<=self.norm<=1+self.slack*2:raise ValueError('invalid norm bound')
        if meta['storage'] not in ('int8','int16') or meta['projection_storage'] not in ('int8','int16'):raise ValueError('invalid storage')
        spec={'code':((self.m,self.d),meta['storage']),'scale':((self.m,),'float32'),'error':((self.m,),'float32')}
        if self.r:spec.update({'projection':((self.m,self.r),meta['projection_storage']),'p_scale':((self.m,),'float32'),'p_error':((self.m,),'float32'),'residual':((self.m,),'float32'),'basis':((self.d,self.r),'float64')})
        if 'source_rows.npy' in meta['files']:spec['source_rows']=((self.m,),'int64')
        for name in ['rows.f64',*[k+'.npy' for k in spec]]:
            p=self.root/name;v=meta['files'][name]
            if p.stat().st_size!=v['bytes'] or (verify and digest(p)!=v['sha256']):raise ValueError('store integrity failure: '+name)
        for name,(shape,dt) in spec.items():
            z=np.load(self.root/(name+'.npy'),allow_pickle=False)
            if z.shape!=shape or z.dtype!=np.dtype(dt) or not np.all(np.isfinite(z)):raise ValueError('invalid array: '+name)
            if name in ('scale','p_scale') and np.any(z<=0):raise ValueError('nonpositive scale')
            if name in ('error','p_error','residual') and np.any(z<0):raise ValueError('negative radius')
            z.flags.writeable=False;setattr(self,name,z)
        self.index_bytes=sum(getattr(self,k).nbytes for k in spec)
        self.backing_bytes=8*self.m*self.d;self.file=open(self.root/'rows.f64','rb',buffering=0)
        try:
            self.identity=self._identity()
            if self.identity[2]!=self.backing_bytes:raise ValueError('backing size mismatch')
            self.page_size=os.sysconf('SC_PAGE_SIZE') if hasattr(os,'sysconf') else 4096
        except BaseException:
            self.file.close();raise

    def _identity(self):
        s=os.fstat(self.file.fileno());return s.st_dev,s.st_ino,s.st_size,s.st_mtime_ns,s.st_ctime_ns

    def _check(self):
        if self._identity()!=self.identity:raise ValueError('backing file changed; reopen/rebuild')

    def close(self):self.file.close()
    def __enter__(self):return self
    def __exit__(self,*_):self.close()

    def advise(self,cold=False):
        """Best-effort OS advice, never a guarantee of physical cold storage."""
        if not hasattr(os,'posix_fadvise'):return False
        try:
            os.posix_fadvise(self.file.fileno(),0,0,os.POSIX_FADV_DONTNEED if cold else os.POSIX_FADV_NORMAL);return True
        except OSError:return False

    def _read(self,a,end,stats):
        n=(end-a)*8*self.d;self.file.seek(a*8*self.d);b=self.file.read(n)
        self._record(a,end,b,stats)
        return np.frombuffer(b,dtype='<f8').reshape(end-a,self.d)

    def _record(self,a,end,data,stats):
        n=(end-a)*8*self.d
        if len(data)!=n:raise OSError('short backing read')
        stats['read_calls']+=1;stats['read_bytes']+=n;stats['rows_read']+=end-a
        off=a*8*self.d;stats['page_visits']+=(off+n-1)//self.page_size-off//self.page_size+1

    def _gather(self,ids,gap,stats,pool=None):
        # Sorted unique rows. Read contiguous runs with an optional gap budget.
        cuts=np.flatnonzero(np.diff(ids)>gap+1)+1;parts=np.split(ids,cuts)
        if pool is None:
            for part in parts:
                a=int(part[0]);end=int(part[-1])+1
                z=self._read(a,end,stats);yield part,z[part-a]
            return
        # Positional reads avoid racing the file seek position. Limit outstanding
        # work to 2*workers; score order and I/O accounting remain deterministic.
        def fetch(part):
            a=int(part[0]);end=int(part[-1])+1
            data=os.pread(self.file.fileno(),(end-a)*8*self.d,a*8*self.d)
            return part,a,end,data
        width=2*stats['read_workers']
        for start in range(0,len(parts),width):
            for part,a,end,data in pool.map(fetch,parts[start:start+width]):
                self._record(a,end,data,stats)
                z=np.frombuffer(data,dtype='<f8').reshape(end-a,self.d)
                yield part,z[part-a]

    def search(self,queries,k=20,*,mode='cascade',tile_rows=4096,query_block=8,gap_rows=0,read_workers=1):
        """Two-pass bound screen; full explicit-read and mmap baselines.

        The first pass derives a global kth lower bound. In cascade mode,
        projected intervals select which rows need full-code refinement.
        Second pass intersects intervals, then coalesces exact reads across
        the query block. Memory for score panels is O(tile_rows*query_block),
        plus the original-vector tiles; no M-by-B persistent score matrix.
        """
        self._check();q=normalize(queries)
        if q.ndim==1:q=q[None,:]
        if (q.shape[1]!=self.d or not len(q) or not isinstance(k,int) or not 1<=k<=self.m
                or mode not in ('full','mmap','code','projection','cascade') or (mode in ('projection','cascade') and not self.r)
                or not all(isinstance(t,int) and not isinstance(t,bool) for t in (tile_rows,query_block,gap_rows,read_workers)) or min(tile_rows,query_block)<1 or gap_rows<0 or not 1<=read_workers<=8 or (read_workers>1 and not hasattr(os,'pread'))):raise ValueError('invalid search arguments')
        s=dict(read_workers=read_workers,read_calls=0,read_bytes=0,rows_read=0,page_visits=0,mapped_bytes=0,mapped_page_visits=0,
               projection_pairs=0,code_pairs=0,exact_pairs=0,rescore_pairs=0,candidate_pairs=0,query_coordinates=0,max_panel_cells=0)
        results=[];scores=[];m,d=self.m,self.d;slack=self.slack
        mapped=None;pool=None
        if mode=='mmap':mapped=np.memmap(self.file,dtype='<f8',mode='r',shape=(m,d))
        if read_workers>1 and mode not in ('full','mmap'):
            from concurrent.futures import ThreadPoolExecutor
            pool=ThreadPoolExecutor(max_workers=read_workers)
        try:
            for qb in range(0,len(q),query_block):
                z=q[qb:qb+query_block];b=len(z);qcode=_query(z,slack);pquery=None
                if mode in ('projection','cascade'):
                    p=z@self.basis;pquery=_query(p,slack);qr=np.sqrt(np.maximum(0,1-np.sum(p*p,axis=1)+slack));s['query_coordinates']+=b*d*self.r
                best=[(np.empty(0,dtype=np.int64),np.empty(0)) for _ in z]
                def update(rows,vec,mask=None):
                    v=vec@z.T;s['exact_pairs']+=v.size;s['max_panel_cells']=max(s['max_panel_cells'],v.size)
                    for j in range(b):
                        take=np.arange(len(rows)) if mask is None else np.flatnonzero(mask[:,j])
                        if not len(take):continue
                        ii=rows[take];ss=v[take,j]
                        # Stable canonical per-row summation for all potential
                        # top-k contenders, independent of GEMM tile shapes.
                        if len(ss)>k:
                            kth=np.partition(ss,len(ss)-k)[len(ss)-k];keep=ss>=kth-4*slack;take=take[keep];ii=ii[keep]
                        ss=np.sum(vec[take]*z[j],axis=1,dtype=np.float64);s['rescore_pairs']+=len(ii)
                        oi,os_=best[j];best[j]=rank_top(np.concatenate((os_,ss)),np.concatenate((oi,ii)),k)
                def panel(a,end,threshold,previous_lower=None):
                    l=end-a
                    if mode=='code':
                        v,e=_panel(self.code[a:end],self.scale[a:end],self.error[a:end],qcode,self.norm,slack);s['code_pairs']+=l*b;s['max_panel_cells']=max(s['max_panel_cells'],v.size)
                        return v-e-2*slack,v+e+2*slack
                    v,e=_panel(self.projection[a:end],self.p_scale[a:end],self.p_error[a:end],pquery,self.norm,slack)
                    e+=self.residual[a:end,None].astype(float)*qr[None,:];s['projection_pairs']+=l*b
                    lo,hi=v-e-2*slack,v+e+2*slack
                    if mode=='cascade':
                        # Rows whose coarse bound can still reach a current
                        # global lower bound; score their union across queries.
                        if previous_lower is not None:
                            threshold=np.maximum(threshold,np.min(_top_merge(previous_lower,lo,k),axis=0))
                        local=np.flatnonzero(np.any(hi>=threshold[None,:],axis=1))
                        if len(local):
                            ids=a+local;v,e=_panel(self.code[ids],self.scale[ids],self.error[ids],qcode,self.norm,slack);s['code_pairs']+=v.size
                            lo[local]=np.maximum(lo[local],v-e-2*slack);hi[local]=np.minimum(hi[local],v+e+2*slack)
                    s['max_panel_cells']=max(s['max_panel_cells'],l*b)
                    return lo,hi
                if mode in ('full','mmap'):
                    for a in range(0,m,tile_rows):
                        end=min(m,a+tile_rows)
                        if mapped is None:vec=self._read(a,end,s)
                        else:
                            vec=mapped[a:end];n=(end-a)*8*d;off=a*8*d;s['mapped_bytes']+=n;s['mapped_page_visits']+=(off+n-1)//self.page_size-off//self.page_size+1
                        update(np.arange(a,end),vec);s['candidate_pairs']+=(end-a)*b
                else:
                    top=np.full((k,b),-np.inf)
                    for a in range(0,m,tile_rows):
                        end=min(m,a+tile_rows);lo,_=panel(a,end,np.min(top,axis=0),top);top=_top_merge(top,lo,k)
                    threshold=np.min(top,axis=0)
                    for a in range(0,m,tile_rows):
                        end=min(m,a+tile_rows);_,hi=panel(a,end,threshold);mask=hi>=threshold[None,:]-slack
                        s['candidate_pairs']+=int(mask.sum());s['max_panel_cells']=max(s['max_panel_cells'],mask.size)
                        ids=a+np.flatnonzero(np.any(mask,axis=1))
                        if len(ids):
                            # Gather each union row once per tile/query block.
                            # One contiguous GEMM avoids per-read scoring overhead.
                            gathered=np.empty((len(ids),d),dtype=float);offset=0
                            for part,vec in self._gather(ids,gap_rows,s,pool):
                                gathered[offset:offset+len(part)]=vec;offset+=len(part)
                            update(ids,gathered,mask[ids-a])
                        # Each existing exact kth score is another valid lower bound.
                        threshold=np.maximum(threshold,[ss[-1]-slack if len(ss)>=k else -np.inf for _,ss in best])
                for ii,ss in best:
                    if len(ii)!=k:raise RuntimeError('insufficient candidates; disable screening')
                    results.append(ii.tolist());scores.append(ss.tolist())
        finally:
            if pool is not None:pool.shutdown(wait=True)
            if mapped is not None:mapped._mmap.close()
        self._check();s['coordinate_work']=(s['exact_pairs']+s['rescore_pairs']+s['code_pairs'])*d+s['projection_pairs']*self.r+s['query_coordinates']
        s['backing_logical_bytes']=s['read_bytes']+s['mapped_bytes'];s['backing_page_visits']=s['page_visits']+s['mapped_page_visits']
        return dict(indices=results,scores=scores,stats=s)
