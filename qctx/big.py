from __future__ import annotations
import json, math, re
from pathlib import Path
from typing import Iterable, Iterator, List, Optional
from .loader import preprocess_markdown
from .html_text import html_to_text_stream
try:
    from sentence_transformers import SentenceTransformer
except Exception:
    SentenceTransformer=None
from sklearn.decomposition import IncrementalPCA
import numpy as np

def stream_lines(path:str, assume_html:Optional[bool]=None)->Iterable[str]:
    p=Path(path); is_html=assume_html if assume_html is not None else p.suffix.lower() in {'.html','.htm'}
    with p.open('r', encoding='utf-8', errors='ignore') as f:
        if is_html: yield from html_to_text_stream(f)
        else:
            for line in f: yield line

def stream_chunker(lines:Iterable[str], limit_bytes:int=2048)->Iterator[str]:
    buf=bytearray(); ws_re=re.compile(r'(\s+)')
    for line in lines:
        line=line.replace('`','')
        for part in ws_re.split(line):
            b=part.encode('utf-8', errors='ignore')
            if len(buf)+len(b)>limit_bytes and len(buf)>0:
                yield buf.decode('utf-8', errors='ignore'); buf=bytearray()
            buf.extend(b)
    if buf: yield buf.decode('utf-8', errors='ignore')

def write_jsonl_chunks(input_path:str, output_jsonl:str, limit_bytes:int=2048, assume_html:Optional[bool]=None)->int:
    i=0
    with open(output_jsonl,'w',encoding='utf-8') as out:
        for chunk in stream_chunker(stream_lines(input_path, assume_html=assume_html), limit_bytes=limit_bytes):
            chunk=preprocess_markdown(chunk)
            if chunk.strip():
                out.write(json.dumps({'id':i,'text':chunk}, ensure_ascii=False)+'\n'); i+=1
    return i

def embed_jsonl(jsonl_path:str, output_jsonl:str, model_name:str='all-MiniLM-L6-v2', batch_size:int=64)->int:
    if SentenceTransformer is None: raise ImportError('sentence-transformers is not installed.')
    model=SentenceTransformer(model_name)
    cache_texts=[]; cache_ids=[]; total=0
    with open(jsonl_path,'r',encoding='utf-8') as f, open(output_jsonl,'w',encoding='utf-8') as out:
        for line in f:
            obj=json.loads(line); cache_ids.append(int(obj['id'])); cache_texts.append(obj['text'])
            if len(cache_texts)>=batch_size:
                vecs=model.encode(cache_texts, batch_size=batch_size)
                for i,v in zip(cache_ids, vecs):
                    out.write(json.dumps({'id':i,'vector': (v.tolist() if hasattr(v,'tolist') else list(map(float,v)))})+'\n')
                total+=len(cache_texts); cache_texts.clear(); cache_ids.clear()
        if cache_texts:
            vecs=model.encode(cache_texts, batch_size=batch_size)
            for i,v in zip(cache_ids, vecs):
                out.write(json.dumps({'id':i,'vector': (v.tolist() if hasattr(v,'tolist') else list(map(float,v)))})+'\n')
            total+=len(cache_texts)
    return total

def incremental_pca_jsonl(vectors_jsonl:str, output_jsonl:str, n_components:int=2, batch_size:int=1024)->int:
    dim=None; n=0
    with open(vectors_jsonl,'r',encoding='utf-8') as f:
        for line in f:
            obj=json.loads(line); v=obj['vector']; dim=len(v) if dim is None else dim; n+=1
    if dim is None: raise ValueError('No vectors found.')
    ipca=IncrementalPCA(n_components=n_components)
    X_batch=[]
    with open(vectors_jsonl,'r',encoding='utf-8') as f:
        for line in f:
            obj=json.loads(line); X_batch.append(obj['vector'])
            if len(X_batch)>=batch_size:
                ipca.partial_fit(np.asarray(X_batch,dtype=float)); X_batch=[]
        if X_batch: ipca.partial_fit(np.asarray(X_batch,dtype=float))
    wrote=0; X_batch=[]; ids=[]
    with open(vectors_jsonl,'r',encoding='utf-8') as f, open(output_jsonl,'w',encoding='utf-8') as out:
        for line in f:
            obj=json.loads(line); ids.append(int(obj['id'])); X_batch.append(obj['vector'])
            if len(X_batch)>=batch_size:
                Y=ipca.transform(np.asarray(X_batch,dtype=float)).tolist()
                for i,y in zip(ids,Y): out.write(json.dumps({'id':i,'vector':y})+'\n')
                wrote+=len(X_batch); X_batch=[]; ids=[]
        if X_batch:
            Y=ipca.transform(np.asarray(X_batch,dtype=float)).tolist()
            for i,y in zip(ids,Y): out.write(json.dumps({'id':i,'vector':y})+'\n')
            wrote+=len(X_batch)
    return wrote

def centroid_amplitude_from_vectors_jsonl(vectors_jsonl:str, take:Optional[int]=None):
    mean=None; count=0; import numpy as np
    with open(vectors_jsonl,'r',encoding='utf-8') as f:
        for k,line in enumerate(f):
            if take is not None and k>=take: break
            obj=json.loads(line); v=np.asarray(obj['vector'],dtype=float)
            mean=v if mean is None else mean+v; count+=1
    if count==0: raise ValueError('No vectors found for centroid.')
    mean=mean/float(count); amp=np.abs(mean)
    L=amp.shape[0]; n=int(math.ceil(math.log2(L))); size=1<<n
    if L<size:
        padded=np.zeros(size,dtype=float); padded[:L]=amp; amp=padded
    elif L>size:
        amp=amp[:size]
    s=np.linalg.norm(amp); 
    if s==0: amp[0]=1.0; s=1.0
    amp=amp/s; return amp

def learn_pca_basis_jsonl(vectors_jsonl:str, basis_out:str, n_components:int=8, batch_size:int=1024)->dict:
    dim=None
    with open(vectors_jsonl,'r',encoding='utf-8') as f:
        for line in f:
            obj=json.loads(line); v=obj['vector']; dim=len(v) if dim is None else dim; break
    if dim is None: raise ValueError('No vectors found.')
    ipca=IncrementalPCA(n_components=n_components)
    X_batch=[]
    with open(vectors_jsonl,'r',encoding='utf-8') as f:
        for line in f:
            obj=json.loads(line); X_batch.append(obj['vector'])
            if len(X_batch)>=batch_size:
                ipca.partial_fit(np.asarray(X_batch,dtype=float)); X_batch=[]
        if X_batch: ipca.partial_fit(np.asarray(X_batch,dtype=float))
    basis={'n_components':int(n_components),'components':ipca.components_.tolist(),'explained_variance_ratio':ipca.explained_variance_ratio_.tolist(),'mean':ipca.mean_.tolist()}
    with open(basis_out,'w',encoding='utf-8') as out: json.dump(basis,out)
    return basis

def pc1_series_from_vectors_jsonl(vectors_jsonl:str, basis_json:dict, batch_size:int=2048)->list:
    import numpy as np
    mean=np.asarray(basis_json['mean'],dtype=float); comp1=np.asarray(basis_json['components'][0],dtype=float)
    series=[]; X_batch=[]
    with open(vectors_jsonl,'r',encoding='utf-8') as f:
        for line in f:
            obj=json.loads(line); X_batch.append(obj['vector'])
            if len(X_batch)>=batch_size:
                X=np.asarray(X_batch,dtype=float)-mean; s=X@comp1
                series.extend(s.tolist()); X_batch=[]
        if X_batch:
            X=np.asarray(X_batch,dtype=float)-mean; s=X@comp1
            series.extend(s.tolist())
    return series

def downsample_series_avg(values:list, target_len:int)->list:
    n=len(values)
    if target_len>=n: return values[:]
    win=n/float(target_len); out=[]
    for k in range(target_len):
        start=int(round(k*win)); end=int(round((k+1)*win))
        if end<=start: end=min(start+1,n)
        seg=values[start:end]; out.append(float(sum(seg)/len(seg) if seg else 0.0))
    return out
