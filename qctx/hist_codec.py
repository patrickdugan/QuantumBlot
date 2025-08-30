from __future__ import annotations
import json
from typing import Dict, List
import numpy as np
from sklearn.cluster import MiniBatchKMeans
from . import big
from .motifs import pull_motifs

def _project_stream_to_pca(vectors_jsonl:str, basis_json:dict, batch_size:int=4096):
    import numpy as _np
    mean=_np.asarray(basis_json['mean'],dtype=float); comps=_np.asarray(basis_json['components'],dtype=float)
    ids=[]; X=[]
    with open(vectors_jsonl,'r',encoding='utf-8') as f:
        for line in f:
            obj=json.loads(line); ids.append(int(obj['id'])); X.append(obj['vector'])
            if len(X)>=batch_size:
                Xb=_np.asarray(X,dtype=float)-mean; Yb=Xb@comps.T
                yield ids, Yb; ids=[]; X=[]
    if X:
        Xb=_np.asarray(X,dtype=float)-mean; Yb=Xb@comps.T
        yield ids, Yb

def _kmeans_fit_predict_stream(vectors_jsonl:str, basis_json:dict, k:int=32, batch_size:int=4096):
    km=MiniBatchKMeans(n_clusters=k, random_state=42, batch_size=batch_size*2, n_init='auto')
    for ids, Y in _project_stream_to_pca(vectors_jsonl, basis_json, batch_size=batch_size): km.partial_fit(Y)
    labels={}
    for ids, Y in _project_stream_to_pca(vectors_jsonl, basis_json, batch_size=batch_size):
        preds=km.predict(Y)
        for i,lab in zip(ids,preds): labels[i]=int(lab)
    return labels, km.cluster_centers_.tolist()

def _collect_cluster_motifs(chunks_jsonl:str, labels_by_id:Dict[int,int], top_k:int=12):
    from collections import defaultdict, Counter
    per_cluster_tokens=defaultdict(list); counts=defaultdict(int)
    with open(chunks_jsonl,'r',encoding='utf-8') as f:
        for line in f:
            obj=json.loads(line); i=int(obj['id']); lab=labels_by_id.get(i)
            if lab is None: continue
            toks=pull_motifs(obj['text'], top_k=top_k*4)
            per_cluster_tokens[lab].extend(toks); counts[lab]+=1
    motifs={}
    for lab,toks in per_cluster_tokens.items():
        c=Counter(toks); motifs[lab]=[w for w,_ in c.most_common(top_k)]
    return counts, motifs

def _summarize_cluster(cluster_id:int, size:int, centroid:List[float], motifs:List[str])->str:
    head=', '.join(motifs[:4]) if motifs else 'misc'
    tail=', '.join(motifs[4:8]) if len(motifs)>4 else ''
    mag=float(np.linalg.norm(np.asarray(centroid)))
    s1=f'Cluster {cluster_id} (~{size} chunks): themes — {head}'+(f'; {tail}.' if tail else '.')
    s2=f'Semantic intensity (||centroid||) ≈ {mag:.2f}; use this as a prior if your query aligns with these motifs.'
    return s1+' '+s2

def _timeline(chunks_jsonl:str, labels_by_id:Dict[int,int], slices:int=10):
    max_id=-1
    with open(chunks_jsonl,'r',encoding='utf-8') as f:
        for line in f: max_id=max(max_id, int(json.loads(line)['id']))
    if max_id<0: return []
    width=max(1,(max_id+1)//slices); buckets=[[] for _ in range(slices)]; texts=[[] for _ in range(slices)]
    with open(chunks_jsonl,'r',encoding='utf-8') as f:
        for line in f:
            obj=json.loads(line); i=int(obj['id']); lab=labels_by_id.get(i); idx=min(i//width, slices-1)
            if lab is not None: buckets[idx].append(lab)
            if obj.get('text',''): texts[idx].append(obj['text'])
    from collections import Counter
    out=[]
    for s in range(slices):
        if not buckets[s]:
            out.append({'range':[s*width, min((s+1)*width-1, max_id)], 'dominant_clusters':[], 'motifs':[]}); continue
        c=Counter(buckets[s]).most_common(3); joined='\n'.join(texts[s][:200]); m=pull_motifs(joined, top_k=10)
        out.append({'range':[s*width, min((s+1)*width-1, max_id)], 'dominant_clusters':[int(x[0]) for x in c], 'motifs': m})
    return out

def build_hist_codec(chunks_jsonl:str, vectors_jsonl:str, basis_json:dict, k:int=32, slices:int=10)->Dict:
    labels_by_id, centers=_kmeans_fit_predict_stream(vectors_jsonl, basis_json, k=k, batch_size=4096)
    counts, motifs=_collect_cluster_motifs(chunks_jsonl, labels_by_id, top_k=12)
    clusters=[]
    for cid in range(k):
        size=int(counts.get(cid,0)); centroid=centers[cid] if cid<len(centers) else [0.0]*len(basis_json['components'])
        mv=motifs.get(cid, []); summary=_summarize_cluster(cid, size, centroid, mv)
        clusters.append({'id':cid,'size':size,'centroid':[round(float(x),5) for x in centroid],'motifs':mv,'summary':summary})
    timeline=_timeline(chunks_jsonl, labels_by_id, slices=slices)
    global_m=[]
    for c in clusters:
        for w in c['motifs'][:6]:
            if w not in global_m: global_m.append(w)
    if not global_m and len(timeline): global_m=timeline[0]['motifs']
    return {'clusters':clusters,'timeline':timeline,'global_motifs':global_m[:400]}
