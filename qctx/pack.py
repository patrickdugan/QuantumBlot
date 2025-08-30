from __future__ import annotations
import json
from typing import Dict
from .raw_codec import build_raw_codec
from . import big
from .hist_codec import build_hist_codec

def build_rope_hints(chunks_jsonl:str, vectors_jsonl:str, out_path:str, n_components:int=8, k_clusters:int=32, shots:int=4096, pc1_len:int=4096, timeline_slices:int=10, model_name:str='all-MiniLM-L6-v2')->Dict:
    raw_pack=build_raw_codec(vectors_jsonl, out_path+'.raw.json', model_name=model_name, n_components=n_components, shots=shots, pc1_target_len=pc1_len)
    basis_path=out_path+'.raw.json.basis.json'
    try:
        basis=json.loads(open(basis_path,'r',encoding='utf-8').read())
    except Exception:
        basis=big.learn_pca_basis_jsonl(vectors_jsonl, basis_out=out_path+'.basis.json', n_components=n_components, batch_size=2048)
    hist=build_hist_codec(chunks_jsonl, vectors_jsonl, basis_json=basis, k=k_clusters, slices=timeline_slices)
    pack={'raw':raw_pack['raw'], 'codec':hist, 'provenance':{'chunks_source':chunks_jsonl,'vectors_source':vectors_jsonl,'pca_components':n_components,'clusters':k_clusters,'pc1_downsample':pc1_len,'shots':shots}}
    with open(out_path,'w',encoding='utf-8') as f: json.dump(pack,f)
    return pack
