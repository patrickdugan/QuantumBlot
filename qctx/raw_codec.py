from __future__ import annotations
import json, math
from typing import Dict
import numpy as np
from .big import
from .qft_pca import fourier_fingerprint

def _infer_dim(vectors_jsonl:str)->int:
    with open(vectors_jsonl,'r',encoding='utf-8') as f:
        for line in f:
            return len(json.loads(line)['vector'])
    raise ValueError('Empty vectors JSONL')

def _quantize_matrix(mat,decimals:int=5):
    return [[round(float(x),decimals) for x in row] for row in mat]

def _quantize_list(lst,decimals:int=5):
    return [round(float(x),decimals) for x in lst]

def _pc1_amplitudes(values, power_of_two:int=4096):
    v=np.asarray(values,dtype=float); v=v-float(np.mean(v)); v=np.abs(v)
    s=np.linalg.norm(v); 
    if s==0: v[0]=1.0; s=1.0
    v=v/s; L=v.shape[0]
    if L<power_of_two:
        pad=np.zeros(power_of_two,dtype=float); pad[:L]=v; v=pad
    elif L>power_of_two:
        v=v[:power_of_two]; v=v/np.linalg.norm(v)
    return v

def build_raw_codec(vectors_jsonl:str, out_path:str, model_name:str='all-MiniLM-L6-v2', n_components:int=8, shots:int=4096, pc1_target_len:int=4096)->Dict:
    basis=big.learn_pca_basis_jsonl(vectors_jsonl, basis_out=out_path+'.basis.json', n_components=n_components, batch_size=2048)
    amp_centroid=big.centroid_amplitude_from_vectors_jsonl(vectors_jsonl, take=None)
    counts_centroid=fourier_fingerprint(amp_centroid, shots=shots)
    series=big.pc1_series_from_vectors_jsonl(vectors_jsonl, basis_json=basis, batch_size=4096)
    ds=big.downsample_series_avg(series, target_len=pc1_target_len)
    amp_pc1=_pc1_amplitudes(ds, power_of_two=pc1_target_len)
    counts_pc1=fourier_fingerprint(amp_pc1, shots=shots)
    dim=_infer_dim(vectors_jsonl)
    pack={'raw':{'embedding_spec':{'model':model_name,'dim':int(dim)},
                 'pca':{'components':int(n_components),'basis':_quantize_matrix(basis['components'],5),'explained_variance':_quantize_list(basis['explained_variance_ratio'],5),'mean':_quantize_list(basis['mean'],5)},
                 'qft':{'centroid':{'shots':int(shots),'n_qubits':int(math.log2(len(amp_centroid))),'counts':counts_centroid},
                        'pc1':{'shots':int(shots),'n_qubits':int(math.log2(pc1_target_len)),'downsample':int(pc1_target_len),'counts':counts_pc1}}}}
    with open(out_path,'w',encoding='utf-8') as f: json.dump(pack,f)
    return pack
