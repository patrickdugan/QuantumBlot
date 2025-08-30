from __future__ import annotations
from typing import List
try:
    from sentence_transformers import SentenceTransformer
except Exception:
    SentenceTransformer=None
class EmbeddingBackendMissing(ImportError): pass

def encode_texts(texts:List[str], model_name:str='all-MiniLM-L6-v2')->List[List[float]]:
    if SentenceTransformer is None:
        raise EmbeddingBackendMissing('sentence-transformers is not installed in this environment')
    model=SentenceTransformer(model_name)
    embs=model.encode(texts)
    return [e.tolist() if hasattr(e,'tolist') else list(map(float,e)) for e in embs]
