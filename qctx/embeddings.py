from __future__ import annotations
from typing import List
try:
    from sentence_transformers import SentenceTransformer
except Exception:
    SentenceTransformer = None

class EmbeddingBackendMissing(ImportError):
    pass

def encode_texts(
    texts: List[str],
    model_name: str = "all-MiniLM-L6-v2",
    batch_size: int = 16
) -> List[List[float]]:
    if SentenceTransformer is None:
        raise EmbeddingBackendMissing("SentenceTransformers not available")
    model = SentenceTransformer(model_name)
    return model.encode(
        texts,
        batch_size=batch_size,
        convert_to_numpy=True,
        show_progress_bar=True,
    ).tolist()
