from __future__ import annotations
import json
import math
import re
from pathlib import Path
from typing import Iterable, Iterator, List, Optional

from .loader import preprocess_markdown
from .html_text import html_to_text_stream

try:
    from sentence_transformers import SentenceTransformer  # type: ignore
except Exception:  # pragma: no cover
    SentenceTransformer = None  # type: ignore

from sklearn.decomposition import IncrementalPCA
import numpy as np


def stream_lines(path: str, assume_html: Optional[bool] = None) -> Iterable[str]:
    """Stream a large file line-by-line; converts HTML to text if needed."""
    p = Path(path)
    is_html = assume_html if assume_html is not None else p.suffix.lower() in {".html", ".htm"}
    with p.open("r", encoding="utf-8", errors="ignore") as f:
        if is_html:
            yield from html_to_text_stream(f)
        else:
            for line in f:
                yield line


def stream_chunker(lines: Iterable[str], limit_bytes: int = 2048) -> Iterator[str]:
    """Chunk a text stream by utf-8 bytes with whitespace preferences, streaming-friendly."""
    buf = bytearray()
    ws_re = re.compile(r"(\s+)")
    for line in lines:
        # minimal markdown scrub per line before chunking
        line = line.replace("`", "")
        parts = ws_re.split(line)
        for part in parts:
            b = part.encode("utf-8", errors="ignore")
            if len(buf) + len(b) > limit_bytes and len(buf) > 0:
                yield buf.decode("utf-8", errors="ignore")
                buf = bytearray()
            buf.extend(b)
    if buf:
        yield buf.decode("utf-8", errors="ignore")


def write_jsonl_chunks(input_path: str, output_jsonl: str, limit_bytes: int = 2048, assume_html: Optional[bool] = None) -> int:
    """Create a JSONL file where each line is {"id": i, "text": chunk}. Returns count."""
    i = 0
    with open(output_jsonl, "w", encoding="utf-8") as out:
        for chunk in stream_chunker(stream_lines(input_path, assume_html=assume_html), limit_bytes=limit_bytes):
            chunk = preprocess_markdown(chunk)
            if chunk.strip():
                rec = {"id": i, "text": chunk}
                out.write(json.dumps(rec, ensure_ascii=False) + "\n")
                i += 1
    return i


def embed_jsonl(jsonl_path: str, output_jsonl: str, model_name: str = "all-MiniLM-L6-v2", batch_size: int = 64) -> int:
    """Read {id,text} JSONL, write {id,vector} JSONL using sentence-transformers in batches."""
    if SentenceTransformer is None:
        raise ImportError("sentence-transformers is not installed.")
    model = SentenceTransformer(model_name)
    cache_texts: List[str] = []
    cache_ids: List[int] = []
    total = 0
    with open(jsonl_path, "r", encoding="utf-8") as f, open(output_jsonl, "w", encoding="utf-8") as out:
        for line in f:
            obj = json.loads(line)
            cache_ids.append(int(obj["id"]))
            cache_texts.append(obj["text"])
            if len(cache_texts) >= batch_size:
                vecs = model.encode(cache_texts, batch_size=batch_size)
                for i, v in zip(cache_ids, vecs):
                    out.write(json.dumps({"id": i, "vector": list(map(float, (v.tolist() if hasattr(v, 'tolist') else v)))}) + "\n")
                total += len(cache_texts)
                cache_texts.clear(); cache_ids.clear()
        if cache_texts:
            vecs = model.encode(cache_texts, batch_size=batch_size)
            for i, v in zip(cache_ids, vecs):
                out.write(json.dumps({"id": i, "vector": list(map(float, (v.tolist() if hasattr(v, 'tolist') else v)))}) + "\n")
            total += len(cache_texts)
    return total


def incremental_pca_jsonl(vectors_jsonl: str, output_jsonl: str, n_components: int = 2, batch_size: int = 1024) -> int:
    """Stream vectors from JSONL and write reduced vectors as {id, vector}."""
    # First pass: determine dimensionality and count
    dim = None
    n = 0
    with open(vectors_jsonl, "r", encoding="utf-8") as f:
        for line in f:
            obj = json.loads(line)
            v = obj["vector"]
            dim = len(v) if dim is None else dim
            n += 1
    if dim is None:
        raise ValueError("No vectors found.")

    ipca = IncrementalPCA(n_components=n_components)
    # Second pass: partial_fit
    X_batch = []
    with open(vectors_jsonl, "r", encoding="utf-8") as f:
        for line in f:
            obj = json.loads(line)
            v = obj["vector"]
            X_batch.append(v)
            if len(X_batch) >= batch_size:
                ipca.partial_fit(np.asarray(X_batch, dtype=float))
                X_batch = []
        if X_batch:
            ipca.partial_fit(np.asarray(X_batch, dtype=float))

    # Third pass: transform and write
    wrote = 0
    X_batch = []
    ids = []
    with open(vectors_jsonl, "r", encoding="utf-8") as f, open(output_jsonl, "w", encoding="utf-8") as out:
        for line in f:
            obj = json.loads(line)
            ids.append(int(obj["id"]))
            X_batch.append(obj["vector'])
            if len(X_batch) >= batch_size:
                Y = ipca.transform(np.asarray(X_batch, dtype=float)).tolist()
                for i, y in zip(ids, Y):
                    out.write(json.dumps({"id": i, "vector": y}) + "\n")
                wrote += len(X_batch)
                X_batch = []; ids = []
        if X_batch:
            Y = ipca.transform(np.asarray(X_batch, dtype=float)).tolist()
            for i, y in zip(ids, Y):
                out.write(json.dumps({"id": i, "vector": y}) + "\n")
            wrote += len(X_batch)
    return wrote


def centroid_amplitude_from_vectors_jsonl(vectors_jsonl: str, take: Optional[int] = None) -> np.ndarray:
    """Build an amplitude vector from the centroid of embeddings for QFT.

    Steps: mean across rows -> abs -> normalize -> pad/trim to nearest power of two.
    """
    import numpy as np
    mean = None
    count = 0
    with open(vectors_jsonl, "r", encoding="utf-8") as f:
        for k, line in enumerate(f):
            if take is not None and k >= take:
                break
            obj = json.loads(line)
            v = np.asarray(obj["vector"], dtype=float)
            mean = v if mean is None else mean + v
            count += 1
    if count == 0:
        raise ValueError("No vectors found for centroid.")
    mean = mean / float(count)
    amp = np.abs(mean)
    # pad/trim to nearest power of two
    L = amp.shape[0]
    n = int(math.ceil(math.log2(L)))
    size = 1 << n
    if L < size:
        padded = np.zeros(size, dtype=float)
        padded[:L] = amp
        amp = padded
    elif L > size:
        amp = amp[:size]
    # normalize
    s = np.linalg.norm(amp)
    if s == 0:
        amp[0] = 1.0
        s = 1.0
    amp = amp / s
    return amp
