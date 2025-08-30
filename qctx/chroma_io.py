from __future__ import annotations
import os, json
from typing import List, Dict, Any, Optional

def _import_chromadb():
    try:
        import chromadb  # type: ignore
        from chromadb.config import Settings  # type: ignore
    except Exception as e:
        raise ImportError("chromadb is not installed. pip install chromadb") from e
    return chromadb, Settings

def _client(persist_dir: str):
    chromadb, Settings = _import_chromadb()
    return chromadb.Client(Settings(chroma_db_impl="duckdb+parquet", persist_directory=persist_dir))

def create_or_get_collection(persist_dir: str, name: str = "qctx", metadata: Optional[Dict[str, Any]] = None):
    client = _client(persist_dir)
    try:
        col = client.get_collection(name=name)
    except Exception:
        col = client.create_collection(name=name, metadata=metadata or {"hnsw:space": "cosine"})
    return col

def _read_jsonl(path: str):
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                yield json.loads(line)

def _snippet(text: str, limit: int = 320) -> str:
    text = text.strip().replace("\n", " ")
    if len(text) > limit:
        return text[: limit - 1] + "…"
    return text

def load_into_chroma(chunks_jsonl: str, vectors_jsonl: str, persist_dir: str, collection: str = "qctx", batch: int = 512) -> int:
    col = create_or_get_collection(persist_dir, name=collection)
    vecs = {}
    for obj in _read_jsonl(vectors_jsonl):
        vecs[int(obj["id"])] = obj["vector"]
    ids, embeddings, documents, metadatas = [], [], [], []
    count = 0
    for obj in _read_jsonl(chunks_jsonl):
        i = int(obj["id"])
        v = vecs.get(i)
        if v is None:
            continue
        ids.append(str(i))
        embeddings.append(v)
        documents.append(_snippet(obj.get("text","")))
        metadatas.append({"pos": i})
        if len(ids) >= batch:
            col.upsert(ids=ids, embeddings=embeddings, documents=documents, metadatas=metadatas)
            count += len(ids)
            ids, embeddings, documents, metadatas = [], [], [], []
    if ids:
        col.upsert(ids=ids, embeddings=embeddings, documents=documents, metadatas=metadatas)
        count += len(ids)
    try:
        col._client.persist()
    except Exception:
        pass
    return count

def embed_query(text: str, model_name: str = "all-MiniLM-L6-v2"):
    try:
        from sentence_transformers import SentenceTransformer  # type: ignore
    except Exception as e:
        raise ImportError("sentence-transformers is not installed.") from e
    model = SentenceTransformer(model_name)
    v = model.encode([text])[0]
    return v.tolist() if hasattr(v, "tolist") else list(map(float, v))

def search(persist_dir: str, collection: str, query: str, k: int = 64, model_name: str = "all-MiniLM-L6-v2"):
    col = create_or_get_collection(persist_dir, name=collection)
    qv = embed_query(query, model_name=model_name)
    res = col.query(query_embeddings=[qv], n_results=k)
    out = []
    for idx in range(len(res["ids"][0])):
        out.append({
            "id": int(res["ids"][0][idx]),
            "score": float(res["distances"][0][idx]) if "distances" in res and res["distances"] else None,
            "pos": int(res["metadatas"][0][idx].get("pos", 0)),
            "snippet": res["documents"][0][idx],
        })
    return out

def get_items(persist_dir: str, collection: str, ids: List[int]):
    col = create_or_get_collection(persist_dir, name=collection)
    sids = [str(i) for i in ids]
    res = col.get(ids=sids, include=["embeddings", "documents", "metadatas"])
    out = []
    for i, emb, doc, meta in zip(res["ids"], res.get("embeddings", []), res.get("documents", []), res.get("metadatas", [])):
        out.append({
            "id": int(i),
            "vector": emb,
            "snippet": doc,
            "pos": int(meta.get("pos", 0)) if meta else 0
        })
    return out
