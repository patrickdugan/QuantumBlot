from __future__ import annotations
import os, json, math
from typing import List, Dict, Any
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import numpy as np

from .chroma_io import create_or_get_collection, search as chroma_search, get_items as chroma_get
from .qft_pca import fourier_fingerprint

app = FastAPI(title="qctx MCP server", version="0.1.0")

PERSIST = os.environ.get("QCTX_PERSIST", "./chroma")
COLLECTION = os.environ.get("QCTX_COLLECTION", "qctx")
BASIS_PATH = os.environ.get("QCTX_BASIS_PATH", "./basis.json")  # optional PCA basis

_basis = None
if os.path.exists(BASIS_PATH):
    try:
        _basis = json.loads(open(BASIS_PATH, "r", encoding="utf-8").read())
    except Exception:
        _basis = None

class SearchReq(BaseModel):
    query: str
    k: int = 64
    model: str = "all-MiniLM-L6-v2"

class GetReq(BaseModel):
    ids: List[int]

class SliceItem(BaseModel):
    id: int
    vector: List[float]
    pos: int

class SliceReq(BaseModel):
    items: List[SliceItem]
    pc1_len: int = 4096
    shots: int = 2048

@app.post("/vector/search")
def vector_search(req: SearchReq):
    hits = chroma_search(PERSIST, COLLECTION, req.query, k=req.k, model_name=req.model)
    return {"hits": hits}

@app.post("/vector/get")
def vector_get(req: GetReq):
    items = chroma_get(PERSIST, COLLECTION, req.ids)
    return {"items": items}

def _centroid_qft(vectors: List[List[float]], shots: int = 2048):
    v = np.abs(np.mean(np.asarray(vectors, float), axis=0))
    n = int(math.ceil(math.log2(len(v)))); size = 1 << n
    if len(v) < size:
        pad = np.zeros(size); pad[:len(v)] = v; v = pad
    elif len(v) > size:
        v = v[:size]
    v = v / (np.linalg.norm(v) or 1.0)
    return fourier_fingerprint(v, shots=shots)

def _pc1_qft(vectors: List[List[float]], basis: Dict[str, Any], positions: List[int], target_len: int = 4096, shots: int = 2048):
    comps = np.asarray(basis["components"], float)
    mean  = np.asarray(basis["mean"], float)
    X = np.asarray(vectors, float) - mean
    pc1 = X @ comps[0]
    if positions:
        order = np.argsort(np.asarray(positions))
        pc1 = pc1[order]
    n = len(pc1)
    if n > target_len:
        win = n / float(target_len)
        ds = [pc1[round(i*win):round((i+1)*win)].mean() for i in range(target_len)]
    else:
        ds = pc1.tolist()
        if n < target_len:
            ds += [0.0]*(target_len-n)
    a = np.abs(np.asarray(ds) - float(np.mean(ds)))
    a = a / (np.linalg.norm(a) or 1.0)
    return fourier_fingerprint(a, shots=shots)

@app.post("/spectral/slice_hint")
def spectral_slice(req: SliceReq):
    if not req.items:
        raise HTTPException(400, "No items provided.")
    vectors = [it.vector for it in req.items]
    positions = [it.pos for it in req.items]
    centroid = _centroid_qft(vectors, shots=req.shots)
    pc1 = None
    if _basis is not None:
        pc1 = _pc1_qft(vectors, _basis, positions, target_len=req.pc1_len, shots=req.shots)
    return {"centroid_qft": {"counts": centroid}, "pc1_qft": {"counts": pc1} if pc1 else None}

# Run with: uvicorn qctx.mcp_server:app --host 0.0.0.0 --port 8000
