from __future__ import annotations
import asyncio, ast, json
from pathlib import Path
from typing import Optional
import typer
from .loader import slurp_text, preprocess_markdown, slice_bytes
from .embeddings import encode_texts, EmbeddingBackendMissing
from .dimred import pca_project
from .qft_pca import fourier_fingerprint
from . import big
from .raw_codec import build_raw_codec
from .pack import build_rope_hints
app=typer.Typer(help='Quantum Context Toolkit CLI')
@app.command('ingest')
def cmd_ingest(path:str, limit:int=2048):
    text=asyncio.run(slurp_text(path)); cleaned=preprocess_markdown(text)
    chunks=slice_bytes(cleaned, limit=limit)
    typer.echo(json.dumps({'count':len(chunks),'chunks':chunks}, ensure_ascii=False))
@app.command('embed')
def cmd_embed(path:str, model:str='all-MiniLM-L6-v2'):
    text=asyncio.run(slurp_text(path)); cleaned=preprocess_markdown(text)
    parts=[p.strip() for p in cleaned.split('\n\n') if p.strip()]
    try: vecs=encode_texts(parts, model_name=model)
    except EmbeddingBackendMissing as e: raise typer.BadParameter(str(e))
    typer.echo(json.dumps({'count':len(vecs),'vectors':vecs}))
@app.command('reduce')
def cmd_reduce(n_components:int=2, vectors_path:Optional[str]=None):
    data=json.loads(Path(vectors_path).read_text(encoding='utf-8')) if vectors_path else json.loads(typer.get_text_stream('stdin').read())
    reduced=pca_project(data, n_components=n_components).tolist()
    typer.echo(json.dumps(reduced))
@app.command('qft')
def cmd_qft(vector:str, shots:int=1024):
    import numpy as np
    arr=np.asarray(ast.literal_eval(vector), dtype=complex).ravel()
    counts=fourier_fingerprint(arr, shots=shots)
    typer.echo(json.dumps(counts))
@app.command('ingest-big')
def cmd_ingest_big(path:str, out:str='chunks.jsonl', limit:int=2048, assume_html:bool=None):
    count=big.write_jsonl_chunks(path, out, limit_bytes=limit, assume_html=assume_html)
    typer.echo(json.dumps({'count':count,'out':out}))
@app.command('embed-big')
def cmd_embed_big(chunks:str='chunks.jsonl', out:str='vectors.jsonl', model:str='all-MiniLM-L6-v2', batch_size:int=64):
    total=big.embed_jsonl(chunks, out, model_name=model, batch_size=batch_size)
    typer.echo(json.dumps({'count':total,'out':out,'model':model}))
@app.command('reduce-big')
def cmd_reduce_big(vectors:str='vectors.jsonl', out:str='reduced.jsonl', n_components:int=2, batch_size:int=1024):
    total=big.incremental_pca_jsonl(vectors, out, n_components=n_components, batch_size=batch_size)
    typer.echo(json.dumps({'count':total,'out':out,'n_components':n_components}))
@app.command('qft-centroid')
def cmd_qft_centroid(vectors:str='vectors.jsonl', shots:int=1024, take:int=None):
    amp=big.centroid_amplitude_from_vectors_jsonl(vectors, take=take)
    counts=fourier_fingerprint(amp, shots=shots)
    typer.echo(json.dumps(counts))
@app.command('qft-pc1')
def cmd_qft_pc1(vectors:str='vectors.jsonl', shots:int=2048, target_len:int=4096):
    import numpy as np
    basis=big.learn_pca_basis_jsonl(vectors, basis_out=vectors+'.pc1.basis.json', n_components=1, batch_size=4096)
    series=big.pc1_series_from_vectors_jsonl(vectors, basis_json=basis, batch_size=4096)
    ds=big.downsample_series_avg(series, target_len=target_len)
    v=np.asarray(ds,dtype=float); v=v-float(np.mean(v)); v=np.abs(v)
    s=np.linalg.norm(v) or 1.0; v=v/s
    if len(v)!=target_len:
        import numpy as _np
        if len(v)<target_len:
            pad=_np.zeros(target_len,dtype=float); pad[:len(v)] = v; v=pad
        else:
            v=v[:target_len]; v=v/_np.linalg.norm(v)
    counts=fourier_fingerprint(v, shots=shots)
    typer.echo(json.dumps(counts))
@app.command('pack-raw')
def cmd_pack_raw(vectors:str='vectors.jsonl', out:str='raw_codec.json', model:str='all-MiniLM-L6-v2', n_components:int=8, shots:int=4096, pc1_target_len:int=4096):
    pack=build_raw_codec(vectors, out, model_name=model, n_components=n_components, shots=shots, pc1_target_len=pc1_target_len)
    typer.echo(json.dumps({'out':out,'pca_components':n_components,'shots':shots,'pc1_len':pc1_target_len}))
@app.command('pack-rope-hints')
def cmd_pack_rope(chunks:str='chunks.jsonl', vectors:str='vectors.jsonl', out:str='rope_hint.json', n_components:int=8, k_clusters:int=32, shots:int=4096, pc1_len:int=4096, timeline_slices:int=10):
    pack=build_rope_hints(chunks, vectors, out, n_components=n_components, k_clusters=k_clusters, shots=shots, pc1_len=pc1_len, timeline_slices=timeline_slices)
    typer.echo(json.dumps({'out':out,'k':k_clusters,'pc1_len':pc1_len,'shots':shots,'n_components':n_components}))

# --- Chroma integration ---
from .chroma_io import load_into_chroma as _chroma_load, search as _chroma_search, get_items as _chroma_get

@app.command("chroma-load")
def cmd_chroma_load(chunks: str = "chunks.jsonl", vectors: str = "vectors.jsonl", persist: str = "./chroma", collection: str = "qctx", batch: int = 512):
    """Load vectors+chunks into a Chroma collection."""
    n = _chroma_load(chunks, vectors, persist_dir=persist, collection=collection, batch=batch)
    typer.echo(json.dumps({"loaded": n, "collection": collection, "persist": persist}))

@app.command("chroma-search")
def cmd_chroma_search(query: str, k: int = 64, persist: str = "./chroma", collection: str = "qctx", model: str = "all-MiniLM-L6-v2"):
    """Search Chroma with a fresh embedding of the query."""
    hits = _chroma_search(persist, collection, query, k=k, model_name=model)
    typer.echo(json.dumps({"hits": hits}))

@app.command("chroma-get")
def cmd_chroma_get(ids: str, persist: str = "./chroma", collection: str = "qctx"):
    """Fetch items by ids (comma-separated)."""
    id_list = [int(x) for x in ids.split(",") if x.strip()]
    items = _chroma_get(persist, collection, id_list)
    typer.echo(json.dumps({"items": items}))
