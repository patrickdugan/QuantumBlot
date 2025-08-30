from __future__ import annotations
import asyncio
import ast
import json
from pathlib import Path
from typing import Optional, List

import typer

from .loader import slurp_text, preprocess_markdown, slice_bytes
from .embeddings import encode_texts, EmbeddingBackendMissing
from .dimred import pca_project
from .qft_pca import fourier_fingerprint
from . import big


app = typer.Typer(help="Quantum Context Toolkit CLI")

@app.command("ingest-big")
def cmd_ingest_big(path: str, out: str = "chunks.jsonl", limit: int = 2048, assume_html: bool = None):
    """Stream-ingest a huge file to JSONL chunks {id,text}."""
    count = big.write_jsonl_chunks(path, out, limit_bytes=limit, assume_html=assume_html)
    typer.echo(json.dumps({"count": count, "out": out}))

@app.command("embed-big")
def cmd_embed_big(chunks: str = "chunks.jsonl", out: str = "vectors.jsonl", model: str = "all-MiniLM-L6-v2", batch_size: int = 64):
    """Embed chunks JSONL -> vectors JSONL (streaming)."""
    total = big.embed_jsonl(chunks, out, model_name=model, batch_size=batch_size)
    typer.echo(json.dumps({"count": total, "out": out, "model": model}))

@app.command("reduce-big")
def cmd_reduce_big(vectors: str = "vectors.jsonl", out: str = "reduced.jsonl", n_components: int = 2, batch_size: int = 1024):
    """Incremental PCA on vectors JSONL -> reduced JSONL."""
    total = big.incremental_pca_jsonl(vectors, out, n_components=n_components, batch_size=batch_size)
    typer.echo(json.dumps({"count": total, "out": out, "n_components": n_components}))

@app.command("qft-centroid")
def cmd_qft_centroid(vectors: str = "vectors.jsonl", shots: int = 1024, take: int = None):
    """Compute centroid amplitude from vectors JSONL and run QFT fingerprint."""
    import numpy as np
    amp = big.centroid_amplitude_from_vectors_jsonl(vectors, take=take)
    counts = fourier_fingerprint(amp, shots=shots)
    typer.echo(json.dumps(counts))



@app.command("ingest")
def cmd_ingest(path: str, limit: int = 2048):
    """Read a file, scrub markdown, and emit chunked JSON."""
    text = asyncio.run(slurp_text(path))
    cleaned = preprocess_markdown(text)
    chunks = slice_bytes(cleaned, limit=limit)
    typer.echo(json.dumps({"count": len(chunks), "chunks": chunks}, ensure_ascii=False))


@app.command("embed")
def cmd_embed(path: str, model: str = "all-MiniLM-L6-v2"):
    """Embed file by paragraph break."""
    text = asyncio.run(slurp_text(path))
    cleaned = preprocess_markdown(text)
    parts = [p.strip() for p in cleaned.split("\n\n") if p.strip()]
    try:
        vecs = encode_texts(parts, model_name=model)
    except EmbeddingBackendMissing as e:
        raise typer.BadParameter(str(e))
    typer.echo(json.dumps({"count": len(vecs), "vectors": vecs}))


@app.command("reduce")
def cmd_reduce(n_components: int = 2, vectors_path: Optional[str] = None):
    """Reduce embeddings via PCA. Provide a JSON file of vectors or read stdin."""
    if vectors_path:
        data = json.loads(Path(vectors_path).read_text(encoding="utf-8"))
    else:
        data = json.loads(typer.get_text_stream("stdin").read())
    reduced = pca_project(data, n_components=n_components).tolist()
    typer.echo(json.dumps(reduced))


@app.command("qft")
def cmd_qft(vector: str = typer.Argument(..., help="Python/JSON literal list, e.g., [0,1,0,0]"), shots: int = 1024):
    """Run QFT fingerprint on a provided amplitude vector."""
    arr = np_from_literal(vector)
    counts = fourier_fingerprint(arr, shots=shots)
    typer.echo(json.dumps(counts))


def np_from_literal(lit: str):
    import numpy as np
    try:
        obj = ast.literal_eval(lit)
    except Exception as e:
        raise typer.BadParameter(f"Could not parse vector literal: {e}")
    arr = np.asarray(obj, dtype=complex).ravel()
    return arr


def main():
    app()


if __name__ == "__main__":
    main()
