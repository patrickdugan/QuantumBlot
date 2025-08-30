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


app = typer.Typer(help="Quantum Context Toolkit CLI")


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
