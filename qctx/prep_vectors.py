import json, sys
from pathlib import Path
from html_text import html_to_text_stream
from loader import slice_bytes
from embeddings import encode_texts

def html_to_chunks(path: str, limit: int = 2048):
    with open(path, "r", encoding="utf-8") as f:
        lines = f.readlines()
    text_iter = html_to_text_stream(lines)
    full_text = "".join(text_iter)
    return slice_bytes(full_text, limit=limit)

def main(html_path: str, out_path: str, model: str = "all-MiniLM-L6-v2"):
    chunks = html_to_chunks(html_path, limit=2048)
    print(f"Loaded {len(chunks)} chunks")
    vecs = encode_texts(chunks, model_name=model)
    with open(out_path, "w", encoding="utf-8") as f:
        for i, (chunk, vec) in enumerate(zip(chunks, vecs)):
            f.write(json.dumps({"id": i, "text": chunk, "vector": vec}) + "\n")
    print(f"Wrote {len(vecs)} embeddings → {out_path}")

if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: python prep_vectors.py chat.html vectors.jsonl")
        sys.exit(1)
    main(sys.argv[1], sys.argv[2])
