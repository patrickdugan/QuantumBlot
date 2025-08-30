import json, sys
from embeddings import encode_texts
from loader import slice_bytes

def json_to_messages(path: str):
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)   # conversations.json is a list
    texts = []
    for conv in data:
        mapping = conv.get("mapping", {})
        for node in mapping.values():
            msg = node.get("message")
            if not msg:
                continue
            content = msg.get("content", {})
            if content.get("content_type") == "text":
                for part in content.get("parts", []):
                    if part.strip():
                        texts.append(part.strip())
    return texts

def main(json_path: str, out_path: str, model: str = "all-MiniLM-L6-v2"):
    texts = json_to_messages(json_path)
    print(f"Loaded {len(texts)} messages")
    chunks = []
    for t in texts:
        chunks.extend(slice_bytes(t, limit=2048))
    print(f"Split into {len(chunks)} chunks")
    vecs = encode_texts(chunks, model_name=model)
    with open(out_path, "w", encoding="utf-8") as f:
        for i, (chunk, vec) in enumerate(zip(chunks, vecs)):
            f.write(json.dumps({"id": i, "text": chunk, "vector": vec}) + "\n")
    print(f"Wrote {len(vecs)} embeddings → {out_path}")

if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: python prep_vectors_json.py conversations.json vectors.jsonl")
        sys.exit(1)
    main(sys.argv[1], sys.argv[2])
