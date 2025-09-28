import argparse
import json
import numpy as np
from tqdm import tqdm
from sentence_transformers import SentenceTransformer

def embed_texts(
    texts, model_name="intfloat/e5-base-v2", batch_size=32, device="cuda"
):
    model = SentenceTransformer(model_name, device=device)
    embeddings = model.encode(
        texts,
        batch_size=batch_size,
        convert_to_numpy=True,
        show_progress_bar=True,
        normalize_embeddings=True,  # recommended for cosine similarity
    )
    return embeddings

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="JSONL with {id,text}")
    parser.add_argument("--output-jsonl", required=True, help="Where to save embeddings")
    parser.add_argument("--output-npy", help="Optional: save .npy array")
    parser.add_argument("--batch-size", type=int, default=32)
    args = parser.parse_args()

    # 1. Load input JSONL
    ids, texts = [], []
    with open(args.input, "r", encoding="utf-8") as f:
        for line in f:
            obj = json.loads(line)
            if "text" in obj and obj["text"].strip():
                ids.append(obj.get("id"))
                texts.append(obj["text"].strip())

    print(f"Loaded {len(texts)} texts from {args.input}")

    # 2. Embed
    embeddings = embed_texts(texts, batch_size=args.batch_size)

    # 3. Save JSONL
    with open(args.output_jsonl, "w", encoding="utf-8") as f:
        for i, (id_, vec) in enumerate(zip(ids, embeddings)):
            out = {"id": id_ or f"row-{i}", "vector": vec.tolist(), "text": texts[i]}
            f.write(json.dumps(out) + "\n")

    print(f"Saved JSONL embeddings to {args.output_jsonl}")

    # 4. Optionally save .npy
    if args.output_npy:
        np.save(args.output_npy, embeddings)
        print(f"Saved raw embeddings to {args.output_npy}")

if __name__ == "__main__":
    main()
