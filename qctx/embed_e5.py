import argparse
import json
from typing import List
from sentence_transformers import SentenceTransformer
import numpy as np


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="Input file (.jsonl or .txt)")
    parser.add_argument("--output-jsonl", required=True, help="Where to save embeddings JSONL")
    parser.add_argument("--output-npy", required=True, help="Where to save embeddings NPY")
    parser.add_argument("--batch-size", type=int, default=32)
    args = parser.parse_args()

    texts, meta = [], []
    print(f"[INFO] Loading input: {args.input}")

    with open(args.input, "r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            line = line.strip()
            if not line:
                continue

            if args.input.endswith(".jsonl"):
                obj = json.loads(line)
                text = obj.get("text", "")
                out = {"id": obj.get("id", str(i)), "text": text}
            else:
                # Treat .txt lines as raw text
                out = {"id": str(i), "text": line}

            texts.append(out["text"])
            meta.append(out)

    print(f"[INFO] Loaded {len(texts)} texts for embedding")

    model = SentenceTransformer("intfloat/e5-base-v2")

    embeddings = model.encode(
        texts,
        batch_size=args.batch_size,
        convert_to_numpy=True,
        show_progress_bar=True,
    )

    print("[INFO] Writing JSONL and NPY outputs")
    with open(args.output_jsonl, "w", encoding="utf-8") as f_out:
        for i, emb in enumerate(embeddings):
            obj = {
                "id": meta[i]["id"],
                "text": meta[i]["text"],
                "embedding": emb.tolist(),
            }
            f_out.write(json.dumps(obj) + "\n")

            # Log every 500 entries
            if (i + 1) % 500 == 0:
                print(f"[INFO] Written {i+1}/{len(embeddings)} embeddings")

    np.save(args.output_npy, embeddings)
    print(f"[INFO] Done. Saved {len(embeddings)} embeddings")


if __name__ == "__main__":
    main()
