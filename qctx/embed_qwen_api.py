# embed_qwen_api.py
import argparse, json, time, requests

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="JSONL with {id,text}")
    parser.add_argument("--output-jsonl", required=True)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--wait", type=float, default=0.5, help="Seconds to sleep between calls")
    parser.add_argument("--token", required=True, help="Your HF token")
    args = parser.parse_args()

    API_URL = "https://api-inference.huggingface.co/models/Qwen/Qwen3-Embedding-8B"
    headers = {"Authorization": f"Bearer {args.token}"}

    def embed_batch(batch_texts):
        payload = {"inputs": batch_texts}
        resp = requests.post(API_URL, headers=headers, json=payload)
        if resp.status_code != 200:
            raise RuntimeError(f"HF API error {resp.status_code}: {resp.text}")
        return resp.json()

    with open(args.input) as f_in, open(args.output_jsonl, "w") as f_out:
        buffer = []
        for line in f_in:
            obj = json.loads(line)
            buffer.append(obj)
            if len(buffer) == args.batch_size:
                embs = embed_batch([x["text"] for x in buffer])
                for i, entry in enumerate(buffer):
                    f_out.write(json.dumps({"id": entry["id"], "vector": embs[i]}) + "\n")
                f_out.flush()
                buffer = []
                time.sleep(args.wait)
        if buffer:
            embs = embed_batch([x["text"] for x in buffer])
            for i, entry in enumerate(buffer):
                f_out.write(json.dumps({"id": entry["id"], "vector": embs[i]}) + "\n")

if __name__ == "__main__":
    main()
