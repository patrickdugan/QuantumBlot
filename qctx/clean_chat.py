import json
import datetime

def extract_bucketed_texts(path: str, output_path: str):
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    with open(output_path, "w", encoding="utf-8") as f_out:
        for convo in data:
            title = convo.get("title", "Untitled Thread")
            create_time = convo.get("create_time")
            if create_time:
                date_str = datetime.datetime.fromtimestamp(
                    float(create_time)
                ).strftime("%Y-%m-%d %H:%M:%S")
            else:
                date_str = "Unknown Date"

            # Write header
            f_out.write(f"\n=== {title} ({date_str}) ===\n")

            mapping = convo.get("mapping", {})
            for node in mapping.values():
                msg = node.get("message")
                if not msg:
                    continue
                author = msg.get("author", {}).get("role", "unknown")
                content = msg.get("content", {})
                if content.get("content_type") == "text":
                    parts = content.get("parts", [])
                    for p in parts:
                        if p and p.strip():
                            f_out.write(f"{author.upper()}: {p.strip()}\n")

    print(f"Cleaned and bucketed file written → {output_path}")

if __name__ == "__main__":
    extract_bucketed_texts("context/conversations.json", "conversations_bucketed.txt")
