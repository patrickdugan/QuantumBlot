import asyncio
import json
import re
from pathlib import Path
from typing import Iterable, List, Tuple

DEFAULT_LIMIT = 2048  # bytes


async def slurp_text(path: str) -> str:
    """Read a file's text asynchronously and pretty-print JSON when possible."""
    raw = await asyncio.to_thread(Path(path).read_text, encoding="utf-8")
    return _pretty_json_if_possible(raw)


def preprocess_markdown(text: str) -> str:
    """Lightweight markdown scrubber with conservative rules."""
    # Strip fenced code blocks
    text = re.sub(r"```[\s\S]*?```", "", text)
    # Images ![alt](url)
    text = re.sub(r"!\[[^\]]*\]\([^\)]*\)", "", text)
    # Links [label](url) -> label
    text = re.sub(r"\[([^\]]+)\]\([^\)]+\)", r"\1", text)
    # Inline code backticks
    text = text.replace("`", "")
    # Emphasis markers
    text = re.sub(r"\*\*(.*?)\*\*", r"\1", text)
    text = re.sub(r"\*(.*?)\*", r"\1", text)
    # Headings leading #'s
    text = re.sub(r"^\s*#+\s*", "", text, flags=re.M)
    return text


def slice_bytes(text: str, limit: int = DEFAULT_LIMIT) -> List[str]:
    """Chunk text by UTF-8 bytes with soft boundaries on whitespace."""
    data = text.encode("utf-8")
    out: List[str] = []
    i = 0
    n = len(data)
    while i < n:
        j = min(i + limit, n)
        # walk backwards to try to break at a space (within 80 bytes)
        k = max(i, j - 80)
        while j > k and j < n and data[j:j+1] & b"\xC0" == b"\x80":  # avoid splitting UTF-8 continuation
            j -= 1
        # prefer whitespace split
        ws = data.rfind(b" ", i, j)
        if ws != -1 and ws > i:
            j = ws
        out.append(data[i:j].decode("utf-8", errors="ignore"))
        i = j
    return out


def _pretty_json_if_possible(raw: str) -> str:
    try:
        obj = json.loads(raw)
    except Exception:
        return raw
    if isinstance(obj, (dict, list)):
        return json.dumps(obj, indent=2, ensure_ascii=False)
    if isinstance(obj, str):
        return obj
    return raw
