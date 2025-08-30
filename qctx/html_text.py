from __future__ import annotations
import re
from typing import Iterable, Iterator, Optional

try:
    from bs4 import BeautifulSoup  # type: ignore
except Exception:  # pragma: no cover
    BeautifulSoup = None  # type: ignore


def html_to_text_stream(html_iter: Iterable[str]) -> Iterator[str]:
    """Yield plain text from HTML lines with a light dependency footprint.
    - If BeautifulSoup is present, use it once for the whole doc (buffered).
    - Otherwise, apply a simple regex-based tag stripper line by line.
    """
    if BeautifulSoup is not None:
        buf = []
        for line in html_iter:
            buf.append(line)
        soup = BeautifulSoup("".join(buf), "html.parser")
        text = soup.get_text(separator="\n")
        for line in text.splitlines(True):
            yield line
        return

    # Fallback: very basic stripper
    tag_re = re.compile(r"<[^>]+>")
    for line in html_iter:
        # remove script/style blocks crudely
        line = re.sub(r"<script[\s\S]*?</script>", "", line, flags=re.I)
        line = re.sub(r"<style[\s\S]*?</style>", "", line, flags=re.I)
        # strip tags
        line = tag_re.sub("", line)
        yield line
