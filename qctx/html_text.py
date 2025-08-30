from __future__ import annotations
import re
from typing import Iterable, Iterator
try:
    from bs4 import BeautifulSoup
except Exception:
    BeautifulSoup=None
def html_to_text_stream(html_iter: Iterable[str]) -> Iterator[str]:
    if BeautifulSoup is not None:
        buf=[]; 
        for line in html_iter: buf.append(line)
        soup=BeautifulSoup(''.join(buf),'html.parser')
        text=soup.get_text(separator='\n')
        for line in text.splitlines(True): yield line
        return
    tag_re=re.compile(r'<[^>]+>')
    for line in html_iter:
        line=re.sub(r'<script[\s\S]*?</script>','',line,flags=re.I)
        line=re.sub(r'<style[\s\S]*?</style>','',line,flags=re.I)
        yield tag_re.sub('',line)
