from __future__ import annotations
import re
from typing import Iterable, List
_STOP=set('a an and are as at be but by for if in into is it of on or s so such that the their then there these this to was will with you your'.split())

def _tokens(text:str)->Iterable[str]:
    for tok in re.findall(r"[A-Za-z][A-Za-z0-9_-]{2,}", text.lower()):
        if tok not in _STOP: yield tok

def pull_motifs(text:str, top_k:int=10)->List[str]:
    freq={}
    for t in _tokens(text): freq[t]=freq.get(t,0)+1
    return [w for w,_ in sorted(freq.items(), key=lambda kv:(-kv[1],kv[0]))[:top_k]]
