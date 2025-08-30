import asyncio, json, re
from pathlib import Path
from typing import List
DEFAULT_LIMIT=2048
async def slurp_text(path:str)->str:
    raw=await asyncio.to_thread(Path(path).read_text,encoding='utf-8')
    return _pretty_json_if_possible(raw)
def preprocess_markdown(text:str)->str:
    text=re.sub(r"```[\s\S]*?```",'',text);text=re.sub(r"!\[[^\]]*\]\([^\)]*\)",'',text);text=re.sub(r"\[([^\]]+)\]\([^\)]+\)",r"\1",text);text=text.replace('`','');text=re.sub(r"\*\*(.*?)\*\*",r"\1",text);text=re.sub(r"\*(.*?)\*",r"\1",text);text=re.sub(r"^\s*#+\s*",'',text,flags=re.M);return text
def slice_bytes(text:str,limit:int=DEFAULT_LIMIT)->List[str]:
    data=text.encode('utf-8');out=[];i=0;n=len(data)
    while i<n:
        j=min(i+limit,n);k=max(i,j-80)
        while j>k and j<n and (data[j]&0xC0)==0x80:j-=1
        ws=data.rfind(b' ',i,j)
        if ws!=-1 and ws>i:j=ws
        out.append(data[i:j].decode('utf-8',errors='ignore'));i=j
    return out
def _pretty_json_if_possible(raw:str)->str:
    try: obj=json.loads(raw)
    except Exception: return raw
    if isinstance(obj,(dict,list)):return json.dumps(obj,indent=2,ensure_ascii=False)
    if isinstance(obj,str):return obj
    return raw
