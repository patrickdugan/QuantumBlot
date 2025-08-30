#!/usr/bin/env python3
"""
emit_prompt.py — produce a compact system prompt & payload skeleton
from a RoPE pack ("rope_hint.json") and optional background blots.

Usage examples:
  python emit_prompt.py --rope rope_hint.json --out system_prompt.txt
  python emit_prompt.py --rope rope_hint.json --emit-payload request_skeleton.json
  python emit_prompt.py --rope rope_hint.json --background domain1.json --background domain2.json --counts-top 24 --basis-top 16 --clusters 16

No external dependencies; stdlib only.
"""

import json, os, math, argparse, textwrap
from typing import List, Dict, Any, Tuple

def _load_json(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def _top_counts(counts: Dict[str, int], topn: int) -> Dict[str, int]:
    if not isinstance(counts, dict):
        return {}
    items = sorted(counts.items(), key=lambda kv: (-kv[1], kv[0]))
    return {k: int(v) for k, v in items[:topn]}

def _basis_topk(basis: List[List[float]], topk: int) -> List[Dict[str, Any]]:
    """Return top-|weight| entries per component as {idx, w} lists to keep prompt tiny."""
    out = []
    for comp_idx, row in enumerate(basis):
        pairs = [(i, abs(v), float(v)) for i, v in enumerate(row)]
        pairs.sort(key=lambda t: -t[1])
        top = [{"idx": i, "w": round(w, 5)} for (i, _, w) in pairs[:topk]]
        out.append({"component": comp_idx, "top": top})
    return out

def _cluster_digest(codec: Dict[str, Any], limit: int) -> List[Dict[str, Any]]:
    clusters = codec.get("clusters", [])
    # order by size desc
    clusters = sorted(clusters, key=lambda c: (-int(c.get("size", 0)), c.get("id", 0)))[:limit]
    out = []
    for c in clusters:
        centroid = c.get("centroid", [])
        mag = (sum(v*v for v in centroid) ** 0.5) if centroid else 0.0
        out.append({
            "id": int(c.get("id", 0)),
            "size": int(c.get("size", 0)),
            "motifs": c.get("motifs", [])[:8],
            "centroid_norm": round(float(mag), 4),
            "summary": c.get("summary", ""),
        })
    return out

def _timeline_digest(codec: Dict[str, Any], slices: int) -> List[Dict[str, Any]]:
    tl = codec.get("timeline", [])[:slices]
    out = []
    for t in tl:
        out.append({
            "range": t.get("range", []),
            "dominant_clusters": t.get("dominant_clusters", [])[:3],
            "motifs": t.get("motifs", [])[:8]
        })
    return out

def _build_prior(rope: Dict[str, Any], counts_top: int, basis_top: int, clusters: int, tl_slices: int) -> Dict[str, Any]:
    raw = rope.get("raw", {})
    codec = rope.get("codec", {})
    pca = raw.get("pca", {})
    qft = raw.get("qft", {})
    prior = {
        "embedding_spec": raw.get("embedding_spec", {}),
        "pca": {
            "components": int(pca.get("components", 0)),
            "explained_variance": [round(float(x),5) for x in pca.get("explained_variance", [])],
            "basis_top": _basis_topk(pca.get("basis", []), basis_top) if pca.get("basis") else []
        },
        "qft": {
            "centroid": {
                "n_qubits": qft.get("centroid", {}).get("n_qubits", None),
                "top_counts": _top_counts(qft.get("centroid", {}).get("counts", {}), counts_top)
            },
            "pc1": {
                "n_qubits": qft.get("pc1", {}).get("n_qubits", None),
                "downsample": qft.get("pc1", {}).get("downsample", None),
                "top_counts": _top_counts(qft.get("pc1", {}).get("counts", {}), counts_top)
            }
        },
        "clusters": _cluster_digest(codec, clusters),
        "global_motifs": codec.get("global_motifs", [])[:64],
        "timeline": _timeline_digest(codec, tl_slices)
    }
    return prior

def _merge_background_blots(prior: Dict[str, Any], backgrounds: List[Dict[str, Any]], counts_top: int) -> Dict[str, Any]:
    """Attach background blots as small named blocks; do not attempt numeric merging here (keep deterministic)."""
    bg = []
    for i, b in enumerate(backgrounds):
        name = b.get("name", f"background_{i}")
        raw = b.get("raw", {})
        qft = raw.get("qft", {})
        entry = {
            "name": name,
            "centroid_top_counts": _top_counts(qft.get("centroid", {}).get("counts", {}), counts_top),
            "pc1_top_counts": _top_counts(qft.get("pc1", {}).get("counts", {}), counts_top),
            "notes": b.get("notes", ""),
        }
        bg.append(entry)
    if bg:
        prior["background_blots"] = bg
    return prior

SYSTEM_POLICY_TEMPLATE = """You are a retrieval+reasoning agent with a spectral prior.

Use the following policy for each user query:
1) Propose a concrete retrieval query.
2) Call TOOL `vector.search` (k=64 by default). Then call `vector.get` on those ids to obtain vectors/snippets/positions.
3a) Call TOOL `spectral.slice_hint` with the retrieved items to compute a local spectral blot (centroid QFT and PC1 QFT).
3b) Align the slice blot with the GLOBAL prior (below) and prefer clusters whose motifs/centroid norms align with dominant peaks.
4) Synthesize an answer. Cite item ids. Do not quote long text; summarize. If evidence is weak, say so and propose follow-ups.
5) When context is large or uncertain, repeat 2–3 with a narrower query (rerank by alignment to peaks and cluster motifs).

TOOLS available via MCP (typical names):
- vector.search(query, k) -> hits[{id, score, pos, snippet?}]
- vector.get(ids) -> items[{id, vector, pos, snippet}]
- spectral.slice_hint(items, pc1_len, shots) -> {centroid_qft, pc1_qft}

Use TOOLs early. If a tool is required to proceed, do not speculate; call the tool.

Output rules:
- Provide a short rationale section describing how the slice blot influenced ranking.
- Cite sources as [id:###].
- If no strong match, say so and suggest a refined query.
"""

def _render_prompt(prior: Dict[str, Any]) -> str:
    prior_json = json.dumps(prior, ensure_ascii=False)
    tip = (
        "Tip: treat 'basis_top' as soft semantic axes; treat 'centroid_top_counts' and 'pc1_top_counts' as cadence hints. "
        "Favor clusters with higher 'centroid_norm' and motifs matching the query."
    )
    return textwrap.dedent(f"""\
    ### Quantum Blot Prior + Retrieval Policy
    {SYSTEM_POLICY_TEMPLATE}

    GLOBAL PRIOR (compact JSON):
    {prior_json}

    {tip}
    """)

def _payload_skeleton(prior: Dict[str, Any], first_tool: str, mcp_server: str) -> Dict[str, Any]:
    """Return a minimal request skeleton showing tool_choice and the system prompt."""
    system_prompt = _render_prompt(prior)
    tools = [
        {"type": "mcp", "server": mcp_server, "name": "vector.search"},
        {"type": "mcp", "server": mcp_server, "name": "vector.get"},
        {"type": "mcp", "server": mcp_server, "name": "spectral.slice_hint"}
    ]
    skeleton = {
        "model": "gpt-5-reasoner",  # placeholder
        "tool_choice": {"type": "tool", "name": first_tool},
        "input": [{"role": "system", "content": system_prompt}],
        "tools": tools
    }
    return skeleton

def main():
    ap = argparse.ArgumentParser(description="Emit a compact system prompt from a RoPE pack.")
    ap.add_argument("--rope", required=True, help="Path to rope_hint.json")
    ap.add_argument("--background", action="append", default=[], help="Optional background blot JSON(s)")
    ap.add_argument("--out", help="Write system prompt to this file (text). If omitted, prints to stdout.")
    ap.add_argument("--emit-payload", dest="emit_payload", help="Also emit a payload skeleton JSON to this path.")
    ap.add_argument("--first-tool", default="vector.search", help="Initial tool to force (vector.search | spectral.slice_hint | custom)")
    ap.add_argument("--mcp-server", default="https://YOUR-MCP", help="MCP server id/url placeholder for the payload skeleton")
    ap.add_argument("--counts-top", type=int, default=24, help="Top-N QFT bins to include per fingerprint")
    ap.add_argument("--basis-top", type=int, default=16, help="Top-K weights per PCA component to include")
    ap.add_argument("--clusters", type=int, default=12, help="Max clusters to include in the prior block")
    ap.add_argument("--timeline-slices", type=int, default=6, help="Max timeline slices to include")
    args = ap.parse_args()

    rope = _load_json(args.rope)
    prior = _build_prior(rope, counts_top=args.counts_top, basis_top=args.basis_top, clusters=args.clusters, tl_slices=args.timeline_slices)

    # Attach background blots, if any
    bgs = []
    for p in args.background:
        try:
            bgs.append(_load_json(p))
        except Exception:
            pass
    prior = _merge_background_blots(prior, bgs, counts_top=args.counts_top)

    prompt = _render_prompt(prior)

    if args.out:
        with open(args.out, "w", encoding="utf-8") as f:
            f.write(prompt)
    else:
        print(prompt)

    if args.emit_payload:
        payload = _payload_skeleton(prior, first_tool=args.first_tool, mcp_server=args.mcp_server)
        with open(args.emit_payload, "w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, indent=2)

if __name__ == "__main__":
    main()
