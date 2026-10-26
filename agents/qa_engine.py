"""agents/qa_engine.py — Symbol-index RAG: retrieval and question answering.

The WP1 project index (tools/index_builder.py) is flattened into searchable
documents — one per script, scene, class_name and autoload. Scoring is pure
Python: token overlap plus a bonus for exact symbol-name hits, so "MiningShip"
always ranks the MiningShip script first without any embedding model.

answer_question() feeds the top-k retrieved source files (truncated) to the
LLM and asks for file:line citations. With no client (or on LLM failure) it
degrades honestly to a structured summary of the retrieval results — it never
invents an answer.

lookup_symbol() is LLM-free: it scans class_names, autoloads, script
signals/methods and scene nodes for a name and returns structured hits.

Usage:
    from agents.qa_engine import IndexRetriever, answer_question, lookup_symbol

    hits = lookup_symbol(index, "MiningShip")
    result = answer_question(index, project_root, "采矿船怎么收集矿石", client)
"""

from __future__ import annotations

import math
import re
from pathlib import Path

# ── Tokenisation ──────────────────────────────────────────────────────────────

_TOKEN_RE = re.compile(r"[A-Za-z0-9]+|[一-鿿]+")
_CAMEL_BOUNDARY_RE = re.compile(r"(?<=[a-z0-9])(?=[A-Z])")

# Exact name match outweighs token overlap by a wide margin.
_EXACT_HIT_BONUS = 5.0


def _tokenize(text: str) -> list[str]:
    """Lowercase word tokens; CamelCase identifiers also split into parts."""
    tokens: list[str] = []
    for raw in _TOKEN_RE.findall(text.lower()):
        tokens.append(raw)
        for part in _CAMEL_BOUNDARY_RE.split(raw):
            if part != raw:
                tokens.append(part)
    return tokens


# ── Document flattening ───────────────────────────────────────────────────────

def _flatten_index(index: dict) -> list[dict]:
    """Turn the WP1 index into retrieval documents.

    Each doc: {"kind", "path", "names": [...], "text": str}.  `names` holds
    the exact identifiers (class names, file stems, signal/method names) used
    for exact-hit scoring; `text` is the bag-of-words source.
    """
    docs: list[dict] = []
    scripts: dict = index.get("scripts", {})

    for res_path, entry in scripts.items():
        class_name = entry.get("class_name") or ""
        methods = list(entry.get("methods", {}))
        signals = list(entry.get("signals", []))
        stem = res_path.rsplit("/", 1)[-1].replace(".gd", "")
        names = [n for n in [class_name, stem] if n]
        text = " ".join([
            res_path, class_name, entry.get("extends") or "",
            " ".join(methods), " ".join(signals),
        ])
        summary_bits = [res_path]
        if class_name:
            summary_bits.append(f"class {class_name}")
        if entry.get("extends"):
            summary_bits.append(f"extends {entry['extends']}")
        if signals:
            summary_bits.append(f"signals: {', '.join(signals)}")
        if methods:
            summary_bits.append(f"methods: {', '.join(methods[:8])}"
                                + ("…" if len(methods) > 8 else ""))
        docs.append({
            "kind": "script", "path": res_path, "names": names,
            "text": text, "summary": " | ".join(summary_bits),
        })

    for res_path, scene in index.get("scenes", {}).items():
        nodes = list(scene.get("nodes", {}))
        stem = res_path.rsplit("/", 1)[-1].replace(".tscn", "")
        root_class = scene.get("root_class") or ""
        names = [n for n in [root_class, stem] if n]
        text = " ".join([res_path, root_class, " ".join(nodes)])
        summary = (f"{res_path} | scene root: {root_class or '(unknown)'}"
                   + (f" | nodes: {', '.join(nodes[:8])}"
                      + ("…" if len(nodes) > 8 else "") if nodes else ""))
        docs.append({
            "kind": "scene", "path": res_path, "names": names,
            "text": text, "summary": summary,
        })

    for name, res_path in index.get("class_names", {}).items():
        entry = scripts.get(res_path, {})
        docs.append({
            "kind": "class_name", "path": res_path, "names": [name],
            "text": f"{name} {res_path} {entry.get('extends') or ''}",
            "summary": f"class_name {name} → {res_path}",
        })

    for name, res_path in index.get("autoloads", {}).items():
        docs.append({
            "kind": "autoload", "path": res_path, "names": [name],
            "text": f"{name} autoload singleton {res_path}",
            "summary": f"autoload {name} → {res_path}",
        })

    return docs


# ── Retriever ─────────────────────────────────────────────────────────────────

class IndexRetriever:
    """Token-overlap + exact-name retrieval over the flattened index."""

    def __init__(self, index: dict) -> None:
        self._docs = _flatten_index(index)
        # Token sets are cached per doc; document frequency gives a light IDF.
        self._doc_tokens = [set(_tokenize(d["text"])) for d in self._docs]
        df: dict[str, int] = {}
        for tokens in self._doc_tokens:
            for tok in tokens:
                df[tok] = df.get(tok, 0) + 1
        self._df = df
        self._n_docs = max(len(self._docs), 1)

    def _idf(self, token: str) -> float:
        return math.log1p(self._n_docs / (1 + self._df.get(token, 0)))

    def retrieve(self, query: str, k: int = 5) -> list[dict]:
        """Score every doc, return the top-k as {kind, path, score, summary}."""
        q_tokens = _tokenize(query)
        q_lower = query.lower()
        scored: list[tuple[float, dict]] = []
        for doc, doc_tokens in zip(self._docs, self._doc_tokens):
            overlap = sum(self._idf(t) for t in q_tokens if t in doc_tokens)
            exact = sum(1 for name in doc["names"] if name.lower() in q_lower)
            score = overlap + _EXACT_HIT_BONUS * exact
            if score > 0:
                scored.append((score, doc))
        scored.sort(key=lambda sd: (-sd[0], sd[1]["path"]))
        return [{"kind": d["kind"], "path": d["path"], "score": round(s, 4),
                 "summary": d["summary"]} for s, d in scored[:k]]


def retrieve(index: dict, query: str, k: int = 5) -> list[dict]:
    """Convenience wrapper: build a one-shot retriever and query it."""
    return IndexRetriever(index).retrieve(query, k=k)


# ── Symbol lookup (LLM-free) ──────────────────────────────────────────────────

def lookup_symbol(index: dict, name: str) -> dict:
    """Direct index scan for a symbol name. Returns {"query", "hits"}.

    Hit kinds: class_name, autoload, script, signal, method, scene, scene_node.
    Matching is case-insensitive on equality, plus substring on signal/method
    names so "ore" still finds "ore_collected".
    """
    hits: list[dict] = []
    target = name.lower()

    def _matches(candidate: str) -> bool:
        return candidate.lower() == target

    def _contains(candidate: str) -> bool:
        return target in candidate.lower()

    for cls, res_path in index.get("class_names", {}).items():
        if _matches(cls):
            hits.append({"kind": "class_name", "name": cls, "path": res_path,
                         "detail": f"class_name {cls} declared in {res_path}"})

    for al, res_path in index.get("autoloads", {}).items():
        if _matches(al):
            hits.append({"kind": "autoload", "name": al, "path": res_path,
                         "detail": f"autoload singleton {al} → {res_path}"})

    for res_path, entry in index.get("scripts", {}).items():
        stem = res_path.rsplit("/", 1)[-1].replace(".gd", "")
        if _matches(stem) or _matches(entry.get("class_name") or ""):
            hits.append({
                "kind": "script", "name": entry.get("class_name") or stem,
                "path": res_path,
                "detail": f"script {res_path}"
                          + (f" (class {entry['class_name']})"
                             if entry.get("class_name") else ""),
            })
        for sig in entry.get("signals", []):
            if _matches(sig) or _contains(sig):
                hits.append({"kind": "signal", "name": sig, "path": res_path,
                             "detail": f"signal {sig} declared in {res_path}"})
        for meth, info in entry.get("methods", {}).items():
            if _matches(meth) or _contains(meth):
                line = info.get("line") if isinstance(info, dict) else None
                loc = f"{res_path}:{line}" if line else res_path
                hits.append({"kind": "method", "name": meth, "path": res_path,
                             "detail": f"method {meth}() at {loc}"})

    for res_path, scene in index.get("scenes", {}).items():
        stem = res_path.rsplit("/", 1)[-1].replace(".tscn", "")
        if _matches(stem) or _matches(scene.get("root_class") or ""):
            hits.append({"kind": "scene", "name": stem, "path": res_path,
                         "detail": f"scene {res_path}"
                                   f" (root: {scene.get('root_class') or '?'})"})
        for node_key in scene.get("nodes", {}):
            node_name = node_key.rsplit("/", 1)[-1]
            if _matches(node_name):
                hits.append({"kind": "scene_node", "name": node_name,
                             "path": res_path,
                             "detail": f"node {node_key} in scene {res_path}"})

    return {"query": name, "hits": hits}


# ── Question answering ────────────────────────────────────────────────────────

_MAX_SOURCE_LINES = 200
_ANSWER_MAX_TOKENS = 1024

_QA_PROMPT = """You are answering a question about a Godot 4 game project using
retrieved source files. Rules:
  - Cite evidence as file:line for every claim you make.
  - If the retrieved code does not answer the question, say so plainly —
    do not guess.

Question: {question}

Retrieved sources:
{sources}"""


def _read_source(project_root: Path, res_path: str) -> str | None:
    """Read a res:// path under project_root, truncated to _MAX_SOURCE_LINES."""
    rel = res_path[len("res://"):] if res_path.startswith("res://") else res_path
    try:
        text = (project_root / rel).read_text(encoding="utf-8", errors="replace")
    except OSError:
        return None
    lines = text.splitlines()
    if len(lines) > _MAX_SOURCE_LINES:
        lines = lines[:_MAX_SOURCE_LINES] + [
            f"# … truncated ({len(text.splitlines())} lines total)"]
    return "\n".join(lines)


def _degraded_answer(question: str, results: list[dict]) -> str:
    """Structured retrieval summary used when no LLM is available."""
    if not results:
        return ("未找到与问题相关的索引条目（未调用 LLM）。"
                "可以换个问法，或改用符号查询（例如直接给出类名/方法名）。")
    lines = ["（未调用 LLM，以下为符号索引检索到的相关条目）", ""]
    for r in results:
        lines.append(f"- [{r['kind']}] {r['summary']}")
    lines.append("")
    lines.append("如需逐行解释，请配置 LLM API key 后重试。")
    return "\n".join(lines)


def answer_question(index: dict, project_root: Path, question: str,
                    client=None, k: int = 3) -> dict:
    """Retrieve top-k index docs, then answer with the LLM (or degrade).

    Returns {"answer", "sources", "llm_used"}.  Never raises on LLM failure.
    """
    results = retrieve(index, question, k=k)
    sources = [{"kind": r["kind"], "path": r["path"], "score": r["score"]}
               for r in results]

    if client is None:
        return {"answer": _degraded_answer(question, results),
                "sources": sources, "llm_used": False}

    blocks: list[str] = []
    for r in results:
        body = _read_source(Path(project_root), r["path"])
        if body is None:
            # scenes and virtual docs may have no readable source; use summary
            body = f"# (no source file) {r['summary']}"
        blocks.append(f"### {r['path']}\n{body}")
    if not blocks:
        return {"answer": _degraded_answer(question, results),
                "sources": sources, "llm_used": False}

    prompt = _QA_PROMPT.format(question=question, sources="\n\n".join(blocks))
    try:
        answer = client.chat([{"role": "user", "content": prompt}],
                             max_tokens=_ANSWER_MAX_TOKENS).strip()
    except Exception:
        return {"answer": _degraded_answer(question, results),
                "sources": sources, "llm_used": False}

    if not answer:
        return {"answer": _degraded_answer(question, results),
                "sources": sources, "llm_used": False}
    return {"answer": answer, "sources": sources, "llm_used": True}
