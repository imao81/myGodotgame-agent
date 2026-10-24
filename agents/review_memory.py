"""agents/review_memory.py — test-time learning memory for code review.

Inspired by Dynamic Cheatsheet (Suzgun et al., EACL 2026), with one key
difference: DC's curator has no correctness oracle, while our Propose-Verify
pipeline produces a ground-truth verdict (confirmed / refuted) for every
finding.  Memory writes are therefore gated by the static verifier — only
verifier-backed conclusions are ever stored.

Write path:  verified findings → curate() → .kimi/memory/review_memory.jsonl
Read path:   retrieve() → injected into the review prompt as known
             false-positive patterns, so the LLM stops re-reporting them.

No LLM calls, no external dependencies (pure-Python TF-IDF retrieval).
"""

from __future__ import annotations

import json
import math
import re
from datetime import datetime, timezone
from pathlib import Path

_MEMORY_FILE = "review_memory.jsonl"
_ARCHIVE_FILE = "review_memory.archive.jsonl"

# Retrieval tuning
_SAME_FILE_BOOST = 1.5      # memory entry recorded in the same file under review
_ENTITY_MATCH_BOOST = 2.0   # entity name appears among the file's declared symbols
_HITS_WEIGHT = 0.2          # mild log bonus for repeatedly-seen entries
_DECAY_PER_DAY = 0.95       # hits decay per day since last_seen_at
_ARCHIVE_THRESHOLD = 0.5    # effective hits below this → archived on save
_EST_CHARS_PER_TOKEN = 4


def _tokenize(text: str) -> list[str]:
    return re.findall(r"[a-z0-9_]+", text.lower())


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _days_since(iso_ts: str) -> float:
    try:
        then = datetime.fromisoformat(iso_ts)
        if then.tzinfo is None:
            then = then.replace(tzinfo=timezone.utc)
        return max(0.0, (datetime.now(timezone.utc) - then).total_seconds() / 86400)
    except (ValueError, TypeError):
        return 0.0


def _entity_key(assertion: dict) -> str | None:
    """Stable entity identifier for (assertion_type, entity) dedup, or None."""
    atype = assertion.get("type", "")
    if atype == "method_is_not_async":
        cls = assertion.get("target_class") or assertion.get("file_hint") or ""
        return f"{cls}.{assertion.get('method', '')}".strip(".")
    if atype in ("node_type_mismatch", "node_not_found"):
        return f"{assertion.get('node_path', '')}@{assertion.get('scene_file', '')}"
    if atype == "identifier_not_defined":
        return assertion.get("identifier", "")
    if atype == "signal_not_declared":
        return f"{assertion.get('signal_name', '')}@{assertion.get('file', '')}"
    if atype == "file_not_exists":
        return assertion.get("path", "")
    return None  # "unverifiable" has no stable entity — not curated


class ReviewMemory:
    """Verifier-gated, decaying, deduplicated memory of past review outcomes."""

    def __init__(self, memory_dir: Path, max_inject_tokens: int = 800):
        self.memory_dir = Path(memory_dir)
        self.path = self.memory_dir / _MEMORY_FILE
        self.max_inject_tokens = max_inject_tokens
        self.entries: list[dict] = self._load()

    # ── persistence ──────────────────────────────────────────────────────────

    def _load(self) -> list[dict]:
        if not self.path.is_file():
            return []
        entries = []
        for line in self.path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                entries.append(json.loads(line))
            except json.JSONDecodeError:
                continue
        return entries

    def save(self) -> None:
        self.memory_dir.mkdir(parents=True, exist_ok=True)
        active, archived = [], []
        for e in self.entries:
            (active if self._effective_hits(e) >= _ARCHIVE_THRESHOLD else archived).append(e)
        self.path.write_text(
            "".join(json.dumps(e, ensure_ascii=False) + "\n" for e in active),
            encoding="utf-8",
        )
        if archived:
            with (self.memory_dir / _ARCHIVE_FILE).open("a", encoding="utf-8") as f:
                for e in archived:
                    f.write(json.dumps(e, ensure_ascii=False) + "\n")
        self.entries = active

    # ── scoring ──────────────────────────────────────────────────────────────

    @staticmethod
    def _effective_hits(entry: dict) -> float:
        return entry.get("hits", 1) * _DECAY_PER_DAY ** _days_since(
            entry.get("last_seen_at", "")
        )

    def _tfidf_cosine(self, query_tokens: list[str], entry_tokens: list[str],
                      df: dict[str, int], n_docs: int) -> float:
        def tfidf(tokens: list[str]) -> dict[str, float]:
            tf: dict[str, int] = {}
            for t in tokens:
                tf[t] = tf.get(t, 0) + 1
            return {
                t: (1 + math.log(c)) * math.log((n_docs + 1) / (df.get(t, 0) + 0.5))
                for t, c in tf.items()
            }

        qv, ev = tfidf(query_tokens), tfidf(entry_tokens)
        dot = sum(w * ev.get(t, 0.0) for t, w in qv.items())
        qn = math.sqrt(sum(w * w for w in qv.values()))
        en = math.sqrt(sum(w * w for w in ev.values()))
        return dot / (qn * en) if qn and en else 0.0

    # ── read path ────────────────────────────────────────────────────────────

    def retrieve(
        self,
        file_rel: str,
        index: dict,
        k: int = 3,
        min_score: float = 0.15,
        kinds: tuple[str, ...] = ("false_positive",),
    ) -> list[dict]:
        """Return top-k memory entries relevant to the file under review.

        score = TF-IDF cosine × same-file boost × entity-match boost
                × (1 + 0.2·ln(1+decayed hits))
        Results are capped by a hard token budget (max_inject_tokens).
        """
        candidates = [e for e in self.entries if e.get("kind") in kinds]
        if not candidates:
            return []

        script = index.get("scripts", {}).get(f"res://{file_rel}", {})
        query_text = " ".join([
            file_rel,
            script.get("class_name") or "",
            " ".join(script.get("signals", [])),
            " ".join(list(script.get("methods", {}).keys())[:20]),
        ])
        query_tokens = _tokenize(query_text)
        # Symbols actually declared in this file — an entity mentioned in memory
        # that also appears here is a strong exact-match signal.
        file_symbols = {
            *(script.get("signals", [])),
            *script.get("methods", {}).keys(),
            script.get("class_name") or "",
        } - {""}

        n_docs = len(candidates)
        df: dict[str, int] = {}
        entry_tokens: list[list[str]] = []
        for e in candidates:
            toks = _tokenize(e.get("retrieval_text", ""))
            entry_tokens.append(toks)
            for t in set(toks):
                df[t] = df.get(t, 0) + 1

        scored: list[tuple[float, dict]] = []
        for e, toks in zip(candidates, entry_tokens):
            sim = self._tfidf_cosine(query_tokens, toks, df, n_docs)
            if sim <= 0:
                continue
            boost = (
                _SAME_FILE_BOOST
                if e.get("file_context", {}).get("file") == file_rel else 1.0
            )
            # Entity head (e.g. "damaged" from "damaged@res://...") found among
            # the file's declared symbols → exact-match boost
            entity_head = e.get("entity", "").split("@")[0].rsplit(".", 1)[-1]
            if entity_head and entity_head in file_symbols:
                boost *= _ENTITY_MATCH_BOOST
            score = sim * boost * (1 + _HITS_WEIGHT * math.log1p(self._effective_hits(e)))
            if score >= min_score:
                scored.append((score, e))

        scored.sort(key=lambda x: x[0], reverse=True)

        results, tokens_used = [], 0
        for score, e in scored[:k]:
            est = len(e.get("summary", "") + e.get("entity", "")) // _EST_CHARS_PER_TOKEN
            if tokens_used + est > self.max_inject_tokens:
                break
            tokens_used += est
            results.append({**e, "_score": round(score, 3)})
        return results

    # ── write path ───────────────────────────────────────────────────────────

    def curate(self, enriched_findings: list[dict], run_id: str) -> dict:
        """Fold verifier-backed findings into memory.  Returns stats dict.

        Only findings with a verifier verdict are stored:
          refuted  → kind="false_positive"   (do not re-report)
          confirmed → kind="confirmed_pattern" (stored for stats/future use)
        Unverifiable findings are skipped — no ground truth, no memory.
        """
        added = updated = skipped = 0
        by_key = {
            (e.get("assertion_type"), e.get("entity")): e for e in self.entries
        }

        for f in enriched_findings:
            verdict = f.get("verified")
            if verdict not in ("confirmed", "refuted"):
                skipped += 1
                continue
            assertion = f.get("assertion", {})
            entity = _entity_key(assertion)
            if not entity:
                skipped += 1
                continue

            kind = "false_positive" if verdict == "refuted" else "confirmed_pattern"
            key = (assertion.get("type", ""), entity)

            if key in by_key:
                entry = by_key[key]
                entry["hits"] = entry.get("hits", 1) + 1
                entry["last_seen_at"] = _now_iso()
                entry["evidence"] = {
                    "verifier_status": verdict,
                    "verifier_details": f.get("_verifier", {}).get("details", ""),
                    "run_id": run_id,
                }
                updated += 1
                continue

            entry = {
                "id": f"mem_{len(self.entries) + 1:04d}",
                "kind": kind,
                "assertion_type": assertion.get("type", ""),
                "entity": entity,
                "summary": f.get("message", "")[:300],
                "evidence": {
                    "verifier_status": verdict,
                    "verifier_details": f.get("_verifier", {}).get("details", ""),
                    "run_id": run_id,
                },
                "file_context": {"file": f.get("file", "")},
                "retrieval_text": " ".join([
                    assertion.get("type", ""), entity,
                    f.get("file", ""), f.get("message", "")[:200],
                ]),
                "hits": 1,
                "created_at": _now_iso(),
                "last_seen_at": _now_iso(),
            }
            self.entries.append(entry)
            by_key[key] = entry
            added += 1

        self.save()
        return {
            "added": added, "updated": updated, "skipped": skipped,
            "total": len(self.entries),
        }

    # ── introspection ────────────────────────────────────────────────────────

    def stats(self) -> dict:
        return {
            "total_entries": len(self.entries),
            "false_positive": sum(1 for e in self.entries if e.get("kind") == "false_positive"),
            "confirmed_pattern": sum(
                1 for e in self.entries if e.get("kind") == "confirmed_pattern"
            ),
            "memory_file": str(self.path),
        }
