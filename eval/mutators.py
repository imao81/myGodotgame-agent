"""eval/mutators.py — mechanical bug injection for recall measurement.

The golden set measures precision (it labels findings the pipeline produced).
It cannot measure recall: bugs no run ever found are invisible. Mutation
testing closes that gap — inject a known bug, and detection becomes binary
ground truth.

Each mutator scans a file for an injection site, applies a minimal text
transform, and returns (mutated_text, record) or None when no site exists.
Records carry everything detection needs: file, line, the assertion type a
correct reviewer should raise, and the entity the finding should name.

Design rules:
  - one mutant per file (mutants in the same file could mask each other)
  - mutations must keep the file parseable (we test review recall, not the
    compiler) — except break_preload_path, which is a load-time issue by
    nature; it still parses
  - every mutant corresponds to a bug class the pipeline claims to catch
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Callable


@dataclass
class MutantRecord:
    mutant_id: str
    file: str                 # project-relative posix path
    operator: str
    line: int                 # 1-based line of the mutation
    expected_assertion: str   # assertion type a correct reviewer should raise
    expected_entity: str      # entity the finding should name
    description: str

    def to_dict(self) -> dict:
        return dict(self.__dict__)


_EMIT_RE = re.compile(r"^(\s*)Events\.(\w+)\.emit\(", re.MULTILINE)
_PRELOAD_RE = re.compile(r'preload\("(res://[^"]+)"\)')
_CLASS_REF_RE = re.compile(r"\b(\w+)\.new\(\)")


def _rename_emitted_signal(text: str, file: str, mid: str):
    """Events.x.emit() → Events.x_broken.emit() — undeclared signal."""
    m = _EMIT_RE.search(text)
    if not m:
        return None
    sig = m.group(2)
    broken = f"{sig}_broken"
    new = text[:m.start()] + m.group(0).replace(
        f"Events.{sig}.emit", f"Events.{broken}.emit") + text[m.end():]
    line = text[:m.start()].count("\n") + 1
    return new, MutantRecord(
        mid, file, "rename_emitted_signal", line,
        "signal_not_declared", broken,
        f"Emits Events.{broken} which is not declared in the Events autoload")


def _dead_guard_emission(text: str, file: str, mid: str):
    """Wrap an emission in `if false:` — dead signal path (runtime-verifiable)."""
    m = _EMIT_RE.search(text)
    if not m:
        return None
    indent, sig = m.group(1), m.group(2)
    line_start = text.rfind("\n", 0, m.start()) + 1
    line_end = text.find("\n", m.start())
    line_end = len(text) if line_end == -1 else line_end
    emit_line = text[line_start:line_end]
    guarded = f"{indent}if false:  # MUTANT\n{indent}\t{emit_line.lstrip()}"
    new = text[:line_start] + guarded + text[line_end:]
    line = text[:line_start].count("\n") + 1
    return new, MutantRecord(
        mid, file, "dead_guard_emission", line,
        "runtime_signal_not_emitted", sig,
        f"Events.{sig}.emit() is unreachable behind `if false:`")


def _break_preload_path(text: str, file: str, mid: str):
    """preload(res://x) → preload of a nonexistent path."""
    m = _PRELOAD_RE.search(text)
    if not m:
        return None
    path = m.group(1)
    broken = path.replace(".", "_missing.", 1)  # res://a/b.gd → res://a/b_missing.gd
    if broken == path:
        broken = path + "_missing"
    new = text[:m.start(1)] + broken + text[m.end(1):]
    line = text[:m.start()].count("\n") + 1
    return new, MutantRecord(
        mid, file, "break_preload_path", line,
        "file_not_exists", broken,
        f"Preloads {broken} which does not exist on disk")


def _rename_class_reference(text: str, file: str, mid: str,
                            known_classes: set[str]):
    """SomeClass.new() → SomeClassMissing.new() — undefined identifier."""
    for m in _CLASS_REF_RE.finditer(text):
        cls = m.group(1)
        if cls not in known_classes:
            continue
        broken = f"{cls}Missing"
        new = text[:m.start(1)] + broken + text[m.end(1):]
        line = text[:m.start()].count("\n") + 1
        return new, MutantRecord(
            mid, file, "rename_class_reference", line,
            "identifier_not_defined", broken,
            f"References undefined class {broken}")
    return None


# Operator registry. known_classes is supplied by the project index.
MUTATORS: dict[str, Callable] = {
    "rename_emitted_signal": _rename_emitted_signal,
    "dead_guard_emission": _dead_guard_emission,
    "break_preload_path": _break_preload_path,
    "rename_class_reference": _rename_class_reference,
}


def apply_mutator(operator: str, text: str, file: str, mutant_id: str,
                  known_classes: set[str] | None = None):
    """Apply one mutator; returns (mutated_text, MutantRecord) or None."""
    fn = MUTATORS[operator]
    if operator == "rename_class_reference":
        return fn(text, file, mutant_id, known_classes or set())
    return fn(text, file, mutant_id)
