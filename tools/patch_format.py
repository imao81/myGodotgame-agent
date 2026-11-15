"""Surgical edit-block format for LLM-generated fixes (Aider-style).

The model never rewrites a whole file; it emits one or more edit blocks:

    <<<<<<< ORIGINAL
    <exact text from the file>
    =======
    <replacement text>
    >>>>>>> FIXED

An ORIGINAL section must occur in the file exactly once — a hallucinated or
ambiguous edit location is rejected, never partially applied. When the model
ignores the format and returns a whole file instead, we compute the diff
ratio as the boundary metric (large rewrites are out-of-boundary).
"""

from __future__ import annotations

import difflib
import re
from dataclasses import dataclass, field

_BEGIN = "<<<<<<< ORIGINAL"
_SEP = "======="
_END = ">>>>>>> FIXED"

# Whole-file fallback: a response whose diff touches more than this fraction
# of the file's lines is considered out-of-boundary ("drive-by refactoring").
_MAX_DIFF_RATIO = 0.4


@dataclass
class EditBlock:
    original: str
    fixed: str


@dataclass
class ParseResult:
    blocks: list[EditBlock] = field(default_factory=list)
    leftover: str = ""        # non-block text (whole-file fallback candidate)
    errors: list[str] = field(default_factory=list)


def parse_blocks(text: str) -> ParseResult:
    """Extract edit blocks from LLM output; tolerate chatter around blocks."""
    result = ParseResult()
    # Block regex: tolerate trailing whitespace on marker lines.
    pattern = re.compile(
        re.escape(_BEGIN) + r"\s*\n(.*?)\n?" + re.escape(_SEP) + r"\s*\n(.*?)\n?"
        + re.escape(_END),
        re.DOTALL,
    )
    spans: list[tuple[int, int]] = []
    for m in pattern.finditer(text):
        result.blocks.append(EditBlock(original=m.group(1), fixed=m.group(2)))
        spans.append((m.start(), m.end()))
    leftover = text
    for start, end in reversed(spans):
        leftover = leftover[:start] + leftover[end:]
    result.leftover = leftover.strip()
    # Malformed markers: a BEGIN without a matching END
    if _BEGIN in result.leftover or _END in result.leftover:
        result.errors.append("malformed edit block (unbalanced markers)")
    return result


@dataclass
class ApplyResult:
    new_text: str | None
    applied: int
    errors: list[str] = field(default_factory=list)


def apply_blocks(file_text: str, blocks: list[EditBlock]) -> ApplyResult:
    """Apply blocks sequentially. All-or-nothing: any failure → new_text None."""
    errors: list[str] = []
    for i, block in enumerate(blocks):
        count = file_text.count(block.original)
        if count == 0:
            errors.append(f"block {i}: ORIGINAL text not found in file")
        elif count > 1:
            errors.append(f"block {i}: ORIGINAL text matches {count} locations "
                          f"(ambiguous — must be unique)")
    if errors:
        return ApplyResult(new_text=None, applied=0, errors=errors)
    new_text = file_text
    for block in blocks:
        new_text = new_text.replace(block.original, block.fixed, 1)
    return ApplyResult(new_text=new_text, applied=len(blocks), errors=[])


def diff_ratio(original: str, fixed: str) -> float:
    """Fraction of original lines touched by the diff (whole-file boundary)."""
    original_lines = original.splitlines()
    if not original_lines:
        return 1.0 if fixed else 0.0
    matcher = difflib.SequenceMatcher(None, original_lines, fixed.splitlines())
    changed = sum(max(i2 - i1, j2 - j1)
                  for tag, i1, i2, j1, j2 in matcher.get_opcodes()
                  if tag != "equal")
    return changed / len(original_lines)


def boundary_report(original: str, llm_output: str, blocks: list[EditBlock],
                    leftover: str) -> dict:
    """Boundary verdict for one LLM response.

    Edit-block mode: boundary is exact by construction (we only substitute the
    declared regions). Whole-file fallback: boundary = diff ratio threshold.
    """
    if blocks:
        applied = apply_blocks(original, blocks)
        return {
            "mode": "edit_blocks",
            "boundary_ok": applied.new_text is not None,
            "diff_ratio": None,
            "apply_errors": applied.errors,
        }
    # Whole-file fallback: treat the leftover as the new file content.
    ratio = diff_ratio(original, leftover)
    return {
        "mode": "whole_file",
        "boundary_ok": ratio <= _MAX_DIFF_RATIO,
        "diff_ratio": round(ratio, 3),
        "apply_errors": [],
    }
