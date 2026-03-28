"""Parse and write Godot .tres resource files.

.tres format is line-oriented key=value inside a [resource] section:
    [gd_resource type="Resource" ...]
    [ext_resource path="..." type="..." id=N]
    [resource]
    key = value
    _key = 100.0

This parser uses regex to extract float/int typed values.
Writes back by replacing only changed lines in place (preserves formatting).
"""

import re
from pathlib import Path


_VALUE_RE = re.compile(r"^(\w+)\s*=\s*(.+)$")
_FLOAT_RE = re.compile(r"^-?\d+\.\d+$")
_INT_RE = re.compile(r"^-?\d+$")
_SECTION_RE = re.compile(r"^\[(\w+)")


def parse(path: str) -> dict:
    """Return {key: value} for all numeric fields in the [resource] section."""
    result = {}
    in_resource = False
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        line = line.strip()
        m = _SECTION_RE.match(line)
        if m:
            in_resource = m.group(1) == "resource"
            continue
        if not in_resource:
            continue
        vm = _VALUE_RE.match(line)
        if not vm:
            continue
        key, raw = vm.group(1), vm.group(2).strip()
        # Strip leading underscore (Godot private export convention)
        clean_key = key.lstrip("_")
        if _FLOAT_RE.match(raw):
            result[clean_key] = float(raw)
        elif _INT_RE.match(raw):
            result[clean_key] = int(raw)
    return result


def write(path: str, changes: dict) -> None:
    """Apply {clean_key: new_value} changes in-place.

    Matches both `key = value` and `_key = value` forms.
    """
    text = Path(path).read_text(encoding="utf-8")
    lines = text.splitlines(keepends=True)
    in_resource = False
    result = []
    for line in lines:
        stripped = line.strip()
        m = _SECTION_RE.match(stripped)
        if m:
            in_resource = m.group(1) == "resource"
            result.append(line)
            continue
        if in_resource:
            vm = _VALUE_RE.match(stripped)
            if vm:
                key = vm.group(1)
                clean_key = key.lstrip("_")
                if clean_key in changes:
                    new_val = changes[clean_key]
                    # Preserve float formatting
                    if isinstance(new_val, float) and not isinstance(new_val, bool):
                        formatted = f"{new_val:.1f}" if new_val == int(new_val) else str(new_val)
                    else:
                        formatted = str(new_val)
                    indent = line[: len(line) - len(line.lstrip())]
                    result.append(f"{indent}{key} = {formatted}\n")
                    continue
        result.append(line)
    Path(path).write_text("".join(result), encoding="utf-8")


def parse_all(project_path: str, subdir: str = "") -> dict[str, dict]:
    """Return {relative_path: {key: value}} for every .tres under project_path."""
    root = Path(project_path)
    search = root / subdir if subdir else root
    result = {}
    for tres in sorted(search.rglob("*.tres")):
        try:
            values = parse(str(tres))
            if values:  # skip empty (non-balance) resources
                result[str(tres.relative_to(root))] = values
        except Exception:
            pass
    return result
