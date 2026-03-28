"""Per-file context summary builder.

Given a .gd file path and the project index, generates a compact
comment block (≤800 tokens) that tells the LLM about cross-file
dependencies so it can avoid false positives.
"""

from __future__ import annotations

from pathlib import Path


def build_file_context(file_path: str, project_index: dict, project_path: str) -> str:
    """Return a comment block summarising cross-file context for file_path.

    Args:
        file_path: Project-relative path, e.g. "ships/enemies/pirate/pirate_ship.gd"
        project_index: Output of project_indexer.build_index()
        project_path: Absolute path to the Godot project root (for reading file content)

    Returns:
        A string of GDScript-style comments to prepend to the review prompt,
        or "" if no useful context was found.
    """
    # Normalise to forward-slash res:// path for index lookups
    rel_posix = file_path.replace("\\", "/")
    res_path = f"res://{rel_posix}"

    # Read the file content for reference-based filtering
    abs_path = Path(project_path) / rel_posix
    try:
        file_content = abs_path.read_text(encoding="utf-8", errors="replace")
    except OSError:
        file_content = ""

    lines: list[str] = [f"# === Project context for: {file_path} ==="]

    # ── Scene attachment ───────────────────────────────────────────────────────
    scene_res = project_index.get("script_to_scene", {}).get(res_path)
    if scene_res:
        scene_name = scene_res.split("/")[-1]
        lines.append(f"# Attached to scene: {scene_name}")

        scene_nodes = project_index.get("scene_nodes", {}).get(scene_res, {})
        if scene_nodes:
            lines.append("# Scene node tree (node_name: NodeType):")
            for node_name, node_type in scene_nodes.items():
                lines.append(f"#   {node_name}: {node_type}")

    # ── Available class_names ──────────────────────────────────────────────────
    # Only include class_names that are actually referenced in this file
    all_classes = project_index.get("class_names", {})
    referenced_classes = [
        cls for cls in all_classes
        if cls in file_content and all_classes[cls] != res_path
    ]
    if referenced_classes:
        lines.append(f"# Available class_names (defined elsewhere, usable without preload):")
        for cls in sorted(referenced_classes):
            src = all_classes[cls].replace("res://", "")
            lines.append(f"#   {cls}  →  {src}")

    # ── Autoload singletons ────────────────────────────────────────────────────
    autoloads = project_index.get("autoloads", [])
    referenced_autoloads = [a for a in autoloads if a in file_content]
    if referenced_autoloads:
        lines.append(f"# Autoload singletons used: {', '.join(referenced_autoloads)}")

    # ── Reverse dependencies (files that depend on this one) ──────────────────
    rev_deps = project_index.get("reverse_deps", {}).get(res_path, [])
    if rev_deps:
        dep_names = [d.split("/")[-1] for d in rev_deps[:6]]  # cap at 6
        lines.append(f"# Files that preload/depend on this: {', '.join(dep_names)}")

    # ── Signals declared in this file ─────────────────────────────────────────
    declared = project_index.get("signal_decls", {}).get(res_path, [])
    if declared:
        lines.append(f"# Signals declared here: {', '.join(declared)}")

    # ── Signals this file emits ────────────────────────────────────────────────
    emitted = project_index.get("signal_emits", {}).get(res_path, [])
    # Filter to only signals that are actually declared somewhere (avoid noise)
    all_declared_signals: set[str] = set()
    for sig_list in project_index.get("signal_decls", {}).values():
        all_declared_signals.update(sig_list)
    real_emits = [s for s in emitted if s in all_declared_signals]
    if real_emits:
        lines.append(f"# Signals emitted here: {', '.join(dict.fromkeys(real_emits))}")

    # If only the header line was added, there's nothing useful — return empty
    if len(lines) <= 1:
        return ""

    lines.append("# === END project context ===")
    return "\n".join(lines)
