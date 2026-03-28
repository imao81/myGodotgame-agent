"""Vision analyzer — analyzes game screenshots via the configured LLM provider."""

import base64
import json
import re
from pathlib import Path

from agents.llm_client import LLMClient

_VISION_PROMPT = (
    "This is a screenshot from Harvester, a 2D top-down space mining game (Godot 4).\n"
    "Expected game state: {expected_state}\n\n"
    "Identify any visual bugs, rendering artifacts, UI misalignments, "
    "overlapping elements, missing sprites, or states that don't match the expected state.\n"
    'Reply with a JSON object: {{"ok": true/false, "issues": ["issue1", "issue2"]}} '
    "where issues is empty if everything looks correct."
)


def analyze_screenshot(client: LLMClient, image_path: str, expected_state: str) -> dict:
    """Analyze a game screenshot and return findings.

    Returns {"path": str, "expected": str, "issues": list[str], "ok": bool}
    """
    img_bytes = Path(image_path).read_bytes()
    img_b64 = base64.standard_b64encode(img_bytes).decode()

    ext = Path(image_path).suffix.lower()
    media_type = {".png": "image/png", ".jpg": "image/jpeg", ".jpeg": "image/jpeg"}.get(
        ext, "image/png"
    )

    prompt = _VISION_PROMPT.format(expected_state=expected_state)

    try:
        text = client.chat_vision(img_b64, media_type, prompt, max_tokens=512).strip()
    except Exception as exc:
        return {
            "path": image_path,
            "expected": expected_state,
            "ok": False,
            "issues": [f"Vision API error: {exc}"],
        }

    match = re.search(r"\{.*\}", text, re.DOTALL)
    if match:
        try:
            data = json.loads(match.group())
            return {"path": image_path, "expected": expected_state, **data}
        except json.JSONDecodeError:
            pass

    return {
        "path": image_path,
        "expected": expected_state,
        "ok": False,
        "issues": [f"Could not parse vision response: {text[:200]}"],
    }


def analyze_all(client: LLMClient, screenshots: list[dict]) -> list[dict]:
    """Analyze a list of {"path": str, "expected": str} dicts.

    Returns list of analysis results.
    """
    return [analyze_screenshot(client, s["path"], s["expected"]) for s in screenshots]
