"""Config Generator Agent.

Reads all .tres balance resource files, builds a balance profile, then
calls Claude to suggest one data-driven config variant based on playtest findings.
If no balance issues were found, reports "balance looks nominal."
"""

import json

from agents.llm_client import LLMClient
from tools.tres_parser import parse_all

_CONFIG_PROMPT = """\
You are a game balance consultant for Harvester, a Godot 4 top-down space mining game.

Game summary:
- Player mines asteroids for iron, returns to station to deposit and unlock upgrades
- Pirates spawn around asteroids as iron is deposited, increasing difficulty over time
- Player upgrades: health, speed, rotation speed, cargo space, weapon damage

Current balance values (from .tres resource files):
{balance_profile}

Playtest balance findings:
{balance_issues}

Based on these findings, propose ONE suggested config variant with targeted changes.
If the playtest found no balance issues, respond with {{"verdict": "nominal", "suggestions": []}}.

Otherwise respond with:
{{
  "verdict": "adjustments_needed",
  "suggestions": [
    {{
      "resource": "<relative path>",
      "changes": {{"<field>": <new_value>}},
      "reason": "<one sentence why>"
    }}
  ]
}}

Only output the JSON object, nothing else.
"""


def run(state: dict) -> dict:
    project_path = state["project_path"]
    balance_issues = state.get("balance_issues", [])

    # Parse all .tres files to build balance profile
    all_resources = parse_all(project_path)

    # Filter to balance-relevant resources (exclude icon/effect resources)
    balance_resources = {
        path: values
        for path, values in all_resources.items()
        if any(kw in path for kw in ("stats", "gun", "cargo"))
    }

    if not balance_resources:
        return {
            "config_suggestions": [],
            "agent_status": {"config_gen": "ok"},
        }

    balance_profile = json.dumps(balance_resources, indent=2)
    issues_text = "\n".join(balance_issues) if balance_issues else "No balance issues detected."

    client = LLMClient()
    try:
        text = client.chat(
            [{
                "role": "user",
                "content": _CONFIG_PROMPT.format(
                    balance_profile=balance_profile,
                    balance_issues=issues_text,
                ),
            }],
            max_tokens=1024,
        ).strip()

        import re
        match = re.search(r"\{.*\}", text, re.DOTALL)
        if match:
            data = json.loads(match.group())
            suggestions = data.get("suggestions", [])
            verdict = data.get("verdict", "unknown")
        else:
            suggestions = []
            verdict = "parse_error"

    except Exception as e:
        suggestions = []
        verdict = "error"
        # Return immediately — no point attaching current_values to empty list
        return {
            "config_suggestions": [],
            "config_verdict": verdict,
            "config_error": str(e),
            "agent_status": {"config_gen": "failed"},
        }

    # Attach current values to each suggestion for reference
    for s in suggestions:
        path = s.get("resource", "")
        s["current_values"] = balance_resources.get(path, {})

    return {
        "config_suggestions": suggestions,
        "config_verdict": verdict,
        "config_error": None,
        "agent_status": {"config_gen": "ok"},
    }
