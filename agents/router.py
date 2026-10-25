"""agents/router.py — Cost-aware model routing via a contextual bandit (LinUCB).

Each in-scope file is routed to a model tier ("arm") before review. After the
review, the verifier's confirmed/refuted outcomes provide a *free* online
reward signal — no human labels needed:

    reward = n_confirmed − n_refuted − λ · cost_norm
    cost_norm = (total_tokens / 1000) × arm relative cost

Zero-finding files earn −λ·cost_norm, which naturally pushes clean files
toward cheaper tiers. Known limitation: an empty result cannot distinguish
"file is genuinely clean" from "model was lazy" — the verifier only sees what
was proposed. This asymmetry is accepted in v1.

Algorithm: disjoint LinUCB (one linear model per arm). Chosen over neural
approaches because sample sizes are small (tens of files per run), the math
is explainable, and selection is fully deterministic (no seed dependence).

Cold start: arms never pulled are assigned round-robin before UCB kicks in.
Persistence: .kimi/router/linucb.json (gitignored) — learning compounds
across runs.

References: "Learning to Route LLMs from Bandit Feedback" (arXiv 2510.07429),
RouteLLM (arXiv 2406.18665).
"""

from __future__ import annotations

import json
import math
import os
from pathlib import Path

import numpy as np

# ── Arm table ─────────────────────────────────────────────────────────────────
# Ordered cheap → expensive; the order doubles as the escalation ladder.
# rel_cost is a *relative* weight for the reward's cost term — deliberately not
# a real price list (prices change; routing only needs the ordering+ratio).
# Override the set with ROUTER_ARMS="kimi-highspeed,claude-sonnet".

_ARM_TABLE: list[dict] = [
    {"name": "kimi-highspeed", "provider": "kimi",
     "model": "kimi-k2.7-code-highspeed", "rel_cost": 1.0,
     "requires_key": "MOONSHOT_API_KEY"},
    {"name": "kimi-standard", "provider": "kimi",
     "model": "kimi-k2.7-code", "rel_cost": 2.0,
     "requires_key": "MOONSHOT_API_KEY"},
    {"name": "kimi-k3", "provider": "kimi",
     "model": "kimi-k3", "rel_cost": 4.0,
     "requires_key": "MOONSHOT_API_KEY"},
    {"name": "claude-sonnet", "provider": "claude",
     "model": "claude-sonnet-4-6", "rel_cost": 8.0,
     "requires_key": "ANTHROPIC_API_KEY"},
]

_DEFAULT_LAMBDA = 0.05
_DEFAULT_ALPHA = 0.5
N_FEATURES = 9


def available_arms() -> list[dict]:
    """Arms usable in this environment, filtered by API keys and ROUTER_ARMS."""
    whitelist = os.environ.get("ROUTER_ARMS", "").strip()
    allowed = {n.strip() for n in whitelist.split(",") if n.strip()} or None
    arms = [
        a for a in _ARM_TABLE
        if os.environ.get(a["requires_key"])
        and (allowed is None or a["name"] in allowed)
    ]
    return arms


def escalation_ladder(arms: list[dict], assigned: str) -> list[str]:
    """Arm names from the assigned tier upward — retry escalation order."""
    names = [a["name"] for a in arms]
    try:
        return names[names.index(assigned):]
    except ValueError:
        return names


# ── Context features ───────────────────────────────────────────────────────────

def file_features(rel_path: str, index: dict, n_lines: int) -> np.ndarray:
    """9-dim structural feature vector for one file (bias term included)."""
    entry = index.get("scripts", {}).get(f"res://{rel_path}", {})
    extends = entry.get("extends") or ""
    depth = 0
    seen: set[str] = set()
    # Walk the extends chain through class_names to estimate inheritance depth
    current = extends
    while current and current not in seen:
        seen.add(current)
        nxt_path = index.get("class_names", {}).get(current)
        if not nxt_path:
            break
        depth += 1
        current = index.get("scripts", {}).get(nxt_path, {}).get("extends") or ""

    return np.array([
        1.0,                                    # bias
        math.log1p(n_lines),                    # file size
        math.log1p(len(entry.get("methods", {}))),
        float(len(entry.get("signals", []))),
        float(len(entry.get("preloads", []))),
        float(len(entry.get("onready_vars", []))),
        1.0 if entry.get("has_async_methods") else 0.0,
        1.0 if entry.get("is_tool") else 0.0,
        float(depth),
    ], dtype=float)


# ── Reward ─────────────────────────────────────────────────────────────────────

def compute_reward(
    findings: list[dict],
    usage: dict | None,
    rel_cost: float,
    lam: float = _DEFAULT_LAMBDA,
) -> float:
    """Verifier outcomes minus cost. usage=None → cost term is 0 (unknown)."""
    base = sum(
        1.0 if f.get("verified") == "confirmed"
        else -1.0 if f.get("verified") == "refuted"
        else 0.0
        for f in findings
    )
    if not usage:
        return base
    total = usage.get("prompt_tokens", 0) + usage.get("completion_tokens", 0)
    return base - lam * (total / 1000.0) * rel_cost


# ── LinUCB ─────────────────────────────────────────────────────────────────────

class LinUCB:
    """Disjoint LinUCB: per-arm A (d×d), b (d); UCB = θ·x + α·√(xᵀA⁻¹x)."""

    def __init__(self, arm_names: list[str], n_features: int = N_FEATURES,
                 alpha: float = _DEFAULT_ALPHA):
        self.arm_names = list(arm_names)
        self.n_features = n_features
        self.alpha = alpha
        self._A = {a: np.eye(n_features) for a in arm_names}
        self._b = {a: np.zeros(n_features) for a in arm_names}
        self.pulls = {a: 0 for a in arm_names}
        self.total_reward = {a: 0.0 for a in arm_names}

    # ── selection ──────────────────────────────────────────────────────────

    def _ucb(self, arm: str, x) -> float:
        x = np.asarray(x, dtype=float)
        a_inv = np.linalg.inv(self._A[arm])
        theta = a_inv @ self._b[arm]
        return float(theta @ x + self.alpha * math.sqrt(x @ a_inv @ x))

    def select(self, contexts: dict[str, np.ndarray]) -> dict[str, str]:
        """Assign each file an arm. Un-pulled arms get cold-start priority."""
        plan: dict[str, str] = {}
        cold = [a for a in self.arm_names if self.pulls[a] == 0]
        cold_i = 0
        for fname in sorted(contexts):
            if cold:
                plan[fname] = cold[cold_i % len(cold)]
                cold_i += 1
                if cold_i >= len(cold):
                    cold = []
            else:
                x = contexts[fname]
                plan[fname] = max(self.arm_names, key=lambda a: self._ucb(a, x))
        return plan

    # ── learning ───────────────────────────────────────────────────────────

    def update(self, arm: str, x, reward: float) -> None:
        if arm not in self._A:
            return
        x = np.asarray(x, dtype=float)
        self._A[arm] += np.outer(x, x)
        self._b[arm] += reward * x
        self.pulls[arm] += 1
        self.total_reward[arm] += reward

    # ── persistence ────────────────────────────────────────────────────────

    def to_dict(self) -> dict:
        return {
            "arm_names": self.arm_names,
            "n_features": self.n_features,
            "alpha": self.alpha,
            "A": {a: self._A[a].tolist() for a in self.arm_names},
            "b": {a: self._b[a].tolist() for a in self.arm_names},
            "pulls": self.pulls,
            "total_reward": self.total_reward,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "LinUCB":
        bandit = cls(data["arm_names"], data["n_features"], data["alpha"])
        for a in bandit.arm_names:
            bandit._A[a] = np.array(data["A"][a], dtype=float)
            bandit._b[a] = np.array(data["b"][a], dtype=float)
        bandit.pulls = {a: int(p) for a, p in data["pulls"].items()}
        bandit.total_reward = {a: float(r) for a, r in data["total_reward"].items()}
        return bandit

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(".tmp")
        tmp.write_text(json.dumps(self.to_dict()), encoding="utf-8")
        tmp.replace(path)  # atomic

    @classmethod
    def load(cls, path: Path, arm_names: list[str],
             n_features: int = N_FEATURES) -> "LinUCB":
        """Load bandit state; fresh instance if missing or arm set changed."""
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
            if sorted(data.get("arm_names", [])) != sorted(arm_names):
                return cls(arm_names, n_features)  # arm set changed — start over
            return cls.from_dict(data)
        except (OSError, json.JSONDecodeError, KeyError, ValueError):
            return cls(arm_names, n_features)
