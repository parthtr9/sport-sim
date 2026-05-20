from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

import streamlit as st

DATA_DIR = Path(__file__).parent / "data"
STATE_FILE = DATA_DIR / "state.json"
AGENT_MEMORY_DIR = DATA_DIR / "agent_memory"

CATEGORIES = ["macro", "markets", "crypto", "culture", "policy", "company"]


def _empty_state() -> dict[str, list]:
    return {"markets": [], "news": [], "edges": []}


def load_state() -> dict[str, list]:
    if STATE_FILE.exists():
        return json.loads(STATE_FILE.read_text(encoding="utf-8"))
    return _empty_state()


def save_state(state: dict[str, list]) -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    STATE_FILE.write_text(
        json.dumps(state, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )


def get_state() -> dict[str, list]:
    """Load state into session_state on first call, return it on subsequent calls."""
    if "app_state" not in st.session_state:
        st.session_state.app_state = load_state()
    return st.session_state.app_state


def persist() -> None:
    """Write the current session state back to disk."""
    save_state(st.session_state.app_state)


def slugify(text: str) -> str:
    slug = re.sub(r"[^a-z0-9]+", "-", text.lower()).strip("-")
    return slug[:64]


# ── Lookup helpers ────────────────────────────────────────────────────────

def find_market(state: dict, market_id: str) -> dict | None:
    return next((m for m in state["markets"] if m["id"] == market_id), None)


def find_news(state: dict, news_id: str) -> dict | None:
    return next((n for n in state["news"] if n["id"] == news_id), None)


def edges_targeting(state: dict, target_id: str) -> list[dict]:
    return [e for e in state["edges"] if e["target_id"] == target_id]


def edges_from(state: dict, source_id: str) -> list[dict]:
    return [e for e in state["edges"] if e["source_id"] == source_id]


def news_edges_for_market(state: dict, market_id: str) -> list[dict]:
    return [
        e for e in state["edges"]
        if e["target_id"] == market_id and e["source_type"] == "news"
    ]


def market_edges_for_market(state: dict, market_id: str) -> list[dict]:
    return [
        e for e in state["edges"]
        if e["target_id"] == market_id and e["source_type"] == "market"
    ]


# ── CRUD helpers ──────────────────────────────────────────────────────────

def add_market(state: dict, market: dict[str, Any]) -> None:
    state["markets"].append(market)


def remove_market(state: dict, market_id: str) -> None:
    state["markets"] = [m for m in state["markets"] if m["id"] != market_id]
    state["edges"] = [
        e for e in state["edges"]
        if e["source_id"] != market_id and e["target_id"] != market_id
    ]


def add_news(state: dict, news: dict[str, Any]) -> None:
    state["news"].append(news)


def remove_news(state: dict, news_id: str) -> None:
    state["news"] = [n for n in state["news"] if n["id"] != news_id]
    state["edges"] = [
        e for e in state["edges"]
        if e["source_id"] != news_id
    ]


def add_edge(state: dict, edge: dict[str, Any]) -> None:
    state["edges"].append(edge)


def remove_edge(state: dict, source_id: str, target_id: str) -> None:
    state["edges"] = [
        e for e in state["edges"]
        if not (e["source_id"] == source_id and e["target_id"] == target_id)
    ]


# ── Agent memory persistence (cross-event) ────────────────────────────────

def save_agent_reactions(market_id: str, reactions: list[dict[str, Any]]) -> None:
    """Save agent reactions to JSON for reuse across news events.
    
    Enables delta-reasoning: next news event can pass these prior reactions
    as context, avoiding full re-evaluation and reducing LLM token usage.
    """
    AGENT_MEMORY_DIR.mkdir(parents=True, exist_ok=True)
    memory_file = AGENT_MEMORY_DIR / f"{market_id}_reactions.json"
    memory_file.write_text(
        json.dumps(reactions, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )


def load_agent_reactions(market_id: str) -> list[dict[str, Any]] | None:
    """Load previously saved agent reactions for a market."""
    memory_file = AGENT_MEMORY_DIR / f"{market_id}_reactions.json"
    if memory_file.exists():
        return json.loads(memory_file.read_text(encoding="utf-8"))
    return None


def clear_agent_reactions(market_id: str) -> None:
    """Clear saved reactions for a market (e.g., when resetting simulation)."""
    memory_file = AGENT_MEMORY_DIR / f"{market_id}_reactions.json"
    if memory_file.exists():
        memory_file.unlink()


def clear_all_agent_memory() -> None:
    """Clear all agent memory across all markets."""
    if AGENT_MEMORY_DIR.exists():
        import shutil
        shutil.rmtree(AGENT_MEMORY_DIR)
