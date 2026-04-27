"""Terminal logging helpers for the multimodal pipeline.

Produces clearly-formatted, human-readable output so you can follow
exactly what each agent is doing at every step.

No external dependencies — just print() with ANSI colors and box chars.
"""
from __future__ import annotations

import time
from typing import Any


# ---------------------------------------------------------------------------
# ANSI color codes (safe on macOS/Linux terminals)
# ---------------------------------------------------------------------------
RESET  = "\033[0m"
BOLD   = "\033[1m"
DIM    = "\033[2m"

CYAN   = "\033[96m"
GREEN  = "\033[92m"
YELLOW = "\033[93m"
RED    = "\033[91m"
BLUE   = "\033[94m"
MAGENTA= "\033[95m"
WHITE  = "\033[97m"

# Agent name → colour
AGENT_COLORS = {
    "vision":     CYAN,
    "prompt":     BLUE,
    "generation": MAGENTA,
    "critique":   YELLOW,
    "pipeline":   GREEN,
}


def _color(agent: str, text: str) -> str:
    c = AGENT_COLORS.get(agent, WHITE)
    return f"{c}{text}{RESET}"


def _trunc(s: str, n: int = 120) -> str:
    s = s.replace("\n", " ")
    return s if len(s) <= n else s[:n] + "…"


# ---------------------------------------------------------------------------
# Public helpers
# ---------------------------------------------------------------------------

def section(title: str) -> None:
    """Print a top-level section banner (used by run_demo.py)."""
    bar = "═" * 72
    print(f"\n{BOLD}{GREEN}{bar}{RESET}")
    print(f"{BOLD}{GREEN}  {title}{RESET}")
    print(f"{BOLD}{GREEN}{bar}{RESET}")


def agent_start(agent: str, label: str, **details: Any) -> float:
    """Print the opening line for an agent step. Returns the start time."""
    t = time.time()
    prefix = _color(agent, f"┌─[{agent.upper()} AGENT]")
    print(f"\n{prefix} {BOLD}{label}{RESET}")
    for k, v in details.items():
        print(f"  {DIM}{k}:{RESET} {_trunc(str(v))}")
    return t


def agent_call(agent: str, endpoint: str, **payload: Any) -> None:
    """Print what's being sent to the API."""
    prefix = _color(agent, "  ├─► API CALL")
    print(f"{prefix}  {BOLD}{endpoint}{RESET}")
    for k, v in payload.items():
        print(f"  │   {DIM}{k}:{RESET} {_trunc(str(v))}")


def agent_response(agent: str, elapsed_ms: int, **fields: Any) -> None:
    """Print the key fields received back from the API."""
    prefix = _color(agent, "  ├─◄ RESPONSE")
    print(f"{prefix}  {DIM}({elapsed_ms} ms){RESET}")
    for k, v in fields.items():
        print(f"  │   {DIM}{k}:{RESET} {_trunc(str(v))}")


def agent_result(agent: str, elapsed_s: float, **fields: Any) -> None:
    """Print the parsed / final output of an agent step."""
    ms = int(elapsed_s * 1000)
    prefix = _color(agent, f"└─[{agent.upper()} DONE]")
    print(f"{prefix}  {DIM}total {ms} ms{RESET}")
    for k, v in fields.items():
        val_str = _trunc(str(v), 160)
        print(f"    {BOLD}{k}:{RESET} {val_str}")


def agent_warn(agent: str, msg: str) -> None:
    print(f"  {YELLOW}⚠  {msg}{RESET}")


def agent_error(agent: str, msg: str) -> None:
    print(f"  {RED}✖  [{agent.upper()}] {msg}{RESET}")


def agent_skip(agent: str, msg: str) -> None:
    print(f"  {DIM}↷  [{agent.upper()}] {msg}{RESET}")


def stub_notice(agent: str) -> None:
    print(f"  {DIM}[{agent.upper()}] stub mode — no API call{RESET}")
