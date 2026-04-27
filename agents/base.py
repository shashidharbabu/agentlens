"""Base class for all agents. Enforces the modular-design rubric item."""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any


class BaseAgent(ABC):
    """Every agent implements .run() and has a .name."""

    name: str = "base"

    @abstractmethod
    def run(self, *args: Any, **kwargs: Any) -> Any:
        """Execute the agent and return a typed dataclass output."""
        raise NotImplementedError
