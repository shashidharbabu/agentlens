"""Custom exceptions so failure modes are explicit and testable.

Maps directly to rubric 3.5.5 (Failure Handling):
  - Ambiguous prompt  -> AmbiguousPromptError
  - Poor-quality image -> LowQualityImageError (warning, not fatal)
  - Generation error  -> GenerationError
"""
from __future__ import annotations


class AgentError(Exception):
    """Base class for all agent errors."""


class AmbiguousPromptError(AgentError):
    """Raised when the user instruction cannot be refined confidently."""


class LowQualityImageError(AgentError):
    """Raised when the input image is unusable (decode failure, tiny, etc.)."""


class GenerationError(AgentError):
    """Raised when image generation fails (API timeout, NSFW, OOM, ...)."""


class CritiqueError(AgentError):
    """Raised when the critique agent can't produce any signal at all."""
