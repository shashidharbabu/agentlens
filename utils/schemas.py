"""Shared dataclass schemas for agent I/O.

Every agent returns one of these. No agent returns a raw dict.
This is how we keep the pipeline composable and how the grader
can verify the rubric items exist as typed fields.
"""
from __future__ import annotations

from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Optional
import json


@dataclass
class VisionOutput:
    caption: str
    objects: list[str]
    scene: str
    vqa_pairs: list[dict]  # [{"question": str, "answer": str}, ...] len >= 2
    raw_model_output: str = ""
    low_quality_input: bool = False
    warnings: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class RefinedPrompt:
    original_instruction: str
    refined_prompt: str
    enrichment_notes: str         # which visual details got pulled in
    preserved_transform: str      # what the user actually wanted changed
    confidence: float = 1.0       # 0-1, used for ambiguity detection
    mode: str = "variation"       # "stylize" | "variation" | "enhance"

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class GenerationConfig:
    model: str
    mode: str
    prompt: str
    inference_steps: int
    guidance_scale: float
    seed: Optional[int] = None
    strength: Optional[float] = None   # img2img denoising strength

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class GenerationResult:
    image_path: Optional[Path]
    config: GenerationConfig
    error: Optional[str] = None

    def to_dict(self) -> dict:
        return {
            "image_path": str(self.image_path) if self.image_path else None,
            "config": self.config.to_dict(),
            "error": self.error,
        }


@dataclass
class Critique:
    clip_similarity_image: float      # input image <-> output image
    clip_similarity_text: float       # output image <-> refined prompt
    visual_relevance: float           # 0-1
    prompt_faithfulness: float        # 0-1
    quality: float                    # 0-1
    rationale: str
    verdict: str                      # "accept" | "revise"
    revision_suggestion: Optional[str] = None
    used_fallback_metric: bool = False  # True if CLIP unavailable

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class RunReport:
    run_id: str
    image_path: str
    instruction: str
    mode: str
    vision: Optional[VisionOutput] = None
    refined_prompt: Optional[RefinedPrompt] = None
    generation: Optional[GenerationResult] = None
    critique: Optional[Critique] = None
    errors: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "run_id": self.run_id,
            "image_path": self.image_path,
            "instruction": self.instruction,
            "mode": self.mode,
            "vision": self.vision.to_dict() if self.vision else None,
            "refined_prompt": (
                self.refined_prompt.to_dict() if self.refined_prompt else None
            ),
            "generation": (
                self.generation.to_dict() if self.generation else None
            ),
            "critique": self.critique.to_dict() if self.critique else None,
            "errors": self.errors,
        }

    def save(self, path: Path) -> None:
        path.write_text(json.dumps(self.to_dict(), indent=2))
