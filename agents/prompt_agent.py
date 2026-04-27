"""Prompt Engineering Agent — rubric 3.2.

Takes the user instruction + VisionOutput, produces a RefinedPrompt
that (a) rewrites clearly, (b) enriches with visual details from the
scene, and (c) explicitly preserves the intended transformation.
"""
from __future__ import annotations

import json
import re
from typing import Optional

import config
from agents.base import BaseAgent
from utils.errors import AmbiguousPromptError
from utils.logger import (
    agent_call, agent_response, agent_result, agent_start, agent_warn, stub_notice,
)
from utils.schemas import RefinedPrompt, VisionOutput


PROMPT_SYSTEM = """You are a prompt engineering agent for an image generation pipeline.
Given: (1) the user's natural-language instruction, (2) a structured vision analysis of the input image, and (3) a generation mode (stylize|variation|enhance).

Your job: produce a refined image-generation prompt that:
  - rewrites the user's request clearly
  - grounds it in concrete visual details from the scene analysis
  - preserves the user's intended transformation exactly
  - is 30-80 words, descriptive, concrete, no hedging

Return STRICT JSON:
{
  "refined_prompt": "...",
  "enrichment_notes": "which visual details from the scene you incorporated",
  "preserved_transform": "the user's core requested change, restated",
  "confidence": 0.0-1.0
}
Only output the JSON, no prose."""


VALID_MODES = {"stylize", "variation", "enhance"}


class PromptAgent(BaseAgent):
    name = "prompt"

    def __init__(self, stub: Optional[bool] = None):
        self.stub = config.using_stub() if stub is None else stub

    def run(
        self,
        user_instruction: str,
        vision: VisionOutput,
        mode: str = "variation",
    ) -> RefinedPrompt:
        import time
        if mode not in VALID_MODES:
            mode = "variation"

        instruction = (user_instruction or "").strip()

        t0 = agent_start(
            "prompt", "Prompt engineering",
            instruction=instruction or "(empty)",
            mode=mode,
            vision_caption=vision.caption[:100],
            mode_flag="real LLM" if not self.stub else "stub (no API)",
        )

        if len(instruction) < 3:
            from utils.errors import AmbiguousPromptError
            agent_warn("prompt", f"Instruction too short ({len(instruction)} chars) — raising AmbiguousPromptError")
            raise AmbiguousPromptError(
                f"Instruction too short to refine: {instruction!r}"
            )

        if self.stub:
            stub_notice("prompt")
            refined = self._stub_refine(instruction, vision, mode)
        else:
            try:
                refined = self._llm_refine(instruction, vision, mode, t0)
            except Exception as exc:
                agent_warn("prompt", f"LLM failed ({exc}); falling back to stub.")
                refined = self._stub_refine(instruction, vision, mode)

        if refined.confidence < 0.5 or len(refined.refined_prompt) < 10:
            agent_warn("prompt", f"Low confidence ({refined.confidence:.2f}) — consider clarifying instruction.")
            refined.enrichment_notes += " [low confidence — consider clarifying]"

        agent_result(
            "prompt", time.time() - t0,
            refined_prompt=refined.refined_prompt[:160],
            enrichment_notes=refined.enrichment_notes[:120],
            preserved_transform=refined.preserved_transform[:120],
            confidence=f"{refined.confidence:.2f}",
            mode=refined.mode,
        )
        return refined

    # -------------------------------------------------------------------
    def _llm_refine(
        self, instruction: str, vision: VisionOutput, mode: str, t0: float
    ) -> RefinedPrompt:
        import time
        from anthropic import Anthropic

        client = Anthropic(api_key=config.ANTHROPIC_API_KEY)
        user_msg = json.dumps(
            {
                "user_instruction": instruction,
                "mode": mode,
                "vision": {
                    "caption": vision.caption,
                    "objects": vision.objects,
                    "scene": vision.scene,
                },
            },
            indent=2,
        )

        agent_call(
            "prompt",
            f"POST anthropic/messages  model={config.ANTHROPIC_MODEL}",
            user_instruction=instruction,
            mode=mode,
            vision_objects=", ".join(vision.objects[:5]),
            max_tokens=512,
        )

        t_call = time.time()
        resp = client.messages.create(
            model=config.ANTHROPIC_MODEL,
            max_tokens=512,
            system=PROMPT_SYSTEM,
            messages=[{"role": "user", "content": user_msg}],
        )
        elapsed = int((time.time() - t_call) * 1000)
        raw = "".join(
            b.text for b in resp.content if getattr(b, "type", "") == "text"
        )
        usage = resp.usage
        agent_response(
            "prompt", elapsed,
            input_tokens=usage.input_tokens,
            output_tokens=usage.output_tokens,
            raw_json=raw[:200],
        )

        parsed = _robust_json_parse(raw)
        return RefinedPrompt(
            original_instruction=instruction,
            refined_prompt=parsed.get("refined_prompt", instruction),
            enrichment_notes=parsed.get("enrichment_notes", ""),
            preserved_transform=parsed.get("preserved_transform", instruction),
            confidence=float(parsed.get("confidence", 0.8)),
            mode=mode,
        )

    def _stub_refine(
        self, instruction: str, vision: VisionOutput, mode: str
    ) -> RefinedPrompt:
        objs = ", ".join(vision.objects[:5]) if vision.objects else "the main subject"
        scene_snip = (vision.scene or vision.caption or "").strip()
        if len(scene_snip) > 120:
            scene_snip = scene_snip[:120].rsplit(" ", 1)[0] + "..."

        mode_hint = {
            "stylize": "rendered in a distinct artistic style",
            "variation": "as a creative variation",
            "enhance": "enhanced with sharper detail and improved lighting",
        }[mode]

        refined = (
            f"{instruction.rstrip('.')}, {mode_hint}. "
            f"Scene: {scene_snip} Key elements: {objs}. "
            f"High quality, coherent composition."
        )
        return RefinedPrompt(
            original_instruction=instruction,
            refined_prompt=refined,
            enrichment_notes=(
                f"Incorporated scene description and {len(vision.objects)} key objects "
                f"from the vision agent."
            ),
            preserved_transform=instruction,
            confidence=0.7,
            mode=mode,
        )


# ---------------------------------------------------------------------
def _robust_json_parse(raw: str) -> dict:
    try:
        return json.loads(raw)
    except Exception:
        pass
    m = re.search(r"\{.*\}", raw, re.DOTALL)
    if m:
        try:
            return json.loads(m.group(0))
        except Exception:
            pass
    return {}
