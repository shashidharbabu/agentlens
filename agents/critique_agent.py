"""Critique and Evaluation Agent — rubric 3.4.

Assesses the generated image on:
  - visual relevance (input <-> output)
  - prompt faithfulness (output <-> refined prompt)
  - quality of the transformation
Uses CLIP similarity as the automatic signal and an LLM (Claude) for
the qualitative rationale. Produces an accept/revise verdict.
"""
from __future__ import annotations

import base64
import io
import json
import re
from pathlib import Path
from typing import Optional

from PIL import Image

import config
from agents.base import BaseAgent
from utils.clip_utils import (
    clip_image_image_similarity,
    clip_image_text_similarity,
)
from utils.errors import CritiqueError
from utils.io_utils import load_image
from utils.logger import (
    agent_call, agent_response, agent_result, agent_start, agent_warn, stub_notice,
)
from utils.schemas import Critique, GenerationResult, RefinedPrompt


CRITIQUE_SYSTEM = """You are a critique agent. You are given an input image, a generated output image, and the refined prompt used for generation.

Score each of the following on a 0.0-1.0 scale:
  - visual_relevance: how well the output preserves relevant content of the input
  - prompt_faithfulness: how well the output follows the refined prompt
  - quality: overall technical quality of the transformation

Return STRICT JSON:
{
  "visual_relevance": 0.0-1.0,
  "prompt_faithfulness": 0.0-1.0,
  "quality": 0.0-1.0,
  "rationale": "2-3 sentence critique",
  "verdict": "accept" | "revise",
  "revision_suggestion": "one-line concrete fix if verdict is revise, else null"
}"""


# Accept thresholds (also documented in plan.md section 5.4)
MIN_CLIP_TEXT = 0.22
MIN_QUALITY = 0.55


class CritiqueAgent(BaseAgent):
    name = "critique"

    def __init__(self, stub: Optional[bool] = None):
        self.stub = config.using_stub() if stub is None else stub

    def run(
        self,
        input_image_path: str | Path,
        generation: GenerationResult,
        refined: RefinedPrompt,
    ) -> Critique:
        import time
        t0 = agent_start(
            "critique", "Critique and evaluation",
            input_image=str(input_image_path),
            generated_image=str(generation.image_path) if generation.image_path else "NONE (generation failed)",
            mode="real LLM + CLIP" if not self.stub else "stub (CLIP only)",
        )

        # If generation failed, issue the verdict immediately.
        if generation.image_path is None or generation.error:
            agent_warn("critique", f"Generation failed — auto-issuing 'revise' verdict: {generation.error}")
            c = Critique(
                clip_similarity_image=0.0,
                clip_similarity_text=0.0,
                visual_relevance=0.0,
                prompt_faithfulness=0.0,
                quality=0.0,
                rationale=f"Generation failed: {generation.error or 'no image produced'}",
                verdict="revise",
                revision_suggestion="Retry generation; consider simpler prompt or different backend.",
                used_fallback_metric=True,
            )
            agent_result("critique", time.time() - t0, verdict=c.verdict, rationale=c.rationale[:120])
            return c

        in_img = load_image(input_image_path)
        out_img = load_image(generation.image_path)

        # 1. Automatic CLIP signals
        print(f"  \033[93m├─► CLIP\033[0m  computing image↔image and image↔text similarity…")
        t_clip = time.time()
        sim_img, fb1 = clip_image_image_similarity(in_img, out_img)
        sim_txt, fb2 = clip_image_text_similarity(out_img, refined.refined_prompt)
        used_fallback = fb1 or fb2
        clip_ms = int((time.time() - t_clip) * 1000)
        fb_note = " (SSIM fallback — CLIP unavailable)" if used_fallback else " (CLIP ViT-B-32)"
        print(f"  \033[93m│   CLIP done\033[0m  {clip_ms} ms{fb_note}")
        print(f"  \033[93m│   image↔image similarity:\033[0m  {sim_img:.4f}  (1.0 = identical)")
        print(f"  \033[93m│   image↔text  similarity:\033[0m  {sim_txt:.4f}  (threshold ≥ {MIN_CLIP_TEXT})")

        # 2. LLM qualitative reasoning
        if self.stub:
            stub_notice("critique")
            scores = self._stub_score(sim_img, sim_txt)
        else:
            try:
                scores = self._llm_score(in_img, out_img, refined.refined_prompt, t0)
            except Exception as exc:
                agent_warn("critique", f"LLM scoring failed ({exc}); falling back to stub scores.")
                scores = self._stub_score(sim_img, sim_txt)

        visual_relevance = scores["visual_relevance"]
        prompt_faithfulness = scores["prompt_faithfulness"]
        quality = scores["quality"]

        # 3. Verdict rule
        accept = (sim_txt >= MIN_CLIP_TEXT) and (quality >= MIN_QUALITY)
        if used_fallback:
            accept = quality >= MIN_QUALITY and prompt_faithfulness >= 0.5

        verdict = "accept" if accept else "revise"
        revision = (
            None
            if verdict == "accept"
            else scores.get(
                "revision_suggestion",
                "Regenerate with a more specific prompt or adjust the transformation strength.",
            )
        )

        c = Critique(
            clip_similarity_image=round(sim_img, 4),
            clip_similarity_text=round(sim_txt, 4),
            visual_relevance=round(visual_relevance, 3),
            prompt_faithfulness=round(prompt_faithfulness, 3),
            quality=round(quality, 3),
            rationale=scores["rationale"],
            verdict=verdict,
            revision_suggestion=revision,
            used_fallback_metric=used_fallback,
        )

        verdict_color = "\033[92m" if verdict == "accept" else "\033[91m"
        agent_result(
            "critique", time.time() - t0,
            **{
                "clip_img↔img":   f"{c.clip_similarity_image:.4f}",
                "clip_img↔text":  f"{c.clip_similarity_text:.4f}  (threshold ≥ {MIN_CLIP_TEXT})",
                "visual_relevance":    f"{c.visual_relevance:.3f}",
                "prompt_faithfulness": f"{c.prompt_faithfulness:.3f}",
                "quality":             f"{c.quality:.3f}  (threshold ≥ {MIN_QUALITY})",
                "rationale":      c.rationale[:200],
                "verdict":        f"{verdict_color}{verdict.upper()}\033[0m"
                                  + (f"  →  {revision}" if revision else ""),
            },
        )
        return c

    # -------------------------------------------------------------------
    def _llm_score(
        self, in_img: Image.Image, out_img: Image.Image, prompt: str, t0: float = 0.0
    ) -> dict:
        import time
        from anthropic import Anthropic

        client = Anthropic(api_key=config.ANTHROPIC_API_KEY)

        agent_call(
            "critique",
            f"POST anthropic/messages  model={config.ANTHROPIC_MODEL}",
            content="input image + generated image + refined prompt",
            max_tokens=600,
        )

        t_call = time.time()
        resp = client.messages.create(
            model=config.ANTHROPIC_MODEL,
            max_tokens=600,
            system=CRITIQUE_SYSTEM,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "INPUT IMAGE:"},
                        _img_block(in_img),
                        {"type": "text", "text": "GENERATED OUTPUT IMAGE:"},
                        _img_block(out_img),
                        {"type": "text", "text": f"REFINED PROMPT:\n{prompt}"},
                    ],
                }
            ],
        )
        elapsed = int((time.time() - t_call) * 1000)
        raw = "".join(b.text for b in resp.content if getattr(b, "type", "") == "text")
        usage = resp.usage
        agent_response(
            "critique", elapsed,
            input_tokens=usage.input_tokens,
            output_tokens=usage.output_tokens,
            raw_json=raw[:200],
        )

        parsed = _robust_json_parse(raw)
        return {
            "visual_relevance": float(parsed.get("visual_relevance", 0.5)),
            "prompt_faithfulness": float(parsed.get("prompt_faithfulness", 0.5)),
            "quality": float(parsed.get("quality", 0.5)),
            "rationale": parsed.get("rationale", "No rationale returned."),
            "revision_suggestion": parsed.get("revision_suggestion"),
        }

    def _stub_score(self, sim_img: float, sim_txt: float) -> dict:
        # Map CLIP scores into the 0-1 rubric score space with reasonable ceilings.
        visual_relevance = max(0.0, min(1.0, (sim_img + 1) / 2))
        prompt_faithfulness = max(0.0, min(1.0, sim_txt * 2.5 + 0.2))
        # quality: if image looks similar AND prompt aligned, call it decent
        quality = 0.5 + 0.3 * visual_relevance + 0.2 * prompt_faithfulness
        quality = min(1.0, quality)
        rationale = (
            f"[stub critique] CLIP image-image={sim_img:.3f}, "
            f"image-text={sim_txt:.3f}. "
            "The transformation appears consistent with the input and prompt "
            "based on automatic signals; LLM reasoning not available in stub mode."
        )
        return {
            "visual_relevance": visual_relevance,
            "prompt_faithfulness": prompt_faithfulness,
            "quality": quality,
            "rationale": rationale,
            "revision_suggestion": "Increase prompt specificity or adjust generation strength.",
        }


# ---------------------------------------------------------------------
def _img_block(img: Image.Image) -> dict:
    buf = io.BytesIO()
    img.convert("RGB").save(buf, format="JPEG", quality=85)
    b64 = base64.b64encode(buf.getvalue()).decode("ascii")
    return {
        "type": "image",
        "source": {"type": "base64", "media_type": "image/jpeg", "data": b64},
    }


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
