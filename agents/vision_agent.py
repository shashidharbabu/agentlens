"""Vision Understanding Agent — rubric 3.1.

Produces:
  - short caption
  - list of key objects/entities
  - brief scene description
  - answers to >= 2 visual questions

Uses a multimodal LLM (Claude) in real mode, or a deterministic stub
based on image stats in stub mode.
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
from utils.errors import LowQualityImageError
from utils.io_utils import is_low_quality, load_image
from utils.logger import (
    agent_call, agent_response, agent_result, agent_start, agent_warn, stub_notice,
)
from utils.schemas import VisionOutput


DEFAULT_VQA_QUESTIONS = [
    "What is the dominant color palette of this image?",
    "What is happening in the scene?",
]


VISION_SYSTEM_PROMPT = """You are a vision understanding agent. Analyze the image and return STRICT JSON with these keys:
{
  "caption": "one-sentence caption",
  "objects": ["object1", "object2", ...],   // up to 10 key objects/entities
  "scene": "2-3 sentence scene description",
  "vqa": [{"question": "...", "answer": "..."}, ...]   // one answer per question provided
}
Answer ONLY with the JSON object, no prose, no code fences."""


class VisionAgent(BaseAgent):
    name = "vision"

    def __init__(self, stub: Optional[bool] = None):
        self.stub = config.using_stub() if stub is None else stub

    def run(
        self,
        image_path: str | Path,
        vqa_questions: Optional[list[str]] = None,
    ) -> VisionOutput:
        import time
        img = load_image(image_path)
        w, h = img.size
        t0 = agent_start(
            "vision", "Image understanding",
            image=str(image_path),
            size=f"{w}x{h} px",
            mode="real LLM" if not self.stub else "stub (no API)",
        )

        low_q, reason = is_low_quality(img)
        warnings: list[str] = []
        if low_q:
            agent_warn("vision", reason)
            warnings.append(reason)

        questions = vqa_questions or DEFAULT_VQA_QUESTIONS
        if len(questions) < 2:
            questions = list(questions) + DEFAULT_VQA_QUESTIONS[len(questions):2]

        if self.stub:
            stub_notice("vision")
            out = self._stub_analyze(img, questions)
        else:
            try:
                out = self._llm_analyze(img, questions, t0)
            except Exception as exc:
                agent_warn("vision", f"LLM failed ({exc}); falling back to stub.")
                warnings.append(f"LLM failed ({exc}); using stub.")
                out = self._stub_analyze(img, questions)

        out.low_quality_input = low_q
        out.warnings = warnings

        import time as _t
        agent_result(
            "vision", _t.time() - t0,
            caption=out.caption,
            objects=", ".join(out.objects[:6]) + ("…" if len(out.objects) > 6 else ""),
            scene=out.scene[:120] + ("…" if len(out.scene) > 120 else ""),
            vqa_pairs=f"{len(out.vqa_pairs)} answered",
        )
        for pair in out.vqa_pairs:
            print(f"      Q: {pair['question']}")
            print(f"      A: {pair['answer'][:160]}")
        return out

    # -------------------------------------------------------------------
    # Real-model path
    # -------------------------------------------------------------------
    def _llm_analyze(
        self, img: Image.Image, questions: list[str], t0: float
    ) -> VisionOutput:
        import time
        from anthropic import Anthropic

        client = Anthropic(api_key=config.ANTHROPIC_API_KEY)
        b64 = _image_to_base64_jpeg(img)
        img_kb = len(b64) * 3 // 4 // 1024  # approx decoded bytes

        user_text = (
            "Analyze this image.\n"
            "Also answer these visual questions:\n"
            + "\n".join(f"- {q}" for q in questions)
        )

        agent_call(
            "vision",
            f"POST anthropic/messages  model={config.ANTHROPIC_MODEL}",
            image_size=f"~{img_kb} KB (JPEG)",
            vqa_questions="\n            ".join(questions),
            max_tokens=1024,
        )

        t_call = time.time()
        resp = client.messages.create(
            model=config.ANTHROPIC_MODEL,
            max_tokens=1024,
            system=VISION_SYSTEM_PROMPT,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image",
                            "source": {
                                "type": "base64",
                                "media_type": "image/jpeg",
                                "data": b64,
                            },
                        },
                        {"type": "text", "text": user_text},
                    ],
                }
            ],
        )
        elapsed = int((time.time() - t_call) * 1000)
        raw = "".join(
            block.text for block in resp.content if getattr(block, "type", "") == "text"
        )
        usage = resp.usage
        agent_response(
            "vision", elapsed,
            input_tokens=usage.input_tokens,
            output_tokens=usage.output_tokens,
            raw_json=raw[:200],
        )

        parsed = _robust_json_parse(raw)
        return VisionOutput(
            caption=parsed.get("caption", ""),
            objects=list(parsed.get("objects", []))[:10],
            scene=parsed.get("scene", ""),
            vqa_pairs=parsed.get("vqa", []),
            raw_model_output=raw,
        )

    # -------------------------------------------------------------------
    # Stub path (no API, deterministic, still produces all fields)
    # -------------------------------------------------------------------
    def _stub_analyze(
        self, img: Image.Image, questions: list[str]
    ) -> VisionOutput:
        w, h = img.size
        orientation = "landscape" if w > h else ("portrait" if h > w else "square")
        avg = _avg_rgb(img)
        palette = _describe_palette(avg)
        brightness = sum(avg) / 3

        caption = f"A {orientation} image with a predominantly {palette} palette."
        scene = (
            f"This is a {orientation}-oriented image, {w}x{h} pixels, "
            f"with an average brightness of {brightness:.0f}/255 and a "
            f"{palette} color cast. The content appears visually coherent."
        )
        objects = ["subject", "background", "foreground elements"]
        vqa_pairs = []
        for q in questions:
            if "color" in q.lower():
                a = f"The dominant palette is {palette}."
            elif "happen" in q.lower() or "scene" in q.lower():
                a = f"A {orientation} composition with {palette} tones."
            else:
                a = f"Based on the image, the answer relates to its {palette} {orientation} composition."
            vqa_pairs.append({"question": q, "answer": a})

        return VisionOutput(
            caption=caption,
            objects=objects,
            scene=scene,
            vqa_pairs=vqa_pairs,
            raw_model_output="[stub mode — no LLM called]",
        )


# ---------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------
def _image_to_base64_jpeg(img: Image.Image) -> str:
    buf = io.BytesIO()
    img.convert("RGB").save(buf, format="JPEG", quality=90)
    return base64.b64encode(buf.getvalue()).decode("ascii")


def _robust_json_parse(raw: str) -> dict:
    """Try direct JSON, then extract the first {...} block."""
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
    return {"caption": raw[:120], "objects": [], "scene": raw, "vqa": []}


def _avg_rgb(img: Image.Image) -> tuple[int, int, int]:
    small = img.resize((32, 32)).convert("RGB")
    pixels = list(small.getdata())
    r = sum(p[0] for p in pixels) // len(pixels)
    g = sum(p[1] for p in pixels) // len(pixels)
    b = sum(p[2] for p in pixels) // len(pixels)
    return r, g, b


def _describe_palette(rgb: tuple[int, int, int]) -> str:
    r, g, b = rgb
    if max(r, g, b) - min(r, g, b) < 20:
        brightness = (r + g + b) / 3
        if brightness < 80:
            return "dark neutral"
        if brightness > 180:
            return "bright neutral"
        return "muted neutral"
    if r > g and r > b:
        return "warm red"
    if g > r and g > b:
        return "green"
    if b > r and b > g:
        return "cool blue"
    if r > 150 and g > 150 and b < 100:
        return "warm yellow"
    return "mixed"
