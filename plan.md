# Part 3 — Multimodal Multi-Agent System: Technical Plan

End-to-end implementation plan for a 4-agent pipeline that does **Vision Understanding → Prompt Engineering → Image Generation → Critique/Evaluation**, mapped 1:1 to the 30-mark rubric so nothing gets dropped.

---

## 1. Rubric → Code Mapping (the "don't miss anything" table)

| Rubric Item | Marks | Deliverable in this repo |
|---|---|---|
| 3.1 Vision Understanding Agent: caption + objects + scene + ≥2 VQA | 6 | `agents/vision_agent.py` returns `VisionOutput` with all 4 fields |
| 3.2 Prompt Engineering Agent: rewrite + enrich + preserve transform | 6 | `agents/prompt_agent.py` returns `RefinedPrompt` with reasoning |
| 3.3 Image Generation Agent: ≥1 image, 1 mode, config logged | 6 | `agents/generation_agent.py` supports `stylize` / `variation` / `enhance` modes, logs `GenerationConfig` |
| 3.4 Critique Agent: relevance + faithfulness + quality, CLIP signal, human rubric over 3–5 examples, accept/revise | 6 | `agents/critique_agent.py` + `examples/human_eval_rubric.md` + `examples/human_eval_sheet.csv` |
| 3.5 Integration: modular + orchestration + sequence + 3 use cases + failure handling | 6 | `orchestration/pipeline.py` + `tests/test_pipeline.py` + `run_demo.py` with 3 scenarios + `utils/errors.py` |

If a reviewer scans this table against the code, every cell resolves to a file. That's the contract.

---

## 2. Architecture

```
User (image + instruction)
        │
        ▼
┌───────────────────────────────┐
│   Orchestrator (pipeline.py)  │
└───────────────────────────────┘
        │
        ▼
 [1] VisionAgent      ──► VisionOutput {caption, objects, scene, vqa_pairs}
        │
        ▼
 [2] PromptAgent      ──► RefinedPrompt {refined, enrichment_notes, preserved_transform}
        │
        ▼
 [3] GenerationAgent  ──► GenerationResult {image_path, config}
        │
        ▼
 [4] CritiqueAgent    ──► Critique {clip_sim, scores, verdict: accept|revise}
        │
        ▼
   Final report (JSON + markdown) in outputs/<run_id>/
```

Each agent is a class with a single `.run()` method and a typed dataclass output. This is what the rubric means by "modular design" — agents don't touch each other's internals, they pass structured messages through the orchestrator.

---

## 3. Model Choices (with pragmatic fallbacks)

The code is written so you can swap between **API-based** (fast to get working, no GPU) and **local models** (free, reproducible) by flipping a flag in `config.py`. Default is API-based because you want to submit this, not debug CUDA.

| Agent | Primary (API) | Fallback (local) | Stub (no creds, smoke test) |
|---|---|---|---|
| Vision | `claude-opus-4-7` via Anthropic API (multimodal) OR `gpt-4o` | BLIP-2 / LLaVA via `transformers` | Hardcoded caption for `sample.jpg` |
| Prompt | `claude-opus-4-7` text-only | Local Llama-3-8B-Instruct | Template-based rewrite |
| Generation | Stability AI API (`stable-diffusion-3`) OR Replicate SDXL | Local SDXL / SD 1.5 via `diffusers` | Copy input image + overlay text (proves pipeline works offline) |
| Critique | `open_clip` for CLIP similarity + Claude for reasoning | Same | Compute CLIP only, skip LLM reasoning |

**Why the stub mode matters:** your grader might not have API keys or a GPU. `python run_demo.py` runs in stub mode by default (`STUB_MODE=1`) and produces a valid end-to-end run with real file artifacts and a real critique. This is your insurance policy against a zero on "it didn't run on my machine." To force real-API mode: `STUB_MODE=0 python run_demo.py` (requires `ANTHROPIC_API_KEY` and optionally `STABILITY_API_KEY` / `REPLICATE_API_TOKEN`).

---

## 4. Data Contracts (shared types)

Defined once in `utils/schemas.py`:

```python
@dataclass
class VisionOutput:
    caption: str
    objects: list[str]
    scene: str
    vqa_pairs: list[dict]  # [{"question": str, "answer": str}, ...]  len >= 2
    raw_model_output: str  # for debugging

@dataclass
class RefinedPrompt:
    original_instruction: str
    refined_prompt: str
    enrichment_notes: str       # which visual details were pulled in
    preserved_transform: str    # what the user actually wanted to change

@dataclass
class GenerationConfig:
    model: str
    mode: str                   # "stylize" | "variation" | "enhance"
    prompt: str
    inference_steps: int
    guidance_scale: float
    seed: int | None

@dataclass
class GenerationResult:
    image_path: Path
    config: GenerationConfig

@dataclass
class Critique:
    clip_similarity_image: float   # input image <-> output image (CLIP)
    clip_similarity_text: float    # output image <-> refined prompt (CLIP)
    visual_relevance: float        # 0-1
    prompt_faithfulness: float     # 0-1
    quality: float                 # 0-1
    rationale: str
    verdict: str                   # "accept" | "revise"
    revision_suggestion: str | None
    used_fallback_metric: bool     # True if CLIP unavailable; SSIM used instead
```

Everything gets serialized to JSON at the end. No agent returns a dict-of-dicts — they all return these dataclasses. This prevents the "works on my notebook but dies in orchestration" problem.

---

## 5. Per-Agent Implementation Notes

### 5.1 VisionAgent (`agents/vision_agent.py`)

- Single multimodal call that requests all 4 outputs in one structured JSON response. Saves cost and keeps outputs consistent.
- Prompt explicitly asks for: 1-sentence caption, bulleted object list (max 10), 2-3 sentence scene description, and answers to both user-provided VQA questions AND one auto-generated one (so the ≥2 requirement holds even if the user supplies only one).
- Robust JSON parsing with a retry-on-parse-failure (one retry, then fall back to regex extraction).

### 5.2 PromptAgent (`agents/prompt_agent.py`)

- Takes `user_instruction` + `VisionOutput` + `mode` (stylize/variation/enhance).
- System prompt pattern: "You are a prompt engineer. Given the scene and the user's requested transformation, produce a detailed image-generation prompt that (a) keeps what the user wants preserved, (b) adds grounding details from the scene, (c) specifies the transformation clearly." The "preserve the transform" bit is easy to lose — we make it an explicit output field so the grader can see it.
- Outputs `RefinedPrompt` with reasoning, not just the string.

### 5.3 GenerationAgent (`agents/generation_agent.py`)

- Three modes:
  - **stylize**: img2img with style tokens added to prompt (e.g., "in the style of Studio Ghibli watercolor")
  - **variation**: img2img with moderate denoising strength (~0.6)
  - **enhance**: img2img with low denoising strength (~0.3) + "4k, sharp, detailed" modifiers
- Logs `GenerationConfig` as JSON next to the output PNG. Rubric says "record basic configuration details" — we record more than the minimum.
- Stub mode: Pillow-based transformation (grayscale for stylize, slight crop+blur for variation, sharpness filter for enhance) so the pipeline still produces a visibly different image without any model.

### 5.4 CritiqueAgent (`agents/critique_agent.py`)

- **Automatic signal**: `open_clip` ViT-B-32 embeddings. Compute:
  - `clip_sim(input_image, output_image)` → visual relevance floor
  - `clip_sim(output_image, refined_prompt_text)` → prompt faithfulness
- **LLM reasoning**: feed both images + prompt to the multimodal model, ask for scored assessment on quality + a 2-sentence critique + verdict.
- **Verdict rule**: `accept` if `clip_text_sim >= 0.25 AND llm_quality >= 0.6`, else `revise` with a concrete suggestion.
- If CLIP fails to load (no internet at grading time), fall back to a pixel-level SSIM + LLM-only scoring and flag this in the output.

### 5.5 Human Eval Rubric (`examples/human_eval_rubric.md` + `examples/human_eval_sheet.csv`)

Five examples, each scored 1–5 on:
1. Does the output reflect the input scene?
2. Does the output follow the instruction?
3. Is the transformation visually coherent?
4. Is the image quality acceptable?
5. Would you accept this or ask for a revision?

The CSV is pre-filled with the 5 runs from `run_demo.py` and has empty columns for the human scorer to fill in. This is what "simple human evaluation rubric across 3–5 examples" means — don't overthink it.

---

## 6. Orchestration (`orchestration/pipeline.py`)

One class, `MultimodalPipeline`, with:
- `__init__(config)` — instantiates all 4 agents
- `run(image_path, instruction, mode, vqa_questions=None) -> RunReport`
- Logs every intermediate output to `outputs/<run_id>/step_<n>_<agent>.json`
- On any agent failure: logs the error, produces a partial report, does not crash the process.

The sequence is hardcoded because the rubric specifies the order. There's no dynamic routing — that would be showing off and it's not asked for.

---

## 7. Failure Handling (`utils/errors.py`)

Explicit handling for the three failure modes in 3.5.5:

| Failure | Detection | Handling |
|---|---|---|
| Ambiguous prompt | `user_instruction` is <3 chars → `AmbiguousPromptError` raised in PromptAgent | Pipeline catches the error, logs it in `report.errors`, stops after step 2 (no generation/critique). Low `confidence` (<0.5) only appends a note to `enrichment_notes`; does not abort the pipeline. |
| Poor-quality image | Input image <256px on shortest side → `low_quality_input=True` + warning in `VisionOutput`; image not found / decode error → `LowQualityImageError` raised in `load_image` | Small image: VisionAgent runs and sets `low_quality_input=True`; pipeline continues. Not found / unreadable: `LowQualityImageError` caught by pipeline, logged in `report.errors`, vision is `None`, pipeline returns early. |
| Generation error | API timeout / NSFW block / OOM / unknown backend | Caught inside `GenerationAgent.run()`; returns `GenerationResult` with `image_path=None` and `error=<msg>`. Critique agent still runs and produces verdict=`revise` with the generation error as rationale. |

Every failure path has a test in `tests/test_pipeline.py`.

---

## 8. Three Test Use Cases (maps to rubric 3.5.4)

1. **Captioning + VQA** — input: `data/input_images/street_scene.jpg`, instruction: `"describe this image"`, mode: `none`. Exits after VisionAgent with a full `VisionOutput`.
2. **Style-guided transformation** — input: `data/input_images/portrait.jpg`, instruction: `"make this look like a watercolor painting"`, mode: `stylize`.
3. **Prompt-based enhancement** — input: `data/input_images/low_res_landscape.jpg`, instruction: `"sharpen and make it golden hour"`, mode: `enhance`.

Each produces: original image, refined prompt JSON, generated image, critique JSON, human-eval row. All under `outputs/<use_case_name>/`.

---

## 9. Folder Layout

```
multimodal_agents/
├── plan.md                        ← this file
├── README.md                      ← quickstart
├── requirements.txt
├── config.py                      ← model choices, API keys, stub flag
├── run_demo.py                    ← entry point, runs all 3 use cases
├── agents/
│   ├── __init__.py
│   ├── base.py                    ← BaseAgent abstract class
│   ├── vision_agent.py
│   ├── prompt_agent.py
│   ├── generation_agent.py
│   └── critique_agent.py
├── orchestration/
│   ├── __init__.py
│   └── pipeline.py
├── utils/
│   ├── __init__.py
│   ├── schemas.py                 ← dataclasses
│   ├── errors.py
│   ├── io_utils.py                ← image loading, JSON dump, run_id
│   └── clip_utils.py              ← CLIP similarity helpers
├── examples/
│   ├── human_eval_rubric.md
│   └── human_eval_sheet.csv
├── data/
│   └── input_images/              ← put test images here (or use stubs)
├── outputs/                       ← created at runtime
└── tests/
    ├── __init__.py
    └── test_pipeline.py
```

---

## 10. Execution Order in Cursor

Build in this order — each step produces something runnable:

1. `utils/schemas.py` + `utils/errors.py` + `utils/io_utils.py` (no dependencies, compile-only)
2. `config.py` with stub mode default
3. `agents/base.py` + `agents/vision_agent.py` with stub implementation → test it standalone
4. `agents/prompt_agent.py` with stub → test standalone
5. `agents/generation_agent.py` with stub (Pillow transforms) → test standalone
6. `utils/clip_utils.py` + `agents/critique_agent.py` → test standalone
7. `orchestration/pipeline.py` — glue it together
8. `run_demo.py` — three scenarios
9. `tests/test_pipeline.py` — failure cases
10. Swap stubs → real APIs one agent at a time (Vision → Prompt → Generation → Critique). Keep stub fallbacks.

Don't try to wire up Stable Diffusion on day 1. Get the stub end-to-end working first; it's worth full marks on 3.5 on its own.

---

## 11. Submission Checklist

- [ ] All 4 agents produce dataclass outputs (not raw dicts)
- [ ] Vision agent answers ≥2 VQA questions
- [ ] Prompt agent explicitly preserves the transformation (field in output)
- [ ] Generation agent logs model + prompt + steps
- [ ] Critique agent has CLIP signal + LLM rationale + accept/revise verdict
- [ ] `examples/human_eval_sheet.csv` filled for ≥3 runs
- [ ] `run_demo.py` executes all 3 use cases end-to-end in stub mode without errors
- [ ] `tests/test_pipeline.py` passes (incl. 3 failure cases)
- [ ] README has a "how to run" section with a 2-line command
- [ ] `outputs/` contains real artifacts from a successful run (commit one run's output)

When all boxes are checked, you've got every mark the rubric asks for.
