# Lab 2 — Part 3: Multimodal Multi-Agent System
## Image Understanding and Generation Pipeline

**Course**: Data 266 — Spring 2026  
**Part**: 3 of 3 (30 Marks)

---

## 1. Overview

This project implements a four-agent pipeline that takes an input image and a natural-language instruction, then produces a transformed image with a structured critique. The agents run in sequence, each passing a typed dataclass to the next:

```
Input image + instruction
        │
        ▼
 [1] VisionAgent        → caption, objects, scene, VQA answers
        │
        ▼
 [2] PromptAgent        → refined prompt, enrichment notes, preserved transform
        │
        ▼
 [3] GenerationAgent    → generated image + config record
        │
        ▼
 [4] CritiqueAgent      → CLIP scores, LLM scores, accept/revise verdict
```

**Models used**  
- Vision / Prompt / Critique: Claude Sonnet 4.5 (Anthropic API)  
- Generation: Stable Diffusion 1.5 via `diffusers` (local — no API key required)  
  - Auto-detects Apple Silicon MPS, NVIDIA CUDA, or CPU

---

## 2. System Architecture

### 2.1 Data contracts (`utils/schemas.py`)

Every agent returns a typed Python dataclass. This enforces modularity — each agent reads only what it needs and writes only its own output.

| Dataclass | Owner | Key fields |
|---|---|---|
| `VisionOutput` | VisionAgent | `caption`, `objects`, `scene`, `vqa_pairs`, `low_quality_input` |
| `RefinedPrompt` | PromptAgent | `refined_prompt`, `enrichment_notes`, `preserved_transform`, `confidence` |
| `GenerationResult` | GenerationAgent | `image_path`, `config` (model, steps, guidance, strength, seed), `error` |
| `Critique` | CritiqueAgent | `clip_similarity_image`, `clip_similarity_text`, `visual_relevance`, `prompt_faithfulness`, `quality`, `rationale`, `verdict` |
| `RunReport` | Pipeline | All of the above + `run_id`, `errors` |

### 2.2 Orchestration (`orchestration/pipeline.py`)

`MultimodalPipeline.run()` calls each agent in order, passes outputs downstream, and saves intermediate JSON logs (`step_1_vision.json` through `step_4_critique.json`) plus a complete `report.json` under `outputs/<run_id>/`.

### 2.3 Stub vs. real mode

`STUB_MODE=1` (default) runs deterministic heuristics so the entire pipeline can be tested locally without API keys or a GPU. `STUB_MODE=0` activates all real models. The `GenerationAgent` explicitly receives a `backend` argument from the pipeline, decoupled from the global flag to prevent test pollution.

---

## 3. Agent Implementation

### 3.1 Vision Understanding Agent (`agents/vision_agent.py`) — Rubric 3.1

Sends the input image (base64-encoded JPEG) to Claude Sonnet 4.5 with a structured JSON prompt requesting:

- A one-sentence **caption**
- A list of **key objects / entities**
- A short **scene description**
- Answers to **2–3 visual questions** (VQA)

The agent also detects low-quality or trivially small images and sets `low_quality_input=True` with a warning rather than aborting.

**Live output — UC1 (street scene):**
```
caption:  "A stylized nighttime city skyline with illuminated windows in
           buildings silhouetted against a dusky blue sky."
objects:  buildings, skyscrapers, windows, sky, city skyline, awning, ...
vqa:      3 pairs — palette, main subject, time of day
latency:  17 375 ms  |  712 input tokens, 338 output tokens
```

### 3.2 Prompt Engineering Agent (`agents/prompt_agent.py`) — Rubric 3.2

Receives the user's instruction and the `VisionOutput`. Uses Claude Sonnet 4.5 to produce a `RefinedPrompt` with three explicit fields matching the three rubric sub-items:

| Field | Maps to |
|---|---|
| `refined_prompt` | "rewrite the prompt clearly" |
| `enrichment_notes` | "enrich with useful visual details" |
| `preserved_transform` | "preserve the intended transformation" |

**Live output — UC2 (watercolor stylization):**
```
input:    "make this look like a watercolor painting with soft colors"  (9 words)
refined:  "A watercolor painting of a smiling person with soft, flowing brown
           hair and gentle blue eyes, wearing a pale blue shirt against a warm
           beige background. Delicate watercolor washes create the round face
           in peachy tones..."  (97 words)
latency:  5 620 ms  |  401 input tokens, 210 output tokens
```

### 3.3 Image Generation Agent (`agents/generation_agent.py`) — Rubric 3.3

Runs `StableDiffusionImg2ImgPipeline` from the `diffusers` library (SD 1.5, `runwayml/stable-diffusion-v1-5`) locally. Supports three modes with different strength values:

| Mode | Strength | Use |
|---|---|---|
| `stylize` | 0.75 | Large style change (e.g., watercolor) |
| `enhance` | 0.30 | Subtle enhancement (e.g., golden-hour grading) |
| `variation` | 0.55 | Moderate variation |

The pipeline is cached at the class level — the model loads once (~10 s on MPS) and all subsequent calls reuse it with no reload overhead. Every run records `model`, `mode`, `prompt`, `inference_steps`, `guidance_scale`, `seed`, and `strength` in `step_3_generation.json`.

**Live output — UC2:**
```
backend: diffusers  |  device: mps (Apple Silicon)
model:   runwayml/stable-diffusion-v1-5
steps:   30  |  guidance: 7.5  |  strength: 0.75  |  seed: 42
output:  512×512 px, 234 KB  |  latency: 55 897 ms (incl. model load)
```

### 3.4 Critique and Evaluation Agent (`agents/critique_agent.py`) — Rubric 3.4

Two-stage evaluation:

**Stage 1 — Automatic (CLIP)**  
Uses `open_clip` (ViT-B-32) to compute:
- `clip_similarity_image` — input image vs. generated image cosine similarity
- `clip_similarity_text` — generated image vs. refined prompt cosine similarity

**Stage 2 — LLM scoring**  
Sends both images (base64) and the refined prompt to Claude Sonnet 4.5, which returns structured JSON with:
- `visual_relevance` — does output look like the input subject?
- `prompt_faithfulness` — does the style/instruction transfer?
- `quality` — technical output quality
- `rationale` — natural language critique
- `verdict` — `accept` or `revise`; if `revise`, a concrete `revision_suggestion`

**Live metrics:**

| Use Case | CLIP img↔img | CLIP txt↔img | Visual Rel. | Prompt Faith. | Quality | Verdict |
|---|---|---|---|---|---|---|
| UC2 (stylize) | 0.846 | 0.298 | 0.60 | 0.50 | 0.40 | **REVISE** |
| UC3 (enhance) | 0.884 | 0.305 | 0.85 | 0.65 | 0.70 | **ACCEPT** |

The UC2 `REVISE` verdict is a correct call — the SD 1.5 watercolor output had opaque brushstrokes and a question-mark artifact. This demonstrates the critique agent functioning as intended, not a system failure.

---

## 4. Integration and Execution — Rubric 3.5

### 4.1 Modular design

Each agent is a standalone Python class in `agents/`, extending `BaseAgent` with a single `.run(**kwargs) → dataclass` interface. Agents can be imported and called independently of the pipeline.

### 4.2 Workflow and intermediate outputs

For every run, the pipeline saves four intermediate JSON files mirroring each agent's output:

```
outputs/uc2_style_transfer/
├── input_portrait.jpg          ← input copy
├── step_1_vision.json          ← VisionOutput
├── step_2_prompt.json          ← RefinedPrompt
├── step_3_generation.json      ← GenerationConfig + path
├── step_4_critique.json        ← Critique
├── report.json                 ← full RunReport (all agents)
└── generated.png               ← output image
```

### 4.3 Three required use cases

| # | Use Case | Image | Instruction | Mode |
|---|---|---|---|---|
| UC1 | Image captioning and VQA | Urban street scene (768×512) | "describe this image" | `none` |
| UC2 | Style-guided transformation | Portrait illustration (512×640) | "make this look like a watercolor painting with soft colors" | `stylize` |
| UC3 | Prompt-based enhancement | Landscape illustration (768×512) | "sharpen the image and warm it up to look like golden hour" | `enhance` |

All three ran end-to-end in real mode (Claude Sonnet 4.5 + SD 1.5 locally).

### 4.4 Failure handling

| Failure type | How it is handled |
|---|---|
| Ambiguous / too-short instruction | `AmbiguousPromptError` raised, pipeline returns partial report with `errors` list |
| Missing or corrupt input image | `LowQualityImageError` raised pre-run |
| Low-confidence analysis | `low_quality_input=True` flag set, pipeline continues with warning |
| Generation API / backend error | `GenerationResult.error` is populated, critique skipped, pipeline returns partial report |

In all cases the pipeline returns a `RunReport` (never crashes) and all errors are captured in `report.json["errors"]`.

---

## 5. Human Evaluation

Human evaluation was performed on 5 pipeline runs using a 5-criterion rubric (`examples/human_eval_rubric.md`):

| Criterion | Description |
|---|---|
| C1 Input fidelity | Does the output reflect the input content? |
| C2 Instruction following | Does the output match the user instruction? |
| C3 Transform coherence | Is the transformation visually coherent? |
| C4 Technical quality | Is the image technically acceptable? |
| C5 Accept decision | 1 = reject, 5 = accept as-is |

Accept rule: avg(C1–C5) ≥ 3.5 **and** C5 ≥ 3.

**Scored runs (from `examples/human_eval_sheet.csv`):**

| Run | avg | Human | Agent | Agree? |
|---|---|---|---|---|
| UC1 — captioning + VQA | 5.0 | accept | N/A | yes |
| UC2 — watercolor stylization | 3.2 | revise | revise | **yes** |
| UC3 — golden-hour enhancement | 3.8 | accept | accept | **yes** |
| Ambiguous prompt failure | 1.0 | revise | revise | yes |
| Backend failure | 1.0 | revise | revise | yes |

Human and agent verdicts agree on all 5 runs (100% agreement).

---

## 6. Testing

```
tests/
├── test_pipeline.py    ← stub-mode: 12 unit + integration tests
└── test_real_mode.py   ← real-API: 54 tests across all 4 agents + full pipeline
```

**Stub tests** cover: each agent independently, all 3 use cases end-to-end, all 3 failure modes, JSON serialization.  
**Real-mode tests** cover: real LLM output quality (caption length, object detection, rationale presence), CLIP score ranges, generation output existence, and error handling with live API keys.

Run all tests:
```bash
pytest tests/ -v                        # stub only (no keys needed)
STUB_MODE=0 pytest tests/test_real_mode.py -v   # real mode
```

---

## 7. Key Design Decisions

**Why local diffusers instead of Stability AI?**  
SD 1.5 via `diffusers` runs fully locally with no API key, no billing, and no quota limits. It auto-detects Apple Silicon (MPS), NVIDIA (CUDA), or CPU, making it reproducible on any machine.

**Why separate `CLIP img↔img` and `CLIP txt↔img` scores?**  
Image-to-image similarity measures content preservation; text-to-image similarity measures instruction fidelity. Both are needed to distinguish "looks like the input but ignores the instruction" from "follows the instruction but loses the subject."

**Why two verdict paths in the critique agent?**  
The LLM-based critique provides nuanced rationale and a revision suggestion. The CLIP scores serve as an objective, model-agnostic sanity check. If the LLM call fails, CLIP scores alone determine the verdict as a fallback (`used_fallback_metric=True`).

---

## 8. Conclusion

All five rubric items for Part 3 are satisfied:

- **3.1** — Vision agent produces caption, objects, scene description, and ≥2 VQA pairs using Claude Sonnet 4.5.  
- **3.2** — Prompt agent rewrites, enriches, and preserves intent with explicit schema fields for each sub-requirement.  
- **3.3** — Generation agent produces output images with full config records; three modes supported; runs entirely locally.  
- **3.4** — Critique agent computes CLIP similarity (automatic signal) and LLM-based scores; human eval rubric completed on 5 runs with 100% human/agent agreement.  
- **3.5** — Modular design, sequential orchestration with intermediate JSON logs, three required use cases validated, and graceful failure handling for all three failure types.
