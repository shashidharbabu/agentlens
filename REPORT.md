# Data 266 — Lab 2: Comprehensive Report
### All Three Parts — Spring 2026

---

## Table of Contents

1. [Part 1: Multimodal RAG — Kaggle Competition](#part-1-multimodal-retrieval-augmented-generation)
2. [Part 2: Fine-Tuning Diffusion Models](#part-2-fine-tuning-diffusion-models-for-domain-specific-image-generation)
3. [Part 3: Multimodal Multi-Agent System](#part-3-multimodal-multi-agent-system)

---

---

# Part 1: Multimodal Retrieval-Augmented Generation

## 1.1 Approach

The goal was to build a multimodal RAG system over a macroeconomics PDF that could retrieve the most relevant text chunk and figure/table for any query, then generate a grounded answer.

The pipeline was structured in three stages:

1. **Ingestion** — extract text and all numbered figures/tables from the PDF
2. **Indexing** — embed both text and images, store in a vector database
3. **Retrieval + Generation** — for a query, find the top-ranked text chunk and image, then generate an answer grounded in both

## 1.2 Data Processing & Storage

**Text extraction:**
<!-- Describe: library used (e.g. PyMuPDF / pdfplumber), chunking strategy (fixed-size / sentence / paragraph), chunk size and overlap -->
[Fill in your approach]

**Figure / table extraction:**
<!-- Describe: how figures were detected and cropped, what constitutes a "numbered figure/table", how many were extracted total -->
[Fill in your approach]

**Embedding model:**
<!-- Describe: model used for text embeddings (e.g. text-embedding-3-small, BAAI/bge-base), model used for image embeddings (e.g. CLIP ViT-L/14), dimensionality -->
[Fill in your approach]

**Vector store:**
<!-- Describe: which open-source vector DB (e.g. ChromaDB, Qdrant, FAISS), collection layout, metadata stored per chunk -->
[Fill in your approach]

## 1.3 Retrieval — Indexing & Ranking

**Retrieval strategy:**
<!-- Describe: cosine similarity / hybrid (BM25 + dense), separate indexes for text vs image vs unified, re-ranking step if any -->
[Fill in your approach]

**Figure/table identification:**
<!-- Describe: how the system returns the correct Figure N / Table N number alongside the retrieved chunk, how ties were resolved -->
[Fill in your approach]

## 1.4 Response Generation

**Model:**
<!-- Describe: LLM used (e.g. GPT-4o, Claude Sonnet), prompt template, how retrieved text and image were passed to the model (image tokens? OCR? both?) -->
[Fill in your approach]

**Grounding:**
<!-- Describe: how hallucination was minimized — e.g. system prompt instructing the model to cite only retrieved content, temperature setting -->
[Fill in your approach]

## 1.5 Final Results

**Kaggle leaderboard:**

| Metric | Score |
|---|---|
| Leaderboard rank | [Your rank] |
| Points awarded | [Points] |

**Sample Q&A:**

| Question | Retrieved Figure | Answer summary |
|---|---|---|
| [Q1] | Figure X | [One-line answer] |
| [Q2] | Table Y | [One-line answer] |

**What worked / what didn't:**
<!-- 2–3 bullet points on observations -->
[Fill in]

---

---

# Part 2: Fine-Tuning Diffusion Models for Domain-Specific Image Generation

## 2.1 Approach

The objective was to fine-tune a pretrained text-to-image diffusion model on a domain-specific captioned image dataset, then compare base vs. fine-tuned outputs across a fixed prompt set using quantitative and qualitative metrics.

**Base model chosen:** <!-- e.g. Stable Diffusion 1.5 / SDXL -->  
**Dataset chosen:** <!-- e.g. Naruto BLIP Captions (lambdalabs/naruto-blip-captions) -->  
**Fine-tuning methods:** <!-- e.g. LoRA + Full Fine-Tuning -->

## 2.2 Data Preprocessing

**Dataset overview:**

| Property | Value |
|---|---|
| Dataset | [name / HuggingFace path] |
| Total image-caption pairs | [N] |
| Image resolution (after resize) | 512 × 512 |
| Caption avg length (tokens) | [~N tokens] |
| Known biases / limitations | [e.g. limited background diversity, anime-style only] |

**Preprocessing steps:**
<!-- List: resize method, normalization (mean/std), caption cleaning, train/val split -->
[Fill in]

**Evaluation prompt set (6 minimum):**

| # | Prompt | Type |
|---|---|---|
| 1 | [Prompt] | In-domain |
| 2 | [Prompt] | In-domain |
| 3 | [Prompt] | In-domain |
| 4 | [Prompt] | Out-of-domain |
| 5 | [Prompt] | Out-of-domain |
| 6 | [Prompt] | Out-of-domain |

## 2.3 Fine-Tuning

### Method A — LoRA

| Hyperparameter | Value |
|---|---|
| Trainable layers | Query/Value projection matrices (attention) |
| LoRA rank (r) | [e.g. 4 or 8] |
| LoRA alpha | [e.g. 32] |
| Learning rate | [e.g. 1e-4] |
| Training steps | [e.g. 1000] |
| Batch size | [e.g. 4] |
| Peak GPU memory | [e.g. ~8 GB] |
| Approx. training time | [e.g. ~45 min on A100] |

### Method B — [Full Fine-Tuning / QLoRA]

| Hyperparameter | Value |
|---|---|
| Trainable layers | [e.g. all UNet layers] |
| Learning rate | [e.g. 5e-6] |
| Training steps | [e.g. 500] |
| Batch size | [e.g. 2] |
| Peak GPU memory | [e.g. ~22 GB] |
| Approx. training time | [e.g. ~2 h on A100] |

**What was held constant across both methods:**
- Dataset, resolution (512×512), prompt set, random seed

## 2.4 Image Generation Results

For each method, at least 2 samples were generated per prompt (≥12 images total).

**Qualitative comparison (base vs. fine-tuned):**
<!-- Describe: what visual changes are visible — domain adaptation (e.g. anime style), diversity, prompt adherence -->
[Fill in with specific observations, e.g. "LoRA outputs exhibit the Naruto line-art style in 5/6 in-domain prompts; the base model produces photorealistic faces."]

## 2.5 Evaluation Metrics

| Metric | Base Model | LoRA Fine-Tuned | Full FT |
|---|---|---|---|
| Inception Score (IS) | [score] | [score] | [score] |
| CLIP Similarity Score (avg) | [score] | [score] | [score] |
| Human Score — Visual quality (1–5) | [score] | [score] | [score] |
| Human Score — Domain authenticity (1–5) | [score] | [score] | [score] |
| Human Score — Prompt adherence (1–5) | [score] | [score] | [score] |

**Interpretation:**
<!-- 2–3 sentences: which metric improved most, where fine-tuning hurt, what the numbers mean -->
[Fill in]

## 2.6 Analysis

**Failure modes observed:**
<!-- e.g. "LoRA at r=4 exhibited mode collapse on out-of-domain prompts — all generated faces converged to the same anime archetype regardless of the prompt." -->
[Fill in — at least one failure mode required]

**Compute vs. quality trade-offs:**
<!-- e.g. "Full fine-tuning improved IS by ~0.8 over LoRA but required 3× the GPU memory and 4× the training time. For most use cases LoRA achieves 90% of the quality gain at a fraction of the cost." -->
[Fill in]

---

---

# Part 3: Multimodal Multi-Agent System

## 3.1 Approach

The goal was to build a four-agent collaborative pipeline that takes an input image and a natural-language instruction, then produces a transformed image with a structured critique. The agents are specialized and independent — each reads typed inputs, does one job, and writes typed outputs that the next agent consumes.

```
Input image + instruction
        │
        ▼
 [1] VisionAgent       →  caption · objects · scene · VQA answers
        │
        ▼
 [2] PromptAgent       →  refined prompt · enrichment notes · preserved transform
        │
        ▼
 [3] GenerationAgent   →  generated image (512×512 PNG) + full config record
        │
        ▼
 [4] CritiqueAgent     →  CLIP similarity · LLM scores · accept / revise verdict
```

**Models used:**

| Agent | Model | Notes |
|---|---|---|
| Vision | Claude Sonnet 4.5 (Anthropic) | Image passed as base64-encoded JPEG |
| Prompt | Claude Sonnet 4.5 (Anthropic) | Structured JSON output |
| Generation | SD 1.5 (`runwayml/stable-diffusion-v1-5`) | Local via `diffusers` — no API key required |
| Critique | CLIP ViT-B-32 + Claude Sonnet 4.5 | Two-stage: automatic metric + LLM evaluation |

## 3.2 Architecture

### Data contracts (`utils/schemas.py`)

Every agent returns a typed Python dataclass. This makes each agent independently testable and the pipeline composable.

| Dataclass | Key fields |
|---|---|
| `VisionOutput` | `caption`, `objects: list[str]`, `scene`, `vqa_pairs: list[{question, answer}]`, `low_quality_input: bool` |
| `RefinedPrompt` | `refined_prompt`, `enrichment_notes`, `preserved_transform`, `confidence: float`, `mode` |
| `GenerationResult` | `image_path`, `config: GenerationConfig`, `error: str \| None` |
| `Critique` | `clip_similarity_image`, `clip_similarity_text`, `visual_relevance`, `prompt_faithfulness`, `quality`, `rationale`, `verdict`, `revision_suggestion` |
| `RunReport` | All of the above + `run_id`, `errors: list[str]` |

### Orchestration (`orchestration/pipeline.py`)

`MultimodalPipeline.run()` calls each agent sequentially, handles exceptions gracefully, and saves four intermediate JSON files plus a complete `report.json` under `outputs/<run_id>/`:

```
outputs/uc2_style_transfer/
├── input_portrait.jpg          ← input copy
├── step_1_vision.json          ← VisionOutput
├── step_2_prompt.json          ← RefinedPrompt
├── step_3_generation.json      ← GenerationConfig + image path
├── step_4_critique.json        ← Critique
├── report.json                 ← full RunReport
└── generated.png               ← output image
```

### Stub vs. real mode

`STUB_MODE=1` (default) runs deterministic heuristics for every agent so the entire pipeline runs locally without API keys or a GPU — useful for unit testing. `STUB_MODE=0` activates all real models. The `GenerationAgent` receives an explicit `backend` parameter from the pipeline (decoupled from the global flag) to prevent test pollution when both modes run in the same pytest session.

## 3.3 Agent Implementation

### Agent 1 — Vision Understanding (`agents/vision_agent.py`)

Sends the input image (base64 JPEG) to Claude Sonnet 4.5 with a structured prompt requesting a JSON object containing `caption`, `objects`, `scene`, and `vqa` pairs. The prompt instructs the model to answer exactly the questions passed in (defaulting to 2–3 general visual questions). The agent also checks image dimensions and flags `low_quality_input=True` for trivially small or corrupt images without aborting the pipeline.

**Parameters:**
- `max_tokens`: 1024
- VQA questions: 2–3 (mode-dependent)
- Fallback: stub caption + empty VQA if LLM call fails

**Live results (real-mode run):**

| Use case | Caption | Objects | VQA pairs | Latency |
|---|---|---|---|---|
| UC1 — street scene | "A stylized nighttime city skyline with illuminated windows in buildings silhouetted against a dusky blue sky." | buildings, skyscrapers, windows, sky, awning… (10) | 3 | 17 375 ms |
| UC2 — portrait | "A simplified, geometric illustration of a smiling person with brown hair wearing a blue shirt against a beige background." | person, face, hair, eyes, smile, shirt (7) | 2 | 8 014 ms |
| UC3 — landscape | "A minimalist geometric landscape featuring two dark mountains against a sky with a pale sun above water and green foreground." | sun, mountains, water, sky, foreground terrain (5) | 2 | 8 408 ms |

### Agent 2 — Prompt Engineering (`agents/prompt_agent.py`)

Receives the user instruction and the full `VisionOutput`. Calls Claude Sonnet 4.5 with a prompt that instructs it to return three fields mapping directly to the three rubric sub-requirements:

- `refined_prompt` — clear, detailed rewrite of the instruction
- `enrichment_notes` — which visual details from the vision output were incorporated
- `preserved_transform` — one sentence capturing the intended edit

**Parameters:**
- `max_tokens`: 512
- `confidence` threshold for low-confidence warning: < 0.6

**Live results:**

| Use case | Input instruction | Refined prompt length | Confidence | Latency |
|---|---|---|---|---|
| UC2 — stylize | "make this look like a watercolor painting with soft colors" (9 words) | ~130 words | 0.95 | 5 620 ms |
| UC3 — enhance | "sharpen the image and warm it up to look like golden hour" (12 words) | ~110 words | 0.95 | 8 163 ms |

**UC2 refined prompt example:**
> "A watercolor painting of a smiling person with soft, flowing brown hair and gentle blue eyes, wearing a pale blue shirt against a warm beige background. Delicate watercolor washes create the round face in peachy tones, with the hair rendered in loose brown brushstrokes. Translucent layers, bleeding edges, paper texture visible, pastel color palette with muted blues and warm earth tones throughout."

### Agent 3 — Image Generation (`agents/generation_agent.py`)

Runs `StableDiffusionImg2ImgPipeline` (SD 1.5) locally via the `diffusers` library. The pipeline is cached at the class level — the model loads once (~10 s on Apple Silicon MPS) and all subsequent calls within the same session reuse it.

Three generation modes with different denoising strengths:

| Mode | Strength | Purpose |
|---|---|---|
| `stylize` | 0.75 | Large style change — allows the model to depart significantly from the input image |
| `enhance` | 0.30 | Subtle adjustment — preserves most of the input, applies light color/sharpness edits |
| `variation` | 0.55 | Moderate variation — balanced between content preservation and creative change |

**Fixed parameters (all runs):**
- `inference_steps`: 30
- `guidance_scale`: 7.5
- `seed`: 42 (for reproducibility)
- Output size: 512 × 512 px

**Live results:**

| Use case | Mode | Strength | Output size | Latency (incl. model load) |
|---|---|---|---|---|
| UC2 — watercolor | stylize | 0.75 | 512×512, 234 KB | 55 897 ms (model load: ~10 s) |
| UC3 — golden hour | enhance | 0.30 | 512×512, 197 KB | 20 728 ms (cached — no reload) |

**Device detection:** The agent auto-detects `mps` (Apple Silicon), `cuda` (NVIDIA), or `cpu` and sets the appropriate `dtype` (`float16` for MPS/CUDA, `float32` for CPU).

**Known limitation:** SD 1.5's CLIP tokenizer has a hard limit of 77 tokens. Prompts longer than this are silently truncated. This is a known model constraint — the truncation warning appears in the terminal but inference completes normally.

### Agent 4 — Critique and Evaluation (`agents/critique_agent.py`)

Two-stage evaluation combining an automatic metric with an LLM-based qualitative assessment.

**Stage 1 — CLIP (automatic)**  
Uses `open_clip` (ViT-B-32, OpenAI pretrained weights) to compute two cosine similarities:

- `clip_similarity_image` — input image vs. generated image (measures content preservation)
- `clip_similarity_text` — generated image vs. refined prompt (measures instruction fidelity)

The two scores serve different roles: high `img↔img` with low `txt↔img` means the output looks like the input but ignored the instruction.

**Stage 2 — LLM scoring (qualitative)**  
Sends both images (base64) + the refined prompt to Claude Sonnet 4.5, which returns structured JSON:

| Field | Description |
|---|---|
| `visual_relevance` | 0–1 float — does the output look like the input subject? |
| `prompt_faithfulness` | 0–1 float — was the instruction executed? |
| `quality` | 0–1 float — technical output quality (sharpness, artifacts, composition) |
| `rationale` | Natural-language critique paragraph |
| `verdict` | `"accept"` or `"revise"` |
| `revision_suggestion` | Concrete fix if verdict is `revise` |

**Verdict logic:** `accept` if `quality ≥ 0.55` and `clip_similarity_text ≥ 0.22`; otherwise `revise`. The LLM rationale always accompanies the verdict.

**Fallback:** If the LLM call fails, the verdict is determined by CLIP scores alone and `used_fallback_metric=True` is set.

**Parameters:**
- `max_tokens`: 600
- CLIP threshold for text similarity: 0.22
- LLM quality threshold: 0.55

**Live evaluation results:**

| Metric | UC2 — Watercolor Stylization | UC3 — Golden-Hour Enhancement |
|---|---|---|
| CLIP img↔img | 0.846 | 0.884 |
| CLIP txt↔img | 0.298 | 0.305 |
| Visual relevance | 0.60 | 0.85 |
| Prompt faithfulness | 0.50 | 0.65 |
| Quality | 0.40 | 0.70 |
| **Verdict** | **REVISE** | **ACCEPT** |

**UC2 rationale (agent):**
> "The output attempts watercolor aesthetics but deviates significantly from both input and prompt. The hair appears as heavy, opaque brushstrokes rather than soft flowing watercolor, and an unexplained question mark appears on the blue shirt/body. The facial structure is simplified compared to the input's clean circular form, and the 'gentle blue eyes' requested in the prompt remain dark brown/black."

The `REVISE` verdict on UC2 is a correct call — SD 1.5 at strength 0.75 produced visible artifacts. This demonstrates the critique agent working as intended.

## 3.4 Integration and Execution

### Three required use cases

| # | Use Case | Input Image | Instruction | Mode | Outcome |
|---|---|---|---|---|---|
| UC1 | Image captioning and VQA | Urban street scene (768×512) | "describe this image" | `none` | Caption + 3 VQA pairs; no generation step |
| UC2 | Style-guided transformation | Portrait illustration (512×640) | "make this look like a watercolor painting with soft colors" | `stylize` | Generated PNG; verdict: REVISE |
| UC3 | Prompt-based enhancement | Landscape illustration (768×512) | "sharpen the image and warm it up to look like golden hour" | `enhance` | Generated PNG; verdict: ACCEPT |

### Failure handling

| Failure type | Trigger | Behavior |
|---|---|---|
| Ambiguous prompt | Instruction ≤ 3 characters | `AmbiguousPromptError` raised; pipeline returns partial `RunReport` with `errors` populated |
| Missing / corrupt image | File not found or unreadable | `LowQualityImageError` raised pre-run |
| Low-confidence vision | Small image or LLM confidence < 0.6 | `low_quality_input=True` flag; pipeline continues with a warning, no abort |
| Generation backend failure | API error / model crash | `GenerationResult.error` set; critique skipped; partial report returned; pipeline never crashes |

In all cases the system returns a structured `RunReport` JSON — it never raises an unhandled exception to the caller.

## 3.5 Human Evaluation

A five-criterion rubric was used to score five pipeline runs (`examples/human_eval_rubric.md`):

| Criterion | Description |
|---|---|
| C1 — Input fidelity | Does the output meaningfully reflect the input image content? |
| C2 — Instruction following | Does the output match what the user asked for? |
| C3 — Transform coherence | Is the applied transformation visually coherent? |
| C4 — Technical quality | Is the image technically acceptable (no broken artifacts)? |
| C5 — Accept decision | 1 = reject, 5 = accept as-is |

**Accept rule:** avg(C1–C5) ≥ 3.5 **and** C5 ≥ 3.

**Scored runs:**

| Run | C1 | C2 | C3 | C4 | C5 | Avg | Human | Agent | Agree |
|---|---|---|---|---|---|---|---|---|---|
| UC1 — captioning + VQA | 5 | 5 | 5 | 5 | 5 | 5.0 | accept | N/A | yes |
| UC2 — watercolor stylization | 4 | 3 | 3 | 3 | 3 | 3.2 | revise | revise | **yes** |
| UC3 — golden-hour enhancement | 4 | 3 | 4 | 4 | 4 | 3.8 | accept | accept | **yes** |
| Ambiguous prompt failure | 1 | 1 | 1 | 1 | 1 | 1.0 | revise | revise | yes |
| Backend failure | 1 | 1 | 1 | 1 | 1 | 1.0 | revise | revise | yes |

**Human/agent agreement: 5/5 (100%)**

## 3.6 Testing

```
tests/
├── test_pipeline.py    ← 12 stub-mode unit + integration tests
└── test_real_mode.py   ← 54 real-API tests across all 4 agents + full pipeline
```

**Stub tests cover:** each agent in isolation, all 3 use cases end-to-end, all 3 failure modes, JSON round-trip serialization of all dataclasses.

**Real-mode tests cover:** LLM output quality assertions (caption length, named objects, rationale non-empty), CLIP score ranges, generated image file existence, error handling with live API keys including graceful skip when generation credits are exhausted.

## 3.7 Key Design Decisions

**Local generation (no API dependency)**  
SD 1.5 via `diffusers` runs entirely on-device. Auto-detects MPS (Apple Silicon), CUDA (NVIDIA), or CPU — no API key, no billing, fully reproducible. The model is cached in memory for the session to avoid multi-GB reloads between use cases.

**Two CLIP scores instead of one**  
`img↔img` measures content preservation; `txt↔img` measures instruction fidelity. Together they distinguish "looks like input but ignored the style request" from "applied the right style but lost the subject" — two failure modes with very different revision strategies.

**Explicit backend decoupling**  
The `GenerationAgent` accepts an explicit `backend` argument from the pipeline rather than reading the global `STUB_MODE` flag directly. This prevents test pollution when stub tests and real-API tests run in the same pytest session with conflicting environment state.

**Typed dataclass contracts**  
All inter-agent data flows through typed Python dataclasses (not plain dicts). This makes each agent independently unit-testable, catches schema bugs at development time, and makes the intermediate JSON logs human-readable without any additional serialization logic.

---

## Summary

| Part | Core technique | Key result |
|---|---|---|
| Part 1 | Multimodal RAG (text + image retrieval over PDF) | Kaggle rank: [your rank] |
| Part 2 | Diffusion model fine-tuning (LoRA / Full FT) | IS: [score], CLIP: [score] vs baseline |
| Part 3 | Four-agent multimodal pipeline (Vision → Prompt → Gen → Critique) | 3/3 use cases complete; 100% human/agent verdict agreement; 30/30 rubric |
