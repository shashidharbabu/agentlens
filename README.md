# Multimodal Multi-Agent System

Four specialized agents that collaborate to analyze an image, refine a prompt, generate a new image, and evaluate the result.

```
Input image + instruction
        │
        ▼
 [1] VisionAgent      →  caption · objects · scene · VQA
        │
        ▼
 [2] PromptAgent      →  refined prompt · enrichment · preserved transform
        │
        ▼
 [3] GenerationAgent  →  generated image (SD 1.5, local — no API key)
        │
        ▼
 [4] CritiqueAgent    →  CLIP scores · LLM scores · accept / revise verdict
```

> **Lab 2 — Part 3 · Data 266 Spring 2026**  
> Full rubric evaluation: [`REPORT.md`](./REPORT.md)

---

## Quickstart (stub mode — no keys, no GPU)

```bash
pip install -r requirements.txt
python run_demo.py
```

Runs all three required use cases end-to-end. Artifacts land in `outputs/<use_case>/`:

```
outputs/uc2_style_transfer/
├── input_portrait.jpg
├── generated.png
├── step_1_vision.json     ← VisionOutput
├── step_2_prompt.json     ← RefinedPrompt
├── step_3_generation.json ← GenerationConfig
├── step_4_critique.json   ← Critique
└── report.json            ← full RunReport
```

---

## Real mode

Copy `.env.example` to `.env`, fill in your key, then:

```bash
cp .env.example .env
# edit .env — add ANTHROPIC_API_KEY at minimum
source .env
python run_demo.py
```

### Generation backends

| Backend | Env var | Notes |
|---|---|---|
| **diffusers** *(default)* | `GENERATION_BACKEND=diffusers` | Local SD 1.5 — no API key. Auto-detects MPS / CUDA / CPU. ~4 GB download on first run. |
| Stability AI | `GENERATION_BACKEND=stability` + `STABILITY_API_KEY=...` | SD3 via API |
| Replicate | `GENERATION_BACKEND=replicate` + `REPLICATE_API_TOKEN=...` | SDXL via API |

---

## Run a single use case

```bash
python run_demo.py --use-case 2   # style transfer only
```

## Use your own image

```python
from orchestration import MultimodalPipeline

pipeline = MultimodalPipeline()
report = pipeline.run(
    image_path="data/input_images/my_photo.jpg",
    instruction="turn this into a watercolor painting",
    mode="stylize",          # stylize | enhance | variation
)
print(report.critique.verdict)   # "accept" or "revise"
```

---

## Tests

```bash
pytest tests/ -v                                      # stub — no keys needed
STUB_MODE=0 pytest tests/test_real_mode.py -v         # real API
```

---

## Project layout

```
multimodal_agents/
├── REPORT.md                  ← full Part 3 lab report
├── plan.md                    ← rubric → code mapping
├── run_demo.py                ← entry point (3 use cases)
├── config.py                  ← mode switches, model choices
├── agents/
│   ├── vision_agent.py        ← rubric 3.1
│   ├── prompt_agent.py        ← rubric 3.2
│   ├── generation_agent.py    ← rubric 3.3
│   └── critique_agent.py      ← rubric 3.4
├── orchestration/pipeline.py  ← rubric 3.5 (orchestration)
├── utils/
│   ├── schemas.py             ← typed dataclass contracts
│   ├── errors.py              ← AmbiguousPromptError, LowQualityImageError
│   ├── clip_utils.py          ← CLIP similarity (ViT-B-32)
│   ├── io_utils.py            ← image load / save helpers
│   └── logger.py              ← color-coded terminal logging
├── tests/
│   ├── test_pipeline.py       ← stub-mode unit + integration tests
│   └── test_real_mode.py      ← real-API eval suite
├── examples/
│   ├── human_eval_rubric.md   ← scoring criteria
│   └── human_eval_sheet.csv   ← 5 scored runs
├── data/input_images/         ← sample images (auto-generated)
├── outputs/                   ← run artifacts (gitignored)
├── .env.example               ← env var template
└── requirements.txt
```

---

## Rubric coverage at a glance

| Section | Implementation |
|---|---|
| 3.1 Vision Understanding | `agents/vision_agent.py` — Claude Sonnet 4.5, 3 VQA pairs |
| 3.2 Prompt Engineering | `agents/prompt_agent.py` — refined prompt + enrichment + preserved transform |
| 3.3 Image Generation | `agents/generation_agent.py` — SD 1.5 local, 3 modes, full config record |
| 3.4 Critique & Eval | `agents/critique_agent.py` — CLIP + LLM scores + human eval CSV |
| 3.5 Integration | `orchestration/pipeline.py` — sequential chain, intermediate JSON, 3 use cases, failure handling |
