"""Project configuration.

STUB_MODE=True lets the whole pipeline run without any API keys or GPU.
This is the default so `python run_demo.py` works out of the box on a
fresh clone. Set STUB_MODE=False (or export ANTHROPIC_API_KEY etc.) to use
real models.
"""
from __future__ import annotations

import os
from pathlib import Path


# --- paths ----------------------------------------------------------
PROJECT_ROOT = Path(__file__).parent
OUTPUTS_ROOT = PROJECT_ROOT / "outputs"
DATA_ROOT = PROJECT_ROOT / "data"
INPUT_IMAGES_DIR = DATA_ROOT / "input_images"

OUTPUTS_ROOT.mkdir(exist_ok=True)
INPUT_IMAGES_DIR.mkdir(parents=True, exist_ok=True)


# --- mode -----------------------------------------------------------
STUB_MODE = os.environ.get("STUB_MODE", "1") == "1"


# --- model selection ------------------------------------------------
# Vision + Prompt + Critique LLM
ANTHROPIC_API_KEY = os.environ.get("ANTHROPIC_API_KEY", "")
ANTHROPIC_MODEL = os.environ.get("ANTHROPIC_MODEL", "claude-sonnet-4-5")

# Image generation
# Options: "stub" (Pillow transforms), "stability" (Stability API),
#          "replicate" (Replicate SDXL), "diffusers" (local)
GENERATION_BACKEND = os.environ.get("GENERATION_BACKEND", "stub")
STABILITY_API_KEY = os.environ.get("STABILITY_API_KEY", "")
REPLICATE_API_TOKEN = os.environ.get("REPLICATE_API_TOKEN", "")

# Local open-source generation (backend="diffusers") — no API key needed.
# Downloads once to ~/.cache/huggingface/ on first use.
# Options: "runwayml/stable-diffusion-v1-5"  (~4 GB, recommended)
#          "stabilityai/stable-diffusion-2-1" (~5 GB, better quality)
LOCAL_SD_MODEL = os.environ.get("LOCAL_SD_MODEL", "runwayml/stable-diffusion-v1-5")


# --- generation defaults --------------------------------------------
DEFAULT_INFERENCE_STEPS = 30
DEFAULT_GUIDANCE_SCALE = 7.5
DEFAULT_SEED = 42

# Strength by mode (for img2img)
STRENGTH_BY_MODE = {
    "stylize": 0.75,
    "variation": 0.6,
    "enhance": 0.3,
}


def using_stub() -> bool:
    """True if we should run in stub mode (no API keys / no GPU)."""
    if STUB_MODE:
        return True
    # If no keys are set, force stub mode so we don't crash.
    if not ANTHROPIC_API_KEY and GENERATION_BACKEND == "stub":
        return True
    return False
