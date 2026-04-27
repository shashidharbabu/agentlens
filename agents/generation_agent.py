"""Image Generation Agent — rubric 3.3.

Supports three modes: stylize / variation / enhance.
Logs GenerationConfig (model, prompt, steps, guidance, seed, strength).
Real backends: Stability API, Replicate, local diffusers.
Stub backend: Pillow-based transformations that produce a visibly
different image without any model — used when no API keys are set.
"""
from __future__ import annotations

import io
import random
from pathlib import Path
from typing import Optional

from PIL import Image, ImageEnhance, ImageFilter, ImageOps

import config
from agents.base import BaseAgent
from utils.errors import GenerationError
from utils.io_utils import load_image
from utils.logger import (
    agent_call, agent_error, agent_response, agent_result, agent_start, stub_notice,
)
from utils.schemas import GenerationConfig, GenerationResult, RefinedPrompt


class GenerationAgent(BaseAgent):
    name = "generation"

    def __init__(self, backend: Optional[str] = None):
        if backend is not None:
            # Explicit backend always wins — allows real-mode tests and pipeline
            # to override stub mode set by test_pipeline.py at module level.
            self.backend = backend
        elif config.using_stub():
            self.backend = "stub"
        else:
            self.backend = config.GENERATION_BACKEND

    def run(
        self,
        refined: RefinedPrompt,
        input_image_path: str | Path,
        output_path: Path,
        seed: Optional[int] = None,
    ) -> GenerationResult:
        import time
        mode = refined.mode
        seed = seed if seed is not None else config.DEFAULT_SEED
        strength = config.STRENGTH_BY_MODE.get(mode, 0.6)

        cfg = GenerationConfig(
            model=self._model_name(),
            mode=mode,
            prompt=refined.refined_prompt,
            inference_steps=config.DEFAULT_INFERENCE_STEPS,
            guidance_scale=config.DEFAULT_GUIDANCE_SCALE,
            seed=seed,
            strength=strength,
        )

        t0 = agent_start(
            "generation", "Image generation",
            backend=self.backend,
            model=cfg.model,
            mode=mode,
            strength=f"{strength} (img2img denoising)",
            steps=cfg.inference_steps,
            guidance=cfg.guidance_scale,
            seed=seed,
            prompt=refined.refined_prompt[:120],
        )

        try:
            if self.backend == "stub":
                stub_notice("generation")
                img = self._stub_generate(input_image_path, mode, seed)
            elif self.backend == "stability":
                img = self._stability_generate(
                    input_image_path, refined.refined_prompt, strength, seed, t0
                )
            elif self.backend == "replicate":
                img = self._replicate_generate(
                    input_image_path, refined.refined_prompt, strength
                )
            elif self.backend == "diffusers":
                img = self._diffusers_generate(
                    input_image_path, refined.refined_prompt, strength, seed
                )
            else:
                raise GenerationError(f"Unknown backend: {self.backend}")

            output_path.parent.mkdir(parents=True, exist_ok=True)
            img.save(output_path)
            w, h = img.size
            file_kb = output_path.stat().st_size // 1024
            agent_result(
                "generation", time.time() - t0,
                output_image=str(output_path),
                dimensions=f"{w}x{h} px",
                file_size=f"{file_kb} KB",
                config=f"model={cfg.model} | steps={cfg.inference_steps} | guidance={cfg.guidance_scale} | strength={strength}",
            )
            return GenerationResult(image_path=output_path, config=cfg)

        except Exception as exc:
            agent_error("generation", str(exc))
            return GenerationResult(
                image_path=None,
                config=cfg,
                error=f"{type(exc).__name__}: {exc}",
            )

    # -------------------------------------------------------------------
    def _model_name(self) -> str:
        if self.backend == "diffusers":
            return config.LOCAL_SD_MODEL  # e.g. "runwayml/stable-diffusion-v1-5"
        return {
            "stub": "pillow-stub-v1",
            "stability": "stable-diffusion-3",
            "replicate": "stability-ai/sdxl",
        }.get(self.backend, "unknown")

    # -------------------------------------------------------------------
    # STUB: produces a visibly-different image per mode, deterministic by seed.
    # -------------------------------------------------------------------
    def _stub_generate(
        self, input_path: str | Path, mode: str, seed: int
    ) -> Image.Image:
        random.seed(seed)
        img = load_image(input_path)

        if mode == "stylize":
            # watercolor-esque: posterize + slight blur + saturation bump
            out = img.filter(ImageFilter.SMOOTH_MORE)
            out = ImageOps.posterize(out, 3)
            out = ImageEnhance.Color(out).enhance(1.4)
            out = ImageEnhance.Contrast(out).enhance(1.1)
        elif mode == "enhance":
            # sharper + brighter + warmer
            out = ImageEnhance.Sharpness(img).enhance(2.0)
            out = ImageEnhance.Contrast(out).enhance(1.15)
            out = ImageEnhance.Brightness(out).enhance(1.05)
            # warm tint
            r, g, b = out.split()
            r = r.point(lambda p: min(255, int(p * 1.05)))
            b = b.point(lambda p: max(0, int(p * 0.95)))
            out = Image.merge("RGB", (r, g, b))
        else:  # variation
            out = img.filter(ImageFilter.GaussianBlur(radius=0.5))
            out = ImageEnhance.Color(out).enhance(1.2)
            # mild seeded shift
            shift = (random.randint(-8, 8), random.randint(-8, 8))
            out = ImageOps.expand(out, border=10, fill=(128, 128, 128))
            out = out.crop(
                (
                    10 + shift[0],
                    10 + shift[1],
                    10 + shift[0] + img.width,
                    10 + shift[1] + img.height,
                )
            )

        return out

    # -------------------------------------------------------------------
    # Stability AI REST API (img2img)
    # -------------------------------------------------------------------
    def _stability_generate(
        self,
        input_path: str | Path,
        prompt: str,
        strength: float,
        seed: int,
        t0: float = 0.0,
    ) -> Image.Image:
        import requests

        import time
        if not config.STABILITY_API_KEY:
            raise GenerationError("STABILITY_API_KEY not set")

        url = "https://api.stability.ai/v2beta/stable-image/generate/sd3"

        agent_call(
            "generation",
            f"POST {url}",
            model="stable-diffusion-3",
            mode="image-to-image",
            strength=strength,
            seed=seed,
            prompt=prompt[:120],
        )

        headers = {
            "Authorization": f"Bearer {config.STABILITY_API_KEY}",
            "Accept": "image/*",
        }
        with open(input_path, "rb") as f:
            files = {"image": f}
            data = {
                "prompt": prompt,
                "mode": "image-to-image",
                "strength": str(strength),
                "seed": str(seed),
                "output_format": "png",
            }
            t_call = time.time()
            r = requests.post(url, headers=headers, files=files, data=data, timeout=60)
        elapsed = int((time.time() - t_call) * 1000)

        if r.status_code != 200:
            agent_response("generation", elapsed, status=f"HTTP {r.status_code} ERROR", body=r.text[:200])
            raise GenerationError(f"Stability API {r.status_code}: {r.text[:200]}")

        img = Image.open(io.BytesIO(r.content)).convert("RGB")
        agent_response(
            "generation", elapsed,
            status=f"HTTP {r.status_code} OK",
            image_bytes=f"{len(r.content) // 1024} KB received",
            dimensions=f"{img.size[0]}x{img.size[1]} px",
        )
        return img

    # -------------------------------------------------------------------
    def _replicate_generate(
        self, input_path: str | Path, prompt: str, strength: float
    ) -> Image.Image:
        import replicate
        import requests

        if not config.REPLICATE_API_TOKEN:
            raise GenerationError("REPLICATE_API_TOKEN not set")

        with open(input_path, "rb") as f:
            output = replicate.run(
                "stability-ai/sdxl:latest",
                input={
                    "prompt": prompt,
                    "image": f,
                    "prompt_strength": strength,
                    "num_inference_steps": config.DEFAULT_INFERENCE_STEPS,
                },
            )
        # replicate returns a list of URLs
        url = output[0] if isinstance(output, list) else output
        r = requests.get(url, timeout=60)
        return Image.open(io.BytesIO(r.content)).convert("RGB")

    # -------------------------------------------------------------------
    # Local open-source backend — no API key, no credits.
    # Uses Stable Diffusion 1.5 (img2img) via the diffusers library.
    # Auto-detects: Apple Silicon (MPS) > NVIDIA (CUDA) > CPU.
    # Model downloads once (~4 GB) to ~/.cache/huggingface/ on first run.
    # -------------------------------------------------------------------
    def _diffusers_generate(
        self,
        input_path: str | Path,
        prompt: str,
        strength: float,
        seed: int,
    ) -> Image.Image:
        import time
        import torch
        from diffusers import StableDiffusionImg2ImgPipeline

        # --- device selection -------------------------------------------
        if torch.backends.mps.is_available():
            device = "mps"
            dtype = torch.float16
        elif torch.cuda.is_available():
            device = "cuda"
            dtype = torch.float16
        else:
            device = "cpu"
            dtype = torch.float32

        model_id = config.LOCAL_SD_MODEL

        agent_call(
            "generation",
            f"local diffusers  model={model_id}",
            device=device,
            dtype=str(dtype),
            strength=strength,
            seed=seed,
            prompt=prompt[:120],
            note="first run downloads ~4 GB to ~/.cache/huggingface/",
        )

        # Lazy-load and cache pipeline on the class so re-runs don't reload weights.
        cache_key = f"_pipe_{model_id}_{device}"
        pipe = getattr(GenerationAgent, cache_key, None)
        if pipe is None:
            print(f"  \033[95m│   Loading {model_id} onto {device}…\033[0m", flush=True)
            t_load = time.time()
            pipe = StableDiffusionImg2ImgPipeline.from_pretrained(
                model_id,
                torch_dtype=dtype,
                safety_checker=None,        # skip NSFW filter for speed
                requires_safety_checker=False,
            )
            pipe = pipe.to(device)
            pipe.set_progress_bar_config(disable=False)
            setattr(GenerationAgent, cache_key, pipe)
            print(f"  \033[95m│   Model loaded in {time.time() - t_load:.1f}s\033[0m", flush=True)
        else:
            print(f"  \033[95m│   Using cached pipeline ({model_id} on {device})\033[0m")

        # SD 1.5 img2img expects 512×512; keep aspect ratio with padding if needed.
        init_image = load_image(input_path).resize((512, 512))

        generator = torch.Generator(device=device).manual_seed(seed)

        t_infer = time.time()
        result = pipe(
            prompt=prompt,
            image=init_image,
            strength=strength,
            num_inference_steps=config.DEFAULT_INFERENCE_STEPS,
            guidance_scale=config.DEFAULT_GUIDANCE_SCALE,
            generator=generator,
        )
        elapsed = int((time.time() - t_infer) * 1000)

        out_img = result.images[0]
        agent_response(
            "generation", elapsed,
            status="inference complete",
            device=device,
            dimensions=f"{out_img.size[0]}x{out_img.size[1]} px",
        )
        return out_img
