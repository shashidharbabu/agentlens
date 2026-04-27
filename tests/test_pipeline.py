"""Tests for the multimodal pipeline.

Covers:
  - Happy path on all three use cases (stub mode)
  - Failure mode 1: ambiguous prompt
  - Failure mode 2: poor-quality / missing image
  - Failure mode 3: generation error (simulated backend failure)
  - Agent modularity (each agent runs standalone)

Run with:  pytest tests/ -v
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

import pytest
from PIL import Image

# Make sure we're in stub mode for tests — no API keys required.
os.environ["STUB_MODE"] = "1"

# Ensure project root is on path when running via `pytest tests/`
sys.path.insert(0, str(Path(__file__).parent.parent))

import config  # noqa: E402
from agents import (  # noqa: E402
    CritiqueAgent,
    GenerationAgent,
    PromptAgent,
    VisionAgent,
)
from data.sample_images import ensure_sample_images  # noqa: E402
from orchestration import MultimodalPipeline  # noqa: E402
from utils.errors import AmbiguousPromptError, LowQualityImageError  # noqa: E402
from utils.schemas import (  # noqa: E402
    Critique,
    GenerationConfig,
    GenerationResult,
    RefinedPrompt,
    VisionOutput,
)


@pytest.fixture(scope="session")
def sample_images():
    return ensure_sample_images(config.INPUT_IMAGES_DIR)


@pytest.fixture(scope="session")
def pipeline():
    return MultimodalPipeline(stub=True)


# ---------------------------------------------------------------------
# Standalone agent tests (rubric 3.5.1 — modular design)
# ---------------------------------------------------------------------
class TestAgentsStandalone:
    def test_vision_agent_standalone(self, sample_images):
        agent = VisionAgent(stub=True)
        out = agent.run(sample_images["street_scene"])
        assert isinstance(out, VisionOutput)
        assert out.caption
        assert isinstance(out.objects, list)
        assert out.scene
        # Rubric 3.1: must answer >= 2 visual questions
        assert len(out.vqa_pairs) >= 2
        for pair in out.vqa_pairs:
            assert "question" in pair and "answer" in pair

    def test_prompt_agent_standalone(self, sample_images):
        vision = VisionAgent(stub=True).run(sample_images["portrait"])
        agent = PromptAgent(stub=True)
        refined = agent.run(
            "make it a watercolor painting", vision, mode="stylize"
        )
        assert isinstance(refined, RefinedPrompt)
        assert refined.refined_prompt
        # Rubric 3.2: must preserve transformation explicitly
        assert refined.preserved_transform
        assert refined.enrichment_notes  # must enrich with scene details
        assert refined.mode == "stylize"

    def test_generation_agent_standalone(self, sample_images, tmp_path):
        refined = RefinedPrompt(
            original_instruction="enhance",
            refined_prompt="A sharp, detailed landscape in golden hour light.",
            enrichment_notes="",
            preserved_transform="enhance the image",
            mode="enhance",
        )
        agent = GenerationAgent(backend="stub")
        out_path = tmp_path / "out.png"
        result = agent.run(refined, sample_images["landscape"], out_path)
        assert isinstance(result, GenerationResult)
        # Rubric 3.3: config must be recorded
        assert isinstance(result.config, GenerationConfig)
        assert result.config.model
        assert result.config.prompt
        assert result.config.inference_steps > 0
        assert result.image_path and result.image_path.exists()

    def test_critique_agent_standalone(self, sample_images, tmp_path):
        # produce an output image to critique
        refined = RefinedPrompt(
            original_instruction="variation",
            refined_prompt="A creative variation of the input scene.",
            enrichment_notes="",
            preserved_transform="variation",
            mode="variation",
        )
        gen_result = GenerationAgent(backend="stub").run(
            refined, sample_images["street_scene"], tmp_path / "out.png"
        )
        critique = CritiqueAgent(stub=True).run(
            input_image_path=sample_images["street_scene"],
            generation=gen_result,
            refined=refined,
        )
        assert isinstance(critique, Critique)
        # Rubric 3.4: all three dimensions + CLIP signal + verdict
        assert 0.0 <= critique.visual_relevance <= 1.0
        assert 0.0 <= critique.prompt_faithfulness <= 1.0
        assert 0.0 <= critique.quality <= 1.0
        assert critique.verdict in {"accept", "revise"}
        assert critique.rationale


# ---------------------------------------------------------------------
# Happy path end-to-end on all three use cases (rubric 3.5.4)
# ---------------------------------------------------------------------
class TestHappyPath:
    def test_use_case_1_captioning_vqa(self, pipeline, sample_images):
        report = pipeline.run(
            image_path=sample_images["street_scene"],
            instruction="describe this image",
            mode="none",
            vqa_questions=[
                "What colors dominate?",
                "What is happening?",
                "What time of day is it?",
            ],
            run_name="test_uc1",
        )
        assert report.vision is not None
        assert len(report.vision.vqa_pairs) >= 2
        # mode="none" -> no generation / critique expected
        assert report.generation is None
        assert report.critique is None
        assert not report.errors

    def test_use_case_2_style_transfer(self, pipeline, sample_images):
        report = pipeline.run(
            image_path=sample_images["portrait"],
            instruction="make it look like a watercolor painting",
            mode="stylize",
            run_name="test_uc2",
        )
        assert report.vision is not None
        assert report.refined_prompt is not None
        assert report.generation is not None
        assert report.generation.image_path is not None
        assert report.generation.image_path.exists()
        assert report.critique is not None
        assert report.critique.verdict in {"accept", "revise"}

    def test_use_case_3_enhance(self, pipeline, sample_images):
        report = pipeline.run(
            image_path=sample_images["landscape"],
            instruction="sharpen and make it golden hour",
            mode="enhance",
            run_name="test_uc3",
        )
        assert report.vision is not None
        assert report.refined_prompt is not None
        assert report.generation is not None
        assert report.critique is not None
        # The stub should at minimum produce an image file
        assert report.generation.image_path.exists()


# ---------------------------------------------------------------------
# Failure handling (rubric 3.5.5)
# ---------------------------------------------------------------------
class TestFailureHandling:
    def test_failure_ambiguous_prompt(self, pipeline, sample_images):
        """Very short/empty instruction should be handled without crashing."""
        report = pipeline.run(
            image_path=sample_images["portrait"],
            instruction="",  # empty -> ambiguous
            mode="stylize",
            run_name="test_fail_ambiguous",
        )
        # Pipeline should not raise; it should record the error.
        assert any("prompt" in e.lower() for e in report.errors)
        # vision still runs successfully
        assert report.vision is not None
        # generation / critique did not run
        assert report.generation is None or report.generation.image_path is None

    def test_failure_missing_image(self, pipeline, tmp_path):
        """Non-existent image should be caught and reported gracefully."""
        bogus_path = tmp_path / "does_not_exist.jpg"
        report = pipeline.run(
            image_path=bogus_path,
            instruction="describe it",
            mode="stylize",
            run_name="test_fail_missing_image",
        )
        assert any("vision" in e.lower() for e in report.errors)
        assert report.vision is None

    def test_failure_low_quality_image_flagged(self, tmp_path):
        """Tiny image should trigger the low-quality warning but still run."""
        tiny_path = tmp_path / "tiny.jpg"
        Image.new("RGB", (64, 64), (128, 128, 128)).save(tiny_path, "JPEG")

        vision = VisionAgent(stub=True).run(tiny_path)
        assert vision.low_quality_input is True
        assert vision.warnings  # should contain a warning string

    def test_failure_generation_error_handled(self, sample_images, tmp_path):
        """A generation failure must not crash the pipeline; critique should
        still run and issue a 'revise' verdict."""

        class BrokenGenerationAgent(GenerationAgent):
            def run(self, refined, input_image_path, output_path, seed=None):
                cfg = GenerationConfig(
                    model="broken-stub",
                    mode=refined.mode,
                    prompt=refined.refined_prompt,
                    inference_steps=0,
                    guidance_scale=0.0,
                    seed=seed,
                )
                return GenerationResult(
                    image_path=None,
                    config=cfg,
                    error="Simulated backend failure",
                )

        # Build a pipeline with the broken generation agent swapped in.
        pipeline = MultimodalPipeline(stub=True)
        pipeline.generation = BrokenGenerationAgent(backend="stub")

        report = pipeline.run(
            image_path=sample_images["landscape"],
            instruction="enhance this image",
            mode="enhance",
            run_name="test_fail_generation",
        )
        assert report.generation is not None
        assert report.generation.image_path is None
        assert report.generation.error
        # Critique should still run and recommend revision
        assert report.critique is not None
        assert report.critique.verdict == "revise"
        assert "failed" in report.critique.rationale.lower()


# ---------------------------------------------------------------------
# Intermediate output logging (rubric 3.5.2)
# ---------------------------------------------------------------------
def test_pipeline_writes_intermediate_logs(pipeline, sample_images):
    report = pipeline.run(
        image_path=sample_images["street_scene"],
        instruction="make it warmer",
        mode="enhance",
        run_name="test_logs",
    )
    run_dir = config.OUTPUTS_ROOT / "test_logs"
    assert (run_dir / "step_1_vision.json").exists()
    assert (run_dir / "step_2_prompt.json").exists()
    assert (run_dir / "step_3_generation.json").exists()
    assert (run_dir / "step_4_critique.json").exists()
    assert (run_dir / "report.json").exists()
