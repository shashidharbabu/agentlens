"""Real-model eval suite for the multimodal pipeline.

All tests in this file make live API calls:
  - VisionAgent / PromptAgent / CritiqueAgent → Anthropic (claude-sonnet-4-5)
  - GenerationAgent → Stability AI (stable-diffusion-3)

Tests are auto-skipped when the required env vars are absent, so the file
is safe to import in CI even without credentials.

Run stub-only tests:   pytest tests/test_pipeline.py -v
Run real-mode tests:   source .env && pytest tests/test_real_mode.py -v
Run everything:        source .env && pytest tests/ -v

Marks used:
  @needs_anthropic  — skipped if ANTHROPIC_API_KEY is unset
  @needs_stability  — skipped if STABILITY_API_KEY is unset (generation tests)
  @needs_all        — skipped unless both keys are present (end-to-end)
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import pytest
from PIL import Image

# Project root on path when running via `pytest tests/`
sys.path.insert(0, str(Path(__file__).parent.parent))

import config  # noqa: E402
from agents import CritiqueAgent, GenerationAgent, PromptAgent, VisionAgent  # noqa: E402
from data.sample_images import ensure_sample_images  # noqa: E402
from orchestration import MultimodalPipeline  # noqa: E402
from utils.schemas import (  # noqa: E402
    Critique,
    GenerationConfig,
    GenerationResult,
    RefinedPrompt,
    RunReport,
    VisionOutput,
)

# ---------------------------------------------------------------------------
# Skip markers
# ---------------------------------------------------------------------------
_HAS_ANTHROPIC = bool(os.environ.get("ANTHROPIC_API_KEY", "").strip())
_HAS_STABILITY = bool(os.environ.get("STABILITY_API_KEY", "").strip())
_HAS_ALL = _HAS_ANTHROPIC and _HAS_STABILITY

needs_anthropic = pytest.mark.skipif(
    not _HAS_ANTHROPIC, reason="ANTHROPIC_API_KEY not set — real LLM tests skipped"
)
needs_stability = pytest.mark.skipif(
    not _HAS_STABILITY, reason="STABILITY_API_KEY not set — generation tests skipped"
)
needs_all = pytest.mark.skipif(
    not _HAS_ALL,
    reason="Both ANTHROPIC_API_KEY and STABILITY_API_KEY required for end-to-end tests",
)


# ---------------------------------------------------------------------------
# Shared session fixtures (compute once, reuse across tests to save API cost)
# ---------------------------------------------------------------------------
@pytest.fixture(scope="session")
def sample_images():
    return ensure_sample_images(config.INPUT_IMAGES_DIR)


@pytest.fixture(scope="session")
def real_vision_portrait(sample_images):
    """Real Claude analysis of the portrait image (used by Vision + Prompt tests)."""
    agent = VisionAgent(stub=False)
    return agent.run(sample_images["portrait"])


@pytest.fixture(scope="session")
def real_vision_landscape(sample_images):
    """Real Claude analysis of the landscape image."""
    agent = VisionAgent(stub=False)
    return agent.run(sample_images["landscape"])


@pytest.fixture(scope="session")
def real_vision_street(sample_images):
    """Real Claude analysis of the street_scene image with custom VQA."""
    agent = VisionAgent(stub=False)
    return agent.run(
        sample_images["street_scene"],
        vqa_questions=[
            "What is the dominant color palette of this image?",
            "What is the main subject and what is it doing?",
            "What time of day does this appear to be?",
        ],
    )


@pytest.fixture(scope="session")
def real_refined_stylize(real_vision_portrait):
    """Real Claude prompt refinement for stylize mode."""
    agent = PromptAgent(stub=False)
    return agent.run(
        "make this look like a watercolor painting with soft colors",
        real_vision_portrait,
        mode="stylize",
    )


@pytest.fixture(scope="session")
def real_refined_enhance(real_vision_landscape):
    """Real Claude prompt refinement for enhance mode."""
    agent = PromptAgent(stub=False)
    return agent.run(
        "sharpen the image and warm it up to look like golden hour",
        real_vision_landscape,
        mode="enhance",
    )


def _skip_if_quota_error(result, label: str = "Generation"):
    """Skip the calling test when the API returns a quota/credit error (402).

    This prevents billing-limit failures from masking real code bugs.
    """
    if result.error and any(
        kw in result.error.lower()
        for kw in ("402", "credits", "quota", "insufficient", "billing")
    ):
        pytest.skip(f"{label} skipped — API quota/credits exhausted: {result.error[:120]}")


@pytest.fixture(scope="session")
def real_generated_stylize(sample_images, real_refined_stylize, tmp_path_factory):
    """Real Stability AI generation for stylize mode (portrait).

    Automatically skips dependent tests when Stability credits run out.
    """
    out = tmp_path_factory.mktemp("real_gen") / "stylize_out.png"
    agent = GenerationAgent(backend="stability")
    result = agent.run(real_refined_stylize, sample_images["portrait"], out)
    _skip_if_quota_error(result, "Stylize generation")
    return result


@pytest.fixture(scope="session")
def real_generated_enhance(sample_images, real_refined_enhance, tmp_path_factory):
    """Real Stability AI generation for enhance mode (landscape).

    Automatically skips dependent tests when Stability credits run out.
    """
    out = tmp_path_factory.mktemp("real_gen") / "enhance_out.png"
    agent = GenerationAgent(backend="stability")
    result = agent.run(real_refined_enhance, sample_images["landscape"], out)
    _skip_if_quota_error(result, "Enhance generation")
    return result


# ---------------------------------------------------------------------------
# A. Vision Agent — real-model quality tests
# ---------------------------------------------------------------------------
class TestVisionAgentReal:
    @needs_anthropic
    def test_caption_is_non_stub(self, real_vision_portrait):
        """Real caption should not look like the stub fallback format."""
        c = real_vision_portrait.caption
        assert len(c) > 20, f"Caption too short: {c!r}"
        assert "predominantly" not in c, "Caption looks like stub output"
        assert "palette" not in c.lower() or len(c) > 60, "Caption may be stub"

    @needs_anthropic
    def test_caption_is_complete_sentence(self, real_vision_portrait):
        c = real_vision_portrait.caption
        assert c[0].isupper(), "Caption should start with a capital letter"
        assert len(c.split()) >= 5, "Caption too short to be a real sentence"

    @needs_anthropic
    def test_objects_are_specific(self, real_vision_portrait):
        """Real objects should be actual named things, not stub placeholders."""
        objects = real_vision_portrait.objects
        assert len(objects) >= 2, "Expected at least 2 objects"
        stub_words = {"subject", "foreground elements", "background"}
        real_objects = [o for o in objects if o.lower() not in stub_words]
        assert len(real_objects) >= 1, f"All objects look like stubs: {objects}"

    @needs_anthropic
    def test_scene_is_detailed(self, real_vision_portrait):
        scene = real_vision_portrait.scene
        assert len(scene) > 50, f"Scene too short: {scene!r}"
        assert "pixels" not in scene, "Scene looks like stub (contains pixel dimensions)"

    @needs_anthropic
    def test_vqa_count_meets_rubric(self, real_vision_street):
        """Must answer at least 2 VQA questions — PDF rubric 3.1 requirement."""
        assert len(real_vision_street.vqa_pairs) >= 2

    @needs_anthropic
    def test_vqa_answers_are_specific(self, real_vision_street):
        """VQA answers from Claude should be substantive, not stub templates."""
        for pair in real_vision_street.vqa_pairs:
            assert "question" in pair and "answer" in pair
            ans = pair["answer"]
            assert len(ans) > 10, f"VQA answer too short: {ans!r}"
            assert "relates to its" not in ans, f"Answer looks like stub: {ans!r}"

    @needs_anthropic
    def test_vqa_answers_reference_question_topic(self, real_vision_street):
        """Spot-check: color-palette question should mention a color in the answer."""
        color_q = next(
            (p for p in real_vision_street.vqa_pairs if "color" in p["question"].lower()),
            None,
        )
        assert color_q is not None
        color_words = {"blue", "red", "green", "yellow", "gray", "grey", "brown",
                       "white", "black", "orange", "purple", "pink", "warm", "cool",
                       "dark", "light", "muted", "neutral", "palette", "tone", "hue"}
        answer_lower = color_q["answer"].lower()
        assert any(w in answer_lower for w in color_words), (
            f"Color VQA answer doesn't mention any color: {color_q['answer']!r}"
        )

    @needs_anthropic
    def test_raw_model_output_is_present(self, real_vision_portrait):
        raw = real_vision_portrait.raw_model_output
        assert raw and raw != "[stub mode — no LLM called]"
        assert len(raw) > 20

    @needs_anthropic
    def test_low_quality_flag_is_false_for_normal_image(self, real_vision_portrait):
        assert real_vision_portrait.low_quality_input is False

    @needs_anthropic
    def test_vision_output_is_json_serializable(self, real_vision_portrait):
        d = real_vision_portrait.to_dict()
        dumped = json.dumps(d)
        assert len(dumped) > 50


# ---------------------------------------------------------------------------
# B. Prompt Agent — real-model quality tests
# ---------------------------------------------------------------------------
class TestPromptAgentReal:
    @needs_anthropic
    def test_refined_prompt_is_longer_than_instruction(self, real_refined_stylize):
        original = "make this look like a watercolor painting with soft colors"
        refined = real_refined_stylize.refined_prompt
        assert len(refined) > len(original), (
            f"Refined ({len(refined)} chars) is not longer than original ({len(original)} chars)"
        )

    @needs_anthropic
    def test_refined_prompt_has_minimum_length(self, real_refined_stylize):
        refined = real_refined_stylize.refined_prompt
        assert len(refined.split()) >= 10, f"Refined prompt too short: {refined!r}"

    @needs_anthropic
    def test_enrichment_notes_reference_scene(self, real_refined_stylize):
        notes = real_refined_stylize.enrichment_notes
        assert len(notes) > 10, f"Enrichment notes too short: {notes!r}"

    @needs_anthropic
    def test_preserved_transform_is_specific(self, real_refined_stylize):
        pt = real_refined_stylize.preserved_transform
        assert len(pt) > 10, f"preserved_transform too short: {pt!r}"
        watercolor_words = {"watercolor", "painting", "style", "color", "soft",
                            "artistic", "convert", "transform", "illustration"}
        pt_lower = pt.lower()
        assert any(w in pt_lower for w in watercolor_words), (
            f"preserved_transform doesn't mention the watercolor transform: {pt!r}"
        )

    @needs_anthropic
    def test_confidence_is_above_threshold(self, real_refined_stylize):
        assert real_refined_stylize.confidence >= 0.5, (
            f"Confidence too low for valid instruction: {real_refined_stylize.confidence}"
        )

    @needs_anthropic
    def test_mode_is_preserved_in_output(self, real_refined_stylize):
        assert real_refined_stylize.mode == "stylize"

    @needs_anthropic
    def test_enhance_mode_prompt_mentions_sharpening_or_light(self, real_refined_enhance):
        prompt_lower = real_refined_enhance.refined_prompt.lower()
        expected_words = {"sharp", "golden", "warm", "hour", "light", "enhance",
                          "detail", "clarity", "bright", "amber", "sun"}
        assert any(w in prompt_lower for w in expected_words), (
            f"Enhance prompt doesn't reference expected concepts: {real_refined_enhance.refined_prompt!r}"
        )

    @needs_anthropic
    def test_refined_prompt_output_is_json_serializable(self, real_refined_stylize):
        d = real_refined_stylize.to_dict()
        dumped = json.dumps(d)
        assert len(dumped) > 30

    @needs_anthropic
    def test_ambiguous_prompt_raises_gracefully(self, real_vision_portrait):
        """Empty instruction should raise AmbiguousPromptError even in real mode."""
        from utils.errors import AmbiguousPromptError
        agent = PromptAgent(stub=False)
        with pytest.raises(AmbiguousPromptError):
            agent.run("", real_vision_portrait, mode="stylize")

    @needs_anthropic
    def test_variation_mode_produces_valid_output(self, real_vision_street, sample_images):
        agent = PromptAgent(stub=False)
        refined = agent.run(
            "create a variation of this scene",
            real_vision_street,
            mode="variation",
        )
        assert refined.mode == "variation"
        assert len(refined.refined_prompt) > 10


# ---------------------------------------------------------------------------
# C. Generation Agent — real Stability AI tests
# ---------------------------------------------------------------------------
class TestGenerationAgentReal:
    @needs_stability
    def test_generated_image_file_exists(self, real_generated_stylize):
        assert real_generated_stylize.image_path is not None
        assert real_generated_stylize.image_path.exists(), (
            f"Generated image not found at {real_generated_stylize.image_path}"
        )

    @needs_stability
    def test_generated_image_is_valid_png(self, real_generated_stylize):
        img = Image.open(real_generated_stylize.image_path)
        assert img.mode in ("RGB", "RGBA", "L")
        w, h = img.size
        assert w > 0 and h > 0

    @needs_stability
    def test_generated_image_has_meaningful_size(self, real_generated_stylize):
        """File must be > 5 KB — rules out empty or placeholder outputs."""
        size = real_generated_stylize.image_path.stat().st_size
        assert size > 5_000, f"Generated image too small ({size} bytes) — likely empty"

    @needs_stability
    def test_generation_config_is_complete(self, real_generated_stylize):
        """Rubric 3.3 requires model, prompt, and inference_steps to be logged."""
        cfg = real_generated_stylize.config
        assert cfg.model and cfg.model != "unknown"
        assert cfg.prompt and len(cfg.prompt) > 10
        assert cfg.inference_steps > 0
        assert cfg.guidance_scale > 0
        assert cfg.mode == "stylize"

    @needs_stability
    def test_generation_config_is_json_serializable(self, real_generated_stylize):
        d = real_generated_stylize.config.to_dict()
        dumped = json.dumps(d, default=str)
        assert "model" in dumped and "prompt" in dumped

    @needs_stability
    def test_generation_error_field_is_none_on_success(self, real_generated_stylize):
        assert real_generated_stylize.error is None

    @needs_stability
    def test_enhance_mode_produces_image(self, real_generated_enhance):
        assert real_generated_enhance.image_path is not None
        assert real_generated_enhance.image_path.exists()
        assert real_generated_enhance.config.mode == "enhance"

    @needs_stability
    def test_generation_result_is_json_serializable(self, real_generated_stylize):
        d = real_generated_stylize.to_dict()
        dumped = json.dumps(d, default=str)
        assert len(dumped) > 30


# ---------------------------------------------------------------------------
# D. Critique Agent — real-model quality tests
# ---------------------------------------------------------------------------
class TestCritiqueAgentReal:
    @pytest.fixture(scope="class")
    def real_critique_stylize(self, sample_images, real_generated_stylize, real_refined_stylize):
        agent = CritiqueAgent(stub=False)
        return agent.run(
            input_image_path=sample_images["portrait"],
            generation=real_generated_stylize,
            refined=real_refined_stylize,
        )

    @needs_all
    def test_clip_scores_are_in_valid_range(self, real_critique_stylize):
        assert 0.0 <= real_critique_stylize.clip_similarity_image <= 1.0
        assert 0.0 <= real_critique_stylize.clip_similarity_text <= 1.0

    @needs_all
    def test_llm_scores_are_in_valid_range(self, real_critique_stylize):
        assert 0.0 <= real_critique_stylize.visual_relevance <= 1.0
        assert 0.0 <= real_critique_stylize.prompt_faithfulness <= 1.0
        assert 0.0 <= real_critique_stylize.quality <= 1.0

    @needs_all
    def test_rationale_is_llm_generated(self, real_critique_stylize):
        rationale = real_critique_stylize.rationale
        assert "[stub critique]" not in rationale, (
            "Rationale looks like a stub response; LLM reasoning not used"
        )
        assert len(rationale) > 30, f"Rationale too short: {rationale!r}"

    @needs_all
    def test_verdict_is_valid_enum(self, real_critique_stylize):
        assert real_critique_stylize.verdict in {"accept", "revise"}

    @needs_all
    def test_revision_suggestion_present_when_verdict_is_revise(self, real_critique_stylize):
        if real_critique_stylize.verdict == "revise":
            assert real_critique_stylize.revision_suggestion is not None
            assert len(real_critique_stylize.revision_suggestion) > 5

    @needs_all
    def test_clip_image_score_is_nonzero_for_valid_output(self, real_critique_stylize):
        """For a successful generation, image-image similarity must be > 0."""
        assert real_critique_stylize.clip_similarity_image > 0.0

    @needs_all
    def test_critique_output_is_json_serializable(self, real_critique_stylize):
        d = real_critique_stylize.to_dict()
        dumped = json.dumps(d, default=str)
        assert "verdict" in dumped and "rationale" in dumped

    @needs_all
    def test_critique_handles_generation_failure_gracefully(self, sample_images, real_refined_stylize):
        """Critique issued with image_path=None must return verdict=revise."""
        failed_gen = GenerationResult(
            image_path=None,
            config=GenerationConfig(
                model="broken", mode="stylize", prompt="...",
                inference_steps=0, guidance_scale=0.0
            ),
            error="Simulated API failure",
        )
        agent = CritiqueAgent(stub=False)
        critique = agent.run(
            input_image_path=sample_images["portrait"],
            generation=failed_gen,
            refined=real_refined_stylize,
        )
        assert critique.verdict == "revise"
        assert "failed" in critique.rationale.lower()


# ---------------------------------------------------------------------------
# E. Full Pipeline — real end-to-end tests
# ---------------------------------------------------------------------------
class TestFullPipelineReal:
    @pytest.fixture(scope="class")
    def real_pipeline(self):
        return MultimodalPipeline(stub=False)

    @needs_anthropic
    def test_uc1_captioning_vqa_real(self, real_pipeline, sample_images):
        """UC1: captioning+VQA with real Claude — no generation step."""
        report = real_pipeline.run(
            image_path=sample_images["street_scene"],
            instruction="describe this image",
            mode="none",
            vqa_questions=[
                "What is the dominant color palette?",
                "What is the main subject?",
                "What time of day is depicted?",
            ],
            run_name="real_test_uc1",
        )
        assert report.vision is not None, f"Vision failed: {report.errors}"
        assert not report.errors, f"Unexpected errors: {report.errors}"
        assert len(report.vision.vqa_pairs) >= 2
        assert report.generation is None  # mode=none stops after vision
        assert report.critique is None
        # Content quality: real Claude should produce a meaningful caption
        assert len(report.vision.caption) > 20
        assert "pixels" not in report.vision.caption  # not a stub

    @needs_all
    def test_uc2_style_transfer_real(self, real_pipeline, sample_images):
        """UC2: full pipeline with real Claude + Stability AI."""
        report = real_pipeline.run(
            image_path=sample_images["portrait"],
            instruction="make this look like a watercolor painting with soft colors",
            mode="stylize",
            run_name="real_test_uc2",
        )
        assert report.vision is not None, f"Vision failed: {report.errors}"
        assert report.refined_prompt is not None, f"Prompt failed: {report.errors}"
        assert report.generation is not None, f"Generation step missing: {report.errors}"
        # Skip if Stability credits ran out — not a code bug.
        _skip_if_quota_error(report.generation, "UC2 style-transfer generation")
        assert report.critique is not None, f"Critique failed: {report.errors}"
        assert report.generation.image_path is not None
        assert report.generation.image_path.exists()
        assert report.generation.image_path.stat().st_size > 5_000
        assert "[stub critique]" not in report.critique.rationale
        assert report.critique.verdict in {"accept", "revise"}

    @needs_all
    def test_uc3_enhance_real(self, real_pipeline, sample_images):
        """UC3: enhancement pipeline with real APIs."""
        report = real_pipeline.run(
            image_path=sample_images["landscape"],
            instruction="sharpen the image and warm it up to look like golden hour",
            mode="enhance",
            run_name="real_test_uc3",
        )
        assert report.vision is not None
        assert report.refined_prompt is not None
        assert report.generation is not None
        _skip_if_quota_error(report.generation, "UC3 enhance generation")
        assert report.generation.image_path is not None
        assert report.generation.image_path.exists()
        assert report.critique is not None

    @needs_all
    def test_variation_mode_real(self, real_pipeline, sample_images):
        """Test the variation mode — not exercised by the demo use cases."""
        report = real_pipeline.run(
            image_path=sample_images["street_scene"],
            instruction="create an artistic variation of this scene",
            mode="variation",
            run_name="real_test_variation",
        )
        assert report.generation is not None
        _skip_if_quota_error(report.generation, "Variation-mode generation")
        assert report.generation.config.mode == "variation"
        if report.generation.image_path:
            assert report.generation.image_path.exists()

    @needs_anthropic
    def test_full_pipeline_no_unexpected_errors(self, real_pipeline, sample_images):
        """A valid run should produce an empty errors list."""
        report = real_pipeline.run(
            image_path=sample_images["portrait"],
            instruction="describe this image",
            mode="none",
            run_name="real_test_no_errors",
        )
        assert report.errors == [], f"Unexpected errors in clean run: {report.errors}"

    @needs_all
    def test_intermediate_step_logs_contain_real_content(
        self, real_pipeline, sample_images
    ):
        """Step JSON files must contain real model output, not stub placeholders."""
        report = real_pipeline.run(
            image_path=sample_images["portrait"],
            instruction="stylize as oil painting",
            mode="stylize",
            run_name="real_test_logs",
        )
        run_dir = config.OUTPUTS_ROOT / "real_test_logs"

        # Steps 1 & 2 (Anthropic) always produce real content.
        step1 = json.loads((run_dir / "step_1_vision.json").read_text())
        assert step1["raw_model_output"] != "[stub mode — no LLM called]"
        assert len(step1["caption"]) > 20

        step2 = json.loads((run_dir / "step_2_prompt.json").read_text())
        assert len(step2["refined_prompt"]) > 20
        assert len(step2["preserved_transform"]) > 10

        # Step 3 (Stability) — skip assertions if credits ran out.
        step3 = json.loads((run_dir / "step_3_generation.json").read_text())
        if step3.get("error") and any(
            kw in str(step3["error"]).lower()
            for kw in ("402", "credits", "quota", "insufficient")
        ):
            pytest.skip("Stability credits exhausted — step_3 generation assertions skipped")
        assert step3["config"]["model"] != "pillow-stub-v1"
        assert step3["config"]["inference_steps"] > 0

        # Step 4 (Anthropic critique) — always runs even on generation failure;
        # only check LLM rationale when generation succeeded.
        step4 = json.loads((run_dir / "step_4_critique.json").read_text())
        assert step4["verdict"] in ("accept", "revise")
        if step3.get("image_path"):  # generation succeeded
            assert "[stub critique]" not in step4["rationale"]

    @needs_all
    def test_run_report_is_fully_serializable(self, real_pipeline, sample_images):
        """RunReport.to_dict() must produce a JSON-serializable structure."""
        report = real_pipeline.run(
            image_path=sample_images["landscape"],
            instruction="make it warmer",
            mode="enhance",
            run_name="real_test_serial",
        )
        d = report.to_dict()
        dumped = json.dumps(d, default=str)
        loaded = json.loads(dumped)
        assert loaded["run_id"] == "real_test_serial"
        assert loaded["vision"] is not None
        assert loaded["mode"] == "enhance"


# ---------------------------------------------------------------------------
# F. Schema + data-contract tests (no API required — just validates structure)
# ---------------------------------------------------------------------------
class TestSchemaContracts:
    """These run in any mode; they validate the dataclass contracts."""

    def test_vision_output_all_fields_present(self):
        v = VisionOutput(
            caption="test caption",
            objects=["obj1", "obj2"],
            scene="test scene",
            vqa_pairs=[{"question": "q?", "answer": "a"}],
        )
        d = v.to_dict()
        for key in ("caption", "objects", "scene", "vqa_pairs",
                    "raw_model_output", "low_quality_input", "warnings"):
            assert key in d, f"Missing field in VisionOutput.to_dict(): {key}"

    def test_refined_prompt_all_fields_present(self):
        r = RefinedPrompt(
            original_instruction="test",
            refined_prompt="test refined",
            enrichment_notes="notes",
            preserved_transform="transform",
        )
        d = r.to_dict()
        for key in ("original_instruction", "refined_prompt", "enrichment_notes",
                    "preserved_transform", "confidence", "mode"):
            assert key in d, f"Missing field in RefinedPrompt.to_dict(): {key}"

    def test_generation_config_all_fields_present(self):
        cfg = GenerationConfig(
            model="test", mode="stylize", prompt="test prompt",
            inference_steps=30, guidance_scale=7.5
        )
        d = cfg.to_dict()
        for key in ("model", "mode", "prompt", "inference_steps", "guidance_scale"):
            assert key in d

    def test_critique_all_fields_present(self):
        c = Critique(
            clip_similarity_image=0.9,
            clip_similarity_text=0.3,
            visual_relevance=0.8,
            prompt_faithfulness=0.7,
            quality=0.75,
            rationale="test rationale",
            verdict="accept",
        )
        d = c.to_dict()
        for key in ("clip_similarity_image", "clip_similarity_text",
                    "visual_relevance", "prompt_faithfulness", "quality",
                    "rationale", "verdict", "used_fallback_metric"):
            assert key in d, f"Missing field in Critique.to_dict(): {key}"

    def test_run_report_all_fields_present(self):
        r = RunReport(run_id="test", image_path="img.jpg",
                      instruction="test", mode="stylize")
        d = r.to_dict()
        for key in ("run_id", "image_path", "instruction", "mode",
                    "vision", "refined_prompt", "generation", "critique", "errors"):
            assert key in d, f"Missing field in RunReport.to_dict(): {key}"

    def test_vqa_pairs_accept_multiple_questions(self):
        questions = ["Q1?", "Q2?", "Q3?"]
        vqa = [{"question": q, "answer": f"A for {q}"} for q in questions]
        v = VisionOutput(caption="c", objects=[], scene="s", vqa_pairs=vqa)
        assert len(v.vqa_pairs) == 3

    def test_generation_result_null_path_serializes(self):
        """GenerationResult with image_path=None must serialize without error."""
        cfg = GenerationConfig(
            model="m", mode="stylize", prompt="p", inference_steps=1, guidance_scale=7.5
        )
        gr = GenerationResult(image_path=None, config=cfg, error="some error")
        d = gr.to_dict()
        assert d["image_path"] is None
        assert d["error"] == "some error"
        json.dumps(d, default=str)  # must not raise


# ---------------------------------------------------------------------------
# G. Error handling in real mode
# ---------------------------------------------------------------------------
class TestErrorHandlingReal:
    @needs_anthropic
    def test_ambiguous_prompt_recorded_in_report(self, sample_images):
        """Empty instruction should be caught and recorded in report.errors."""
        pipeline = MultimodalPipeline(stub=False)
        report = pipeline.run(
            image_path=sample_images["portrait"],
            instruction="",
            mode="stylize",
            run_name="real_test_ambiguous",
        )
        assert any("prompt" in e.lower() for e in report.errors)
        assert report.generation is None

    @needs_anthropic
    def test_missing_image_recorded_in_report(self, tmp_path):
        """Non-existent image path should be caught gracefully."""
        pipeline = MultimodalPipeline(stub=False)
        report = pipeline.run(
            image_path=tmp_path / "nonexistent.jpg",
            instruction="describe it",
            mode="stylize",
            run_name="real_test_missing_img",
        )
        assert report.vision is None
        assert any("vision" in e.lower() for e in report.errors)

    @needs_anthropic
    def test_bad_generation_key_handled(self, sample_images, real_refined_stylize, tmp_path):
        """Invalid Stability key should not crash the pipeline; error is captured.

        We patch config.STABILITY_API_KEY directly (no importlib.reload needed)
        because GenerationAgent reads config.STABILITY_API_KEY inside
        _stability_generate() at call time, not at __init__ time.
        """
        original = config.STABILITY_API_KEY
        try:
            config.STABILITY_API_KEY = "sk-invalid-key-for-testing"
            # Pass backend="stability" explicitly so the stub-mode guard is bypassed.
            agent = GenerationAgent(backend="stability")
            out_path = tmp_path / "bad_key_out.png"
            result = agent.run(real_refined_stylize, sample_images["portrait"], out_path)
            # GenerationAgent.run() catches the API error and returns a result
            # with error set rather than raising.
            assert result.error is not None, (
                "Expected an error result for invalid Stability API key"
            )
            assert result.image_path is None
        finally:
            config.STABILITY_API_KEY = original

    def test_low_quality_image_real_flag(self, tmp_path):
        """Tiny image (64×64) must set low_quality_input=True in any mode."""
        tiny = tmp_path / "tiny.jpg"
        Image.new("RGB", (64, 64), (200, 100, 50)).save(tiny, "JPEG")
        # This works in stub mode — no API call needed
        agent = VisionAgent(stub=True)
        out = agent.run(tiny)
        assert out.low_quality_input is True
        assert len(out.warnings) >= 1
