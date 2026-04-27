"""Multi-agent pipeline orchestration — rubric 3.5.

Sequence: Vision -> Prompt -> Generation -> Critique.
Each step logs its output to outputs/<run_id>/step_<n>_<agent>.json.
Agent failures are caught and recorded; the pipeline always returns
a RunReport even on partial failure.
"""
from __future__ import annotations

import shutil
from pathlib import Path
from typing import Optional

import config
from agents import CritiqueAgent, GenerationAgent, PromptAgent, VisionAgent
from utils.errors import (
    AmbiguousPromptError,
    GenerationError,
    LowQualityImageError,
)
from utils.io_utils import (
    dump_json,
    make_run_dir,
    new_run_id,
    save_step_log,
)
from utils.schemas import RunReport


class MultimodalPipeline:
    """Wires the four agents together in the rubric-specified order."""

    def __init__(self, stub: Optional[bool] = None):
        stub = config.using_stub() if stub is None else stub
        self.vision = VisionAgent(stub=stub)
        self.prompt = PromptAgent(stub=stub)
        # Pass backend explicitly so an external STUB_MODE env override (e.g.
        # from test_pipeline.py) cannot silently downgrade a stub=False pipeline.
        self.generation = GenerationAgent(backend="stub" if stub else config.GENERATION_BACKEND)
        self.critique = CritiqueAgent(stub=stub)
        self.stub = stub

    # ------------------------------------------------------------------
    def run(
        self,
        image_path: str | Path,
        instruction: str,
        mode: str = "variation",
        vqa_questions: Optional[list[str]] = None,
        run_name: Optional[str] = None,
    ) -> RunReport:
        run_id = run_name or new_run_id()
        run_dir = make_run_dir(config.OUTPUTS_ROOT, run_id)

        # Copy the input image so the output folder is self-contained.
        input_copy = run_dir / f"input_{Path(image_path).name}"
        try:
            shutil.copy(image_path, input_copy)
        except Exception:
            input_copy = Path(image_path)

        report = RunReport(
            run_id=run_id,
            image_path=str(input_copy),
            instruction=instruction,
            mode=mode,
        )

        # --- step 1: vision ------------------------------------------
        try:
            report.vision = self.vision.run(input_copy, vqa_questions=vqa_questions)
            save_step_log(run_dir, 1, "vision", report.vision)
        except LowQualityImageError as exc:
            report.errors.append(f"vision: {exc}")
            report.save(run_dir / "report.json")
            return report
        except Exception as exc:
            report.errors.append(f"vision unexpected: {exc}")
            report.save(run_dir / "report.json")
            return report

        # If mode is "none" (pure captioning/VQA use case), stop after vision.
        if mode == "none":
            report.save(run_dir / "report.json")
            return report

        # --- step 2: prompt ------------------------------------------
        try:
            report.refined_prompt = self.prompt.run(
                instruction, report.vision, mode=mode
            )
            save_step_log(run_dir, 2, "prompt", report.refined_prompt)
        except AmbiguousPromptError as exc:
            report.errors.append(f"prompt ambiguous: {exc}")
            report.save(run_dir / "report.json")
            return report
        except Exception as exc:
            report.errors.append(f"prompt unexpected: {exc}")
            report.save(run_dir / "report.json")
            return report

        # --- step 3: generation --------------------------------------
        out_img_path = run_dir / "generated.png"
        try:
            report.generation = self.generation.run(
                refined=report.refined_prompt,
                input_image_path=input_copy,
                output_path=out_img_path,
            )
            save_step_log(run_dir, 3, "generation", report.generation)
            if report.generation.error:
                report.errors.append(f"generation: {report.generation.error}")
        except GenerationError as exc:
            report.errors.append(f"generation: {exc}")
        except Exception as exc:
            report.errors.append(f"generation unexpected: {exc}")

        # --- step 4: critique (always runs, even on generation failure)
        try:
            report.critique = self.critique.run(
                input_image_path=input_copy,
                generation=report.generation,
                refined=report.refined_prompt,
            )
            save_step_log(run_dir, 4, "critique", report.critique)
        except Exception as exc:
            report.errors.append(f"critique unexpected: {exc}")

        # Final full report
        report.save(run_dir / "report.json")
        return report
