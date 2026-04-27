"""Entry point: run the three required use cases end-to-end.

  1. Captioning + VQA                (mode="none")
  2. Style-guided transformation     (mode="stylize")
  3. Prompt-based enhancement        (mode="enhance")

Usage:
  python run_demo.py                 # uses default sample images
  python run_demo.py --use-case 2    # run only the style-transfer case
  STUB_MODE=0 python run_demo.py     # use real APIs (needs keys)
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import config
from data.sample_images import ensure_sample_images
from orchestration import MultimodalPipeline
from utils.logger import section
from utils.schemas import RunReport


USE_CASES = [
    {
        "name": "uc1_captioning_vqa",
        "description": "Image captioning and visual questions",
        "image_key": "street_scene",
        "instruction": "describe this image",
        "mode": "none",
        "vqa_questions": [
            "What is the dominant color palette of this image?",
            "What is the main subject and what is it doing?",
            "What time of day does this appear to be?",
        ],
    },
    {
        "name": "uc2_style_transfer",
        "description": "Style-guided image transformation (watercolor)",
        "image_key": "portrait",
        "instruction": "make this look like a watercolor painting with soft colors",
        "mode": "stylize",
        "vqa_questions": None,
    },
    {
        "name": "uc3_enhance",
        "description": "Prompt-based enhancement (golden hour + sharpen)",
        "image_key": "landscape",
        "instruction": "sharpen the image and warm it up to look like golden hour",
        "mode": "enhance",
        "vqa_questions": None,
    },
]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--use-case",
        type=int,
        choices=[0, 1, 2, 3],
        default=0,
        help="0 = run all (default), 1-3 = specific use case",
    )
    args = parser.parse_args()

    stub = config.using_stub()
    print(f"\n\033[1m[config]\033[0m  STUB_MODE={stub}  |  MODEL={config.ANTHROPIC_MODEL}  |  GENERATION_BACKEND={config.GENERATION_BACKEND}")
    if stub:
        print("  \033[93m⚠  Running in STUB mode — no API calls will be made.\033[0m")
        print("  \033[93m   To use real models: source .env && python run_demo.py\033[0m")

    image_paths = ensure_sample_images(config.INPUT_IMAGES_DIR)
    pipeline = MultimodalPipeline()

    to_run = USE_CASES if args.use_case == 0 else [USE_CASES[args.use_case - 1]]

    reports: list[RunReport] = []
    for uc in to_run:
        section(f"USE CASE {to_run.index(uc) + 1}/{len(to_run)}: {uc['description'].upper()}")
        print(f"  \033[1mimage:\033[0m       {image_paths[uc['image_key']]}")
        print(f"  \033[1minstruction:\033[0m {uc['instruction']}")
        print(f"  \033[1mmode:\033[0m        {uc['mode']}")
        print()

        report = pipeline.run(
            image_path=image_paths[uc["image_key"]],
            instruction=uc["instruction"],
            mode=uc["mode"],
            vqa_questions=uc["vqa_questions"],
            run_name=uc["name"],
        )
        reports.append(report)

        _print_report_summary(report)

    # Also write a combined summary for the human eval sheet.
    summary = [r.to_dict() for r in reports]
    summary_path = config.OUTPUTS_ROOT / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, default=str))

    section("ALL RUNS COMPLETE")
    print(f"  Combined summary  →  {summary_path}")
    print(f"  Per-run artifacts →  {config.OUTPUTS_ROOT}/<use_case>/")
    print(f"  Step logs         →  step_1_vision.json … step_4_critique.json")
    print()


def _print_report_summary(report: RunReport) -> None:
    print("\n--- Report summary ---")
    if report.vision:
        print(f"Caption:      {report.vision.caption}")
        print(f"Objects:      {', '.join(report.vision.objects[:5])}")
        print(f"VQA pairs:    {len(report.vision.vqa_pairs)}")
        for vp in report.vision.vqa_pairs[:2]:
            print(f"  Q: {vp['question']}")
            print(f"  A: {vp['answer']}")
    if report.refined_prompt:
        print(f"Refined:      {report.refined_prompt.refined_prompt[:120]}...")
        print(f"Preserved:    {report.refined_prompt.preserved_transform}")
    if report.generation:
        cfg = report.generation.config
        print(
            f"Generated:    {report.generation.image_path} "
            f"(model={cfg.model}, mode={cfg.mode}, steps={cfg.inference_steps})"
        )
        if report.generation.error:
            print(f"  ERROR: {report.generation.error}")
    if report.critique:
        c = report.critique
        print(
            f"Critique:     verdict={c.verdict} "
            f"(img-sim={c.clip_similarity_image:.3f}, "
            f"txt-sim={c.clip_similarity_text:.3f}, "
            f"quality={c.quality:.2f})"
        )
        print(f"  Rationale:  {c.rationale[:160]}...")
        if c.revision_suggestion:
            print(f"  Revision:   {c.revision_suggestion}")
    if report.errors:
        print(f"Errors:       {report.errors}")


if __name__ == "__main__":
    main()
