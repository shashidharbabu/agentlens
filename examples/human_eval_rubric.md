# Human Evaluation Rubric

Used to manually score at least 3–5 runs from the pipeline (rubric item 3.4).
Each run is scored across 5 criteria on a **1–5 scale** (1 = poor, 5 = excellent).

## Criteria

| # | Criterion | Question |
|---|---|---|
| C1 | **Input fidelity** | Does the output image meaningfully reflect the content of the input image? |
| C2 | **Instruction following** | Does the output reflect the user's original instruction? |
| C3 | **Transformation coherence** | Is the applied transformation visually coherent (not broken / warped / incomplete)? |
| C4 | **Technical quality** | Is the image technically acceptable (resolution, artifacts, composition)? |
| C5 | **Acceptance decision** | Would you accept this output as-is, or request a revision? (1 = reject, 5 = accept as-is) |

## Scoring scale

- **5** — Excellent. No issues.
- **4** — Good. Minor issues that don't hurt usability.
- **3** — Acceptable. Some issues but the result is usable.
- **2** — Poor. Significant issues; likely needs rework.
- **1** — Unusable.

## Accept / Revise rule

- **Accept** if average(C1..C5) ≥ 3.5 **and** C5 ≥ 3.
- **Revise** otherwise. Note the weakest criterion and suggest a concrete fix.

## How to score

1. Run `python run_demo.py` — this produces 3 runs under `outputs/` (`uc1_captioning_vqa`, `uc2_style_transfer`, `uc3_enhance`).
2. (Optional) Add 2 more runs with your own images for a total of 5.
3. For each run, open the input image and the `generated.png` side-by-side and fill in the row in `human_eval_sheet.csv`.
4. Compare your human verdict to the agent's verdict in `report.json` (`critique.verdict`) — note any disagreements.

## Why this rubric

These five criteria are the minimum needed to cover the three dimensions the rubric specifies (visual relevance, prompt faithfulness, transformation quality) plus an explicit accept/revise decision. Keeping it to 5 integer scores per run makes it fast to do and trivial to aggregate.
