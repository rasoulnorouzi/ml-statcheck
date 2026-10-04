# Version 4 brief — the human gold set, and what to improve with it

Date: 2026-10-04. Branch: `v4-gold`. Status: development and testing only. Nothing is
pushed and nothing is published. The owner decides when to publish.

This brief is the input for the manager. It records what the owner asked for, what
was measured before the team started, and the limits the team works inside.

## What the owner supplied

`data/manual_references.xlsx`, ignored by git. Three human coders (MS, ME, MN) coded
409 papers of our corpus by hand, with the quote copied from the PDF.

- 2704 coded results in 185 papers.
- 224 papers marked "no result reported in the text". These measure false finds.
- The coders follow the statcheck scope. They skip a correlation without degrees of
  freedom, which our guideline marks. They also code CI, Wald, and `b` rows that our
  guideline does not mark.
- The owner says the human labels are not always right. 135 of 2704 rows hold a typed
  value that is not in the quote. Most are a correction of broken PDF spacing (the
  typed value is right). A few dozen are typing errors.

## What exists already, uncommitted, on this branch

| File | Content |
|---|---|
| `statcheck-ml/pipeline/14_gold.py` | Builds `dataset/gold/gold.jsonl` from the sheet and the PyMuPDF text |
| `statcheck-ml/pipeline/14_gold_eval.py` | Scores regex, model, cascade, and model-on-passages on gold; bootstrap over papers |
| `statcheck-ml/dataset/gold/gold.jsonl` | 409 papers, 2704 results, 2664 located in the text |
| `statcheck-ml/results/gold_eval.json` | The first gold scores |
| `.gitignore` | ignores the sheet |
| `statcheck-ml/data/clean/` | all 3100 PDFs converted with `pipeline/00_convert.py` (ignored) |

The environment: `.venv/` at the repository root (Python 3.12, pinned requirements,
`statcheck-ml` installed editable). R 4.5 with statcheck 1.5.0 in the user library. The
committed baseline used R 4.6.1 with statcheck 1.5.0. Node 24. Both ports are cloned
under `ports/`.

## Measured before the team started

All numbers below use the shipped model `gru-crf-s0`, the in-scope test types (t, F,
r, z, chi2, Q), and gold located in the text.

### Gold, the 228 papers with no bronze training window ("unseen")

| System | Recall | 95% CI | Damaged operator | Finds in 167 no-result papers |
|---|---|---|---|---|
| regex (Python port, repaired) | 0.464 | 0.303–0.603 | 0.454 | 2 |
| shipped cascade | 0.838 | 0.781–0.890 | 0.773 | 216 |
| model on merged passages | 0.842 | 0.784–0.896 | 0.766 | 209 |

On all 409 papers the prefilter keeps 0.949 of gold results. Recall by test for the
cascade on unseen papers: F 0.96, chi2 0.85, t 0.80, Q 0.70, z 0.56.

### Bronze holdout (unchanged, separate track)

Cascade precision 0.949, recall 0.870, F1 0.908. statcheck repaired: 0.993, 0.467, 0.636.

### Where bronze and gold meet

200 gold results fall inside a bronze window. The raters found 191 (95.5 %). The bronze
results with no gold row are mostly correlations without degrees of freedom.

### Faults found

1. `Pipeline.run_text` drops a result when an earlier result in the paper has the same
   statistic value (`seen` is keyed on the statistic alone). 124 gold rows share a
   statistic with another row in the same paper. The cost is about 1.8 points of recall.
2. The model reports bare numbers with no test name: about 180 such finds in papers with
   results, and most of the 304 finds in no-result papers.
3. `results/eval.json` names the bronze holdout `gold`. Rename it `bronze`.

### Window size, measured without training

| Unit | Recall | Text read | Finds in no-result papers |
|---|---|---|---|
| overlapping 250-character windows (shipped) | 0.862 | 28.1 M | 304 |
| the same windows merged into passages | 0.863 | 15.8 M | 287 |
| passages from 1000-character windows | 0.877 | 24.6 M | 530 |

A merged passage reads each character once and costs nothing. A longer context gains
1.5 points but the model, trained on short windows, then fires on tables.

### The noise layer, measured without training

`gru-crf-s0` (trained with noise) against `gru-crf-noaug-s0` (without), on gold:

| Damage in the quote | n | with noise | without |
|---|---|---|---|
| none | 1245 | 0.911 | 0.914 |
| control character | 495 | 0.715 | 0.719 |
| `¼` for `=` | 294 | 0.857 | 0.861 |
| operator missing | 287 | 0.892 | 0.913 |
| letter for operator | 209 | 0.928 | 0.957 |
| spaced digits | 93 | 0.516 | 0.527 |

Recall 0.857 with noise, 0.863 without. Finds in no-result papers 292 with noise, 216
without. One seed of the no-noise model exists, so the recall gap is within noise; the
false-find gap is not. Real damage is per paper: spaced digits come from 10 papers,
control characters from 47.

## The owner's decisions

1. No BERT, no DistilBERT, no transformer of any kind. Character models only.
2. Gold is a test set. Split it by paper: a frozen gold test, never trained on, and a
   gold-train part that may join training. The 49 papers already in the bronze holdout
   go to the gold test.
3. Annotation moves to a larger unit: the merged passage, at about paragraph level.
4. The coders act as a fourth vote beside the three LLM raters. Their labels are not
   treated as always right.
4a. The training set is a mix: the existing bronze training windows (1962 windows,
   1517 results) stay in it, and the new passages from gold-train papers join them.
   The bronze windows agreed with the coders on 191 of 200 shared results. The gold
   test and the gold-train dev papers are never in it.
5. Rebuild the noise layer: damage a whole paper with one consistent font map, apply
   the same damage to text that holds no result, add spaced digits and a missing
   operator, and keep a family only when gold shows it helps.
6. Benchmark against the original statcheck R package, run on the same text.
7. Report the gold test, the bronze holdout, and their combination, always apart.
8. Remove files and code that are unnecessary, broken, or useless. List every removal.
9. Keep the three packages (Python, R, web) working. Re-export the kits and run the
   port tests after any spec or model change.
10. Training runs on the local CPU. The owner has a GPU in a terminal and runs commands
    there on request, so a task may hand the owner exact commands instead.
11. No push, no release, no publication.

## Limits for the team

- Never delete a file that git does not track, unless the owner asks. Git cannot
  restore it. Task M1.1 deleted the owner's untracked notebook on 2026-10-04. The
  owner did not need it, but the rule stands: a removal applies to tracked files only,
  and each one is listed in the ledger.

- At most twenty agents run at the same time (the owner raised the limit from ten on
  2026-10-04). One workflow runs at most six agents on this 8-CPU machine. Work that
  waits on the API and not on the CPU, such as the M4 rater batches, may run as up to
  three workflows side by side. Plan M4 so that its batches can be split that way.
- The manager runs on fable and chooses the agent and the model for each task. The
  model in an agent's frontmatter is the cost decision of CLAUDE.md. Do not raise it
  because a task feels hard. Split the task.
- Every agent prompt is recorded in `docs/v4/prompts/`, and every accepted task in
  `docs/v4/LEDGER.md`, so the process can be read and repeated.
- Every number in a result file comes from a script that reads a committed file.
- Raters work blind, exactly as in phase 6. They receive the passage and the guideline,
  never the project, the gold labels, or the model output.
