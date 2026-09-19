# statcheck-ml v2 — reproducible annotation, benchmark, and report

Date: 2026-09-19. Status: approved by the owner.

## Context

The first version of the project reached a useful result: a small character model
plus the statcheck regex finds 0.937 of the results in the holdout, against 0.187
for the regex alone. The process that produced that number is not reproducible.

- Annotation ran as two agent passes inside Claude Code. Batches were made by hand.
  No annotation record carries the rater, the model, the guideline version, or a
  timestamp. Adjudication was manual and left no script and no agent file.
- Agreement was a raw overlap and a Jaccard index. No chance-corrected metric, no
  per-field agreement, no confidence interval.
- The schema document does not match the implemented tag set.
- The train/dev/test split is recomputed from a hash and never written down.
- Every evaluation result, checkpoint, and holdout file is ignored by git. The
  numbers in `REPORT.md` are typed by hand. There are no figures.
- Two transformer baselines (MobileBERT, DistilBERT) exist without checkpoints and
  with an undeclared dependency. They cannot ship in R and are not the product.

This design replaces the process. The goal is that a third party can rerun every
stage after the annotation from the committed files, and can rerun the annotation
itself from a documented protocol with full provenance.

## Goals

1. Every number in the final report is produced by a script from a committed file.
2. Three independent raters, chance-corrected agreement, scripted adjudication.
3. A benchmark of lightweight character models with seeds, confidence intervals,
   and paired tests. No transformer baseline.
4. One numbered pipeline, one `reproduce.sh`, pinned dependencies.
5. A scientific report in Markdown with tables and figures, generated from JSON.

## Non-goals

- No human gold set. Labels stay bronze. The report states this limit.
- No change to the p-value mathematics, the prefilter spec, or the JS and R ports.
- No silver tier. See "Retired: silver labels".

## 1. Annotation protocol v2

### Inputs, frozen

- `dataset/windows/train.json`: the 1955 windows of `data/sample_round2/windows.json`.
- `dataset/windows/holdout.json`: the 576 windows of `data/holdout/windows.json`.
- `dataset/windows/key.json`: pool, journal, document, and line for every window.
  Raters never see this file.
- `dataset/MANIFEST.json`: sha256 of every committed dataset file, the command
  that produced it, the seed, and the timestamp.

The same windows as the previous rounds, so the new numbers are comparable.

### Guideline

`docs/GUIDELINE.md`, version 2.0. It is the written artifact the rater agents
embed. Its sha256 is recorded on every annotation record. Content comes from the
present `result-annotator` prompt: the task, the damaged-operator examples, the
mark / do-not-mark table, the output contract.

### Raters

Three agents, `annotator-haiku`, `annotator-sonnet`, `annotator-opus`, in
`.claude/agents/`. Same body, only `model:` differs. Tools: `[Read, Write]`. A
rater reads one batch file and writes one output file. The prompt tells it to read
nothing else. This limit of blindness is recorded in `CONTEXT.md`.

Different model families give raters that do not share one training run, and give
one extra finding: which cheap model annotates best against the final labels.

### Batches and collection

- `pipeline/02_chunk.py`: 20 windows per batch, same order for every rater, 127
  batches. Writes `data/annotation/<set>/batches/batch_NNN.json`.
- The manager dispatches one agent per batch per rater. The output path is
  `data/annotation/<set>/<rater>/batch_NNN.json`.
- `pipeline/03_collect.py`: validates each output (parses, exactly N objects, ids
  match the batch, every result has the nine keys, quote aligns to the window text
  when whitespace is ignored). Stamps `{rater, model_alias, model_id,
  guideline_sha, agent_sha, batch_id, collected_at, retries}`. Lists failed batches
  for re-dispatch. Merges into `dataset/annotations/<set>/<rater>.json`.

## 2. Agreement

`pipeline/04_agree.py` reads the three rater files and writes
`dataset/agreement/<set>.json`.

| Level | Unit | Metric |
|---|---|---|
| Window | `contains_result` | raw %, pairwise Cohen κ, Fleiss κ, Krippendorff α nominal |
| Result | aligned char span | strict pairwise span F1 (exact span), lenient (≥ 50 % overlap), mean pairwise, per-rater F1 against final |
| Field | results all three found | Krippendorff α nominal for `test_type`, `p_operator`, `damaged`; exact-match % for `statistic`, `df1`, `df2`, `n`, `p_value` as normalised numbers |

Every metric carries a bootstrap 95 % interval over windows, 1000 resamples,
seed 0. Fleiss κ and Cohen κ are implemented in `src/statcheck_ml/agreement.py`
with unit tests against published examples. Krippendorff α uses the `krippendorff`
package, pinned.

## 3. Adjudication

`pipeline/05_adjudicate.py`:

1. Consensus: a result is kept when at least two raters found it (lenient match).
   Each field takes the majority value.
2. Disputes: a result found by one rater, or a field with three different values.
   Batched, 20 per file, each with the window text and all three rater outputs.
3. The `adjudicator` agent (opus, `[Read, Write]`) returns `{keep, fields, reason}`
   per dispute. Collected and stamped like annotations.
4. Merge into `dataset/annotations/<set>/final.json`. Every result carries `tier`:
   `unanimous`, `majority`, or `adjudicated`. All are bronze.

## 4. Dataset and splits

`pipeline/06_dataset.py` writes `dataset/train.jsonl`, `dataset/holdout.jsonl`,
and `dataset/splits.json` (document → `train` or `dev`, dev share 0.15, seeded
shuffle by document). Holdout documents are disjoint from training documents; the
script asserts it. The alignment gate is enforced: exit 1 when fewer than 98 % of
quotes align.

`docs/SCHEMA.md` is rewritten to match `src/statcheck_ml/labels.py`: 9 entities,
37 BIOES tags, operator folded into the entity, no block layer. It states why the
block layer was dropped.

### Retired: silver labels

The regex has recall near 0.2. A window it labels holds, on average, four unlabeled
true results for each labeled one. Training on that teaches the model to miss what
the regex misses. Silver labels are therefore not built. `to_bioes.py` is deleted;
`data.py` is the one path from rows to tags.

## 5. Model benchmark

All models are character-level and read `spec/charmap.json`, so all port to JS and
R. No transformer.

| Family | Head | Note |
|---|---|---|
| BiLSTM | softmax, CRF | existing |
| BiGRU | softmax, CRF | existing |
| char-CNN | softmax, CRF | new: embed 64 → four Conv1d(k=5, 128 ch, dilation 1, 2, 4, 8) → linear |

Protocol:

1. Seed 0 trains all six configurations with augmentation on. Selection on dev F1.
2. The top three by dev F1 train again with seeds 1 and 2.
3. One ablation: the best configuration without augmentation, seed 0.
4. 13 runs, about 40 min each on CPU, three in parallel with four threads each.

`pipeline/07_train.py --grid pipeline/grid.json --parallel 3` is resumable and
writes `models/runs.json`: configuration, seed, torch version, git sha, wall time,
best epoch, dev F1. `models/runs.json` is committed.

`pipeline/08_export.py` exports each run to ONNX (opset 17), checks parity against
torch (tag agreement 100 %, max logit difference below 1e-4 on dev), records size
and median CPU latency over 100 windows, and reports the int8 dynamic quantisation
delta. The seed-0 ONNX of the top three is committed under `models/zoo/`.

Removed: `train_bert.py`, the MobileBERT and DistilBERT rows in every document and
agent file, every reference to `transformers`.

## 6. Evaluation and statistics

`pipeline/09_evaluate.py` writes `results/eval.json`. Holdout only: 576 windows,
315 results, evaluated once per final model.

Systems: statcheck-R raw; statcheck-R with repair; each of the top three models,
three seeds; the cascade with the best model.

Subsets: damaged and undamaged operator; per test type; per damage family;
checkable results; verdict agreement with R statcheck on checkable results.

Statistics:

- Bootstrap over holdout documents, 2000 resamples, seed 0 → 95 % percentile
  interval for precision, recall, F1.
- Paired bootstrap on ΔF1: top 1 vs top 2, top 1 vs top 3, cascade vs
  statcheck-with-repair.
- McNemar exact test on per-result hit or miss: cascade vs statcheck-with-repair.
- Mean and standard deviation over the three seeds.
- Wilson interval for recall in each damage family.

`bench_engines.py` stays as the PDF-engine gate (spread ≤ 0.06). Its table goes in
the report.

R: install R with `winget install RProject.R`, install `statcheck`, regenerate the
baseline CSV, run `tests/parity.R` and `tests/smoke.R`. If the install fails, the
existing CSV is committed as a frozen baseline and the report says so.

## 7. Figures and report

`pipeline/10_figures.py` (matplotlib, Agg) writes `results/figures/`:

1. rater agreement heatmap and per-rater F1 against final
2. benchmark F1 with 95 % interval, seed spread for the top three
3. recall by damage family with Wilson intervals: statcheck, best model, cascade
4. precision and recall per system
5. size, latency, and F1 for the zoo
6. dev learning curves for the top three

`pipeline/11_report.py` fills `docs/report_template.md` (`{{key}}` placeholders,
prose in ASD-STE100) from `results/eval.json` and the agreement files, and writes
`results/REPORT.md`. No number in the report is typed by hand.

## 8. Layout

```
statcheck-ml/
  pipeline/            00_convert.py … 11_report.py, grid.json
  src/statcheck_ml/    library only
  spec/ js/ r/ tests/  unchanged; tests gain pytest for agreement, splits, CNN export
  dataset/             windows/ annotations/ agreement/ splits.json train.jsonl holdout.jsonl MANIFEST.json
  models/              zoo/ (three ONNX) runs.json
  results/             eval.json figures/ REPORT.md
  docs/                GUIDELINE.md SCHEMA.md PROTOCOL.md CORPUS.md report_template.md
  reproduce.sh         stages 04, 06, 09, 10, 11 offline; --train adds 07, 08
  requirements.txt     pinned
```

Deleted: `EXPERIMENTS.md`, `TRAINING_PLAN.md`, `REPORT.md`, `SAMPLING.md`,
`evaluate.py`, `to_bioes.py`, `train_bert.py`, the root scripts (moved into
`pipeline/`).

## 9. Team

| Role | Agent | Model | Reason |
|---|---|---|---|
| Manager: dispatch, review, accept or reject, scientific interpretation | session | fable | — |
| Guideline v2, schema sync, adjudication rules | label-architect | opus | a silent mistake propagates |
| Raters | annotator-haiku, -sonnet, -opus | haiku, sonnet, opus | independent families |
| Dispute judge | adjudicator | opus | judgment |
| Chunk, collect, dataset, manifest, reproduce.sh, requirements, layout move | data-engineer | haiku | mechanical |
| Agreement metrics, evaluation statistics, figures | eval-engineer | sonnet | bounded engineering, the mathematics must be right |
| char-CNN, grid runner, export parity | ml-trainer | sonnet | bounded engineering |
| Report template, README, PROTOCOL, PLAN and CONTEXT updates | doc-writer | haiku | STE prose, numbers from JSON |
| Code review before acceptance | cavecrew-reviewer | sonnet | cheap second eye |

### Definition of done, every deliverable

1. Runs from the pinned venv with the documented command and exits 0.
2. Deterministic: the agent shows two runs with the same output hash.
3. New code has unit tests in `tests/`; `python -m pytest tests` passes.
4. No hand-typed number in any Markdown file.
5. One stage per script, about 200 lines or fewer, no spec rule restated.
6. The report to the manager holds: the command, the output tail, the files
   produced, and the checklist above with evidence. "Should work" is rejected.

### Manager review

The manager reruns the command, spot-checks the output, runs the reviewer on the
diff, and returns specific defects. After two rework rounds the manager splits the
task or fixes it.

## 10. Order

1. Foundation, in parallel: guideline and schema; layout move and chunk/collect;
   char-CNN and grid runner; agreement module and tests.
2. Annotation: 381 dispatches, ten in parallel → collect → agree → adjudicate →
   dataset and splits.
3. Training grid in the background (~4 h) → export → evaluate → figures.
4. Report → docs → PLAN, CONTEXT, README → review → commits.

## Verification

- `python -m pytest tests` passes.
- `node tests/parity.mjs` and `node tests/smoke.mjs` pass. `Rscript tests/parity.R`
  passes when R is installed.
- `bash reproduce.sh` regenerates `results/eval.json`, every figure, and
  `results/REPORT.md` from committed files, with the same hashes.
- `dataset/MANIFEST.json` hashes match the committed files.
- Every annotation record carries all provenance fields.
- `grep -ri bert` over the repository returns nothing.
