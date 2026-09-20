# statcheck-ml — project plan

Single source of truth. Update the Status column as work lands. Do not track progress
anywhere else.

Version 2, 2026-09-19. Version 1 reached a working cascade but its process was not
reproducible. `docs/superpowers/specs/2026-09-19-reproducible-pipeline-design.md`
holds the approved design for version 2. `CONTEXT.md` holds the reasons.

## Goal

Replace the regex extraction of statcheck with a learned extractor, keep the p-value
recomputation exact, and ship the result as ONNX. The tool runs in three places: a
browser, R, and Python.

Version 2 adds one requirement: every number in the final report comes from a script
that reads a committed file.

## Non-goal

Machine learning does not decide whether a result is inconsistent. The model only
finds statistical results in text. The comparison stays deterministic.

## Pipeline — three stages

| Stage | Method | Purpose |
|---|---|---|
| 0. Prefilter | regex, deliberately loose | select candidate windows of text |
| 1. Extract | learned token classifier, exported to ONNX | find spans and group them into results |
| 2. Check | closed-form math | recompute p, compare, classify the error |

### Stage 0 — the prefilter, and its risk

Statistical sentences carry an unusual density of non-letter characters. A cheap
filter can therefore drop most of a document before the model runs. This matters most
in the browser, where running the model over a whole paper is the main performance
mistake.

The risk is that the prefilter becomes the new bottleneck. Text the filter discards
can never be recovered by the model, so the filter sets the recall ceiling of the
whole system. Therefore the prefilter is tuned for recall, not precision. Prefilter
recall on the holdout is a gate. Below the agreed threshold, the system is reported as
prefilter-limited and model tuning stops.

### The processing unit is a window, not a sentence

One result can span a sentence boundary. Tables split results further apart. So the
prefilter selects overlapping windows and expands each one by context lines on both
sides. The model tags across the window.

## Three ports — one spec

The prefilter, the character vocabulary, the normalisation, the repair rules, and the
p-value math behave identically in Python, JavaScript, and R. The rules live in shared
JSON files under `src/statcheck_ml/spec/`, and each port reads them. A port holds
only the code that applies the spec.

This drives the model choice. A character model needs a character map, which is
trivial in all three languages. A transformer needs a tokenizer ported three times.
Version 2 therefore drops the transformer baselines entirely.

## The hybrid cascade

On the holdout the real statcheck package reaches a precision near 1.0 and a recall
near 0.19. A learned extractor should not replace it. It should follow it.

| Step | Component | Why |
|---|---|---|
| 1 | prefilter | 1 line in 700 holds a result, so this makes the rest affordable |
| 2 | statcheck, with operator repair | accept everything it finds, because its precision is near 1.0 |
| 3 | the model | read the same candidates, add only what statcheck did not find |
| 4 | the mathematics | recompute and compare, the same code for both sources |

The cascade is scored beside statcheck alone and the model alone, on the holdout,
with confidence intervals and a paired test.

## Model zoo

Every candidate is character-level, so every candidate ports to all three runtimes.

| Family | Heads | Status |
|---|---|---|
| char-BiLSTM | softmax, CRF | exists |
| char-BiGRU | softmax, CRF | exists |
| char-CNN, dilated | softmax, CRF | version 2 |

Six configurations are screened with seed 0. The top three by dev F1 train again with
seeds 1 and 2. The three shipped ONNX files are the seed-0 exports of those three.
Accuracy decides which is recommended. Size and latency are reported, not constrained.

Training runs on local CPU. No GPU is assumed. Every training script checkpoints and
resumes.

## Label tiers

| Tier | Source | Allowed use |
|---|---|---|
| Bronze | three rater agents, majority vote, scripted adjudication | training, development, and the holdout |
| Gold | human labels, supplied by the project owner | held-out test only, when it exists |

There is no human gold set today. Every reported number is measured against bronze
labels, and the report says so. Annotations produced by a language model are never
reported as gold.

The silver tier of version 1 is retired. The regex has recall near 0.2, so a window it
labels holds about four unlabeled true results for each labeled one. That is label
noise, not supervision.

## Adversarial and noise layer

Applied to training spans only, so the model generalises past the text it has seen.

| Family | Content |
|---|---|
| OCR corruption | confusable characters, ligatures, spacing lost in PDF extraction |
| Unicode variance | Greek letter versus spelled name, superscript versus plain digit, dash forms |
| Operator damage | the operator replaced by a control character, as the PDF conversion does |
| Format perturbation | prose phrasing, a result split across a line break, a table row |
| Hard negatives | text that resembles a result and is not, such as a page range or a citation year |

## Phases, version 2

Status values: `blocked`, `ready`, `active`, `done`.

| M | # | Phase | Agent | Model | Status |
|---|---|---|---|---|---|
| M1 | 1 | Guideline v2, schema synced with `labels.py`, adjudication rules | label-architect | opus | done |
| M1 | 2 | Layout: `pipeline/` stages, `reproduce.sh`, pinned requirements, transformer baselines removed | data-engineer | haiku | done |
| M1 | 3 | Frozen windows, manifest, chunk and collect scripts with provenance | data-engineer | haiku | done |
| M1 | 4 | Agreement module: κ, α, span F1, bootstrap, unit tests | eval-engineer | sonnet | done |
| M1 | 5 | char-CNN, grid runner with seeds and parallel runs, export parity | ml-trainer | sonnet | done |
| M2 | 6 | Annotate every window three times, blind | annotator-haiku, -sonnet, -opus | haiku, sonnet, opus | done |
| M2 | 7 | Collect, agreement report, adjudicate disputes, final labels | data-engineer, adjudicator | haiku, opus | done |
| M2 | 8 | Dataset, splits, alignment gate | data-engineer | haiku | done |
| M3 | 9 | Train the grid: 6 screens, 3 × 3 seeds, 1 ablation | ml-trainer | sonnet | done |
| M3 | 10 | Export the zoo, parity, size, latency, quantisation delta | ml-trainer | sonnet | done |
| M4 | 11 | Evaluate on the holdout: systems, subsets, bootstrap, paired tests, McNemar | eval-engineer | sonnet | done |
| M4 | 12 | R baseline regenerated, engine gate, JS and R parity tests green | regex-porter | sonnet | done |
| M4 | 13 | Figures | eval-engineer | sonnet | done |
| M4 | 14 | Report template and generated report, README, PROTOCOL, CONTEXT | doc-writer | haiku | done |
| M4 | 15 | Final review: reproduce.sh from clean checkout, hashes match | manager | fable | done |

All fifteen phases are done (2026-09-20). A clean clone reproduced `results/eval.json`,
every figure, and `results/REPORT.md` byte for byte, apart from the commit id the
report names. Open work: the p-value core adopts statcheck's statistic-rounding rule
(section 7 of the report); a human gold set, when the owner supplies one.

## Version 1 phases, for the record

Version 1 delivered: the corpus conversion with PyMuPDF, the ported regex (74 %
parity), the prefilter spec and its three ports, the normalisation and repair specs,
the p-value core in Python, the augmentation layer, two annotation passes, the
BiLSTM and BiGRU models with CRF, the cascade, and the JS and R normalisation ports
with parity tests. All of that code is kept. Only the process around it changes.

## Phases, version 3 — the ports

Plan: `docs/superpowers/plans/2026-09-20-ports.md`. One agent per task; the manager
accepts on evidence.

| Track | # | Task | Agent | Model | Status |
|---|---|---|---|---|---|
| A | A1 | `weights.json` per zoo model, numpy reference forward pass, 1e-3 parity | ml-trainer | sonnet | done |
| A | A2 | Parity cases for nine stages, Python self-test | eval-engineer | sonnet | done |
| A | A3 | The kit: spec, model, parity, manifest | runtime-engineer | sonnet | done |
| A | A4 | Python package: console script, packaged model, ONNX-only install | runtime-engineer | sonnet | done |
| A | A5 | Layout, CONTEXT, PLAN, CLAUDE | manager | fable | done |
| A | A6 | Python package proof: wheel, twine, CI matrix on three OS × four Pythons, install from the repository address | runtime-engineer | sonnet | done |
| B | B1 | R package skeleton, kit, normalise, extract | runtime-engineer | sonnet | active |
| B | B2 | R model forward pass and Viterbi | stats-core | opus | blocked on B1 |
| B | B3 | R prefilter, repair, grouping, p-value, pipeline | runtime-engineer | sonnet | blocked on B2 |
| B | B4 | R PDF input, docs, CI | runtime-engineer, doc-writer | sonnet, haiku | blocked on B3 |
| C | C1 | Web package skeleton, kit, normalise, extract | runtime-engineer | sonnet | active |
| C | C2 | Web p-value core | stats-core | opus | blocked on C1 |
| C | C3 | Web model, grouping, prefilter, repair, pipeline | runtime-engineer | sonnet | blocked on C2 |
| C | C4 | Web PDF.js, demo page, Pages, docs | runtime-engineer, doc-writer | sonnet, haiku | blocked on C3 |
| D | D1 | Retire the copies in the mother repository | manager | fable | blocked on B4, C4 |
| D | D2 | Report section on the ports, reproduction check | manager | fable | blocked on D1 |
| D | D3 | Push the port repositories, CI green | manager | fable | blocked on the owner |
| D | D4 | Tutorials for Python, R and the web, from executed examples | runtime-engineer | sonnet | blocked on B4, C4, A6 |

## Waiting on the owner

- The human-labeled gold set, whenever it is ready.
- Two empty private repositories, `statcheck-ml-r` and `statcheck-ml-web`, and the
  `git push` deny rule lifted in `.claude/settings.json` for the duration of track D.
