# Three ports of statcheck-ml — design

Date: 2026-09-20. Owner decisions: GPL-3 for all three repositories; names
`statcheck-ml-r` and `statcheck-ml-web` under `github.com/rasoulnorouzi`; private until
parity passes. This repository (`ml-statcheck`) stays the mother: specification, models,
dataset, evaluation, report, and the Python package.

## Goal

A user runs one command in Python, one function in R, or one drop on a web page, and
gets the checked results of a PDF. Every port applies the same rules, from the same
files, and proves it with the same committed cases.

## What exists and what is missing

| Stage | Python (reference) | R today | Web today |
|---|---|---|---|
| extract text from PDF | `pipeline.py` (PyMuPDF, PDFium) | `run_statcheck.R` uses none | none |
| normalise | `normalize.py` | `r/normalize.R` | `js/normalize.js` |
| repair operators | `repair_validated.py` | none | none |
| prefilter, windows | `prefilter.py` | none | none |
| regex extraction | `extract.py` | `r/extract.R` | `js/extract.js` |
| model tagging | `onnx_runtime.py` (ONNX + Viterbi) | none | none |
| group tags into results | `pipeline.py` `find_with_model` | none | none |
| p-value check | `pvalue.py` (scipy) | none | none |
| cascade and report | `pipeline.py` `run_text` | none | none |
| parity suite | `tests/make_parity_cases.py` (normalise only) | `tests/parity.R` | `tests/parity.mjs` |

Missing in both ports: repair, prefilter, model, grouping, p-value, pipeline, PDF
input, packaging. The parity suite covers one stage of nine.

## Architecture

### The kit is the contract

`pipeline/08_port_kit.py` copies into a port repository:

```
kit/
  spec/prefilter.json  normalize.json  repair.json  charmap.json  font_table.json
  model/<config>/tagger.onnx  decoder.json  charmap.json  weights.json
  parity/cases.json
  manifest.json        version, git commit of the mother, sha256 of every file
```

`weights.json` is new: the raw tensors of the shipped model (embedding, each RNN
layer and direction, the output layer, the CRF transitions) as nested float lists with
their shapes and the PyTorch gate order written down. R runs the model from it. The
web runs `tagger.onnx` through onnxruntime-web. Both decode the CRF from
`decoder.json` with the same Viterbi as `onnx_runtime.py`.

A port verifies `manifest.json` when it loads and refuses a stale kit.

### Parity cases cover every stage

`tests/make_parity_cases.py` grows from one stage to seven. The Python reference
produces the expected value; every port must reproduce it. One JSON file,
`parity/cases.json`, holds all sections:

| Section | Input | Expected |
|---|---|---|
| normalise | text excerpt | normalised text (as today) |
| repair | text excerpt | repaired text, replacement count |
| prefilter | document text | kept window starts and ends |
| extract | window text | results from the regex |
| model | window text | tag sequence from the zoo model (ONNX, Python decode) |
| group | tag sequence + text | results built from the spans |
| pvalue | test, statistic, df1, df2, operator, p text | computed p, verdict |
| pipeline | document text | result list of `run_text` (statistic, p, verdict, source) |

Hand-written edge cases join the corpus excerpts, as today. The model section stores
tags, not logits, and a separate `model_logits` section stores the logits of three
windows to 1e-4, so a wrong gate order in R is caught where it happens.

### R port: `statcheckml`

An R package, CRAN layout. No Python, no ONNX runtime, no compiled code.

- Text: `pdftools::pdf_text`, joined with form feeds as the Python engine does.
- Model: a pure R forward pass of the two-layer bidirectional GRU or LSTM from
  `weights.json`, batched over all windows of a document, one matrix product per
  time step per direction, so a document costs seconds, not minutes. Gate order and
  formulas follow PyTorch (`GRU`: r, z, n; `LSTM`: i, f, g, o). The dilated CNN is
  not ported: it is not in the zoo.
- CRF: Viterbi over `decoder.json`.
- p-value: `pt`, `pf`, `pchisq`, `pnorm` — base R, exact.
- API: `sc_check(pdf)`, `sc_check_text(text)`, returning a data frame with one row per
  result: source, test type, statistic, degrees of freedom, operator, reported p,
  computed p, verdict, line. `sc_kit_info()` prints the kit version and the model.
- Tests: testthat, one test per parity section, reading `inst/kit/parity/cases.json`.
- CI: GitHub Actions, `R CMD check` on Windows, macOS and Linux.

### Web port: `statcheck-ml-web`

An npm package of ES modules plus a static demo page.

- Text: PDF.js in the browser; the page reflows positioned items into lines as the
  normalise spec expects (the reflow rule is in `normalize.json`).
- Model: onnxruntime-web (WASM), `tagger.onnx` from the kit; Viterbi over
  `decoder.json` in JavaScript.
- p-value: own implementation of the regularised incomplete beta and gamma functions
  (continued fractions, Numerical Recipes style), tested against the parity table to
  1e-9 relative. This is the one numerically delicate piece and it goes to the opus
  agent.
- API: `checkText(text, kit)`, `checkPdf(file, kit)`, returning the same record shape
  as R and Python. The demo page: drop a PDF, see a table, download JSON. Deployed to
  GitHub Pages from `main` by Actions.
- Tests: vitest (Node), one test per parity section; a browser smoke test is manual.
- CI: GitHub Actions, Node 20 and 24.

### Python: the package in the mother repository

`statcheck_ml` becomes installable and usable without torch: `pip install .` gives
`statcheck-ml check paper.pdf [--json]`, ONNX inference only. torch stays an extra
for training. `pyproject.toml` gains the console script and the extras; the README
gains a three-line usage.

### Retiring the copies

When a port passes all parity sections, its old copy in the mother repository
(`statcheck-ml/js/`, `statcheck-ml/r/`) and the two runners (`tests/parity.mjs`,
`tests/parity.R`) are deleted. The rule "one spec, three thin ports" then has one
place per port. `CLAUDE.md` and `README.md` change with it. `r/run_statcheck.R` stays:
it is the R baseline runner for the evaluation, not a port.

## Repositories on disk

The two port repositories are nested under `ports/` in this checkout,
`ports/statcheck-ml-r` and `ports/statcheck-ml-web`, each its own git repository with
its own remote. `ports/` is gitignored in the mother. Reason: every agent tool call
stays inside this project directory, where the permission rules allow it. On another
machine: clone the mother, then clone the two ports into `ports/`.

## Team and models

| Work | Agent | Model | Why |
|---|---|---|---|
| kit: `weights.json` export, manifest, parity generator for seven sections | ml-trainer / eval-engineer | sonnet | bounded; the reference values come from existing Python |
| R model forward pass and Viterbi | stats-core | opus | a wrong gate order is silent; parity to 1e-4 |
| web p-value core (incomplete beta, gamma) | stats-core | opus | numerics; a silent error changes verdicts |
| R and web: prefilter, repair, extract, grouping, pipeline, PDF input | runtime-engineer | sonnet | rule-driven from the spec JSON, parity-tested |
| packaging: R package layout, npm package, demo page, Actions, Python CLI | runtime-engineer | sonnet | conventional |
| READMEs, LICENSE, CHANGELOG, kit docs | doc-writer | haiku | mechanical |
| review, acceptance, parity sign-off, the report update | manager | fable | evidence-based acceptance |

One agent per port at a time. The two ports advance in parallel only when the token
budget allows; otherwise R first, then web.

## Acceptance

A port is accepted when: every parity section passes in its own CI; the kit manifest
verifies; `sc_check` or `checkPdf` on `examples/sample_paper_damaged.pdf` returns the
same results as the Python pipeline; the package installs from a clean clone. The
mother repository's report gains one section: the three ports, their parity status,
and a timing per document.

## Out of scope

Retraining, new labels, CRAN or npm publication (the repositories are made ready for
it; the owner publishes), the CNN family in the ports, PDF engines other than the one
named per port.
