# statcheck-ml

![Python package](https://github.com/rasoulnorouzi/ml-statcheck/actions/workflows/python.yml/badge.svg)

A machine-learned replacement for the extraction step of the R package
[`statcheck`](https://github.com/MicheleNuijten/statcheck).

`statcheck` does two jobs. It **finds** reported statistical results with regular
expressions, and it **recomputes** the p-value to check them. The mathematics is
correct. The regular expression fails when the PDF conversion damages the text,
and about half of the results in a real corpus are damaged that way.

This project replaces the first job with a small character model. It does not
touch the second.

> **Machine learning finds results. It never judges them.**
> No model output reaches a verdict. The comparison stays closed-form
> mathematics in every port.

---

<!-- readme_block:start -->
## Results

Holdout: unseen documents, bronze labels from three rater agents, 95 % bootstrap
intervals over documents. `statcheck_repaired` is the R package after the operator
repair. The cascade runs it first and adds what the model finds beyond it.

| System | P | R | F1 [CI] |
|---|---|---|---|
| statcheck_raw | 0.983 | 0.183 | 0.308 [0.222, 0.385] |
| statcheck_repaired | 0.993 | 0.467 | 0.636 [0.567, 0.698] |
| gru-crf-s0 | 0.952 | 0.861 | 0.904 [0.871, 0.934] |
| gru-softmax-s0 | 0.925 | 0.882 | 0.903 [0.868, 0.933] |
| lstm-softmax-s0 | 0.910 | 0.879 | 0.894 [0.851, 0.932] |
| cascade_gru-crf-s0 | 0.949 | 0.870 | 0.908 [0.875, 0.937] |

The cascade reaches holdout F1 0.908, against 0.636 for statcheck alone.
<!-- readme_block:end -->

Every result number in this file is copied from a generated file. None is typed
by hand. The full method, every definition, the agreement study, the model grid,
the paired tests and the figures are in
**[results/REPORT.md](statcheck-ml/results/REPORT.md)**.

---

## How it works

A PDF goes in, and checked results come out.

| Stage | What it does |
|---|---|
| 1. extract | A named PDF engine turns the PDF into text. |
| 2. normalize | Line width and damaged characters are made engine independent. |
| 3. repair | Arithmetic restores operators the conversion destroyed. |
| 4. prefilter | About 1 line in 700 holds a result. The rest is dropped. |
| 5. find | The regex reads what it can, and the model reads the rest. |
| 6. check | The p-value is recomputed. This stage is mathematics alone. |

Two design rules carry most of the weight:

- **The prefilter caps the recall of the whole system.** Text it discards can
  never be recovered. It is tuned for recall alone.
- **The processing unit is an overlapping window, not a sentence.** One result
  can cross a line break or a sentence boundary.

---

## How the labels were made

Version 2 rebuilt the annotation so that every number can be reproduced.

1. The windows are frozen. `dataset/MANIFEST.json` holds the SHA-256 of every
   file under `dataset/`, and `reproduce.sh` verifies it after it regenerates
   the agreement and the dataset files.
2. Three rater agents read each window blind: one on Haiku, one on Sonnet, one
   on Opus. Each reads only the guideline, `docs/GUIDELINE.md`, and one batch
   of twenty windows. Every record carries the rater, the model id, the guideline
   hash and the batch id.
3. `pipeline/04_agree.py` measures agreement before any label is merged: Cohen
   and Fleiss kappa, Krippendorff alpha, strict and lenient span F1, and a
   bootstrap interval over windows.
4. `pipeline/05_adjudicate.py` keeps a result when two of three raters found it.
   A result one rater alone found, or a field the raters disagree on, goes to an
   adjudicator agent on Opus. The audit trail is `dataset/annotations/*/disputes.json`.
5. The final labels are **bronze**. No human has checked them. There is no gold
   set. The report says so next to every number.

The silver tier of version 1 is retired. The regex finds about one result in
five, so a window it labels holds four unlabeled true results for each labeled
one. That is label noise, not supervision. See [CONTEXT.md](CONTEXT.md).

---

## Models

Every candidate is a character-level tagger, so every candidate ports to all
three runtimes. There is no transformer in version 2.

| Family | Heads |
|---|---|
| char-BiLSTM | softmax, CRF |
| char-BiGRU | softmax, CRF |
| char-CNN, dilated | softmax, CRF |

`pipeline/grid.json` defines the grid. Six configurations are screened with seed
0. The top three by development F1 train again with seeds 1 and 2, and one
ablation trains without augmentation. The three shipped ONNX files in
`models/zoo/` are the seed-0 exports of the top three. Accuracy decides which is
recommended. Size and latency are reported, not constrained.

---

## Set up on a new machine

```
git clone https://github.com/rasoulnorouzi/ml-statcheck.git
cd ml-statcheck
python -m venv .venv                       # Python 3.12
.venv/Scripts/pip install -r statcheck-ml/requirements.txt   # Linux, macOS: .venv/bin/pip
Rscript -e 'install.packages(c("statcheck", "jsonlite"))'    # R 4.6, for the baseline and the R port
node --version                             # Node 24, for the browser port and the parity suite
```

`reproduce.sh` looks for `../.venv/Scripts/python.exe`; on Linux or macOS run it
as `PY=../.venv/bin/python bash reproduce.sh`.

On Windows, `onnxruntime`'s own files can exceed 260 characters under a deep
venv path. Enable long paths, or create the venv at a short path. A fresh-venv
check hit this limit at a 150-character prefix.

Not in the repository, on purpose:

| Item | Why | Where it comes from |
|---|---|---|
| the article corpus (`02_pdfs.zip`, `statcheck-ml/data/`) | copyright | the owner |
| the torch checkpoints (`models/*/model.pt`) | 2 MB each and not needed: every run's ONNX is committed | `reproduce.sh --train` |
| the training logs | `report.json` holds the same numbers | `reproduce.sh --train` |
| the raw rater outputs (`data/annotation/`) | the collected, stamped files in `dataset/annotations/` are the record | `docs/PROTOCOL.md` |
| the example PDFs | generated | `python examples/make_sample_paper.py` |

Everything the report needs is committed, so `bash reproduce.sh` works without
the corpus. Only stages 00 to 03 and the engine gate need it.

The `.claude/` directory is committed too: the agent charters, the hooks and the
permission rules. A clone opened in Claude Code starts with the same team.
`CLAUDE.md` tells an agent to read `PLAN.md` and `CONTEXT.md` first.

---

## Reproduce

```
cd statcheck-ml
bash reproduce.sh            # agreement, dataset, evaluation, figures, report
bash reproduce.sh --train    # also retrain the grid and export the zoo
```

The script prints the SHA-256 of `results/eval.json`, `results/REPORT.md` and
every figure. Two runs on the same machine give the same hashes. Training is
deterministic on CPU with a fixed seed and thread count; a different machine can
give a different model, so the trained runs are committed.

`pipeline/README.md` lists every stage, its inputs and its outputs. The stages
that need the corpus or the rater agents are marked; the rest run from the
committed files.

---

## Three ports, one specification

The project ships from three ports: Python, R, and a browser. The R and web
ports live in their own repositories. This repository stays the source of truth
for the model, the shared rules, and the measurements.

Every rule that more than one port needs lives in one JSON file:

```
statcheck-ml/src/statcheck_ml/spec/
  prefilter.json     which passages can hold a result
  normalize.json     how to make the text engine independent
  repair.json        how to restore an operator the conversion destroyed
  charmap.json       the characters the shipped model reads
  font_table.json    the operator each damaged font produced
```

`charmap.json` is written by one path only: `pipeline/08_export.py --update-spec`
copies the charmap of the recommended model. No other script touches it.

A port also needs the model and the parity cases. `pipeline/08_port_kit.py`
writes all of it as one kit:

```
python pipeline/08_port_kit.py ../ports/statcheck-ml-r/inst/kit --name statcheck-ml-r
python pipeline/08_port_kit.py ../ports/statcheck-ml-web/kit --name statcheck-ml-web
```

The kit holds `spec/*.json`, the shipped model (ONNX for the web, raw weights
for R), `parity/cases.json` with 220 cases over nine stages, and a manifest
with the mother commit and a SHA-256 per file. A port verifies the manifest
when it loads, so a stale copy fails before it produces a wrong answer.

**Never restate a rule in a port.** A port reads the file and applies it, and
proves it against the parity cases in its own test suite.

| Port | Repository | Runs the model with |
|---|---|---|
| Python | this repository, `statcheck-ml/` | onnxruntime |
| R | [rasoulnorouzi-statcheck-ml-r](https://github.com/rasoulnorouzi/rasoulnorouzi-statcheck-ml-r) | pure R, from `weights.json` |
| Web | [rasoulnorouzi-statcheck-ml-web](https://github.com/rasoulnorouzi/rasoulnorouzi-statcheck-ml-web), demo at https://rasoulnorouzi.github.io/rasoulnorouzi-statcheck-ml-web/ | onnxruntime-web (WASM) |

### Each port needs a different PDF engine

No engine serves all three. R is the constraint, because `pdftools` over poppler
is its only maintained reader. `pipeline/12_engines.py` measures every engine
and reports the spread between the best and the worst. A port must not ship
while the spread is above 0.06. The measured spread is in
`results/engines.json` and in the report.

---

## Commands

```
node scripts/list-extensions.js      # every agent, skill and command loads

cd statcheck-ml
python -m pytest tests -q            # the Python units and the parity self-test

python pipeline/12_engines.py <pdf_dir> <key.json> <labels.json> --text-dir <dir> --recursive
```

Install the command line tool. The package ships its own model, so no
checkout is needed. A PyPI release comes later; install from the repository
until then.

```
pip install "statcheck-ml[pdf] @ git+https://github.com/rasoulnorouzi/ml-statcheck.git#subdirectory=statcheck-ml"
statcheck-ml check paper.pdf
```

The `[pdf]` extra adds PyMuPDF, which reads the PDF. Without it the tool
still checks a `.txt` file.

Call the same pipeline from Python code:

```python
from statcheck_ml import Pipeline

pipeline = Pipeline()          # the model packaged with the install
report = pipeline.run_pdf("paper.pdf")
```

The install carries the model, so this needs no checkout and no path. A
checkout can name another configuration, such as `models/zoo/gru-softmax`.

---

## Honest limits

1. **There is no gold set.** Language model agents produced every label. The
   agreement study in the report is the only measure of label quality. Read
   every number against it.
2. **The models are trained on PyMuPDF text alone.** The augmentation holds the
   other engines' alphabets, but no model was trained on their text.
3. **The R port has a lower ceiling than the Python port**, because poppler
   returns less of the corpus and no normalisation recovers text the engine
   never gave.
4. **The baseline is the R package, not a human.** A result the R package and
   the model both miss is invisible to both, and to this report.

---

## Documents

| Path | Content |
|---|---|
| [statcheck-ml/docs/TUTORIAL_PYTHON.md](statcheck-ml/docs/TUTORIAL_PYTHON.md) | the Python tutorial: install, CLI, API, verdicts, model choice, batch use |
| [statcheck-ml/results/REPORT.md](statcheck-ml/results/REPORT.md) | the generated report: method, agreement, grid, benchmarks, statistics |
| [PLAN.md](PLAN.md) | the phase table and the state of each phase |
| [CONTEXT.md](CONTEXT.md) | why each decision was made, and what the corpus showed |
| [CLAUDE.md](CLAUDE.md) | rules for an agent working in this repository |
| [statcheck-ml/docs/GUIDELINE.md](statcheck-ml/docs/GUIDELINE.md) | the annotation guideline the raters read |
| [statcheck-ml/docs/SCHEMA.md](statcheck-ml/docs/SCHEMA.md) | the tag set and the dataset row |
| [statcheck-ml/docs/PROTOCOL.md](statcheck-ml/docs/PROTOCOL.md) | the annotation and adjudication protocol |
| [statcheck-ml/pipeline/README.md](statcheck-ml/pipeline/README.md) | every pipeline stage, its inputs and outputs |
| [statcheck-ml/CORPUS.md](statcheck-ml/CORPUS.md) | what the supplied corpus contains |
| [statcheck-ml/SAMPLING.md](statcheck-ml/SAMPLING.md) | how the windows were sampled |
