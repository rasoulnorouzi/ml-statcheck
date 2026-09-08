# statcheck-ml

A machine-learned replacement for the extraction step of the R package
[`statcheck`](https://github.com/MicheleNuijten/statcheck).

`statcheck` does two jobs. It **finds** reported statistical results with regular
expressions, and it **recomputes** the p-value to check them. The mathematics is
correct. The regular expression fails when the PDF conversion damages the text,
and more than half of the results in a real corpus are damaged that way.

This project replaces the first job with a character model. It does not touch
the second.

> **Machine learning finds results. It never judges them.**
> No model output reaches a verdict. The comparison stays closed-form
> mathematics in every port.

---

## Results

Holdout: 576 passages, 315 results, 164 of them damaged by the PDF conversion.

| | crf-aug + statcheck | gru-crf + statcheck | statcheck alone |
|---|---|---|---|
| Precision | 0.910 | 0.947 | 0.983 |
| **Recall** | **0.937** | 0.911 | 0.187 |
| F1 | 0.923 | 0.929 | 0.315 |
| **F2** | **0.931** | 0.918 | 0.223 |
| Recall on damaged text | **0.926** | 0.914 | **0.012** |
| Parameters | 616,072 | **466,696** | — |

**The cascade finds about five times as many results as the R package.** On
damaged text the regular expression finds 2 results of 164, and the cascade
finds 152.

Use **crf-aug** where recall matters, which is the usual case for a screening
tool. Use **gru-crf** in a browser: it is 24% smaller and 0.013 F2 behind.

Every number, and every definition behind it, is in **[REPORT.md](statcheck-ml/REPORT.md)**.

---

## How it works

A PDF goes in, and checked results come out.

| Stage | What it does |
|---|---|
| 1. extract | A named PDF engine turns the PDF into text. |
| 2. normalize | Line width and damaged characters are made engine independent. |
| 3. repair | Arithmetic restores operators the conversion destroyed. |
| 4. prefilter | About 1 line in 700 holds a result. The rest is dropped. |
| 5. find | The pattern reads what it can, and the model reads the rest. |
| 6. check | The p-value is recomputed. This stage is mathematics alone. |

Two design rules carry most of the weight:

- **The prefilter caps the recall of the whole system.** Text it discards can
  never be recovered. It is tuned for recall alone.
- **The processing unit is an overlapping window, not a sentence.** 18.4% of
  results are separated from their p-value by a line break.

---

## Three ports, one specification

The project ships from three ports: Python, R, and a browser. **Each port will
live in its own repository.** This repository stays the source of truth for the
model, the shared rules, and the measurements.

Every rule that more than one port needs lives in one JSON file:

```
statcheck-ml/src/statcheck_ml/spec/
  prefilter.json     which passages can hold a result
  normalize.json     how to make the text engine independent
  charmap.json       the 175 characters the model reads
  font_table.json    the operator each damaged font produced
```

**Never restate a rule in a port.** A port reads the file and applies it.

### How a port repository stays in step

```
python statcheck-ml/export_port_kit.py ../statcheck-ml-r --name statcheck-ml-r
```

That copies every shared file into the port repository, and writes
`statcheck-ml-manifest.json` beside them. The manifest records the version, the
size and the SHA-256 of each file.

1. The port reads its copy of `spec/*.json`, and never restates a rule.
2. The port checks every checksum in the manifest when it loads. **A stale copy
   is then found before it produces a wrong answer, not after.**
3. The port runs the parity suite in its own language, against the committed
   cases.

Change a rule in this repository and export again. Never edit a spec file inside
a port repository.

The parity suite is not a formality. It caught three real faults between the
JavaScript and R ports on their first day:

- JavaScript searched one character wider than Python for a line break.
- R kept the space it cut on, because R counts from one and Python from zero.
- R broke a tie with locale collation, so the same input gave different answers
  on different machines.

**An unverified port is worse than no port.**

### Each port needs a different PDF engine

No engine serves all three. R is the constraint, because `pdftools` is its only
maintained reader.

| Port | Engine | Licence | Recall |
|---|---|---|---|
| Python | PyMuPDF | AGPL-3.0 or commercial | 0.929 |
| Browser | PDF.js | Apache-2.0 | 0.923 |
| R | pdftools, over poppler | MIT over GPL-2 | 0.912 |

**PDFium is the permissive alternative to PyMuPDF**, at 0.916, and it has both a
Python and a JavaScript build. Use it when the AGPL licence does not suit
distribution.

`bench_engines.py` measures every engine and reports the spread between the best
and the worst. A port must not ship while the spread is above 0.06. It is 0.043.

---

## Status

The models are trained, frozen and measured. **The packages are not built.**

| Item | Status |
|---|---|
| Corpus, annotation, label schema | done |
| Character models, 11 trained | done |
| Evaluation against the R package | done |
| Engine portability study | done |
| Python pipeline, end to end | usable on torch |
| ONNX export | code exists, **never run** |
| p-value core in JavaScript and R | **not started** |
| Browser port | text layer only |
| R port | text layer only |

The p-value core is the right next piece. It unblocks both ports, and it decides
the verdict, so a silent mistake there would do the most harm.

[PLAN.md](PLAN.md) holds the phase table and the current state of each phase.

---

## Commands

```
node scripts/list-extensions.js      # every agent, skill and command loads

cd statcheck-ml
node tests/parity.mjs                # the browser port matches Python
Rscript tests/parity.R               # the R port matches Python
node tests/smoke.mjs                 # the browser port loads and runs
Rscript tests/smoke.R                # the R port loads and runs

python export_port_kit.py <target> --name <port>   # send the rules to a port

python bench_engines.py <pdf_dir> <key.json> <labels.json> --text-dir <dir>
bash run_all_training.sh
python evaluate_all.py <windows.json> <labels.json> <statcheck.csv> <out.json> \
    <name>=<model.pt>[:crf] ...
```

Use the Python port like this:

```python
from statcheck_ml.pipeline import Pipeline

pipeline = Pipeline(model_path="models/final-crf-aug/model.pt", use_crf=True)
report = pipeline.run_pdf("paper.pdf")
```

---

## Honest limits

1. **There is no gold set.** Language model agents produced every label. The
   measured ceiling is 84.1% on training data and 90.5% on the holdout. Read
   every number against it.
2. **The R port has a lower ceiling than the Python port**, because poppler
   returns 4 points less of the corpus and no normalisation recovers text the
   engine never gave.
3. **`pdftools` hangs on some documents**, twice in 198. The R port gives each
   document a time limit.
4. **The models are trained on PyMuPDF text alone.** The generator now holds the
   other engines' alphabets, but no model has been retrained with them.
5. **The largest single improvement to the baseline is not machine learning.**
   The arithmetic repair alone lifts the R package from 59 results to 158.

Section 12 of [REPORT.md](statcheck-ml/REPORT.md) records the wrong turns as
well, including a reported number that was wrong by a factor of 180, and one
regression that took three attempts to diagnose.

---

## Documents

| Path | Content |
|---|---|
| [REPORT.md](statcheck-ml/REPORT.md) | method, metrics, definitions, full results |
| [PLAN.md](PLAN.md) | the phase table and the state of each phase |
| [CONTEXT.md](CONTEXT.md) | why each decision was made, and what the corpus showed |
| [CLAUDE.md](CLAUDE.md) | rules for an agent working in this repository |
| [statcheck-ml/CORPUS.md](statcheck-ml/CORPUS.md) | what the supplied corpus contains |
| [statcheck-ml/SCHEMA.md](statcheck-ml/SCHEMA.md) | the annotation schema |
