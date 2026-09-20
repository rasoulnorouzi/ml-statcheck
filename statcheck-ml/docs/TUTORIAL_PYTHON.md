# Python tutorial

The Python port of statcheck-ml: `pip install`, a command line tool, a
library. It ships one model and needs no PyTorch. Every command below ran
against this repository at commit `e0fe47f`, and the output shown is pasted
from that run.

The other two ports have their own tutorials: R at
[`ports/statcheck-ml-r/vignettes/statcheckml.Rmd`](https://github.com/rasoulnorouzi/rasoulnorouzi-statcheck-ml-r/blob/main/vignettes/statcheckml.Rmd),
the browser at
[`statcheck-ml-web/docs/TUTORIAL.md`](https://github.com/rasoulnorouzi/rasoulnorouzi-statcheck-ml-web/blob/main/docs/TUTORIAL.md).
All three read the same spec files and the same model.
[`results/REPORT.md`](../results/REPORT.md) section 9 measures how close the
PDF text each one starts from actually is.

## 1. What it does and does not do

`statcheck` does two jobs: it **finds** a reported test result with a
regular expression, and it **checks** it by recomputing the p-value from the
test statistic. The regex is the weak link — it cannot read a result whose
operator (`=`, `<`, `>`) a PDF conversion destroyed, which is common when a
publisher sets the operator in a symbol font with no ToUnicode map.

This project replaces only the finding step, with a small character-level
model:

> Machine learning finds results. It never judges them.

The p-value comparison in [`pvalue.py`](../src/statcheck_ml/pvalue.py) is
closed-form mathematics — `scipy.stats` calls and an `if`/`elif` chain, no
learned weight near it. A model's tags decide where a result is and what its
parts are. They never decide whether a paper's p-value is right.

**Labels are bronze.** Three rater language-model agents wrote every label,
reading one guideline, with disagreements sent to a fourth adjudicator
agent. No human checked one, and there is no gold set:

> Never train on the gold set, and never call a machine annotation gold.

Read every number below against that fact. From
[`results/REPORT.md`](../results/REPORT.md) section 6, on a 200-document
holdout the tool never trained on:

| System | P | R | F1 [CI] |
|---|---|---|---|
| statcheck_raw | 0.983 | 0.183 | 0.308 [0.222, 0.385] |
| statcheck_repaired | 0.993 | 0.467 | 0.636 [0.567, 0.698] |
| cascade_gru-crf-s0 | 0.949 | 0.870 | 0.908 [0.875, 0.937] |

`statcheck_repaired` is the R package after its own operator-repair pass,
the strongest form of the R baseline. The cascade runs it first and adds
only what the model finds beyond it: F1 0.908 against 0.636, a 0.272 gain
significant at p = 0.000 by a paired bootstrap over documents (report
section 7). Most of the gap sits in the damaged subset, where
`statcheck_repaired` reaches F1 0.723 and the cascade reaches 0.922.

## 2. Install

**From the repository** — no PyPI release exists yet:

```
pip install "git+https://github.com/rasoulnorouzi/ml-statcheck.git#subdirectory=statcheck-ml"
pip install "statcheck-ml[pdf] @ git+https://github.com/rasoulnorouzi/ml-statcheck.git#subdirectory=statcheck-ml"
```

The repository is private. Without a credential pip stops at
`fatal: could not read Username for 'https://github.com'`. Three ways to give
it one: `gh auth login` (git then uses the stored credential), a personal
access token with `repo` scope in the URL
(`git+https://<token>@github.com/rasoulnorouzi/ml-statcheck.git#subdirectory=statcheck-ml`),
or SSH (`git+ssh://git@github.com/rasoulnorouzi/ml-statcheck.git#subdirectory=statcheck-ml`).
When the repository becomes public, none of this is needed.

The lines above were not run here: they need the network and a credential.
What I can run is the same
install from a local path, which exercises the same `pyproject.toml`. I
built it in a throwaway venv at a short path, to stay clear of Windows'
260-character path limit, and deleted the venv afterward with `shutil`:

```
python -m venv C:\Users\norouzin\AppData\Local\Temp\tut\venv
C:\Users\norouzin\AppData\Local\Temp\tut\venv\Scripts\python.exe -m pip install ^
    C:\Users\norouzin\Downloads\Projects\practice_claude\statcheck-ml
```

Last lines of that install, verbatim:

```
Building wheel for statcheck-ml (pyproject.toml): finished with status 'done'
  Created wheel for statcheck-ml: filename=statcheck_ml-2.0.0-py3-none-any.whl size=1836173 sha256=14a56a4a67a19b203ef5bbde581e361245787d83a69091f3fc69f4742e056ff3
  Stored in directory: c:\users\norouzin\appdata\local\pip\cache\wheels\c7\f6\6d\4c01f11d0bffa4f42cb6f55f49ca3fb019de31ce38dcb89c17
Successfully built statcheck-ml
Installing collected packages: flatbuffers, protobuf, packaging, numpy, scipy, onnxruntime, statcheck-ml
Successfully installed flatbuffers-25.12.19 numpy-2.5.3 onnxruntime-1.30.0 packaging-26.3 protobuf-7.36.2 scipy-1.18.1 statcheck-ml-2.0.0
```

`numpy`, `scipy` and `onnxruntime` are the only run-time dependencies in
`pyproject.toml`. `pymupdf` (the `pdf` extra) and `torch`/`onnx`/`onnxscript`
(the `train` extra, used only by `pipeline/07_train.py` and
`pipeline/08_export.py`) are optional on purpose — the shipped ONNX model
needs neither.

**From PyPI**, once a release exists: `pip install "statcheck-ml[pdf]"`.

**From a built wheel** — `statcheck-ml/dist/statcheck_ml-2.0.0-py3-none-any.whl`
is already in this checkout; `python -m build` or `pip wheel .` produces a
fresh one (wheel builds are not byte-reproducible, so its hash differs from
the one above):

```
pip install statcheck-ml/dist/statcheck_ml-2.0.0-py3-none-any.whl
```

Verified the same way, in a second throwaway venv: `Successfully installed
flatbuffers-25.12.19 numpy-2.5.3 onnxruntime-1.30.0 packaging-26.3
protobuf-7.36.2 scipy-1.18.1 statcheck-ml-2.0.0`.

Requires Python ≥ 3.10 (`requires-python` in `pyproject.toml`). On Windows,
`onnxruntime`'s own installed files can exceed the 260-character path limit
under a deep venv directory — enable long paths, or install near the drive
root, as done above.

## 3. First run

```
statcheck-ml check examples/sample_paper_damaged.pdf
```

`examples/sample_paper_damaged.pdf` is a generated one-page PDF
(`python examples/make_sample_paper.py`) whose five results all have their
operator replaced by a control character, the way a symbol font with no
ToUnicode map does it in a real corpus:

```
5 results found (5 pattern, 0 model) - consistent=4, decision_error=1
    1  t(23) = 2.45, p = 0.022  0.0223  consistent  [pattern]
    1  f(2, 30) = 5.1, p = 0.012  0.0124  consistent  [pattern]
    1  f(1, 118) = 9.2, p = 0.003  0.0030  consistent  [pattern]
    1  t(46) = 1.8, p = 0.04  0.0784  decision_error  [pattern]
    3  t(19) = 4.15, p = 0.001  0.0005  consistent  [pattern]
```

Every result shows `[pattern]`, not `[model]`. That is not a broken demo —
the pipeline repairs the operator (stage 3, arithmetic-chosen) before either
extractor runs, so `t(23) \x03 2.45` (the control character
`make_sample_paper.py` writes in place of each damaged `=`) already reads
`t(23) = 2.45` by the time the pattern sees it. Section 4 shows the damage
that repair alone cannot fix, where the model earns its place.

| Column | Meaning | `Found` field |
|---|---|---|
| line | source line number | `line` |
| test | test name and df, e.g. `t(23)` or `f(2, 30)` | `test_type`, `df1`, `df2` |
| statistic | the test statistic | `statistic` |
| operator | `=`, `<` or `>` before the p-value | `p_operator` |
| reported p | the p-value as printed | `p_value` |
| computed p | the p-value implied by the statistic, to 4 decimals, or `NA` | `computed_p` |
| verdict | `consistent`, `inconsistent`, `decision_error`, `undecidable` | `verdict` |
| source | `pattern` or `model` | `source` |

`--json` prints the full report instead — every `Found` field, plus
`stages`. One entry from the same command with `--json`:

```json
{
 "test_type": "t", "statistic": 2.45, "df1": 23.0, "df2": null,
 "p_operator": "=", "p_value": 0.022, "quote": "t(23) = 2.45, p = .022",
 "source": "pattern", "line": 1, "repaired": false, "verdict": "consistent",
 "computed_p": 0.022315728160948567, "reason": "", "missing": []
}
```

## 4. The stages, on one damaged sentence

The window `hand-control-operator`, from
[`tests/parity_cases.json`](../tests/parity_cases.json) — the same case the
JavaScript and R parity suites are held to:

```python
from statcheck_ml.normalize import normalize
from statcheck_ml.repair_validated import repair_validated
from statcheck_ml.prefilter import Prefilter
from statcheck_ml.extract import extract
from statcheck_ml.onnx_runtime import OnnxTagger
from statcheck_ml.pvalue import Result, check

text = "F(1, 40) \x02 6.20, p = .016"   # \x02 stands where "=" belongs

normalized, norm_info = normalize(text)
repaired, repair_info = repair_validated(normalized)
windows = list(Prefilter().windows(repaired))
matches = extract(repaired)
tags = OnnxTagger("models/zoo/gru-crf").tag_text(repaired)
outcome = check(Result(test_type="f", statistic=6.20, df1=1, df2=40,
                       p_operator="=", p_value=0.016),
                reported_p_text=".016")
```

```
normalize      : 'F(1, 40) \x02 6.20, p = .016'  {'operators_renamed': {}}
repair_validated: 'F(1, 40) = 6.20, p = .016'
  {'reason': 'too few testable results', 'fallback': 'simple rule',
   'chosen': {"'\x02'": '='}, 'replacements': 1}
Prefilter.windows: 1 window: 'F(1, 40) = 6.20, p = .016'
extract        : {'test_type': 'f', 'statistic': '6.20', 'df1': '1',
                   'df2': '40', 'p_operator': '=', 'p_value': '.016'}
tag_text (char: tag), on the repaired text:
  'F':S-TEST '(':O '1':O ',':O ' ':O '4':B-DF2 '0':E-DF2 ')':O ' ':O
  '=':S-POP_EQ ' ':O '6':B-STAT '.':I-STAT '2':I-STAT '0':E-STAT ',':O
  ' ':O 'p':O ' ':O '=':O ' ':O '.':B-PVAL '0':I-PVAL '1':I-PVAL '6':E-PVAL
check          : Check(verdict='inconsistent', computed_p=0.01702995545478932,
                       reported_p=0.016,
                       reason='the reported and computed p-values disagree')
```

Four things worth reading closely.

`normalize` did nothing — the interesting result. `canonicalise` only
renames a character that is *both* at an operator site and unknown to the
model. `\x02` is already in `spec/charmap.json` (id 2, right after PAD and
UNK), because the training corpus is full of exactly this damage, and the
model was trained to read it directly. `operators_renamed` comes back
empty.

`repair_validated` fell back to the simple rule, because one testable
result is not enough for the arithmetic search to trust a mapping
(`MIN_TESTABLE` in [`spec/repair.json`](../src/statcheck_ml/spec/repair.json)).
The fallback (`repair.py`) assumes a damaged character between a
degrees-of-freedom close-paren and a number is `=` — right here.

The pattern reads the repaired text perfectly, because repair already fixed
the operator. `extract.py` has no shared JSON spec — it is a faithful,
deliberately-not-improved Python port of the R package's regexes, the
baseline improvement is measured against.

The model's own tags do not match a human reading of "F(1, 40)": it tags
`40` as `DF2` and leaves the `1` untagged. That is the model's real,
trained behaviour on this exact input, frozen in `tests/parity_cases.json`
for every port to reproduce exactly — not a hand-picked success.

And the check itself disagrees: computed p is 0.017030, reported is `.016`;
at 3 written decimals the tolerance is ±0.0005, and the 0.00103 gap falls
outside it, so the verdict is `inconsistent`. Section 6 covers the rule.

`OnnxTagger` ([`onnx_runtime.py`](../src/statcheck_ml/onnx_runtime.py)) is
the one place in this port that reads exactly what a real port ships —
`tagger.onnx`, `decoder.json`, `charmap.json` — and decodes with plain
numpy, no PyTorch. `pipeline/08_port_kit.py` bundles those three files, the
five `spec/*.json` files, and `tests/parity_cases.json` into one kit for the
R and web ports, with a manifest of a SHA-256 per file that a port checks on
load — a stale copy fails loudly rather than answering wrong. Python does
not consume a kit; it is the repository the kit is built from, so it reads
`src/statcheck_ml/spec/*.json` directly.

## 5. The API

```python
from statcheck_ml.pipeline import Pipeline, summarise

pipeline = Pipeline(model_path="models/zoo/gru-crf")
report = pipeline.run_pdf("examples/sample_paper.pdf")
print(summarise(report))
```
```
examples/sample_paper.pdf
  characters        : 2,019
  windows kept      : 13 of 41 lines
  found by pattern  : 7
  found by model    : 2
  verdicts          : consistent=6, undecidable=2, decision_error=1
  seconds           : 0.13
```

`Pipeline.__init__`:

| Argument | Default | What it changes |
|---|---|---|
| `model_path` | `None` | A zoo directory, an `.onnx` file, or a raw `.pt` checkpoint. `None` skips the model stage; only the pattern runs. |
| `use_crf` | `False` | Ignored for an ONNX zoo directory — `OnnxTagger` reads it from `decoder.json`. Matters only for a raw torch checkpoint. |
| `repair_text` | `True` | Runs the arithmetic-validated repair (stage 3) first. `False` leaves damaged operators as-is, so only the model can read a damaged result. |
| `use_pattern` | `True` | Runs the regex extractor. `False` means every result comes from the model alone. |
| `alpha` | `0.05` | The threshold `check()` uses for `decision_error` vs. `inconsistent`. Never changes whether a result counts as `consistent`. |
| `engine` | `"pymupdf"` | `"pymupdf"`, `"pdfium"`, or `"poppler"` (needs `pdftotext` on PATH). |
| `normalize_text` | `True` | Runs stage 2 (reflow + operator canonicalisation) first. `False` skips it; `stages` then has no `"normalize"` key. |

Two arguments that flip a verdict, both executed:

```python
p2 = Pipeline(model_path="models/zoo/gru-crf", repair_text=False)
r2 = p2.run_text("Recall was lower under pressure, t(46) \x02 1.80, p \x02 .04.")

p3 = Pipeline(model_path="models/zoo/gru-crf", alpha=0.05)
r3 = p3.run_text("Borderline effect, t(46) = 1.70, p = .04.")   # computed_p 0.0959

p4 = Pipeline(model_path="models/zoo/gru-crf", alpha=0.10)
r4 = p4.run_text("Borderline effect, t(46) = 1.70, p = .04.")
```
```
r2["stages"]["find"] -> {'by_pattern': 0, 'by_model': 1}
r3["results"][0]["verdict"] -> 'decision_error'   (0.04 vs 0.05: sig.; 0.0959 vs 0.05: not)
r4["results"][0]["verdict"] -> 'inconsistent'      (0.04 vs 0.10: sig.; 0.0959 vs 0.10: sig. too)
```

**`run_pdf(pdf_path)`** extracts text with the chosen engine, calls
`run_text`, and adds `source` (the path), `engine` (name used),
`characters` (length of extracted text).

**`run_text(text)`** runs stages 2–6, returns
`{"results": [...], "stages": {...}, "seconds": ...}`:

```python
report = pipeline.run_text("The effect was reliable, F(2, 30) = 5.10, p = .012.")
```
```
text
  characters        : 0
  windows kept      : 1 of 1 lines
  found by pattern  : 1
  found by model    : 0
  verdicts          : consistent=1
  seconds           : 0.0
```

(`characters` reads 0 because `summarise` falls back to `report.get(
"characters", 0)`, and only `run_pdf` sets that key.)

Report keys from `run_pdf`: `characters`, `engine`, `results`, `seconds`,
`source`, `stages`. From `run_text`: just `results`, `seconds`, `stages`.
Each result (`dataclasses.asdict(Found)`) carries: `computed_p`, `df1`,
`df2`, `line`, `missing`, `p_operator`, `p_value`, `quote`, `reason`,
`repaired`, `source`, `statistic`, `test_type`, `verdict`. `missing` is a
tuple naming the parts `check()` needed and did not find — `()` when the
result was complete, whatever the verdict.

**`summarise(report)`** — the text form shown above, built entirely from
`report["stages"]`; not a second source of truth.

CLI flags, each already executed above: `--model DIR` (a zoo directory or
checkout path; default is the model packaged with the install), `--json`
(the full report), `--engine {pymupdf,pdfium,poppler}` (default `pymupdf`).

## 6. Reading a verdict

`pvalue.check()` returns one of four verdicts. All four, executed:

```python
from statcheck_ml.pvalue import Result, check

check(Result(test_type="t", statistic=2.45, df1=23, p_operator="=",
             p_value=0.022), reported_p_text=".022")
# -> consistent,     computed_p=0.022315728160948567

check(Result(test_type="t", statistic=2.10, df1=23, p_operator="=",
             p_value=0.001), reported_p_text=".001")
# -> inconsistent,   computed_p=0.046897512147949814

check(Result(test_type="t", statistic=1.80, df1=46, p_operator="=",
             p_value=0.04), reported_p_text=".04")
# -> decision_error, computed_p=0.07842066481562293

check(Result(test_type="r", statistic=0.42, p_operator="=", p_value=0.02),
      reported_p_text=".02")
# -> undecidable,    computed_p=None, missing=('df1',)
```

- **`consistent`** — the computed p-value falls within tolerance of the
  reported one.
- **`inconsistent`** — they disagree, but land on the same side of `alpha`,
  so the paper's conclusion still holds.
- **`decision_error`** — they disagree *and* land on opposite sides of
  `alpha`: the reported p-value calls the result significant (or not), the
  computed one says the reverse. The consequential kind, because it can flip
  what the paper claims.
- **`undecidable`** — a required part is missing (here, `r` needs `df1`),
  so no p-value could be computed. `missing` names the exact part, so a
  reader can tell "the paper omitted this" from "the tool failed to find
  it."

**The rounding rule**: a reported p-value stands for any computed value
that rounds to it at the same number of decimals the author wrote — `.03`
covers `[.025, .035)`. The decimal count is read from the exact text the
paper printed (`reported_p_text`), not from the float, since a bare float
cannot tell `.03` from `.030`.

**Known difference from R statcheck** (`results/REPORT.md` section 7): on
the results the R package itself reports after repair, 135 verdicts agree
and 15 disagree, both for rounding reasons. R accepts a reported p-value
when the *interval* implied by the rounded statistic contains it; this
project does not yet widen the check that way, and it accepts a p-value
reported as zero, or as a small fixed bound, where R flags it. Neither is a
fault in extraction — the report states the next version of the p-value
core will adopt R's rule so the two agree.

## 7. Choosing a model

Three configurations ship in `models/zoo/`: `gru-crf`, `gru-softmax`,
`lstm-softmax` — a BiGRU or BiLSTM over characters, softmax or CRF head,
the top three of a six-configuration grid by *development* F1, each
retrained on three seeds. Holdout F1, seed 0 (`results/REPORT.md` section
6):

| Config | P | R | F1 [CI] |
|---|---|---|---|
| gru-crf-s0 | 0.952 | 0.861 | 0.904 [0.871, 0.934] |
| gru-softmax-s0 | 0.925 | 0.882 | 0.903 [0.868, 0.933] |
| lstm-softmax-s0 | 0.910 | 0.879 | 0.894 [0.851, 0.932] |

Section 7's paired bootstrap finds no real gap: gru-crf-s0 vs.
gru-softmax-s0 differs by 0.001 F1 [-0.022, 0.023] (p = 0.930); vs.
lstm-softmax-s0 by 0.010 [-0.015, 0.036] (p = 0.447) — both intervals
contain zero. `gru-crf` ships as the default because *development* F1, the
score the grid itself is chosen by, ranks it first (0.927 vs. 0.924 vs.
0.921, section 8), at close to the same size (1.87 MB vs. 2.47 MB ONNX) and
latency (5.4 ms vs. 6.1 ms). With three statistically tied holdout scores,
size and latency were reported, not used to break the tie.

Pass a different one from a checkout:

```
statcheck-ml check examples/sample_paper_damaged.pdf --model models/zoo/lstm-softmax
```
```
5 results found (5 pattern, 0 model) - consistent=4, decision_error=1
    1  t(23) = 2.45, p = 0.022  0.0223  consistent  [pattern]
    1  f(2, 30) = 5.1, p = 0.012  0.0124  consistent  [pattern]
    1  f(1, 118) = 9.2, p = 0.003  0.0030  consistent  [pattern]
    1  t(46) = 1.8, p = 0.04  0.0784  decision_error  [pattern]
    3  t(19) = 4.15, p = 0.001  0.0005  consistent  [pattern]
```

Identical to section 3's default run — every result here came from the
pattern, so the model choice made no difference. Section 4's
`hand-control-operator` case is where it would show up.

## 8. Batch: a folder of PDFs to one CSV

```python
"""Run the pipeline once per PDF in a folder and write one CSV.

Usage: python batch_check.py <pdf_dir> <out.csv>
"""
from __future__ import annotations

import csv
import sys
from pathlib import Path

from statcheck_ml.pipeline import Pipeline

FIELDS = ["file", "line", "test_type", "statistic", "df1", "df2",
          "p_operator", "p_value", "computed_p", "verdict", "source"]


def main(pdf_dir: str, out_csv: str) -> None:
    pipeline = Pipeline(model_path="models/zoo/gru-crf")
    rows = []
    for pdf in sorted(Path(pdf_dir).glob("*.pdf")):
        report = pipeline.run_pdf(str(pdf))
        for r in report["results"]:
            row = {k: r.get(k) for k in FIELDS if k != "file"}
            row["file"] = pdf.name
            rows.append(row)

    with open(out_csv, "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    print(f"wrote {out_csv}: {len(rows)} results from "
          f"{len(list(Path(pdf_dir).glob('*.pdf')))} files")


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
```

Run against `examples/`, which holds the two sample PDFs:

```
python batch_check.py examples out_batch.csv
```
```
wrote out_batch.csv: 14 results from 2 files
```

CSV head:

```
file,line,test_type,statistic,df1,df2,p_operator,p_value,computed_p,verdict,source
sample_paper.pdf,14,t,2.45,23.0,,=,0.022,0.022315728160948567,consistent,pattern
sample_paper.pdf,14,f,5.1,2.0,30.0,=,0.012,0.012400181003238748,consistent,pattern
sample_paper.pdf,14,chi2,8.69,1.0,,=,0.003,0.0031996062478553107,consistent,pattern
sample_paper.pdf,17,r,0.42,,,=,0.02,,undecidable,model
sample_paper.pdf,18,t,1.8,46.0,,=,0.04,0.07842066481562293,decision_error,pattern
```

The fourth row has no `computed_p` and `verdict=undecidable` — the `r = .42`
result on line 17 carries no degrees of freedom, so nothing could be
computed. That is the same case `examples/make_sample_paper.py` documents in
its own comment: the sample paper carries one known-checkable result, one
known decision error, and one known-uncheckable one, on purpose.

## 9. Limits and troubleshooting

**The prefilter is a hard ceiling.** `Prefilter.windows` only offers a line
to the model when it is long enough, contains a digit, and has a non-letter
character density at or above 0.20 — a line with numbers and punctuation on
it. Measured on the clean corpus, this keeps 100% of lines holding a
statistic and drops 66% of the rest. Nothing downstream can recover a line
the prefilter drops, so it is tuned for recall alone.

**PDF engines do not return the same text.** `examples/sample_paper.pdf`
splits one result across a line break (`F(3, 92) = 2.71,` / `p = .049,
...`). `pymupdf` and `pdfium` break the line the same way, so the window
cuts the p-value away from the statistic and the result comes back
`undecidable`; `poppler` joins the two source lines differently and keeps
the result whole, so the pattern reads it — and its recomputed p-value
(0.0496) misses the reported `.049` at 3-decimal tolerance, so it comes
back `inconsistent` instead:

```
statcheck-ml check examples/sample_paper.pdf --engine pdfium
#   22  f(3, 92) = 2.71, p = NA  0.0496  undecidable  [model]
statcheck-ml check examples/sample_paper.pdf --engine poppler
#   10  f(3, 92) = 2.71, p = 0.049  0.0496  inconsistent  [pattern]
```

(pdfium's output is otherwise identical to pymupdf's on this simple,
single-column file — the difference shows up on a real, multi-column
article.) The engine gate in report section 9 quantifies this over the full
holdout:

| Engine | Recall |
|---|---|
| pymupdf | 0.911 |
| pdfium | 0.902 |
| poppler | 0.872 |

Spread 0.040 against the project's own limit of 0.060.

**Some damage the model does not recover.** From the per-family recall
table (section 6), one family stands out:

| Damage family | best_model | cascade | statcheck_repaired |
|---|---|---|---|
| decimal point lost | 0.000 (n=3) | 0.000 (n=3) | 0.000 (n=3) |
| control character | 0.744 (n=82) | 0.756 (n=82) | 0.695 (n=82) |

Nobody recovers a lost decimal point — `.016` becoming `016` removes
information no model, regex, or R's own repair can reconstruct from
context. It is a 3-result family, so read 0.000 as a real miss on a small
sample, not a precise rate. `control character`, the largest family (82
results), is where the model's real advantage sits, and even there its
ceiling is 0.744–0.756, not 1.0.

**Install errors.** A `.pdf` file needs the `pdf` extra; without it the CLI
fails cleanly. Executed against an install with no `pymupdf`:

```
statcheck-ml check sample.pdf
```
```
error: reading a PDF needs the 'pdf' extra: pip install 'statcheck-ml[pdf]'
```

(exit code 2.) A `.txt` file needs nothing extra — same install, run
against plain text:

```
statcheck-ml check sample.txt
```
```
1 results found (1 pattern, 0 model) - consistent=1
    0  f(2, 30) = 5.1, p = 0.012  0.0124  consistent  [pattern]
```

On Windows, a long venv path can push `onnxruntime`'s own installed files
past the 260-character limit; a fresh-venv check hit this at a
150-character prefix. Enable long paths, or install near the drive root.
