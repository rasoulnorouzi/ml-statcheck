# statcheck-ml — project plan

Single source of truth. Update the Status column as work lands. Do not track progress
anywhere else.

## Goal

Replace the regex extraction of statcheck with a learned extractor, keep the p-value
recomputation exact, and ship the result as ONNX. The tool runs in three places: a
browser, R, and Python.

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
whole system.

Therefore the prefilter is tuned for recall, not precision. It answers "could this
contain a result", not "does this look like a result". Candidate signals:

- a digit is present
- the density of non-alphabetic characters is above a threshold
- a trigger character is present, such as a parenthesis, a comparison operator, or a
  Greek letter
- a test letter or keyword is present, such as t, F, r, z, chi, Q, p, df, or N

Prefilter recall on the gold set is a gate. Below the agreed threshold, the system is
reported as prefilter-limited and model tuning stops.

### The processing unit is a window, not a sentence

One result can span a sentence boundary, as in "the effect was significant, t(23) =
2.45. The p-value was .02." Tables split results further apart.

So the prefilter selects overlapping windows and expands each one by a sentence on
both sides. The model tags across the window, and result-block grouping links spans
across the internal boundaries.

This is a schema decision. It belongs in phase 3 and is expensive to add later.

## Three ports — one spec

Three components must behave identically in Python, JavaScript, and R: the prefilter,
the character vocabulary, and the p-value math.

These are not written three times. The rules and the vocabulary live in shared JSON
files, and each port reads them. A port holds only the code that applies the spec.

This principle drives the model choice. A character model needs a character map,
which is trivial in all three languages. A transformer needs a tokenizer ported three
times.


## The hybrid cascade

The measurements decide the design. On the round 2 passages the real statcheck
package reaches a precision of 1.000 and a recall near 0.205. It almost never
reports a result that is not there, and it misses about four results in five.

A learned extractor should therefore not replace it. It should follow it.

| Step | Component | Why |
|---|---|---|
| 1 | prefilter | 1 line in 700 holds a result, so this makes the rest affordable |
| 2 | statcheck | accept everything it finds, because its precision is 1.000 |
| 3 | the model | read the same candidates, add only what statcheck did not find |
| 4 | the mathematics | recompute and compare, the same code for both sources |

Two properties matter more than the score.

An existing statcheck user sees no regression. Every result the package reports
today is still reported, and it is still parsed by the package, so the cascade
cannot introduce a disagreement into work that already depends on it.

The model is answerable only for what the package cannot read. That is also
where the evidence is strongest, because 51.9% of what statcheck misses is text
whose operator the conversion from PDF destroyed.

The cascade is scored beside statcheck alone and the model alone, on passages
from documents no model has seen. Three rows, one table, one metric.

## Model zoo

| Tier | Model | Target size | Ported to all three? |
|---|---|---|---|
| `lite` | char-BiLSTM-CRF | < 10 MB | yes, this is the product |
| `balanced` | MobileBERT | ~25 MB | browser and Python only |
| `best` | DistilBERT | ~66 MB | research baseline, not ported |

Accuracy decides which tier is recommended. Size is reported, not constrained. The
`lite` tier must work everywhere regardless of which tier scores best, because it is
the only one that runs natively in R.

Training runs on local CPU. No GPU is assumed. Every training script checkpoints and
resumes.

## Label tiers

| Tier | Source | Allowed use |
|---|---|---|
| Silver | the ported statcheck regex, run at scale | bulk training |
| Bronze | multi-agent annotation by a language model, with agreement scoring | training and development |
| Gold | human labels, supplied by the project owner | held-out test only |

The gold set is frozen on arrival. No model trains on it. Annotations produced by a
language model are never reported as gold.

## Adversarial and noise layer

Applied to silver spans, so the model generalizes past the regex that produced them.

| Family | Content |
|---|---|
| OCR corruption | confusable characters, ligatures, spacing lost in PDF extraction |
| Unicode variance | Greek letter versus spelled name, superscript versus plain digit, dash forms |
| Format perturbation | prose phrasing, a result split across a line break, a table row |
| Hard negatives | text that resembles a result and is not, such as a page range or a citation year |

Hard negatives carry the most value. The regex fires on them. A trained model can
learn not to.

## Phases

Status values: `blocked`, `ready`, `active`, `done`.

| M | # | Phase | Agent | Model | Status |
|---|---|---|---|---|---|
| M1 | 0 | Repository layout, environment, size and latency gate | data-engineer | haiku | ready |
| M1 | 1 | Ingest corpus, RE-CONVERT the PDFs with PyMuPDF, keep both versions | data-engineer | haiku | ready |
| M1 | 2a | Port the statcheck regex to Python, prove parity against R | regex-porter | sonnet | done, 74% only |
| M1 | 2b | Build the prefilter spec, three ports, measure its recall | regex-porter | sonnet | ready |
| M2 | 3 | Label schema, window unit, result-block grouping, silver labels | label-architect | opus | ready |
| M2 | 4 | Noise and adversarial generator | adversarial-engineer | opus | done |
| M2 | 5 | Annotator agents, agreement, adjudication | label-architect | opus | done |
| M3 | 6 | Train the character model and the transformer baselines | ml-trainer | sonnet | active |
| M3 | 7 | ONNX export, quantization, the three-tier zoo | onnx-engineer | sonnet | blocked |
| M4 | 8 | p-value core in Python, JavaScript, and R, with a parity suite | stats-core | opus | ready |
| M4 | 8b | Text normalisation spec, so the PDF engine cannot change the answer | regex-porter | sonnet | done |
| M4 | 9 | Python reference port and the browser web app | runtime-engineer | sonnet | blocked |
| M4 | 10 | R package, native `lite` path | runtime-engineer | sonnet | blocked |
| M4 | 11 | Evaluation on an unseen test part: statcheck, model, cascade | eval-engineer | sonnet | active |
| M4 | 11b | The hybrid cascade, and its ablation against each part alone | eval-engineer | sonnet | active |
| M4 | 12 | Documentation in ASD-STE100 | doc-writer | haiku | blocked |

Phase 1 is blocked until the owner supplies the corpus folder. Phases 6, 7, 9, 10, 11
and 12 are blocked on their predecessors.

Phases 9 and 10 have a second gate. Each port gets a different PDF engine, and the
engines do not read the same text. `bench_engines.py` measures every engine and
reports the spread between the best and the worst. A port must not ship while the
spread is above 0.06. The spread was 0.136 before phase 8b and is 0.050 now.
`CONTEXT.md` holds the measurements and the reason.

Phases 0, 2a, 2b, 3, 4, 5 and 8 need no data. They can start now.

## Waiting on the owner

- Corpus received. See statcheck-ml/CORPUS.md. The supplied text lost all Greek letters, so phase 1 must convert the PDF files again.
- The human-labeled gold set, whenever it is ready.
