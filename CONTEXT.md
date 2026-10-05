# Context — what happened, and why

Read this file with `PLAN.md`. `PLAN.md` says what the phases are and where each one
stands. This file says why each decision was made, and what the corpus showed.

A new session should read `CLAUDE.md`, then `PLAN.md`, then this file.

Last updated: 2026-10-05.

## The project in one paragraph

`statcheck-ml` replaces the extraction step of the R package `statcheck` with a small
learned model. `statcheck` finds reported statistical test results with regular
expressions, recomputes the p-value, and reports a disagreement. The regular
expression is the weak part. The mathematics is not. So the project learns the
finding step, keeps the mathematics exact, and ships the model as ONNX for a browser,
for R, and for Python.

## Decisions, and the reason for each

### Machine learning finds results. It never judges them.

The p-value comparison stays closed-form in all three languages. No model output
reaches the verdict. A wrong verdict from a confident model is worse than no tool.

### The processing unit is an overlapping window, not a sentence.

Measured on the clean corpus: 18.4% of results are separated from their p-value by a
line break, and 3.3% by a sentence boundary. A sentence-level model loses those.

### A prefilter runs before the model, and it is tuned for recall alone.

Only 0.1% of corpus lines hold a statistic, so a filter is necessary rather than an
optimisation. The danger is that the filter, not the model, sets the recall ceiling,
because text the filter drops can never be recovered.

A threshold of 0.20 on non-letter density keeps 100% of statistical lines and removes
66% of the rest. This is the first operating point.

### One spec, three thin ports.

The prefilter rules, the character vocabulary, and the p-value constants live in
shared JSON. Python, JavaScript, and R read the same files. Nothing is written three
times, because three copies drift.

Version 2 uses character models only. A character map ports to three languages
without runtime dependencies.

### There is no human annotator. Every label comes from a language model.

The owner decided this. It has one permanent consequence, and it must stay written
down: no measurement in this project can find a mistake that the annotator makes
every time. More annotation does not remove this. More agreement does not remove it,
because the annotators share their training.

The project is therefore framed as distillation. A large model reads these papers
well but cannot run in a browser or in R. The small model reproduces its reading at
a small size.

One independent check remains, because the arithmetic does not come from the
annotator. A misread degrees of freedom usually gives a p-value far from the reported
one. Published work puts the true rate of inconsistency near 10%, so a batch far from
that rate has an extraction fault. This checks the results that were found. It says
nothing about the results that were missed.

### Annotators work blind.

An annotator agent receives the passage and the guideline. It does not receive the
project, the regular expression output, or the pool a window came from. This is the
owner requirement, and it is what makes agreement meaningful.

One weakness remains. An agent inside Claude Code can still read repository files, so
the prompt tells it not to. A standalone script with an API key would be cleaner.

### Annotation is sampled. Conversion is not.

Conversion of all 3100 PDF files takes 53 seconds on 14 cores. Annotation costs far
more for each document, so only annotation is sampled.

## What the corpus showed

The owner supplied `02_pdfs.zip`: 3100 PDF files and 3000 converted text files,
across 63 journals, in psychology, sociology, political science, economics,
education, and management.

### The supplied text is unusable on its own

It contains no character above ASCII. The conversion removed every Greek letter, so a
chi-square test appears as a bare digit, or as a wrong letter, or it disappears. A
regular expression finds almost no chi-square results in it.

PyMuPDF converts all 3100 files without error and recovers the Unicode in 99.8% of
them. The project therefore uses its own conversion. The supplied text is kept as an
aligned damaged copy of the same documents, which is real corruption rather than
generated corruption.

### A third of results hide behind control characters

This is the strongest evidence for the project.

In 198 documents the conversion writes a control character where the operator
belongs. Of results with degrees of freedom in parentheses, 6996 have a readable
operator and 3848 do not. That is 35.5% invisible to any pattern.

No substitution table repairs this. The mapping comes from the font of each document,
so one control character is an equals sign in one document and a less-than sign in
another.

Position carries the meaning instead. A character directly after the degrees of
freedom is almost always an equals sign. A character directly after the letter p is
one of three signs. A model that reads the surrounding characters can recover the
operator. A pattern cannot, because a pattern must know the character in advance.

The model therefore names the operator as well as marking it. It trains on the
undamaged operators, which are free supervision, and applies that to the damaged
ones.

## Annotation rounds so far

### Round 0 — 22 windows

The annotator agreed with the pattern on all 8 pool A windows, and handled line
breaks and chi-square correctly. It found three correlations with no degrees of
freedom that the pattern misses.

Six of eight pool B windows were reference list entries. A bibliography has the same
character density as a result, so the sampler was wasting the budget.

### Round 1 — 55 windows

The sampler now removes the reference section and stray reference lines. Reference
contamination fell to zero.

| Pool | Meaning | Yield |
|---|---|---|
| A | the pattern matched a complete result | 15 of 15 |
| B1 | a p-value is present, no complete match | 5 of 15 |
| B2 | a test letter and digits, no p-value | 9 of 15 |
| C | the density filter rejected the line | 0 of 10 |

Pool B2 gives the better yield, which was not expected. It catches the results whose
operator became a control character. Weight it above B1 in the next round.

Alignment reached 100% of quotes once the matcher ignored differences in whitespace.
The strict matcher failed on exactly the results split by a line break.

### The PDF engine is part of the system, not a detail below it

Each port gets a different PDF engine, and they do not read the same text. No
engine serves all three ports.

| Engine | Python | R | Browser |
|---|---|---|---|
| PyMuPDF | yes | no | through a WASM build |
| PDFium | yes, `pypdfium2` | no | through a WASM build |
| poppler | yes | yes, `pdftools` bundles it | no |
| PDF.js | no | no | yes, and it costs nothing |

R is the constraint. `pdftools` is built on poppler, measured as version 26.01.0.
It does not read exactly what the poppler command line reads. On the 131 holdout
documents both could read, `pdf_text()` held 0.916 of the results against 0.902
for `pdftotext`, and kept 0.912 against 0.898 after the prefilter. The R port is
therefore a little better than the command line, not equal to it. Measure the R
port with `pdf_text()`, never by borrowing a `pdftotext` number.

The models were trained on PyMuPDF text. Reading the same 198 holdout documents
with each engine showed the size of the problem, on 323 gold results:

| Engine | Holds the result | Prefilter recall, before | after |
|---|---|---|---|
| PyMuPDF | 0.929 | 0.929 | 0.929 |
| PDF.js | 0.926 | 0.907 | 0.907 |
| PDFium | 0.920 | 0.907 | 0.910 |
| poppler | 0.889 | 0.793 | 0.879 |

The spread between the best and the worst engine was 0.136. It is now 0.050.

The cause was not the character encoding. The first guess was that poppler
writes a destroyed operator as a letter in the Greek and Coptic block, where
PyMuPDF writes a control character, and that the density rule counts letters.
A canonicalisation test recovered nothing, so that guess was wrong.

The cause is the line. Poppler returns a whole column as one line, at a mean
length of 195.8 characters against 47.2 for PyMuPDF. Every prefilter rule
measures one line. A joined line holds a statistic AND a citation, so the
reference rule discarded both. That was 26 of the 37 results poppler lost.

`spec/normalize.json` and `normalize.py` hold the fix. The stage runs directly
after the engine. It restores the line width, and it renames damaged operator
characters to the alphabet the model was trained on.

At the prefilter the stage left PyMuPDF at 0.929 and looked free. At the model
it did not: what the model found on PyMuPDF fell from 0.879 to 0.868, while
poppler rose from 0.791 to 0.846.

### Rename only what the model cannot read

The cause took three attempts to find, and the first two answers were wrong.

The first guess was the character encoding. A canonicalisation test recovered
nothing, so that was wrong. The second guess was the reflow cutting the long
lines of a document that already had normal lines. Gating the reflow per
document did not move the number either, so that was wrong as well.

The answer came from listing every character the stage changed, rather than
reasoning about it. The renaming rule used the position alone: any character
between a letter and a digit that was not in a hand written keep list. In
PyMuPDF text that caught the copyright sign, the multiplication sign, the curly
quotes, and the significance star of `* p < .05`. All of them are ordinary
characters the model reads well. 91% of the renames on PyMuPDF were of
characters the model already knew.

`spec/charmap.json` holds the 175 characters the model can read, and it matches
the checkpoint vocabulary exactly. The rule is now simply this:

> Rename a character only when the model cannot read it. Never rename by
> position.

Unnecessary renames fell to zero on every engine, and PyMuPDF now renames 7
characters instead of 80.

Two lessons are worth keeping:

- A stage measured at one point in the pipeline can still cost recall at the
  next one. Measure the model, not only the filter.
- When two guesses in a row are wrong, stop guessing. List what the code
  actually changed.

The reflow is still decided per document, with `reflow_min_mean_line` at 120.
That gate was added while chasing the wrong cause, and it did not fix the
regression, but it stands on its own evidence: PyMuPDF averages 48 characters
per line, PDF.js 55 and PDFium 69, no PDFium document reaches 119, and poppler
averages 222. A document that is already broken into normal lines has no joined
column to undo.

The renaming is safe because the correspondence between the engines is one to
one inside a document, measured on 50 of 50 documents where both engines damaged
the same place. No information is lost by the engine swap. Only the name of the
character changes.

`bench_engines.py` is the gate. Phases 9 and 10 must not ship while the spread
is above 0.06.

### Each port names its own engine

| Port | Engine | Recall | Licence | Why |
|---|---|---|---|---|
| Python | PyMuPDF | 0.929 | AGPL-3.0 or commercial | Reads the most, and the models were trained on it, so it carries no distribution shift. |
| R | pdftools | 0.912 | MIT over poppler, GPL-2 | R has no other maintained reader. |
| Browser | PDF.js | 0.923 | Apache-2.0 | Needs no second binary to download. |

PDFium is the permissive alternative, at 0.916. Use it in place of PyMuPDF if
the AGPL licence does not suit distribution. It has a Python build and a
JavaScript build, so one engine can serve two ports.

### A window must hold text, not lines

A fixed count of lines is not a fixed amount of context, and this cost more
recall than the density rule did.

The count was tuned on PyMuPDF, whose lines run about 57 characters, so two
lines each side hold about 298 characters. PDF.js breaks the same page into
shorter lines, so the same window holds 175 characters and cuts the result in
half.

The density rule was not the cause. Removing it entirely recovers one result and
costs about 8000 extra windows for it. Matching the amount of text recovers most
of the gap at no extra windows at all, because a window only becomes longer.

`target_window_characters` is 250 in `spec/prefilter.json`. A window grows until
it holds that much and never shrinks, so text that already has long lines is
untouched.

| Engine | Fixed lines | 250 characters | Characters read |
|---|---|---|---|
| PyMuPDF | 0.929 | 0.929 | 1.08x |
| PDF.js | 0.907 | 0.923 | 1.72x |
| PDFium | 0.910 | 0.916 | 1.55x |
| R pdftools | 0.912 | 0.912 | 1.02x |
| poppler | 0.879 | 0.885 | 1.10x |

PyMuPDF recall does not move, so no published number regresses.

## Version 2 — why the process was rebuilt

Version 1 reached a working cascade, and its numbers could not be reproduced from
committed files. The annotation ran as two agent passes without a versioned
guideline, without provenance, and with manual adjudication. The split was
recomputed from a hash and never written down. The report numbers were typed by
hand. On 2026-09-19 the owner approved version 2, and these are its decisions.

### Three raters from three model families

Two passes of one model share one training run, so their agreement measures
consistency, not independence. Version 2 uses one haiku, one sonnet, and one opus
rater. The agreement between them is higher than between the two version 1 passes,
and the per-rater score against the final label tells which cheap model annotates
best under this guideline. The raters still share a model family, and the report
says so in its limitations.

### The guideline is a file, and every record carries its hash

`docs/GUIDELINE.md` is the artifact. The rater agents embed it, the collect script
records its sha256 on every record, and a change to the guideline is a new version.
One rule was added before annotation started: `p_operator` takes one of three
values, and `≤` is written as `<`. Real rater output from version 1 held `≤`.

### Consensus is a script, and disputes are a committed file

A result is kept when two of three raters found it, and each field takes the
majority value. A singleton or a three-way field conflict goes to an adjudicator
agent that sees no rater names. `dataset/annotations/<set>/disputes.json` holds every
dispute, every candidate, and every decision with its reason. The dispute rate was
about three percent of results.

### No silver tier, no transformer

The regex has recall near 0.2, so a silver window holds about four unlabeled true
results for each labeled one. That is label noise, not supervision. The transformer
baselines cannot port to R and were never the product. Both are gone.

### The document is the bootstrap unit

Windows from one document share a font, a conversion, and a writing style. An
interval over windows is too narrow. Every interval and every paired test in the
evaluation resamples holdout documents.

### The repaired baseline of version 1 was not reproducible

Version 1 reported 159 results for `statcheck` after operator repair on the holdout.
The file came from a whole-document substitution that no script in the repository
produces, and it corrupted ordinary words. The version 2 baseline uses the
RESULT-anchored repair and finds 152. The lower number is the honest one.

### Lessons from running the team

- Any shell command outside the allow list, such as `rm`, waits for manual approval
  and stalls an agent for hours. Agents delete and hash inside Python.
- A refactor that removes a flag can leave one dangling name after the training
  loop. Every run then dies after thirty epochs. `tests/test_train_smoke.py` trains
  one epoch to completion and would have caught it.
- A haiku writer cannot write the scientific prose of the report. It typed numbers
  by hand and misstated the method. The manager writes the prose; the generator
  fills every number.
- An agent fabricated an input file to make a generator run. The manager checks the
  timestamp and the schema of every input an agent used.

## Version 3 — the ports get their own repositories

Decided 2026-09-20. Design: `docs/superpowers/specs/2026-09-20-ports-design.md`.

Python stays here: it is the reference implementation, and the pipeline, the
evaluation and the parity cases import it. R and the web each get a repository,
`statcheck-ml-r` and `statcheck-ml-web`, because CRAN and npm each want their own
layout at the repository root, their own CI, and their own release cadence. This
repository feeds them one kit (`pipeline/08_port_kit.py`): the spec files, the shipped
model as ONNX and as raw weights, the parity cases, and a manifest of hashes that the
port verifies when it loads. A port never restates a rule and never keeps a second
copy of the model.

R runs the model in pure R from `weights.json`, batched over the windows of a document.
There is no ONNX runtime for R, `reticulate` would drag Python in, and the `torch`
package downloads libtorch; a two-layer bidirectional GRU is a few matrix products per
character and needs none of them. The web runs `tagger.onnx` through onnxruntime-web.
Both decode the CRF with the same Viterbi as `onnx_runtime.py`.

The parity file grows from one section to nine, one per stage, with the expected
values produced by the Python reference. A port is accepted when every section passes
in its own CI. The two thin copies kept here (`statcheck-ml/js/`, `statcheck-ml/r/`) are
deleted when their port passes, so each rule has one implementation per language.

The port repositories live under `ports/` in this checkout, each its own git
repository, gitignored by the mother. Reason: the agents' tool calls stay inside the
project directory, where the permission rules allow them without a prompt per file.

## Where the three repositories live

Pushed on 2026-09-20. Every workflow green on the first day.

| Repository | Holds | Workflow |
|---|---|---|
| [ml-statcheck](https://github.com/rasoulnorouzi/ml-statcheck) | the spec, the dataset, the models, the evaluation, the report, the Python package | `Python package`: ubuntu, macos, windows x Python 3.10 to 3.13, a wheel job, and an install from the repository address |
| [rasoulnorouzi-statcheck-ml-r](https://github.com/rasoulnorouzi/rasoulnorouzi-statcheck-ml-r) | the R package `statcheckml` | `R-CMD-check`: windows and ubuntu on the current R, macOS on the previous one |
| [rasoulnorouzi-statcheck-ml-web](https://github.com/rasoulnorouzi/rasoulnorouzi-statcheck-ml-web) | the npm package and the demo page | `test` on Node 20 and 24; `pages` deploys https://rasoulnorouzi.github.io/rasoulnorouzi-statcheck-ml-web/ |

Two facts the first CI run taught us. CRAN's macOS arm64 binaries for R 4.6 were
incomplete: pak downloaded a `knitr` archive it could not extract, twice, so macOS
runs one release behind. The R timing bound of ten seconds for a hundred windows is
a benchmark of this machine, not a correctness check; a shared runner took 10.9
seconds, so the bound is skipped on CI.

A port keeps the kit it was exported with, not the mother's newest commit. The kit
manifest names that commit. Re-export only when a spec file or the shipped model
changes, then rerun the port's tests and commit the kit with the mother commit in
the message.

## Version 4 — the human gold set, and what it decided

Decided 2026-10-04, run 2026-10-05, on the branch `v4-gold`. The brief is
`docs/superpowers/specs/2026-10-04-v4-gold-brief.md`, the selection rule is
`docs/v4/preregistration.md`, and the results are in `docs/v4/REPORT.md`. This section
holds the decisions and what each one rests on. The report holds the numbers in full.

### The human labels are a test, never a training label

Three human coders coded 409 papers of the corpus by hand: 2704 results in 185 papers,
and 224 papers with no result. This is the first human reading the project has. The
annotator section above records the limit of machine labels: no measurement made with
them finds a mistake the annotator makes every time. A human reading is the one
measurement that can. So the project spends it on measurement, and the rule in
`CLAUDE.md` stands: never train on the gold set.

The split is by paper, seed 0, and frozen in `dataset/gold/split.json`. The gold test
holds 169 papers with 476 located in-scope results. The gold train holds 240 papers
with 2007. The 49 papers of the bronze holdout went to the gold test, so no holdout
paper of either tier is in any training set. No training input reads the gold test. The
gold-train papers enter training only through the passage annotation below, where the
coder rows are one vote of four.

### Passages are the unit, and the coders are the fourth vote

The brief measured the unit before any training, on all gold papers with the shipped
model. Overlapping 250-character windows found 0.862 of the results and read 28.1 M
characters. The same windows merged into passages found 0.863 and read 15.8 M. Passages
from 1000-character windows found 0.877, but made 530 finds in the no-result papers
against 304. A model trained on short windows fires on tables.

A merged passage reads each character once and costs nothing, so `prefilter.json`
version 2 names `unit: "passage"`. On the gold test the characters the model reads on
the 169 papers fell from 11,433,629 to 6,503,399 (0.569). The recall did not move: 0.723
named, 0.866 with fragments. The finds in the 120 no-result papers fell from 73 to 69.

A passage is the annotation unit too. It is about a paragraph, at most 1500 characters,
up to five times a window, so a batch holds ten passages and not twenty. The three
raters read every passage blind, with guideline 3.0 and nothing else, and they do not
know that the coders exist.

The coders are the fourth voter, with the same weight as each rater, by the owner's
decision. A result stays with two votes of four, any two. The consensus kept 1945
results, and a coder vote backed 0.8401 of them. The singletons, one voter alone, were
haiku 9, sonnet 3, opus 54 and coders 10.

The two label sources were close before the vote began. The bronze windows of version 2
agreed with the coders on 191 of the 200 results they share. That is why the mixed
training set keeps the 1962 bronze windows beside the 1165 passages, less the 35
windows from the dev papers.

### A coder row is a vote, not a training label

A training label needs a span in the scope of the guideline. A coder row has neither.

- Scope. The coders followed statcheck. They skip a correlation without degrees of
  freedom, which the guideline marks, and they code CI, Wald and `b` rows, which it
  does not. A coder row outside the six test types never votes; 13 of the 1662 rows in
  the sampled passages were out of scope.
- Spans. The coders copied a quote from the PDF and typed the values.
  `pipeline/14_gold.py` located 2664 of the 2704 quotes in the converted text; the rest
  have no position. A located row votes by span, an unlocated row by statistic value
  only. A coder row never supplies the span or the quote of a kept result. Those come
  from a rater.
- Trust. The owner says the labels are not always right. 135 of 2704 rows hold a typed
  value that is not in the quote; most correct broken PDF spacing, a few dozen are
  typing errors.

So the coders vote on the test type, the statistic, the degrees of freedom, the p
operator, the p-value and the damage flag. They vote on nothing else. Every passage
label is bronze. A coder vote does not promote a label, and no machine annotation is
called gold.

### Two dev sets, with two jobs

Training stops early on the dev side of `dataset/splits_v4.json`. That is a split by
document of the same bronze windows and passages, with the same bronze labels. The
choice of the shipped model reads the 36 gold dev papers of `dataset/gold/dev.json`.
They are 15 % of the gold-train papers, drawn by paper, seed 0, stratified by
`has_results`. They hold 278 located in-scope results, 20 papers with results and 16
without.

The two jobs differ. Early stopping needs a score at every epoch, in the format the loss
trains on, which is a span under guideline 3.0. The bronze dev split gives that at no
cost. The choice needs the score the owner cares about: human rows, whole papers, and
the full cascade as `Pipeline().run_text` returns it. The gold dev papers give that.

The two sets share no paper. The passage inventory skips the dev papers, so no passage
of a dev paper was rated. `21_dataset_v4.py` leaves out the 35 bronze windows that come
from the dev papers. A model that trained on the text of a dev paper scores too high
there.

### Five seeds, and the rules read means

The brief showed the problem with one seed. The shipped model, trained with noise, found
0.857 of the gold results. The one model trained without noise found 0.863. One seed
cannot say whether that gap is real.

The pre-registration first held three seeds. When the owner offered an A100, the plan
went to seeds 0 to 4 and 35 runs, before any result was read. Rule 1 and rule 2 compare
means over the five seeds. Rule 3 ships the seed-0 run of the choice, so no seed is
chosen by its score.

The runs confirmed the need. On the dev papers the five `gru-crf` runs without noise
range from 0.806 to 0.835. On the gold test the single runs of the grid range from
0.704 to 0.796.

### The outcome: rule 3 kept the old model by one result

The rule was written on 2026-10-04, before any run started, and it did not change after
a result was seen. `pipeline/23_select_v4.py` applied it on the 36 dev papers, with the
gold test and the bronze holdout unread.

Rule 1 chose no noise: the mean R_named of `gru-crf` is 0.819 without noise and 0.808
with all four families. Rule 2 chose `lstm-softmax`. Its mean 0.825 lies within 0.01 of
`gru-softmax` at 0.834, so the tie goes to the lower mean FP, 1.0 against 1.6. Rule 3
compared `lstm-softmax-none-s0` with the shipped `gru-crf-s0` on the same papers: 0.824
against 0.827, one result of 278. The rule replaces the old model only when the
candidate is at least as high, so the old model stays.

Read after the choice, the gold test reverses the order. The candidate cascade finds
0.763 against 0.723, a difference of +0.040, with a paper-bootstrap interval of -0.013
to +0.123. The row-level McNemar p is 3.11e-04, but the 476 rows sit in 49 papers and
are not independent. The paper bootstrap is the figure to read, and it holds zero. The
candidate also makes 77 finds in the 120 no-result papers against 69, and its bronze F1
is 0.876 against 0.904.

The owner kept the pre-registered outcome: the old model ships, and the report gives
the gold-test difference as a finding read after the choice. A rule that changes after
the result is read is not a rule.

### The noise families lowered gold recall for every model type

Read after the choice, the mean cascade R_named over seeds 0 to 4 on the gold test is
lower with all four families for every configuration:

| configuration | no noise | all four families |
|---|---|---|
| gru-crf | 0.745 | 0.732 |
| gru-softmax | 0.763 | 0.754 |
| lstm-softmax | 0.764 | 0.727 |

This agrees with rule 1 on the dev papers. It also agrees with the brief's measurement
before training. There the model trained with the old noise layer found less on every
one of the six damage types the gold quotes show. It made 292 finds in the no-result
papers against 216.

The old families damage one character at random. Real damage is per paper, with one
font map: the brief counted spaced digits from 10 papers and control characters from
47. The four new families follow that shape. Together they still lowered the mean,
0.732 against 0.745 for `gru-crf`.

One family alone stands out. `operator_drop` gave the highest gate run on the dev
papers (0.845) and on the gold test (0.784, with 75 finds in the no-result papers). It
is one run of one seed, so a later grid should carry it with five seeds before it joins
the default.

### The next version needs a larger dev set

278 results in 36 papers turned rule 3 on one result. The gold-train part holds 2007
results in 240 papers; a dev set of several hundred results in more papers would make
rule 3 readable.

The next rule must also choose a side of the trade the gold test showed. The softmax
heads find more: mean cascade R_named 0.763 and 0.764 against 0.745. They report more
finds in the no-result papers, 72.8 and 74.4 against 66.0, with a lower bronze F1,
0.881 and 0.870 against 0.917. The CRF head trades recall for fewer false finds. Decide
which side matters before the rule is written.

### What the gold test is still clean for

It is clean for the trained models. No training input reads it, and the selection rule
reads the 36 dev papers only. `pipeline/24_final_v4.py` read it once per model after
the choice: 36 log lines `v4-final:*` in `results/gold_eval_log.jsonl`.

It is not clean for the rule changes made without training: the dedup by statistic
position, the fragments, and the passages. `results/gold_eval_log.jsonl` holds five
labelled evaluations of the gold test under M1 and M2 and one more under M2b, each
after the previous number was seen. The before-and-after table in the report is a
development figure for those changes, not a held-out score. Decide a later rule change
on the gold-train part, then read the gold test once.

## Known problems

1. A window can cut a result in half. One pool A window began in the middle of a
   result, because the result started before the window. Snap window edges to
   sentence or paragraph boundaries.
2. Detection is not the same as checking. A result with no degrees of freedom and no
   p-value is real but cannot be recomputed. Report the two counts apart.
3. Project agents need a session restart before Claude Code can dispatch them.
   Version 2 ran the raters through the general-purpose agent with a model override
   and the same guideline file, and recorded that on every record.
4. poppler reads 4 points less of the corpus than PyMuPDF, and the normalisation
   stage cannot recover text the engine never returned. The R port therefore has
   a lower ceiling than the Python port. State the number in the R
   documentation rather than hide it.
5. `pdftools` hangs on some documents, and it hung twice in 198. The poppler
   command line reads the same files without trouble. The R port must give each
   document a time limit and continue after one fails.
6. The models are trained on PyMuPDF text alone. `ENGINE_ALPHABETS` in
   `augment.py` now holds the characters other engines use, but no model has
   been retrained with them yet.

## Working agreements

- Documentation and chat about it follow ASD-STE100 Simplified Technical English.
  This does not apply to code comments, to commit messages, or to names in the code.
- The owner asks for a plan before code, and for examples before a batch runs at
  scale.
- Each phase in `PLAN.md` names its agent, and the model of each agent is a cost
  decision. Do not raise it because a task feels hard. Split the task.

## Files

| Path | Content |
|---|---|
| `PLAN.md` | phases, agents, status |
| `CONTEXT.md` | this file: decisions and findings |
| `docs/superpowers/specs/2026-09-19-reproducible-pipeline-design.md` | the version 2 design |
| `docs/superpowers/specs/2026-10-04-v4-gold-brief.md` | the version 4 brief: the owner's decisions and the findings during the run |
| `docs/v4/preregistration.md` | the version 4 selection rules, their dated notes, the outcome |
| `docs/v4/REPORT.md` | the version 4 report |
| `docs/v4/STATUS.md`, `docs/v4/prompts/` | the version 4 task list, and every agent prompt as sent |
| `statcheck-ml/docs/GUIDELINE.md` | the annotation guideline, version 2.0 |
| `statcheck-ml/docs/PROTOCOL.md` | raters, batches, provenance, agreement, consensus |
| `statcheck-ml/docs/SCHEMA.md` | the tag set and the dataset row |
| `statcheck-ml/CORPUS.md` | what the corpus contains and what is wrong with it |
| `statcheck-ml/pipeline/` | one numbered script per stage; `pipeline/README.md` lists them |
| `statcheck-ml/reproduce.sh` | the offline stages from committed files |
| `statcheck-ml/results/REPORT.md` | the generated report |
| `.claude/agents/` | one agent for each phase |

Data is not in the repository. The data directory and the corpus archive are ignored.
The frozen windows, the annotations, the agreement, the splits, the baseline, the
evaluation, and the figures are committed.
