# Context — what happened, and why

Read this file with `PLAN.md`. `PLAN.md` says what the phases are and where each one
stands. This file says why each decision was made, and what the corpus showed.

A new session should read `CLAUDE.md`, then `PLAN.md`, then this file.

Last updated: 2026-09-07.

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

This is also why a character model is preferred to a transformer. A character map
ports to three languages. A subword tokenizer does not.

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

## Known problems

1. A window can cut a result in half. One pool A window began in the middle of a
   result, because the result started before the window. Snap window edges to
   sentence or paragraph boundaries.
2. Detection is not the same as checking. A result with no degrees of freedom and no
   p-value is real but cannot be recomputed. Report the two counts apart.
3. Project agents need a session restart before Claude Code can dispatch them.

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
| `statcheck-ml/CORPUS.md` | what the corpus contains and what is wrong with it |
| `statcheck-ml/SAMPLING.md` | annotation pools and budget |
| `statcheck-ml/SCHEMA.md` | annotation format and training format |
| `statcheck-ml/convert.py` | PDF to text with PyMuPDF |
| `statcheck-ml/sample_windows.py` | build windows, strip references, draw pools |
| `statcheck-ml/to_bioes.py` | align annotations, emit character tags |
| `.claude/agents/` | one agent for each phase |

Data is not in the repository. The data directory and the corpus archive are ignored.
