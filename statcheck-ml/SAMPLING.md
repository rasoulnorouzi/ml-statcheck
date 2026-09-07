# Sampling and annotation budget

Conversion and annotation have very different costs. Convert everything. Sample only
the annotation.

| Task | Cost for the whole corpus | Decision |
|---|---|---|
| Convert the PDF files | 53 seconds on 14 cores | convert all 3100 |
| Apply the regex to make silver labels | a few minutes | label all 3100 |
| Annotate with a language model | many hours and a large token cost | sample |
| Annotate by hand | many days | sample a small set |

## The scale of the problem

The clean corpus holds 3,661,147 lines of 20 characters or more. Only 4,989 of these
lines contain a test statistic, which is 0.1%.

The prefilter at a density of 0.20 keeps every statistical line. It still passes about
1,240,000 lines. That is far more than anyone can annotate, so the project samples
from the filtered set.

## Four annotation pools

Each pool answers a different question. Do not merge them.

| Pool | Source | Purpose | Measured yield |
|---|---|---|---|
| A | the pattern matched a complete result | check the silver labels | 100% |
| B1 | a p-value is present, no complete match | find prose and bare-statistic reports | 33% |
| B2 | a test letter and digits, no p-value | find damaged operators | 60% |
| C | the density filter rejected the line | measure what the filter drops | 0% |
| D | a frozen test set, stratified | the only measure of success | not yet drawn |

The yields come from round 1, which annotated 55 windows.

Pool B2 gives a better yield than pool B1, which was not expected. Pool B2 catches
the results whose operator became a control character, and those are the results a
pattern can never read. Weight pool B2 above pool B1 in the next round.

Pool B holds the value of the project. These windows are where a result hides in a
form the regex cannot match.

Enrich pool B before you sample it. Prefer a window that holds a test letter, a
parenthesis, and a digit, but that failed the full pattern. A near miss has a much
higher yield than a random window.

Pool C is small but it cannot be skipped. If 500 rejected windows hold no result, the
miss rate of the filter is below 0.6% with 95% confidence. Without pool C the recall
of the filter cannot be measured at all, because every other pool is drawn from text
the filter already kept.

Pool D is annotated by a person. It is frozen when it arrives, and no model trains on
it.

## Spend the budget in two rounds

Use 70% of the budget for the first round. Keep 30%.

Train a model on the first round. Then spend the rest on the windows where the model
is uncertain, and on the windows where the sentence head and the span head disagree.
A disagreement between the two heads is a strong signal that the window is difficult.

## Stratify the sample

Draw each pool across these dimensions:

- Field: psychology, economics, political science, sociology, education, management.
  Economics and political science differ most from APA style.
- Test type: t, F, r, z, chi-square, and Q.
- Journal, so that no journal dominates a pool.

## Split by document and by journal

Assign whole documents to the training set, the development set, or the test set.
Never split the windows of one document across two sets. Windows from one paper share
an author, a template, and a font, so a split by window leaks.

Hold out 8 complete journals, which is about 400 documents. These journals appear in
no training set. They measure whether the model reads a venue it has never seen.

## Cost estimate

Annotation runs through agents inside Claude Code, not through a script with an API
key. Each call therefore costs much more than an API call, so the totals are smaller.

Batch about 20 windows into one agent call. Annotate about 1000 windows in a round.
One round is therefore about 50 agent calls for each pass.

Run two passes for each round, and a third pass only where the two disagree. This
costs far less than three full passes and finds the same disagreements.

There is no human annotator. Every label comes from a language model. Section
"The limit of a model annotator" states what this costs.

## The limit of a model annotator

No measurement in this project can find a mistake that the annotator makes every
time. If the annotator always misses one way of reporting a result, the small model
learns to miss it too, and every score stays high.

More annotation does not remove this limit. More agreement does not remove it,
because the annotators share their training.

One independent check remains. The arithmetic does not come from the annotator. A
wrong degrees of freedom or a misread statistic usually gives a p-value far from the
reported one. Published work puts the true rate of inconsistency near 10%. A batch
that is far from that rate has an extraction fault, and no annotator was asked.

The arithmetic checks the results that were found. It says nothing about the results
that were missed.
