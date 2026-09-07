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

| Pool | Source | Size | Purpose |
|---|---|---|---|
| A | windows where the regex found a result | 1500 | check the quality of the silver labels |
| B | windows that pass the filter, where the regex found nothing | 6000 | find the results the regex misses |
| C | windows that the filter rejected | 500 | measure what the filter drops |
| D | a frozen test set, stratified | 1000 | the only measure of success |

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

The language model annotates about 8000 windows in three independent passes. This is
roughly 24000 calls and about 17 million tokens. Use a small model.

A person annotates about 1500 windows, which is roughly 10 to 12 hours.
