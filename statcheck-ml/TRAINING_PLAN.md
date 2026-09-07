# Training and evaluation plan

This plan says what is trained, under which conditions, and how the result is
measured. Every number in the final report comes from one frozen test set that
no model has seen.

## The rule that governs everything

A model is measured once, on data it has never seen, against labels made by the
best annotation protocol available. Nothing else counts as a result.

## Step 1 — the frozen test set

The current test part is drawn from the same sampling run as the training data.
The documents do not overlap, but the sampling did. A separate test set removes
that objection.

| Property | Value |
|---|---|
| Source documents | papers not used in round 2, about 1800 of the 3100 remain |
| Size | 600 windows |
| Pools | A, AN, B1, B2 and C, in the round 2 proportions |
| Journals | all 63, with a cap for each journal |
| Annotation | two independent passes, then adjudication of every dispute |
| Adjudicator | the strongest model available |
| Status | frozen when it is built, evaluated once for each candidate |

The annotation protocol is the one that produced the current labels. Two blind
passes agree on 91.5% of windows and 84.1% of results. Every disagreement goes
to a third pass, which in the last round rejected values that both earlier
passes had wrong.

No model trains on this set. No decision is taken on it. Each evaluation
against it is written to a log with the date and the model, so a second
evaluation of a tuned model cannot be presented as a first one.

## Step 2 — the architectures

All character models share the tag set, the loss, the splits and the metric, so
the comparison is about the architecture alone.

| Id | Architecture | Size | Why it is included |
|---|---|---|---|
| A1 | BiLSTM | 0.6 M | the current model, and the reference |
| A2 | BiLSTM with a CRF | 0.6 M | the tag scheme forbids many sequences; the CRF enforces that |
| A3 | BiGRU | 0.5 M | fewer gates than an LSTM, so it is smaller and faster |
| A4 | BiGRU with a CRF | 0.5 M | the same question for the smaller recurrent unit |
| A5 | MobileBERT | 25 M | a transformer that is small enough to consider shipping |
| A6 | DistilBERT | 66 M | the accuracy ceiling, and a research baseline only |

A5 and A6 read sub-words, not characters. Their predictions are mapped back to
character spans before scoring, so every architecture is judged the same way.

Neither transformer is a shipping tier. A transformer cannot run natively in R,
and its tokenizer would have to be ported to JavaScript and to R as well, which
breaks the rule that one specification serves three ports.

## Step 3 — the training scenarios

| Id | Scenario | Content |
|---|---|---|
| S0 | plain | the annotated windows, and nothing else |
| S1 | adversarial | plus two perturbed copies of each window, plus 400 hard negatives |
| S2 | hard negatives only | plus 400 hard negatives, and no perturbation |
| S3 | operator damage only | plus perturbed copies where only the operator is damaged |

Augmentation is applied to the training part only. A perturbed development or
test window measures the generator, not the model.

S2 and S3 exist to answer a question S1 cannot: which family earned its place.
S1 changes several things at once, so a gain from S1 says nothing about why.

## Step 4 — the grid

The full grid is 24 runs, which is more compute than the question needs.

1. Run all six architectures under S0 and S1. That is 12 runs, and it answers
   both "which architecture" and "does adversarial training help".
2. Run S2 and S3 on the best architecture only. That is 2 more runs, and it
   says which noise family did the work.
3. Repeat the best configuration with three seeds. That is 2 more runs, and it
   says whether a difference between two runs is real.

Total: 16 runs. Every run records its seed, its data, and the exact label file.

## Step 5 — the three systems to compare

The report compares three systems, not two.

| System | What it is |
|---|---|
| statcheck | the real R package, version 1.5.0 |
| model | the best learned extractor, alone |
| hybrid | the prefilter, then statcheck, then the model for the rest |

The cascade order follows the measurement. statcheck reaches a precision of
1.000 and a recall near 0.205, so what it finds is accepted without question and
the model supplies the rest. An existing statcheck user therefore sees no
regression, and the model is answerable only for the results the package cannot
read.

## Step 6 — what is reported

**Finding.** Precision, recall and F1 for each of the three systems. Two
matchings are reported, because they answer different questions.

- by value: a result counts when the test statistic matches. This is the only
  matching statcheck can take part in, because it reports values and not
  positions.
- by span: a result counts when every character boundary and every label match.
  This is stricter and applies to the models alone.

**Finding, by subset.** The same table for damaged results and for undamaged
results. The whole argument for a learned extractor rests on the damaged part,
so it is reported apart rather than hidden inside an average.

**Checking.** Verdict agreement between the systems on the results both found,
and the rate of inconsistency each system reports. Published work puts the true
rate near 10%, so a rate far from that indicates an extraction fault rather
than a discovery.

**Cost.** Parameters, file size after export, and the time to read one window on
one core. Accuracy for each megabyte is the number that decides the shipping
tier.

**Ceiling.** Agreement between the two annotation passes, reported beside every
score. A model cannot be shown to exceed the labels it is measured against.

## Step 7 — what is not claimed

Every label comes from a language model. No measurement in this project can
find a mistake that the annotator makes every time.

Robustness is claimed only for the damage present in this corpus. The
adversarial scenarios teach a wider range, but the test set comes from the same
3100 papers, so it cannot prove that the wider range was learned.

One honest robustness test is available and is planned: the text supplied with
the corpus is corrupted in a different way from the conversion used here, and
the two forms are aligned for 3000 documents. A model trained on one form and
measured on the other is a real test of transfer.
