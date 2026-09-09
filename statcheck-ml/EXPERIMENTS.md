# Experiment log

Every training run, what it changed, and what it measured. A run that failed is
kept, because a failed run is the only evidence that a plausible idea does not
work.

Two numbers appear for each run:

| Number | What it is |
|---|---|
| dev F1 | measured during training, on the development split |
| holdout F2 | measured after training, on the holdout, in the cascade with statcheck |

**The holdout F2 decides.** statcheck-ml is a screening tool: a missed result is
invisible to a reader, and a false positive costs a few seconds of attention.
F2 weights recall four times as heavily as precision inside the harmonic mean.

Dev F1 was a poor guide. `final-gru-crf` has the best dev F1 of every character
model and it is not the model that ships.

---

## The models that ship

| Role | Model | Parameters | Holdout F2 |
|---|---|---|---|
| **default** | `final-crf-aug` + statcheck | 616,072 | **0.931** |
| **lite, for a browser** | `final-gru-crf` + statcheck | 466,696 | 0.918 |
| high precision, optional | `v2-gru-crf-aug` + statcheck | 466,696 | 0.896 |

---

## 1. The first chain: architecture and augmentation

Eleven character models. `run_all_training.sh` holds every command.

| Run | Configuration | dev F1 | Holdout F2, in the cascade |
|---|---|---|---|
| `final-softmax` | no CRF | 0.900 | 0.881 |
| `final-crf` | CRF | 0.916 | 0.928 |
| `final-aug` | augmentation, no CRF | 0.875 | 0.916 |
| `final-gru` | GRU, no CRF | 0.897 | 0.894 |
| `final-gru-crf` | GRU with CRF | **0.927** | 0.918 |
| **`final-crf-aug`** | **CRF with augmentation** | 0.917 | **0.931** |
| `final-gru-aug` | GRU with augmentation | 0.906 | 0.922 |
| `abl-hardneg-only` | CRF, hard negatives alone | 0.922 | 0.870 |
| `abl-perturb-only` | CRF, perturbation alone | 0.909 | 0.917 |

**What the ablations show.** Each augmentation family alone gives higher
precision and materially worse recall: 0.848 for hard negatives alone and 0.908
for perturbation alone, against 0.937 for the two together. The families do
complementary work, and neither is redundant.

## 2. The transformers

| Run | dev F1 |
|---|---|
| `final-mobilebert` | 0.333 |
| `final-distilbert` | 0.333 |

**This is not overfitting.** The loss fell to 0.04 while precision stayed at
0.26. A word-piece tokenizer deletes control characters before it splits
anything, and in this corpus a control character is a destroyed operator, so the
transformer read `F(1,17) < 35.72` and `F(1,17) = 35.72` as the same string on
half the holdout.

`train_bert.py` now uses a cased model and swaps each destroyed operator for a
visible character both tokenizers keep. **No transformer was retrained with that
fix.** Treat 0.333 as a measurement of word-piece tokenization on damaged text,
not as the ceiling of a transformer.

## 3. The second chain: two ideas from error analysis, both rejected

Error analysis on the holdout found 28 false positives and 28 false negatives.
Reading all of them suggested two changes.

**Idea 1, from the false negatives.** 20 of 28 misses were damaged, and the
largest pattern was a destroyed character *inside* the number rather than in the
operator:

```
t(8)   \x01 \x034.40   the statistic is -4.40
t(454) \x02 \x031.950  the statistic is -1.950
r = 0\x0592            the statistic is 0.92
```

`damage_number` was written to teach this.

**Idea 2, from the false positives.** Two thirds were not statistics at all: `t`
used as a time index in economics, percentages, effect-size conventions, means
beside means, confidence-interval bounds, section numbers. Sixteen hard-negative
patterns were written, each from a specific observed false positive.

### What the measurements said

| Run | Changed from `final-crf-aug` | P | R | F2 | FP |
|---|---|---|---|---|---|
| **`final-crf-aug`** | nothing; the baseline | 0.910 | **0.937** | **0.931** | 29 |
| `v2-crf-aug` | both ideas, 400 negatives | 0.975 | 0.873 | 0.892 | 7 |
| `v3-crf-hn150` | both ideas, 150 negatives | 0.958 | 0.879 | 0.894 | 12 |
| `v4-crf-nohn` | `damage_number` alone, no negatives | 0.972 | 0.876 | 0.894 | 8 |

**Every variant lost recall, and none beat the baseline on F2.**

### Which idea caused the loss

The first reading blamed the hard negatives, because they are designed to
suppress. `v4-crf-nohn` settles it: it carries **no hard negatives at all** and
still loses recall.

| Run | damaged recall | undamaged recall |
|---|---|---|
| `final-crf-aug` | **0.877** | **0.947** |
| `v4-crf-nohn` | 0.816 | 0.842 |

**`damage_number` is the cause, and it failed at its own purpose**: damaged
recall fell, which is exactly what it was written to raise.

The likely reason is dose, not idea. The transform fires on 97% of examples, so
nearly every augmented copy carries a mangled number, and the model loses its
grip on ordinary ones.

### What was done about it

Both are **switched off by default**, so the shipped configuration reproduces
`final-crf-aug`. Neither was deleted: `Augmenter(number_damage=True)` and
`Augmenter(extra_negatives=True)` bring them back for anyone who wants to try a
smaller dose.

**A retrain from this repository reproduces the model that ships.** That was the
point of gating them rather than leaving them on.

### What is worth keeping from the failure

The hard negatives work at what they were built for. False positives fell from
28 to 6, and `v2-gru-crf-aug` reaches precision 0.986 with 4 false positives
against 16 for the model that ships.

**That is a real variant, not a consolation.** A first pass over thousands of
papers, where a person reads every flag, wants precision. It is documented as an
option and it is not the default.

---

## 4. What would be tried next

1. **A smaller dose of `damage_number`.** Fire it on 10% to 20% of examples
   rather than 97%. The idea is untested at a sensible rate.
2. **The extra hard negatives on their own.** They have only ever been measured
   beside `damage_number`, so their own effect is unknown.
3. **Retrain with `ENGINE_ALPHABETS`.** Family 4 holds the characters other PDF
   engines produce. No model has been trained with it.

---

## 5. How to repeat any of this

```
bash run_all_training.sh          # the first chain, skips finished runs

python evaluate_all.py data/holdout/windows.json \
    data/holdout/annotations_final.json data/holdout/statcheck_r.csv \
    <out.json> <name>=<model.pt>[:crf] ... \
    --repaired=data/holdout/statcheck_repaired2.csv
```

Each run writes `models/<name>.log` with the loss and the development score of
every epoch, and `models/<name>/report.json` with the final tables. Both are
kept for every run named in this file, including the ones that failed.
