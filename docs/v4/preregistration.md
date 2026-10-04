# Version 4 — pre-registered selection rule

Written 2026-10-04, before any version 4 training run started. The commit that adds this
file is earlier than every run in `models/v4/`. The rule does not change after a result
is seen.

## Data

- Training set: `dataset/train_v4.jsonl` with the split `dataset/splits_v4.json`.
- Selection set: the 36 gold-train dev papers of `dataset/gold/dev.json`, scored by
  `pipeline/14_gold_eval.py` on those papers only. No run trains on them.
- Report-only sets: the gold test (169 papers) and the bronze holdout (576 windows). They
  never take part in a choice. Every candidate is reported on both.

## The runs

The runs train on the owner's GPU with `--epochs 30` and seeds as listed.

| Group | Configurations | Seeds | Noise |
|---|---|---|---|
| Noise gate | gru-crf | 0 | none; old default; old default + each new family alone (4 runs); old default + all four new families |
| Grid | gru-crf, gru-softmax, lstm-softmax | 0, 1, 2 | none; old default + all four new families |

The noise gate holds 7 runs and the grid 18; the two groups share 2 runs, so there are
23 runs in all.

## Metrics on the dev papers

- R_named: recall of in-scope gold results found with a test name (the results list).
- R_frag: recall with fragments counted.
- FP: finds reported as results in the dev papers that hold no result.

## Rule 1 — the noise setting

Compare the grid's two noise settings by their gru-crf seed-0 runs.

1. A setting is eligible only when its FP is at most the FP of "none" plus 2.
2. Among the eligible settings, take the higher R_named.
3. When R_named differs by less than 0.01, take the lower FP; then the setting "none".

The seven gate runs are reported per family. They explain the choice; they do not make it.

## Rule 2 — the configuration

With the noise setting of rule 1:

1. For each configuration, take the mean R_named over seeds 0, 1 and 2.
2. Take the configuration with the highest mean.
3. When two means differ by less than 0.01, take the one with the lower mean FP; then the
   one with fewer parameters.

## Rule 3 — the shipped model

The shipped model is the seed-0 run of the chosen configuration and noise setting, as in
version 2. It replaces `gru-crf-s0` only when its R_named on the dev papers is at least
the R_named of `gru-crf-s0` on the same papers. Otherwise the old model stays, and the
report says so.

## What is reported

For every one of the 23 runs and for the old `gru-crf-s0`: R_named, R_frag and FP on the
dev papers, then on the gold test, and precision, recall and F1 on the bronze holdout.
The gold test numbers are read once, after the choice, and they are logged in
`results/gold_eval_log.jsonl`.
