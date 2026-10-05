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

## Clarification, 2026-10-04, before any run finished

The rules read the shipped system, the cascade (pattern first, then the model), as
`Pipeline().run_text` returns it, because "the results list" above is that output.
`pipeline/23_select_v4.py` runs with `--rule-system cascade`; the model alone is
reported beside it. When two means or two recalls are compared, a tie is a gap below
0.01 from the highest value, computed on exact fractions. At the time of this note no
version 4 run had a `report.json` and no dev score of a version 4 run existed.

## Amendment, 2026-10-05, at the owner's request, before any run finished

The owner asked to retrain only the top three version 2 configurations. The plan is
reduced to four runs, all on the CPU of the development machine, seed 0 only:

| Run | Configuration | Noise |
|---|---|---|
| `gate-old+all` | gru-crf | old default + all four new families |
| `gru-softmax-all-s0` | gru-softmax | old default + all four new families |
| `lstm-softmax-all-s0` | lstm-softmax | old default + all four new families |
| `gate-none` | gru-crf | none (the control for rule 1) |

The five single-family gate runs were stopped before they finished and are not
reported. The seeds 1 and 2 are not trained.

- Rule 1 applies as written, to `gate-none` against `gate-old+all`; it sets the gru-crf
  candidate.
- Rule 2 becomes: among the three candidates (the gru-crf run that rule 1 chose,
  `gru-softmax-all-s0`, `lstm-softmax-all-s0`), take the highest R_named; a gap below
  0.01 from the highest is a tie, broken by the lower FP, then by fewer parameters.
- Rule 3 applies as written.

At the time of this note no version 4 run had a `report.json`.

## The amendment withdrawn, 2026-10-05, at the owner's request

The owner then asked to keep all 23 runs and to train them on the owner's GPU. The
amendment above no longer applies; the 23-run plan and rules 1 to 3 as first written
(with the cascade clarification) stand. The four CPU runs of the amendment were stopped
before any finished and were deleted; no version 4 run had a `report.json` and no dev
score of a version 4 run existed.

## Exploratory runs, 2026-10-05, before any run

The owner has GPU time to spare. `pipeline/22_train_v4.py --plan extended` trains 63
runs: the three configurations, seeds 0 to 2, and seven noise settings (none, the old
default, the old default with each new family alone, and the old default with all four).
The 23 runs above are part of it under the same names and the same arguments; a check
in the script enforces this. Rules 1 to 3 read only the 23 runs. The other 40 runs are
exploratory: they are reported in their own table, marked as such, and never change the
choice.

## Batch size, 2026-10-05, before any run

On the owner's A100 a two-epoch smoke run did not finish in four minutes: the CRF steps
through every character in Python, so a GPU waits on launches. Two changes, both before
any version 4 run:

- `crf.py` scores the true path in one vectorised step and backtracks Viterbi on the CPU.
  On five random batches the loss agrees within 1e-4 and the Viterbi paths are identical.
- Every one of the 23 runs (and the 40 exploratory ones) trains with batch size 64, not
  the version 2 default of 16. The learning rate, the epochs, the patience and the noise
  stay as they are. The rules compare version 4 runs with each other and with the old
  model on the dev papers, so one batch size for all runs keeps the comparison fair.

## Five seeds, 2026-10-05, before any result was read

The owner has an A100 to spare and asked for the most reliable plan. Two changes:

- The grid trains seeds 0 to 4, not 0 to 2: 12 more runs (seeds 3 and 4 of the three
  configurations, each with the noise settings none and all), 35 runs in all.
- Rule 1 compares the mean over seeds 0 to 4 of the gru-crf runs with no noise against
  the gru-crf runs with all noise: the mean R_named, and the mean FP for the limit and
  the tie. Rule 2 takes the mean over seeds 0 to 4. Rule 3 is unchanged: the shipped
  model is the seed-0 run of the choice, so no seed is chosen by its score.

State at this note: the owner's first GPU start ran for seven minutes and was stopped.
The owner pasted its progress table, which shows the training dev F1 (the bronze dev
split that stops the training early) for the first epochs of 12 runs. No run had
finished, no run was scored on the 36 gold-train dev papers that the rules read, and the
session has seen no dev-paper score of a version 4 run.

The runner now records the command of each run in `command.json` and skips a finished
run only when its command matches. One run of the owner's earlier start
(`gru-softmax-none-s0`, batch 16, from the kit before the batch-size note) therefore
trains again.
