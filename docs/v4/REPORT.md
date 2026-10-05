# Version 4 report

Date: 2026-10-05. Branch `v4-gold`, written at commit `b1c3747`. Nothing is pushed and
nothing is published. The owner decides whether to publish this report.

Every number below comes from a committed file. The source files are named at the
head of each section. A number that was read from the gold test after the model choice
is marked "read after the choice".

## 1. What version 4 changed, and the result

Version 4 brought three things to the project: a human-coded gold set with a frozen test
part, a larger annotation unit (the merged passage) with a fourth vote from the human
coders, and a pre-registered training grid of 35 runs on the owner's A100. It also
repaired two faults in the extraction pipeline without training. The result:

1. The shipped model stays `gru-crf-s0`. On the 36 dev papers the pre-registered
   candidate `lstm-softmax-none-s0` had a named recall of 0.824 against 0.827 for the
   old model, so rule 3 kept the old model (`results/v4/selection.md`).
2. On the gold test (169 papers, 476 results) the shipped cascade finds 0.723 of the
   results with a test name and 0.866 when fragments count. It reports 69 finds in the
   120 papers with no result, down from 155 before version 4 (`results/v4/compare.md`).
3. R statcheck on the same papers finds 0.479 (passages), 0.443 (repaired text), 0.237
   (PDF) and 0.113 (raw text). The paired difference of the shipped cascade against R
   statcheck on passages is +0.244 [+0.092, +0.394] (`results/v4/final/paired.md`).
4. Read after the choice: the candidate cascade finds 0.763 against 0.723, a difference
   of +0.040 with a paper-bootstrap interval of -0.013 to +0.123, so the gain is not
   supported at paper level. It makes 77 finds in the no-result papers against 69, and
   its bronze F1 is 0.876 against 0.904 (`results/v4/final/summary.md`, `paired.md`).
5. R statcheck's own check on the coders' values gives 73 inconsistencies in 15 papers.
   The shipped cascade flags 32 of them, R statcheck on passages 26, R statcheck on the
   PDF 10. `pvalue.py` agrees with R on 465 of 465 reference verdicts
   (`results/v4/final/verdicts.md`).

## 2. The data

Sources: `docs/superpowers/specs/2026-10-04-v4-gold-brief.md`, `docs/v4/LEDGER.md`
(M1.3, M3.2, M3.2b, M3c), `docs/v4/STATUS.md`, `dataset/passages/inventory_summary.json`,
`dataset/passages/sample_summary.json`, `dataset/passages/consensus.json`,
`dataset/annotations/passages/*.json`.

### 2.1 The gold set

Three human coders (MS, ME, MN) coded 409 papers of the corpus by hand and copied each
quote from the PDF. The sheet holds 2704 coded results in 185 papers. The other 224
papers are marked "no result reported in the text"; they measure false finds.
`pipeline/14_gold.py` located 2664 of the 2704 results in the converted text.

The coders followed the statcheck scope. They skip a correlation without degrees of
freedom, which our guideline marks. They code CI, Wald and `b` rows, which our guideline
does not mark. The owner says the labels are not always right: 135 of 2704 rows hold a
typed value that is not in the quote. Most of these correct broken PDF spacing; a few
dozen are typing errors.

The in-scope test types are t, F, r, z, chi2 and Q. On all 409 papers the prefilter
keeps 0.949 of the gold results; on the gold test it keeps 0.992.

### 2.2 The split

`pipeline/14_gold_split.py` split the gold papers by paper, seed 0. The split is frozen
in `dataset/gold/split.json` (sha256 fbb0214f…); a rerun gives the same bytes.

| Part | Papers | Papers with results | Located in-scope results |
|---|---|---|---|
| gold test | 169 (the 49 papers of the bronze holdout, 120 in no bronze set) | 49 | 476 |
| gold train | 240 (181 in the bronze training set, 59 in no bronze set) | 136 | 2007 |

No training input reads the gold test. `pipeline/14_gold_dev_split.py` then drew the
dev papers from gold train: 15 % by paper, seed 0, stratified by `has_results`. The dev
set holds 36 papers (20 with results, 16 without) and 278 located in-scope results. The
selection rule reads these 36 papers and nothing else.

### 2.3 The passage sample

The processing unit of version 4 is the merged passage (section 3.3). The inventory
(`pipeline/17_passage_inventory.py`) covers the 204 gold-train papers outside the dev
set: 12592 pieces of at most 1500 characters; 767 units were split into pieces, the
largest unit 10021 characters. Of the 1812 located coder rows in these papers, 1725 lie
inside a passage, 80 outside any unit and 7 on a blanked reference line; 32 more rows
were never located. The span check failed on 0 rows. Passages by flag: 621 hold a gold
row, 585 an in-scope gold row, 141 a damaged gold row; the cascade fires in 866; 80 are
hard negatives; 1531 are damaged with no gold row.

`pipeline/18_sample_passages.py` drew from four strata without overlap, seed 0:

| Stratum | Definition (the sampler's own) | Pool | Cap | Drawn |
|---|---|---|---|---|
| gold | passages with in-scope coder rows | 585 | 900 | 585 |
| hard | passages of papers without results, but the cascade fires | 80 | 400 | 80 |
| damaged | passages of damaged papers with no gold, cascade fires first | 1531 | 200 | 200 |
| clean | no gold and the cascade does not fire, round-robin by journal | 10188 | 300 | 300 |
| total | | | | 1165 |

The 1165 passages come from 196 papers and 61 journals. The clean passages come from
all 59 journals, 5 or 6 each. The sample holds 0 gold-test papers and 0 dev papers.
`02_chunk.py --unpadded --size 10` cut it into 117 batches of ten (the last holds 5);
each record holds only `window_id` and `text`.

### 2.4 The four-vote labels

Three language-model raters read every batch blind: the passage and guideline 3.0
(sha256 961f3145…), nothing else. Each ran as a `general-purpose` agent with the model
set: `claude-haiku-4-5-20251001`, `claude-sonnet-5-5`, `claude-opus-5-5`. That gave 351
rater files. Haiku needed a retry on 9 batches (batch_21 twice: one attempt returned 9
outputs for 10 records); opus needed one retry (batch_75: two window ids swapped);
sonnet needed none. Results per rater: haiku 1906 (182 unaligned), sonnet 1920,
opus 1998.

The coders are the fourth vote (`docs/PROTOCOL.md` version 3). Their rows in the
sampled passages: 1662, of which 13 out of scope, 1644 in scope and located, 5 unlocated
and dropped, 24 with a clamped span. Two spans match when they overlap by 50 % or more
of the shorter span. A result stays with two votes of four; any two voters are enough.
A field takes the value with two votes or more and more votes than any other value. A
result with one vote is a singleton and goes to the adjudicator.

`pipeline/19_consensus.py` kept 1945 results: 879 unanimous and 1066 by majority, 61
with a field conflict. A coder vote backs 0.8401 of the kept results (1634 by span, 311
with no coder row). Singletons, one voter alone: haiku 9, sonnet 3, opus 54, coders 10.
Field conflicts by field: damaged 41, statistic 14, df1 4, p_operator 4, test_type 3,
df2 3. The script wrote 145 disputes into 8 batches.

A blind adjudicator (opus) judged the disputes. `pipeline/20_merge.py` then wrote
`dataset/annotations/passages/final.json`: 1994 results in 1165 passages, of which 879
unanimous, 1005 majority and 110 adjudicated; 49 singletons kept, 27 dropped; 69 field
conflicts resolved. Every passage label is bronze. No machine annotation is gold.

`pipeline/21_dataset_v4.py` built the training set `dataset/train_v4.jsonl` from the
1962 bronze windows of `dataset/train.jsonl` and the 1165 passages. It left out the 35
bronze windows that come from the 36 dev papers, so 1927 bronze windows remain. The bronze windows agreed with the coders on 191
of the 200 results they share.

## 3. The changes made without training

Sources: the brief ("Faults found", "Window size"), `docs/v4/LEDGER.md` (M1, M2, M2b),
`results/v4/compare.md`.

### 3.1 Dedup by statistic position (M2.2)

`Pipeline.run_text` dropped a result when an earlier result in the paper had the same
statistic value: `seen` was keyed on the statistic alone. 124 gold rows share a
statistic with another row in the same paper; the cost was about 1.8 points of recall.
Now every find carries `statistic_span`, and a find is a duplicate only when its
interval overlaps one already emitted. On the gold test: regex 0.489 to 0.498, model
0.836 to 0.849, cascade 0.851 to 0.866; finds in the 120 no-result papers 155 to 157.

### 3.2 Finds with no test name become fragments (M2.3)

The model reported bare numbers with no test name: about 180 such finds in papers with
results, and most of the 304 finds in no-result papers. Such a find is now a fragment.
It keeps every result key except the verdict and never reaches `check_one`. On the gold
test the named recall of the cascade fell from 0.866 to 0.723 (model 0.849 to 0.689),
while recall with fragments stayed 0.866 (model 0.849). The drop is a reclassification.
Finds in the no-result papers fell from 157 to 73, plus 84 fragments.

### 3.3 Merged passages as the unit (M2.4)

Before the change, the brief measured the unit without training on all gold papers:

| Unit | Recall | Text read | Finds in no-result papers |
|---|---|---|---|
| overlapping 250-character windows (shipped) | 0.862 | 28.1 M | 304 |
| the same windows merged into passages | 0.863 | 15.8 M | 287 |
| passages from 1000-character windows | 0.877 | 24.6 M | 530 |

A merged passage reads each character once. `prefilter.json` is now version 2 with
`unit: "passage"`. On the gold test the recall did not change (cascade 0.723 named,
0.866 with fragments); finds in the no-result papers fell from 73 to 69; the characters
the model reads on the 169 papers fell from 11,433,629 to 6,503,399 (0.569).

### 3.4 Close-out (M2b)

Passages made the `line` of a result the first line of its passage (the R sample went
from 2, 2, 2, 2, 4 to 0 for all five rows). The line now comes from the statistic
position in all three ports (the R sample gives 2, 3, 4, 5, 6). A flaky import-path
test was fixed. `R CMD check` gave Status OK; with pandoc 3.12 and the vignette built,
Status OK again. The compare stage prints both recalls.

### 3.5 Before and after

Gold test, 169 papers, 476 results (`results/v4/compare.md`). The gold test was read
after each of these steps, so these numbers are not a clean held-out score (section 7).

| System | Recall before | Recall after, named | After, with fragments | Finds in no-result papers, before | After |
|---|---|---|---|---|---|
| regex | 0.489 | 0.498 | 0.498 | 2 | 2 |
| model | 0.836 | 0.689 | 0.849 | 155 | 69 |
| cascade | 0.851 | 0.723 | 0.866 | 155 | 69 |
| statcheck R, raw text | 0.113 | 0.113 | — | 2 | 2 |
| statcheck R, repaired text | 0.443 | 0.443 | — | 2 | 2 |
| statcheck R, passages | 0.479 | 0.479 | — | 2 | 2 |
| statcheck R, PDF | 0.237 | 0.237 | — | 2 | 2 |

Recall on the 109 damaged gold-test rows: cascade 0.817 before, 0.761 after (named).
Gold train, a development figure: cascade 0.843 before, 0.743 named and 0.861 with
fragments after; finds in its no-result papers 149 to 54. Bronze holdout: a delta of
0.000 on every metric for all 16 systems (cascade P 0.949, R 0.870, F1 0.908).

Both ports follow the new spec, the dedup rule, fragments and the line rule. At the
M2b close the suites gave Python 419 passed, R 1280 expectations with 0 failed, web 385
passed. Port commits: R `1e4726d`, web `2ff741a`; mother `dcb3a3b`.

### 3.6 Removals

M1.1 removed tracked dead code: `src/statcheck_ml/hybrid.py` (zero importers),
`repair.py::repair_document` and `repair.py::report` (zero call sites), a duplicate
`.gitignore` line, and the silver row of `SAMPLING.md`. It also deleted the owner's
untracked notebook `tes.ipynb` by mistake; the brief now forbids deleting untracked
files. M1.2 renamed the bronze holdout key from `gold` to `bronze` in every result file;
no bronze number moved. M3.2b removed a dead helper, two unused tuple members and the
`run_seconds` timing from the inventory summary. The ledger lists each removal.

## 4. Training

Sources: `docs/v4/preregistration.md`, `pipeline/22_train_v4.py`, `docs/v4/GPU.md`,
`docs/v4/STATUS.md`, `results/v4/selection.md`, the brief ("The noise layer").

### 4.1 The pre-registration

The selection rule was written on 2026-10-04, before any version 4 run started. The
rule did not change after a result was seen. Five dated notes follow it, each written
before any run finished: the rule reads the cascade; an amendment to four CPU runs at
the owner's request, then withdrawn; an exploratory 63-run plan that never changes the
choice; batch size 64 with a vectorised CRF path score (loss within 1e-4 on five
random batches, Viterbi paths identical); and five seeds instead of three, which made
35 runs.

### 4.2 The 35 runs and the noise families

The noise gate holds 7 runs of `gru-crf`, seed 0: no noise; the version 2 default
(`--augment 2 --hard-negatives 400`); the default plus each new family alone; the
default plus all four. The grid holds `gru-crf`, `gru-softmax` and `lstm-softmax`, seeds
0 to 4, with the two settings `none` and `all`. The two groups share 2 runs.

The four new families follow the owner's decision 5: `font_map` damages a whole paper
with one consistent font map; `negative_damage` applies the same damage to text that
holds no result; `spaced_digits` and `operator_drop` add two damage kinds the gold
quotes show. The brief counted the damage in the gold quotes and measured the old noise
layer on them before training, `gru-crf-s0` (trained with noise) against
`gru-crf-noaug-s0` (without):

| Damage in the quote | n | with noise | without |
|---|---|---|---|
| none | 1245 | 0.911 | 0.914 |
| control character | 495 | 0.715 | 0.719 |
| `¼` for `=` | 294 | 0.857 | 0.861 |
| operator missing | 287 | 0.892 | 0.913 |
| letter for operator | 209 | 0.928 | 0.957 |
| spaced digits | 93 | 0.516 | 0.527 |

Recall 0.857 with noise, 0.863 without; finds in no-result papers 292 with noise, 216
without. One seed of the no-noise model exists, so the recall gap is within noise; the
false-find gap is not. Real damage is per paper: spaced digits come from 10 papers,
control characters from 47.

Every run trains 30 epochs with batch size 64 on `dataset/train_v4.jsonl`.

### 4.3 The GPU

One epoch of one run takes 409 s on the 8-core development machine, so 35 runs of 30
epochs need about 120 hours on its CPU. The owner trained them on an A100 from a 3.3 MB
kit (`run_v4_gpu.sh prereg 12`): a smoke run of 2 epochs in 75 s, then 35 of 35 runs
with exit 0 in 38 min, 12 at a time, mean run 12 min. One run (`gru-softmax-none-s0`,
batch 16 from an older kit) trained again, as the settings check requires. The archive
has sha256 d4f90143…; all 35 ONNX exports pass parity.

### 4.4 Selection on the 36 dev papers

`pipeline/23_select_v4.py --rule-system cascade` scored every run on the dev papers.
The gold test and the bronze holdout were not read.

| run | R_named | R_frag | FP | cascade R_named | cascade FP | parameters |
|---|---|---|---|---|---|---|
| gate-none | 0.824 | 0.853 | 1 | 0.824 | 1 | 467912 |
| gate-old | 0.799 | 0.827 | 0 | 0.799 | 0 | 469000 |
| gate-old+font_map | 0.809 | 0.835 | 0 | 0.809 | 0 | 469000 |
| gate-old+negative_damage | 0.809 | 0.845 | 0 | 0.809 | 0 | 469320 |
| gate-old+spaced_digits | 0.809 | 0.853 | 0 | 0.809 | 0 | 469000 |
| gate-old+operator_drop | 0.845 | 0.878 | 2 | 0.845 | 2 | 469000 |
| gate-old+all | 0.806 | 0.838 | 0 | 0.806 | 0 | 469832 |
| gru-crf-none-s1 | 0.835 | 0.867 | 2 | 0.835 | 2 | 467912 |
| gru-crf-all-s1 | 0.820 | 0.856 | 1 | 0.820 | 1 | 469832 |
| gru-crf-none-s2 | 0.809 | 0.849 | 0 | 0.809 | 0 | 467912 |
| gru-crf-all-s2 | 0.795 | 0.835 | 1 | 0.795 | 1 | 469640 |
| gru-crf-none-s3 | 0.820 | 0.860 | 1 | 0.820 | 1 | 467912 |
| gru-crf-all-s3 | 0.795 | 0.842 | 1 | 0.795 | 1 | 469896 |
| gru-crf-none-s4 | 0.806 | 0.849 | 1 | 0.806 | 1 | 467912 |
| gru-crf-all-s4 | 0.824 | 0.853 | 0 | 0.824 | 0 | 469640 |
| gru-softmax-none-s0 | 0.831 | 0.867 | 1 | 0.831 | 1 | 466469 |
| gru-softmax-all-s0 | 0.824 | 0.856 | 4 | 0.824 | 4 | 468389 |
| gru-softmax-none-s1 | 0.838 | 0.874 | 2 | 0.838 | 2 | 466469 |
| gru-softmax-all-s1 | 0.795 | 0.824 | 8 | 0.795 | 8 | 468389 |
| gru-softmax-none-s2 | 0.849 | 0.881 | 3 | 0.849 | 3 | 466469 |
| gru-softmax-all-s2 | 0.820 | 0.849 | 3 | 0.820 | 3 | 468197 |
| gru-softmax-none-s3 | 0.824 | 0.860 | 1 | 0.824 | 1 | 466469 |
| gru-softmax-all-s3 | 0.820 | 0.849 | 3 | 0.820 | 3 | 468453 |
| gru-softmax-none-s4 | 0.827 | 0.860 | 1 | 0.827 | 1 | 466469 |
| gru-softmax-all-s4 | 0.806 | 0.838 | 3 | 0.806 | 3 | 468197 |
| lstm-softmax-none-s0 | 0.824 | 0.860 | 0 | 0.824 | 0 | 614949 |
| lstm-softmax-all-s0 | 0.817 | 0.853 | 2 | 0.817 | 2 | 616869 |
| lstm-softmax-none-s1 | 0.817 | 0.842 | 1 | 0.817 | 1 | 614949 |
| lstm-softmax-all-s1 | 0.835 | 0.863 | 4 | 0.835 | 4 | 616869 |
| lstm-softmax-none-s2 | 0.831 | 0.860 | 2 | 0.831 | 2 | 614949 |
| lstm-softmax-all-s2 | 0.831 | 0.860 | 3 | 0.831 | 3 | 616677 |
| lstm-softmax-none-s3 | 0.824 | 0.860 | 0 | 0.824 | 0 | 614949 |
| lstm-softmax-all-s3 | 0.813 | 0.849 | 3 | 0.813 | 3 | 616933 |
| lstm-softmax-none-s4 | 0.831 | 0.863 | 2 | 0.831 | 2 | 614949 |
| lstm-softmax-all-s4 | 0.809 | 0.838 | 1 | 0.809 | 1 | 616677 |
| gru-crf-s0 (shipped) | 0.827 | 0.867 | 0 | 0.827 | 0 | 466661 |

Means over seeds 0 to 4 with the noise setting of rule 1:

| configuration | mean R_named | mean FP | parameters |
|---|---|---|---|
| gru-crf | 0.819 | 1.0 | 467912 |
| gru-softmax | 0.834 | 1.6 | 466469 |
| lstm-softmax | 0.825 | 1.0 | 614949 |

- Rule 1 chose the noise setting `none`: the mean R_named of `gru-crf` is 0.819 with no
  noise and 0.808 with `all`, a gap of 0.01 or more, so the higher wins. The FP limit
  was 3.0.
- Rule 2 chose `lstm-softmax`: its mean 0.825 lies within 0.01 of the top (`gru-softmax`
  0.834), so the tie goes to the lower mean FP, 1.0 against 1.6.
- Rule 3 kept the old model: `lstm-softmax-none-s0` has R_named 0.824 on the dev
  papers against 0.827 for `gru-crf-s0`, and the rule replaces the old model only when
  the candidate is at least as high.

The seven gate runs explain the choice; they do not make it. Among them the single
family `operator_drop` had the highest dev R_named (0.845, FP 2).

## 5. Results on the gold test and the bronze holdout

Sources: `results/v4/final/summary.md`, `results/v4/final/paired.md`. The gold test was
read once per model, after the choice, by `pipeline/24_final_v4.py` (36 log lines
`v4-final:*` in `results/gold_eval_log.jsonl`). The gold test and the bronze holdout
are different labels and are never pooled; the tables below keep them in separate
column groups.

### 5.1 The shipped model and the candidate

Read after the choice. "Model" is the tagger alone; "cascade" is the pattern first,
then the model, as `Pipeline().run_text` returns it. The bronze columns are for the
model alone.

| system | gold model R_named [95% CI] | gold model R_frag | gold model R damaged | gold cascade R_named | gold finds in 120 no-result papers | bronze P | bronze R | bronze F1 |
|---|---|---|---|---|---|---|---|---|
| shipped `gru-crf-s0` | 0.689 [0.532, 0.803] | 0.849 | 0.752 | 0.723 | 69 | 0.952 | 0.861 | 0.904 |
| candidate `lstm-softmax-none-s0` | 0.731 [0.561, 0.847] | 0.880 | 0.807 | 0.763 | 77 | 0.899 | 0.854 | 0.876 |

The brief set the recovery of the test name as a main target of the new training data.
The gap between recall with fragments and named recall was 0.143 for the shipped
cascade (0.866 against 0.723). For the model alone the pair is 0.849 and 0.689 for the
shipped model, 0.880 and 0.731 for the candidate.

### 5.2 Means over seeds 0 to 4

Read after the choice. Gold columns are the gold test; the bronze column is the bronze
holdout.

| configuration | noise | gold model R_named | gold cascade R_named | gold finds in no-result papers | bronze F1 |
|---|---|---|---|---|---|
| gru-crf | none | 0.714 | 0.745 | 66.000 | 0.917 |
| gru-crf | all | 0.696 | 0.732 | 64.600 | 0.906 |
| gru-softmax | none | 0.733 | 0.763 | 72.800 | 0.881 |
| gru-softmax | all | 0.721 | 0.754 | 77.200 | 0.887 |
| lstm-softmax | none | 0.735 | 0.764 | 74.400 | 0.870 |
| lstm-softmax | all | 0.696 | 0.727 | 73.400 | 0.880 |

Every one of the 36 models is listed in `results/v4/final/summary.md`. The gold cascade
R_named of single runs ranges from 0.704 (`lstm-softmax-all-s4`) to 0.796
(`lstm-softmax-none-s4`); the seven gate runs range from 0.729 to 0.784
(`gate-old+operator_drop`, with 75 finds in the no-result papers).

### 5.3 R statcheck in its four forms

`pipeline/15_statcheck_gold.py` ran R 4.5.0 with statcheck 1.5.0 on the gold papers:
the raw full text, the repaired full text, the merged passages, and the PDF through
`checkPDF`. 7 PDFs fail inside `checkPDF` and count as misses. The baseline numbers
were read in M1 and did not change; the shipped cascade number is from the M2 close.

| system | gold R_named [95% CI] | gold finds in 120 no-result papers | R damaged |
|---|---|---|---|
| shipped cascade | 0.723 [0.556, 0.836] | 69 | 0.761 |
| Python regex port | 0.498 [0.299, 0.650] | 2 | 0.578 |
| statcheck R, raw full text | 0.113 [0.036, 0.257] | 2 | 0.000 |
| statcheck R, repaired full text | 0.443 [0.238, 0.646] | 2 | 0.606 |
| statcheck R, passages | 0.479 [0.293, 0.654] | 2 | 0.606 |
| statcheck R, PDF (`checkPDF`) | 0.237 [0.055, 0.493] | 2 | 0.000 |

By test on the gold test, repaired statcheck against the cascade (M1.6, before the
M2 changes): F 0.738 against 0.965, t 0.284 against 0.801, chi2 0.198 against 0.889,
z 0.167 against 0.389.

On the bronze holdout, where the R baseline exists only on the windows: statcheck raw
P 0.983, R 0.183, F1 0.308; statcheck repaired 0.993, 0.467, 0.636; the shipped
cascade 0.949, 0.870, 0.908.

### 5.4 Paired tests with the paper bootstrap

Read after the choice. `pipeline/26_paired_v4.py` reads the stored rows under
`results/v4/final/`, seed 0, 2000 paper resamples. Rows within a paper are not
independent, so the paper bootstrap is the main interval. The exact McNemar p treats
rows as independent and is too small when results cluster in a few papers; read it as
a secondary figure.

Population: the 476 gold rows. "Found" means the verdict is not null.

| A | B | A only | B only | both | neither | recall A | recall B | diff A-B [95% paper CI] | McNemar p |
|---|---|---|---|---|---|---|---|---|---|
| cascade | cascade_v4 | 4 | 23 | 340 | 109 | 0.723 | 0.763 | -0.040 [-0.123, +0.013] | 3.11e-04 |
| cascade | statcheck_r_passages | 128 | 12 | 216 | 120 | 0.723 | 0.479 | +0.244 [+0.092, +0.394] | 1.15e-25 |
| cascade | statcheck_r_pdf | 246 | 15 | 98 | 117 | 0.723 | 0.237 | +0.485 [+0.253, +0.684] | 5.18e-55 |
| cascade_v4 | statcheck_r_passages | 149 | 14 | 214 | 99 | 0.763 | 0.479 | +0.284 [+0.131, +0.419] | 1.14e-29 |

The candidate against the shipped cascade: the row-level McNemar p of 0.0003 says the
gain is real, the paper bootstrap says it is not supported (the interval holds zero).
The paper bootstrap is the figure to read. The shipped cascade against R statcheck is
supported in both forms.

### 5.5 The combined table

Gold columns come from the gold test; bronze columns come from the bronze holdout. The
bronze labels are machine annotations, not human validation. Bronze figures for the
two cascades are for the model alone. `statcheck_raw` and `statcheck_repaired` exist
only on the bronze windows.

| system | gold recall (named) [95% CI] | gold finds in no-result papers | bronze precision | bronze recall | bronze F1 |
|---|---|---|---|---|---|
| shipped cascade (gru-crf-s0) | 0.723 [0.556, 0.836] | 69 | 0.952 | 0.861 | 0.904 |
| candidate cascade (lstm-softmax-none-s0), read after the choice | 0.763 [0.594, 0.871] | 77 | 0.899 | 0.854 | 0.876 |
| regex | 0.498 [0.299, 0.650] | 2 | — | — | — |
| statcheck_r_fulltext_raw | 0.113 [0.036, 0.257] | 2 | — | — | — |
| statcheck_r_fulltext_repaired | 0.443 [0.238, 0.646] | 2 | — | — | — |
| statcheck_r_passages | 0.479 [0.293, 0.654] | 2 | — | — | — |
| statcheck_r_pdf | 0.237 [0.055, 0.493] | 2 | — | — | — |
| statcheck_raw (bronze windows only) | — | — | 0.983 | 0.183 | 0.308 |
| statcheck_repaired (bronze windows only) | — | — | 0.993 | 0.467 | 0.636 |

## 6. Verdicts

Source: `results/v4/final/verdicts.md`. The stage `pipeline/25_verdicts_v4.py` ran
after the choice; every number here is read after the choice. Our own p-value code did
not change.

### 6.1 The reference

The owner asked for the original statcheck check, not only its extraction. The stage
builds one APA string per coded gold-test row from the values the coders typed (test,
df, statistic, p operator, p) and runs R statcheck 1.5.0 on it. That gives a reference
verdict from statcheck itself.

| rows | with reference | consistent | inconsistencies | decision errors |
|---|---|---|---|---|
| 476 | 465 | 392 | 73 | 20 |

The inconsistencies include the decision errors. The rows with no reference: 2 with no
degrees of freedom coded, 3 with no p-value, 6 with no statistic. Of the 465 strings,
463 take the statistic and 453 take p as `raw` prints them; 4 are "ns"; 2 write a coded
≤ or ≥ as < or >. In 9 strings `raw` does not print a coded value, so the coded value
is used; 2 of the reference inconsistencies are such rows, and some of them are coding
slips (section 7).

`pvalue.py` gives the R verdict on 465 of 465 rows. The largest relative difference in
the computed p is 3.52e-13; 0 rows exceed 1e-06.

### 6.2 What each system flags

A system is scored on the rows that have a reference. "Flagged" means inconsistent or
decision error. The 73 reference inconsistencies lie in 15 papers.

| system | flagged of 73 | share [95% CI] | not found | found, judged consistent | found, not checked | papers flagged correctly at least once, of 15 |
|---|---|---|---|---|---|---|
| shipped cascade | 32 | 0.438 [0.163, 0.843] | 38 | 0 | 3 | 9 |
| candidate cascade (cascade_v4) | 34 | 0.466 [0.191, 0.865] | 26 | 0 | 13 | 10 |
| regex | 28 | 0.384 [0.125, 0.750] | 45 | 0 | 0 | 8 |
| statcheck R, raw full text | 6 | 0.082 [0.0, 0.295] | 67 | 0 | 0 | 2 |
| statcheck R, repaired full text | 25 | 0.342 [0.095, 0.696] | 47 | 1 | 0 | 7 |
| statcheck R, passages | 26 | 0.356 [0.106, 0.704] | 46 | 1 | 0 | 8 |
| statcheck R, PDF | 10 | 0.137 [0.0, 0.423] | 63 | 0 | 0 | 3 |

"Found, not checked" counts a row the system found without a test name or a p-value,
so that no verdict was possible. No system judged a reference inconsistency consistent
except R statcheck on the repaired text and on passages, 1 row each. A missed
inconsistency is a missed find: the shipped cascade misses 38 of 73 because it did not
find the row.

Of the 20 reference decision errors, the shipped cascade flags 4 as decision errors
and misses 15 (1 found, not checked); the candidate flags 4 and misses 14; the regex
and R statcheck on repaired text or passages flag 2 each; R statcheck on the raw text
and on the PDF flag 0.

On the 392 rows the reference calls consistent, the shipped cascade found 307 and
flagged 1; the candidate found 314 and flagged 0, with 2 flags on finds outside gold in
papers with results. Every other system flagged 0 of the consistent rows. No system
flagged a find in a paper with no result. A find outside gold is not always wrong: the
coders followed statcheck's scope.

Paired on the 73 inconsistencies (paper bootstrap, read after the choice): candidate
against shipped -0.027 [-0.046, +0.000] with 0 rows the shipped cascade alone flags and
2 the candidate alone flags; shipped against R statcheck on passages +0.082 [+0.000,
+0.244]; shipped against R statcheck on the PDF +0.301 [+0.065, +0.613].

## 7. Limits

- The gold test is not a clean held-out score for the rule changes of section 3. The
  log `results/gold_eval_log.jsonl` holds five labelled evaluations of the gold test
  under M1 and M2 (m1-before, m2-dedup, m2-fragments, m2-passages, m2-after), and M2b
  added m2-close. Each measured a rule change after the previous number was seen. The
  gold test is clean for the trained models: no training input reads it, the selection
  rule reads the 36 dev papers only, and `pipeline/24_final_v4.py` read the gold test
  once per model after the choice.
- The dev set of 278 results in 36 papers was too small to separate the models. Rule 3
  turned on 0.824 against 0.827. On the gold test, read after the choice, the order is
  reversed (0.763 against 0.723) and the paper bootstrap does not support the gap
  either way.
- Rows within a paper are not independent. The 476 results sit in 49 papers, and the
  73 reference inconsistencies in 15. The paper bootstrap intervals are wide for that
  reason: 0.556 to 0.836 for the shipped cascade's recall. The exact McNemar p counts
  rows and is too small.
- The coders followed statcheck's scope. They skip a correlation without degrees of
  freedom and code CI, Wald and `b` rows. A find outside gold is therefore not always
  wrong, and the finds in the no-result papers are an upper bound on false finds.
- Coder typing slips affect the reference verdicts. 135 of 2704 rows hold a typed value
  that is not in the quote; the reference strings use the coded value in 9 of 465
  cases, and 2 of the 73 reference inconsistencies are such rows.
- The bronze holdout has machine labels (three language-model raters and an
  adjudicator). Its precision and recall measure agreement with those labels, not with
  a human. The passage labels of version 4 are bronze too: the coders are one vote of
  four, never a gold label.
- The shipped model trained on bronze windows that overlap the gold-train papers. Its
  gold-train figure (0.743 named) is a development figure, not a score.

## 8. What follows

- A later version needs a larger dev set. 278 results in 36 papers gave a rule-3
  decision on 0.824 against 0.827. The gold-train part holds 2007 results in 240
  papers; a dev set of several hundred results in more papers would make rule 3
  readable.
- Read after the choice, the gold test suggests, and does not prove, two things about
  the grid. First, the noise setting `all` has a lower mean cascade R_named than `none`
  for every configuration (gru-crf 0.732 against 0.745, gru-softmax 0.754 against
  0.763, lstm-softmax 0.727 against 0.764), which agrees with rule 1. Second, the
  softmax heads find more (mean cascade R_named 0.763 and 0.764 against 0.745) and
  report more finds in the no-result papers (72.8 and 74.4 against 66.0), with a lower
  bronze F1 (0.881 and 0.870 against 0.917). The CRF head trades recall for fewer false
  finds. A later version should decide which side of that trade it wants before it
  writes the rule.
- Read after the choice, the single family `operator_drop` gave the highest gate run on
  the dev papers (0.845) and on the gold test (cascade 0.784, with 75 finds in the
  no-result papers). It is one run of one seed. A later grid should carry it with five
  seeds before it joins the default.
- The test name is still the main loss. The shipped cascade finds 0.866 of the gold
  test results and names 0.723 of them. Fragments tell the reader where a bare number
  lies, but nothing can be checked without the test name.
- The verdict comparison shows that a missed inconsistency is a missed find: no system
  judged a found inconsistency consistent except R statcheck, 1 row each in two forms.
  Recall of the find is what moves the verdicts.
- M9 (ports) closed in commit `b1c3747`: the parity cases are unchanged, the kits
  changed in provenance only, and `13_ports.py` now reads the real Python test count.
  `results/ports.json`: Python 247 parity tests passed, R 1280, web 385, none failed.
  M10 (documentation and the
  double reproduce) follows.

## 9. Reproduce

The stage table is `statcheck-ml/pipeline/README.md`. Run from `statcheck-ml/` with the
repository `.venv`. Stages that need the corpus, the owner's sheet, R or a rater agent
are marked; their outputs are committed with a manifest of hashes. Every sampling stage
takes `--seed`, default 0, and a rerun gives the same bytes.

1. `python pipeline/00_convert.py` — PDFs to clean text. Needs the corpus.
2. `python pipeline/14_gold.py <sheet.xlsx> <clean_text_dir> --out dataset/gold/gold.jsonl`
   — the gold set from the owner's sheet. Needs the corpus and the sheet.
3. `python pipeline/14_gold_split.py` — `dataset/gold/split.json`, the frozen split.
4. `python pipeline/14_gold_dev_split.py --gold dataset/gold/gold.jsonl ...` —
   `dataset/gold/dev.json`, the 36 dev papers.
5. `python pipeline/15_statcheck_gold.py --gold dataset/gold/gold.jsonl` — R statcheck
   in four forms, `dataset/baseline/gold/*.csv` and `versions.json`. Needs R, statcheck,
   pdftools and the corpus.
6. `python pipeline/14_gold_eval.py --gold dataset/gold/gold.jsonl --out results/gold_eval.json --label <label>`
   — the gold scores; one line per run in `results/gold_eval_log.jsonl`. Needs the
   corpus.
7. `python pipeline/16_compare.py` — `results/v4/compare.json` and `compare.md` from the
   frozen `results/v4/before/` and the current files.
8. `python pipeline/17_passage_inventory.py data/clean/ --gold dataset/gold/gold.jsonl ...`
   — the passage inventory. Needs the corpus.
9. `python pipeline/18_sample_passages.py <clean_text_dir> ...` — the 1165-passage
   sample, `key.json`, `sample_summary.json`. Needs the corpus.
10. `python pipeline/02_chunk.py --unpadded --size 10` — the 117 rater batches.
11. The three raters and the collect step (`03_collect.py`); then
    `python pipeline/19_consensus.py` (the four-vote consensus and the dispute
    batches); the adjudicator; `python pipeline/20_merge.py` (the final labels);
    `python pipeline/21_dataset_v4.py` (`dataset/train_v4.jsonl`, `splits_v4.json`).
    Need rater agents. The README table has no row for stages 19 to 23; the commands
    here are the usage lines of the scripts.
12. `python pipeline/22_train_v4.py --device cuda --parallel 12`, or on the GPU machine
    `bash run_v4_gpu.sh prereg 12` then `bash run_v4_gpu.sh pack` (`docs/v4/GPU.md`).
    Needs a GPU; about 120 CPU hours otherwise.
13. `python pipeline/23_select_v4.py --rule-system cascade` — the 35 ONNX exports, the
    dev scores, rules 1 to 3; `results/v4/selection.json` and `selection.md`. Reads no
    gold-test paper.
14. `python pipeline/24_final_v4.py` — the one read of the gold test and the bronze
    holdout for every model; `results/v4/final/summary.md`. Needs the corpus.
15. `python pipeline/25_verdicts_v4.py --papers gold_test` — the reference verdicts and
    `results/v4/final/verdicts.md`. Needs R, statcheck and the corpus.
16. `python pipeline/26_paired_v4.py` — `results/v4/final/paired.md` from the stored
    rows; reads no paper and no model.
17. After any spec or model change: `python tests/make_parity_cases.py`, then
    `python pipeline/08_port_kit.py ../ports/statcheck-ml-r/inst/kit --name statcheck-ml-r`
    and `python pipeline/08_port_kit.py ../ports/statcheck-ml-web/kit --name statcheck-ml-web`,
    run each port's tests, commit the kit in each port, and
    `python pipeline/13_ports.py` for `results/ports.json`.
18. `python -m pytest tests -q`, then `bash reproduce.sh` twice; compare the SHA-256 of
    every result file between the two runs.
