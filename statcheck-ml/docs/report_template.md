# statcheck-ml — results report

Generated from commit {{git_sha_short}} ({{generated_at}}). Every number in this
report comes from a committed measurement file. `pipeline/11_report.py` fills this
template. No number is typed by hand.

## Abstract

statcheck-ml replaces the regular-expression extraction step of the R package
`statcheck` with a small character-level model, and keeps the p-value arithmetic
exact. Version 2 rebuilt the process so that a third party can rerun every stage after
the annotation from committed files. Three rater agents annotated every window blind
under a versioned guideline. A scripted consensus rule and an adjudicator agent turned
the three readings into one bronze label per result. Six character-level model
configurations were trained with seeds, exported to ONNX, and evaluated once on a
holdout of unseen documents with bootstrap intervals and paired tests. The cascade of
`statcheck` with operator repair followed by the best model reaches holdout F1
{{cascade_f1|3}} {{cascade_ci|ci}}, against {{statcheck_repaired_f1|3}} for `statcheck`
with repair alone and {{statcheck_raw_f1|3}} for `statcheck` on the raw text.

## 1 Data

The corpus holds 3100 open-access articles from psychology, sociology, political
science, economics, education, and management. Each article was converted to text
with PyMuPDF. A window is one candidate line with two context lines on each side,
grown to at least 250 characters. The sampler drew windows from five pools. Pool A:
the `statcheck` pattern matched a complete result. Pool AN: pool A with a negative
statistic. Pool B1: a p-value is present, but no complete match. Pool B2: a test
letter and digits, but no p-value. Pool C: the density prefilter rejected the line.

Three raters annotated {{n_train_windows|int}} training windows from
{{n_train_docs|int}} documents, and {{n_holdout_windows|int}} holdout windows from
{{n_holdout_docs|int}} documents. The holdout documents are disjoint from the
training documents. Every label is bronze: a rater agent wrote it, and no person
checked it.

{{table:dataset_counts}}

Training pools: {{pools_train}}. Holdout pools: {{pools_holdout}}. The damaged share
of training results is {{damaged_share_train|pct1}}, and of holdout results
{{damaged_share_holdout|pct1}}, out of {{n_results_train|int}} and
{{n_results_holdout|int}} results. A damaged result is one where the PDF conversion
replaced an operator or a sign with a wrong character. This share is the reason the
project exists: a pattern cannot read an operator it does not know in advance, and a
model that reads the surrounding characters can.

## 2 Annotation protocol

Three rater agents read every window: one Claude haiku, one Claude sonnet, and one
Claude opus. Different model families give raters that do not share one training run.
Each rater received one batch of twenty windows and the guideline, version 2.0
(`docs/GUIDELINE.md`), and wrote one JSON file per batch. The rater saw no pool label,
no other rater's output, and no repository file. A collect script validated every
output against the guideline contract: the batch length, the window order, the nine
result keys, the operator set, and the confidence set. The script aligned each quoted
result to its character span in the window, with whitespace ignored, and stamped each
record with the rater, the model id, the sha256 of the guideline, the sha256 of the
agent file, the batch id, the collection time, and the retry count. A batch that
failed a rule was dispatched again, and the retry count records it.

One limit of this design stays written down. An agent inside Claude Code holds a Read
tool, so blindness is an instruction, not a sandbox. The three raters also share a
model family, so their agreement cannot detect a mistake that all three make.

Each rater returned {{results_haiku_train|int}} (haiku), {{results_sonnet_train|int}}
(sonnet), and {{results_opus_train|int}} (opus) results on the training windows.

{{table:rater_yield}}

## 3 Agreement

Agreement is measured at three levels. At the window level, the unit is the binary
decision `contains_result`, and the metrics are the raw percent agreement, the
pairwise Cohen kappa, the Fleiss kappa for three raters, and the nominal Krippendorff
alpha. At the result level, the unit is one aligned result span. Two results match
strictly when their spans are equal, and leniently when the overlap covers at least
half of the shorter span. The metric is the pairwise span F1, and the mean over the
three pairs. At the field level, the unit is one result that all three raters found,
and the metric is the nominal Krippendorff alpha for `test_type`, `p_operator`, and
`damaged`, and the exact-match share for the numeric fields. Every value carries a 95
percent percentile bootstrap interval over windows, from one thousand resamples with
seed 0. The convention of Landis and Koch calls a kappa above 0.81 "almost perfect".
The report uses that word as a convention, not as a claim about truth.

Window-level Fleiss kappa is {{fleiss_train|3}} {{fleiss_train_ci|ci}} on training and
{{fleiss_holdout|3}} {{fleiss_holdout_ci|ci}} on holdout; Krippendorff alpha is
{{alpha_train|3}} {{alpha_train_ci|ci}} and {{alpha_holdout|3}} {{alpha_holdout_ci|ci}}.
Mean strict-span result F1 is {{strict_mean_train|3}} {{strict_mean_train_ci|ci}}
(train) and {{strict_mean_holdout|3}} {{strict_mean_holdout_ci|ci}} (holdout); mean
lenient-span F1 is {{lenient_mean_train|3}} {{lenient_mean_train_ci|ci}} and
{{lenient_mean_holdout|3}} {{lenient_mean_holdout_ci|ci}}. The gap between the strict
and the lenient value is the share of results the raters found together but bounded
differently, for example a quote that includes or excludes a trailing p-value.

{{table:agreement_window}}

{{table:agreement_result}}

The field table shows where the raters disagree once they have found the same result.
The test type is almost never disputed. The operator and the `damaged` flag are the
fields with the lowest agreement, because a damaged operator must be inferred from the
context, and the statistic value follows, because a damaged minus sign changes the
copied value.

{{table:agreement_field}}

The last table scores each rater against the final label. It answers a practical
question: which model is the best cheap annotator under this guideline.

{{table:rater_vs_final}}

{{figure:agreement_heatmap.png|Pairwise strict-span F1 and per-rater agreement with the adjudicated label}}

## 4 Adjudication

A scripted consensus rule turns the three readings into one label. Results are
clustered across raters with the lenient matcher. A cluster that holds at least two
raters is kept, and each field takes the value that at least two raters wrote. Two
kinds of cluster go to an adjudicator agent (Claude opus): a singleton, which one rater
alone found, and a field conflict, where all three raters wrote a different value. The
adjudicator sees the passage and the candidate results without rater names, and
returns keep or reject with one reason. Every dispute, every candidate, and every
decision is committed in `dataset/annotations/<set>/disputes.json`.

Each kept result carries a tier. `unanimous`: all three raters found it and agreed on
every field. `majority`: two raters found it, or a field was decided by vote.
`adjudicated`: the adjudicator decided it. All three tiers are bronze.

Training had {{n_disputes_train|int}} disputes, {{n_adjudicated_train|int}} of them
kept after adjudication; holdout had {{n_disputes_holdout|int}} disputes,
{{n_adjudicated_holdout|int}} kept. Kept-result tiers on training:
{{tier_unanimous_train|int}} unanimous, {{tier_majority_train|int}} majority,
{{tier_adjudicated_train|int}} adjudicated. Training disputes split into
{{n_singletons_train|int}} singletons ({{n_kept_singletons_train|int}} kept) and
{{n_field_conflicts_train|int}} field conflicts.

{{table:adjudication}}

## 5 Models

Every candidate model is character level. It reads one character map, which ports to
Python, JavaScript, and R without a tokenizer. Three families were trained: a
bidirectional LSTM, a bidirectional GRU, and a dilated character CNN. Each family has
two heads: a softmax over the 37 BIOES tags, or a linear-chain CRF above the same
emissions. The CRF runs outside the ONNX graph and ships as a small JSON decoder.

The protocol has three stages. First, the six configurations were trained with seed 0
for up to thirty epochs, with augmentation on: two perturbed copies of each training
window and four hundred generated hard negatives. Early stopping used the dev split
with a patience of six epochs. Second, the top three configurations by dev F1 were
trained again with seeds 1 and 2. Third, the best configuration was trained once more
without augmentation. Model selection used the dev split only. The holdout was
evaluated once per final model. All training ran on CPU, and every run records its
wall time, its git commit, and its torch version in `models/runs.json`.

The grid ran {{n_runs|int}} training runs, {{total_train_hours|3}} CPU hours in
total. The best screen configuration by dev F1 is {{best_config}}; its holdout F1 is
{{best_f1|3}} {{best_ci|ci}}. The three configurations shipped to the zoo are:
{{top3}}.

{{table:benchmark_screen}}

The seed table shows how much of the difference between configurations is noise. A
difference smaller than the seed spread of either configuration is not a finding.

{{table:benchmark_seeds}}

Three findings follow from these two tables. First, the three recurrent configurations
in the zoo are one group: the seed spread of each is as large as the gap between them,
and section 7 confirms it with paired tests. Second, the CRF head adds nothing
measurable to the GRU. For the LSTM the single CRF run scores above the three softmax
runs, but with one seed that is not a finding by the rule above. Version 1 reported a
CRF advantage on single-pass labels; on the three-rater labels the emissions alone are
cleaner. Third,
the CNN family is below the recurrent families on every measure, and {{n_divergent|int}}
run diverged during training: {{divergent_runs}}. Early stopping kept the checkpoint
from before the divergence, so the score in the table is that checkpoint's.

The configuration with the highest holdout F1 is {{holdout_top_config}}
({{holdout_top_f1|3}}, dev F1 {{holdout_top_dev_f1|3}}). It is in the zoo: {{holdout_top_in_zoo}}.
Selection used the dev split, as the protocol fixed in advance, and the holdout score
does not change that choice. Its interval overlaps the intervals of the three shipped
configurations.

The ablation, {{ablation_name}}, reaches holdout F1 {{ablation_f1|3}} against
{{best_f1|3}} for the same configuration with augmentation, and recall
{{ablation_damaged_r|3}} against {{best_damaged_r|3}} on the damaged subset. The
augmentation gives no measurable gain on this holdout. The damage families it
imitates are already present in the three-rater training labels at a rate close to the
holdout's (section 1), so the model learns them from real text.

{{figure:benchmark_f1.png|Holdout F1 by model configuration, seed 0, with seeds 1-2 of the top three}}

{{figure:learning_curves.png|Dev F1 per epoch, seed 0, top-three configs}}

## 6 Evaluation

Four kinds of system were scored on the holdout. `statcheck_raw` is the R package on
the raw window text. `statcheck_repaired` is the same package after the arithmetic
operator repair, which tries each operator for a damaged character and keeps the one
whose recomputed p-value agrees with the reported one. Each model run is one ONNX
export, decoded with its own CRF decoder when it has one. The cascade runs
`statcheck_repaired` first and adds only what the model finds beyond it.

A predicted result matches a gold result when both sit in the same window and their
normalised statistic values are equal. This is the version 1 rule, kept so that the
numbers stay comparable. The subsets are: results with a damaged operator and without;
each test type; each damage family, named from the damaged character; and checkable
results, which carry a statistic, a first degrees of freedom, and a p-value.

`statcheck` alone reaches F1 {{statcheck_raw_f1|3}} (recall {{statcheck_raw_r|3}}) on
raw text and {{statcheck_repaired_f1|3}} (recall {{statcheck_repaired_r|3}}) after
operator repair. The cascade reaches {{cascade_f1|3}} {{cascade_ci|ci}}.
The match key is the pair of window and statistic value. The holdout holds
{{n_results_holdout|int}} results; {{gold_unparseable|int}} gold statistics could not be
parsed as a number, and the rest collapse to {{eval_gold_results|int}} distinct keys
({{eval_gold_damaged|int}} damaged, {{eval_gold_checkable|int}} checkable), because two
results in one window can carry the same value. Unparseable results are excluded from
every match in this section, in both directions.

{{table:systems_overall}}

The damaged subset is where the model earns its place. `statcheck` cannot read a
damaged operator, so its recall on this subset is near the share of results whose
damage sits outside the operator.

{{table:systems_damaged}}

{{table:family_recall}}

{{figure:precision_recall.png|Precision vs. recall by system, holdout overall, with iso-F1 curves}}

{{figure:family_recall.png|Recall by damage family, statcheck, best model, and cascade}}

## 7 Statistical analysis

Every interval in the systems tables is a percentile bootstrap over holdout documents,
from two thousand resamples with seed 0. The document is the unit because windows from
one document share a font, a conversion, and a writing style, so they are not
independent. A paired bootstrap on the same document resamples gives the interval and
the p-value of the F1 difference between two systems. McNemar's exact test counts, per
gold result, the discordant pairs: found by one system and missed by the other. The
seed tables give the mean and the standard deviation of holdout F1 over three seeds.
The per-family recall carries a Wilson interval, because several families hold few
results. None of these tests can find a mistake that the raters made every time; they
compare systems against the same bronze labels.

The cascade over `statcheck_repaired` differs by {{paired_cascade_diff|3}}
{{paired_cascade_ci|ci}} F1 (paired bootstrap, p={{paired_cascade_p|3}}). McNemar's
exact test on the same pair: {{mcnemar_cascade_b|int}} results found only by the
cascade, {{mcnemar_cascade_c|int}} found only by `statcheck_repaired`
(p={{mcnemar_cascade_p|3}}).

{{table:paired_tests}}

The three shipped configurations do not differ on this holdout. {{paired_top1_vs_top2_a}}
against {{paired_top1_vs_top2_b}} differs by {{paired_top1_vs_top2_diff|3}}
{{paired_top1_vs_top2_ci|ci}} (p={{paired_top1_vs_top2_p|3}}), and against
{{paired_top1_vs_top3_b}} by {{paired_top1_vs_top3_diff|3}} {{paired_top1_vs_top3_ci|ci}}
(p={{paired_top1_vs_top3_p|3}}). Both intervals contain zero. The choice between them is
a choice of size and latency, not of accuracy.

The p-value arithmetic is compared with the R package on the results the package itself
reports after repair: {{verdict_agree|int}} verdicts agree, {{verdict_disagree|int}}
disagree, and {{verdict_undecidable|int}} carry no p-value to compare. The disagreements
have two causes, and both are conventions of rounding. The R package accepts a reported
p-value when the interval implied by the rounded test statistic contains it, and this
project does not yet widen the check that way. This project accepts a p-value reported
as zero or as a fixed small bound where the R package flags it. Neither convention is a
fault in the extraction, and the next version of the p-value core will adopt the R
package's rule so the two agree.

## 8 Shipped models

The zoo ships the seed-0 ONNX export of each of {{top3}}, chosen by dev F1, never by a
holdout score. Each export was checked against its torch checkpoint on the dev
windows: the decoded tags must agree on every window, and the largest logit difference
is recorded. The table gives the size, the median CPU latency per window with one
thread, and the size and dev F1 change of the int8 dynamically quantised graph.

{{table:zoo}}

{{figure:zoo_size_latency.png|ONNX size vs. holdout F1, marker area by latency, zoo members outlined}}

## 9 PDF engines

Each port reads the PDF with a different engine, and the engines do not return the
same text. The gate measures, on the holdout documents, the share of gold results that
each engine's text still holds, and the spread between the best and the worst engine.
The ports must not ship while the spread is above the limit.

The spread between the best and the worst engine is {{engine_spread|3}}, against a
gate of {{engine_limit|3}}.

{{table:engines}}

## 10 Limitations

- There is no human gold set. Every number in this report is measured against bronze
  labels that rater agents wrote.
- The raters, the adjudicator, and the models could share blind spots. Agreement
  between raters of one model family cannot reveal them.
- The holdout labels are bronze too. A model that copies a rater's systematic mistake
  scores well on it.
- The rater agents ran through a general-purpose agent wrapper that read the same
  guideline file, because the dedicated rater agent files load only after a session
  restart. The wrapper, the model, and the guideline hash are recorded on every record.
- The version 1 repaired baseline came from a whole-document substitution that no
  script in the repository reproduces. The version 2 repaired baseline is the
  reproducible one, and it finds fewer results.
- The project's p-value consistency check and the R package differ in two rounding
  conventions, stated in section 7. Verdict agreement between the two is a comparison
  of conventions, not a measure of extraction.
- The R port reads less of the corpus than the Python port, because its PDF engine
  returns less text. Its ceiling is lower, and the engine table states it.
- Robustness is claimed only for the damage families that the holdout holds. A family
  with few results carries a wide interval, and the table shows it.
- The prefilter sets a hard ceiling. Text it discards is unrecoverable by any model.

## 11 Reproduction

This report was generated from commit {{git_sha_short}} ({{generated_at}}). The
evaluation numbers in it come from `results/eval.json`, itself run at commit
{{git_sha}}, with R {{r_version}} and statcheck {{statcheck_version}}.

`reproduce.sh` runs the offline stages from committed files: agreement, dataset,
evaluation, figures, and this report. `reproduce.sh --train` adds the training grid
and the export. The annotation stages need the rater agents and follow the procedure
in `docs/PROTOCOL.md`. `dataset/MANIFEST.json` holds the sha256 of every dataset file,
`models/runs.json` records every training run, and `models/export.json` records every
export. A run of `reproduce.sh` on a clean checkout must give the same bytes for every figure,
and the same bytes for `results/eval.json` and this file apart from the commit id and
its date, which name the checkout the run was made from.

## 12 Ports

The extraction runs in three languages. Python is the reference, in this repository.
R (`statcheck-ml-r`) and the browser (`statcheck-ml-web`) live in their own
repositories and read one kit that this repository writes: the five spec files, the
shipped model, the parity cases, and a manifest with a hash per file that the port
verifies when it loads. A port never restates a rule.

Parity is one file, {{n_parity_cases|int}} cases in {{n_parity_sections|int}} sections,
one per stage: normalise, repair, prefilter, extract, model tags, model logits,
grouping, p-value ({{n_parity_pvalue|int}} rows), and the whole pipeline on four
documents. The Python reference produces every expected value; each port reproduces
them in its own test suite. R runs the model in pure R from the raw weights, batched
over the windows of a document; the browser runs the ONNX graph through
onnxruntime-web. Both must match the Python logits within {{logit_tolerance}} with
every tag equal; the parity file states the bound. The p-value core in JavaScript is
an own implementation of the incomplete beta and gamma functions, tested against
SciPy values stored with the test; R uses `pt`, `pf`, `pchisq` and `pnorm`.

Writing the ports found two faults in the reference and two in the ports, each now a
parity case: a chi-square with zero degrees of freedom gave NaN and a verdict instead
of `undecidable`; a Greek chi with a space on each side escaped the regex in
JavaScript and in PCRE, whose word boundary is ASCII-only; and the reference's own
rounding of a p-value text differs between Python's `repr` and JavaScript's number
formatting, which the port now reproduces exactly.

{{table:ports}}

Both timings are the second of two runs on one machine with the model loaded and
one thread: the damaged sample PDF end to end, and one hundred windows of three
hundred characters through the model alone. The model is where a port spends its
time; the R port pays for running it in interpreted matrix code, and a long article
with hundreds of windows costs it under a minute.
