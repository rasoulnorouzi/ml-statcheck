# statcheck-ml — results report

Generated from commit {{git_sha_short}} ({{generated_at}}). Every number below comes
from a committed measurement file; `pipeline/11_report.py` fills this template and
writes nothing by hand. Doc-writer: replace each placeholder sentence with the
section's prose. Keep every double-brace value, table and figure block as it is.

## Abstract

[One paragraph on what was measured and what it found.] The cascade reaches holdout
F1 {{cascade_f1|3}} {{cascade_ci|ci}}, against {{statcheck_repaired_f1|3}} for
statcheck alone on the same holdout.

## 1 Data

Three raters annotated {{n_train_windows|int}} training windows from
{{n_train_docs|int}} documents, and {{n_holdout_windows|int}} holdout windows from
{{n_holdout_docs|int}} documents. Every label is bronze: scored by rater agents,
never by a person.

{{table:dataset_counts}}

Training pools: {{pools_train}}. Holdout pools: {{pools_holdout}}. The damaged share
of training results is {{damaged_share_train|pct1}}, and of holdout results
{{damaged_share_holdout|pct1}}, out of {{n_results_train|int}} and
{{n_results_holdout|int}} results respectively.

## 2 Annotation protocol

[Summarise `docs/PROTOCOL.md` here; do not parse it programmatically.] Each rater
returned {{results_haiku_train|int}} (haiku), {{results_sonnet_train|int}} (sonnet),
and {{results_opus_train|int}} (opus) results on the training windows.

{{table:rater_yield}}

## 3 Agreement

Window-level Fleiss kappa is {{fleiss_train|3}} {{fleiss_train_ci|ci}} on training and
{{fleiss_holdout|3}} {{fleiss_holdout_ci|ci}} on holdout; Krippendorff alpha is
{{alpha_train|3}} {{alpha_train_ci|ci}} and {{alpha_holdout|3}} {{alpha_holdout_ci|ci}}.
Mean strict-span result F1 is {{strict_mean_train|3}} {{strict_mean_train_ci|ci}}
(train) and {{strict_mean_holdout|3}} {{strict_mean_holdout_ci|ci}} (holdout); mean
lenient-span F1 is {{lenient_mean_train|3}} {{lenient_mean_train_ci|ci}} and
{{lenient_mean_holdout|3}} {{lenient_mean_holdout_ci|ci}}.

{{table:agreement_window}}

{{table:agreement_result}}

{{table:agreement_field}}

{{table:rater_vs_final}}

{{figure:agreement_heatmap.png|Pairwise strict-span F1 and per-rater agreement with the adjudicated label}}

## 4 Adjudication

Training had {{n_disputes_train|int}} disputes, {{n_adjudicated_train|int}} of them
adjudicated; holdout had {{n_disputes_holdout|int}} disputes,
{{n_adjudicated_holdout|int}} adjudicated. Kept-result tiers on training:
{{tier_unanimous_train|int}} unanimous, {{tier_majority_train|int}} majority,
{{tier_adjudicated_train|int}} adjudicated. Training disputes split into
{{n_singletons_train|int}} singletons ({{n_kept_singletons_train|int}} kept) and
{{n_field_conflicts_train|int}} field conflicts.

{{table:adjudication}}

## 5 Models

The grid ran {{n_runs|int}} training runs, {{total_train_hours|3}} CPU hours in
total. The best screen config by dev F1 is {{best_config}}, at dev-time; its holdout
F1 is {{best_f1|3}} {{best_ci|ci}}. The three seeds-and-screen configs shipped to the
zoo are: {{top3}}.

{{table:benchmark_screen}}

{{table:benchmark_seeds}}

{{figure:benchmark_f1.png|Holdout F1 by model configuration, seed 0, with seeds 1-2 of the top three}}

{{figure:learning_curves.png|Dev F1 per epoch, seed 0, top-three configs}}

## 6 Evaluation

statcheck alone reaches F1 {{statcheck_raw_f1|3}} (recall {{statcheck_raw_r|3}}) on
raw text and {{statcheck_repaired_f1|3}} (recall {{statcheck_repaired_r|3}}) after
operator repair. The cascade reaches {{cascade_f1|3}} {{cascade_ci|ci}}.
{{gold_unparseable|int}} gold statistics could not be parsed as a number at all, and
are excluded from every match in this section, in both directions.

{{table:systems_overall}}

{{table:systems_damaged}}

{{table:family_recall}}

{{figure:precision_recall.png|Precision vs. recall by system, holdout overall, with iso-F1 curves}}

{{figure:family_recall.png|Recall by damage family, statcheck, best model, and cascade}}

## 7 Statistical analysis

The cascade over statcheck_repaired differs by {{paired_cascade_diff|3}}
{{paired_cascade_ci|ci}} F1 (paired bootstrap, p={{paired_cascade_p|3}}). McNemar's
exact test on the same pair: {{mcnemar_cascade_b|int}} results found only by the
cascade, {{mcnemar_cascade_c|int}} found only by statcheck_repaired
(p={{mcnemar_cascade_p|3}}).

{{table:paired_tests}}

## 8 Shipped models

The zoo ships the seed-0 export of each of {{top3}}, chosen by dev F1, never by a
holdout score.

{{table:zoo}}

{{figure:zoo_size_latency.png|ONNX size vs. holdout F1, marker area by latency, zoo members outlined}}

## 9 PDF engines

The spread between the best and the worst engine is {{engine_spread|3}}, against a
gate of {{engine_limit|3}}.

{{table:engines}}

## 10 Limitations

[State what this report does not claim: bronze labels only, no human gold set yet,
robustness claimed only for the damage families actually measured, the prefilter's
recall as a hard ceiling on everything below it.]

## 11 Reproduction

This report was generated from commit {{git_sha_short}} ({{generated_at}}). The
evaluation numbers in it come from `results/eval.json`, itself run at commit
{{git_sha}}, with R {{r_version}} and statcheck {{statcheck_version}}. `reproduce.sh`
rebuilds every file this report reads, from a clean checkout, without retyping any of
these numbers.
