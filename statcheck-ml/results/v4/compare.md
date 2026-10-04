# Version 4: before and after

Gold and bronze are never pooled. Each table names its tier.

## Gold, gold_test (human-coded, recall)

| system | recall before | recall after | delta | CI before | CI after | damaged before | damaged after | finds without results before | after |
|---|---|---|---|---|---|---|---|---|---|
| regex | 0.489 | 0.498 | 0.009 | [0.292, 0.640] | [0.299, 0.650] | 0.578 | 0.578 | 2 | 2 |
| model | 0.836 | 0.689 | -0.147 | [0.773, 0.893] | [0.532, 0.803] | 0.807 | 0.752 | 155 | 69 |
| cascade | 0.851 | 0.723 | -0.128 | [0.787, 0.908] | [0.556, 0.836] | 0.817 | 0.761 | 155 | 69 |
| statcheck_r_fulltext_raw | 0.113 | 0.113 | 0.000 | [0.036, 0.257] | [0.036, 0.257] | 0.000 | 0.000 | 2 | 2 |
| statcheck_r_fulltext_repaired | 0.443 | 0.443 | 0.000 | [0.238, 0.646] | [0.238, 0.646] | 0.606 | 0.606 | 2 | 2 |
| statcheck_r_passages | 0.479 | 0.479 | 0.000 | [0.293, 0.654] | [0.293, 0.654] | 0.606 | 0.606 | 2 | 2 |
| statcheck_r_pdf | 0.237 | 0.237 | 0.000 | [0.055, 0.493] | [0.055, 0.493] | 0.000 | 0.000 | 2 | 2 |

Only before: model_passages. Only after: none.

## Gold, gold_train (human-coded, recall)

| system | recall before | recall after | delta | CI before | CI after | damaged before | damaged after | finds without results before | after |
|---|---|---|---|---|---|---|---|---|---|
| regex | 0.514 | 0.520 | 0.006 | [0.438, 0.591] | [0.444, 0.597] | 0.420 | 0.422 | 0 | 0 |
| model | 0.840 | 0.714 | -0.126 | [0.789, 0.883] | [0.653, 0.771] | 0.684 | 0.578 | 149 | 54 |
| cascade | 0.843 | 0.743 | -0.100 | [0.793, 0.885] | [0.683, 0.803] | 0.684 | 0.578 | 149 | 54 |
| statcheck_r_fulltext_raw | 0.285 | 0.285 | 0.000 | [0.196, 0.374] | [0.196, 0.374] | 0.034 | 0.034 | 0 | 0 |
| statcheck_r_fulltext_repaired | 0.506 | 0.506 | 0.000 | [0.419, 0.591] | [0.419, 0.591] | 0.459 | 0.459 | 0 | 0 |
| statcheck_r_passages | 0.557 | 0.557 | 0.000 | [0.479, 0.634] | [0.479, 0.634] | 0.459 | 0.459 | 0 | 0 |
| statcheck_r_pdf | 0.329 | 0.329 | 0.000 | [0.235, 0.424] | [0.235, 0.424] | 0.034 | 0.034 | 0 | 0 |

Only before: model_passages. Only after: none.

## Bronze holdout (language-model annotation, windows only)

| system | P before | P after | dP | R before | R after | dR | F1 before | F1 after | dF1 |
|---|---|---|---|---|---|---|---|---|---|
| cascade_gru-crf-s0 | 0.949 | 0.949 | 0.000 | 0.870 | 0.870 | 0.000 | 0.908 | 0.908 | 0.000 |
| cnn-crf-s0 | 0.987 | 0.987 | 0.000 | 0.731 | 0.731 | 0.000 | 0.840 | 0.840 | 0.000 |
| cnn-softmax-s0 | 0.822 | 0.822 | 0.000 | 0.873 | 0.873 | 0.000 | 0.847 | 0.847 | 0.000 |
| gru-crf-noaug-s0 | 0.940 | 0.940 | 0.000 | 0.873 | 0.873 | 0.000 | 0.905 | 0.905 | 0.000 |
| gru-crf-s0 | 0.952 | 0.952 | 0.000 | 0.861 | 0.861 | 0.000 | 0.904 | 0.904 | 0.000 |
| gru-crf-s1 | 0.946 | 0.946 | 0.000 | 0.861 | 0.861 | 0.000 | 0.901 | 0.901 | 0.000 |
| gru-crf-s2 | 0.961 | 0.961 | 0.000 | 0.842 | 0.842 | 0.000 | 0.898 | 0.898 | 0.000 |
| gru-softmax-s0 | 0.925 | 0.925 | 0.000 | 0.882 | 0.882 | 0.000 | 0.903 | 0.903 | 0.000 |
| gru-softmax-s1 | 0.902 | 0.902 | 0.000 | 0.879 | 0.879 | 0.000 | 0.890 | 0.890 | 0.000 |
| gru-softmax-s2 | 0.939 | 0.939 | 0.000 | 0.861 | 0.861 | 0.000 | 0.898 | 0.898 | 0.000 |
| lstm-crf-s0 | 0.956 | 0.956 | 0.000 | 0.879 | 0.879 | 0.000 | 0.916 | 0.916 | 0.000 |
| lstm-softmax-s0 | 0.910 | 0.910 | 0.000 | 0.879 | 0.879 | 0.000 | 0.894 | 0.894 | 0.000 |
| lstm-softmax-s1 | 0.909 | 0.909 | 0.000 | 0.864 | 0.864 | 0.000 | 0.886 | 0.886 | 0.000 |
| lstm-softmax-s2 | 0.899 | 0.899 | 0.000 | 0.851 | 0.851 | 0.000 | 0.874 | 0.874 | 0.000 |
| statcheck_raw | 0.983 | 0.983 | 0.000 | 0.183 | 0.183 | 0.000 | 0.308 | 0.308 | 0.000 |
| statcheck_repaired | 0.993 | 0.993 | 0.000 | 0.467 | 0.467 | 0.000 | 0.636 | 0.636 | 0.000 |

Only before: none. Only after: none.

## Steps (gold_test, one row per labelled evaluation in the log)

| label | date | git_sha | system | recall | finds without results |
|---|---|---|---|---|---|
| m1-before | 2026-10-04T17:33:19+00:00 | e51e018 | regex | 0.489 | 2 |
| m1-before | 2026-10-04T17:33:19+00:00 | e51e018 | model | 0.836 | 155 |
| m1-before | 2026-10-04T17:33:19+00:00 | e51e018 | cascade | 0.851 | 155 |
| m1-before | 2026-10-04T17:33:19+00:00 | e51e018 | model_passages | 0.849 | 150 |
| m1-before | 2026-10-04T17:33:19+00:00 | e51e018 | statcheck_r_fulltext_raw | 0.113 | 2 |
| m1-before | 2026-10-04T17:33:19+00:00 | e51e018 | statcheck_r_fulltext_repaired | 0.443 | 2 |
| m1-before | 2026-10-04T17:33:19+00:00 | e51e018 | statcheck_r_passages | 0.479 | 2 |
| m1-before | 2026-10-04T17:33:19+00:00 | e51e018 | statcheck_r_pdf | 0.237 | 2 |
| m2-dedup | 2026-10-04T17:51:44+00:00 | e51e018 | regex | 0.498 | 2 |
| m2-dedup | 2026-10-04T17:51:44+00:00 | e51e018 | model | 0.849 | 157 |
| m2-dedup | 2026-10-04T17:51:44+00:00 | e51e018 | cascade | 0.866 | 157 |
| m2-dedup | 2026-10-04T17:51:44+00:00 | e51e018 | model_passages | 0.849 | 150 |
| m2-dedup | 2026-10-04T17:51:44+00:00 | e51e018 | statcheck_r_fulltext_raw | 0.113 | 2 |
| m2-dedup | 2026-10-04T17:51:44+00:00 | e51e018 | statcheck_r_fulltext_repaired | 0.443 | 2 |
| m2-dedup | 2026-10-04T17:51:44+00:00 | e51e018 | statcheck_r_passages | 0.479 | 2 |
| m2-dedup | 2026-10-04T17:51:44+00:00 | e51e018 | statcheck_r_pdf | 0.237 | 2 |
| m2-fragments | 2026-10-04T18:02:27+00:00 | e51e018 | regex | 0.498 | 2 |
| m2-fragments | 2026-10-04T18:02:27+00:00 | e51e018 | model | 0.689 | 73 |
| m2-fragments | 2026-10-04T18:02:27+00:00 | e51e018 | cascade | 0.723 | 73 |
| m2-fragments | 2026-10-04T18:02:27+00:00 | e51e018 | model_passages | 0.689 | 69 |
| m2-fragments | 2026-10-04T18:02:27+00:00 | e51e018 | statcheck_r_fulltext_raw | 0.113 | 2 |
| m2-fragments | 2026-10-04T18:02:27+00:00 | e51e018 | statcheck_r_fulltext_repaired | 0.443 | 2 |
| m2-fragments | 2026-10-04T18:02:27+00:00 | e51e018 | statcheck_r_passages | 0.479 | 2 |
| m2-fragments | 2026-10-04T18:02:27+00:00 | e51e018 | statcheck_r_pdf | 0.237 | 2 |
| m2-passages | 2026-10-04T18:11:49+00:00 | e51e018 | regex | 0.498 | 2 |
| m2-passages | 2026-10-04T18:11:49+00:00 | e51e018 | model | 0.689 | 69 |
| m2-passages | 2026-10-04T18:11:49+00:00 | e51e018 | cascade | 0.723 | 69 |
| m2-passages | 2026-10-04T18:11:49+00:00 | e51e018 | statcheck_r_fulltext_raw | 0.113 | 2 |
| m2-passages | 2026-10-04T18:11:49+00:00 | e51e018 | statcheck_r_fulltext_repaired | 0.443 | 2 |
| m2-passages | 2026-10-04T18:11:49+00:00 | e51e018 | statcheck_r_passages | 0.479 | 2 |
| m2-passages | 2026-10-04T18:11:49+00:00 | e51e018 | statcheck_r_pdf | 0.237 | 2 |
| m2-after | 2026-10-04T18:25:44+00:00 | e51e018 | regex | 0.498 | 2 |
| m2-after | 2026-10-04T18:25:44+00:00 | e51e018 | model | 0.689 | 69 |
| m2-after | 2026-10-04T18:25:44+00:00 | e51e018 | cascade | 0.723 | 69 |
| m2-after | 2026-10-04T18:25:44+00:00 | e51e018 | statcheck_r_fulltext_raw | 0.113 | 2 |
| m2-after | 2026-10-04T18:25:44+00:00 | e51e018 | statcheck_r_fulltext_repaired | 0.443 | 2 |
| m2-after | 2026-10-04T18:25:44+00:00 | e51e018 | statcheck_r_passages | 0.479 | 2 |
| m2-after | 2026-10-04T18:25:44+00:00 | e51e018 | statcheck_r_pdf | 0.237 | 2 |
