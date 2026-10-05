# Version 4: before and after

Gold and bronze are never pooled. Each table names its tier.

## Gold, gold_test (human-coded, recall)

| system | recall before | recall after | delta | CI before | CI after | damaged before | damaged after | finds without results before | after | with fragments before | with fragments after |
|---|---|---|---|---|---|---|---|---|---|---|---|
| regex | 0.489 | 0.498 | 0.009 | [0.292, 0.640] | [0.299, 0.650] | 0.578 | 0.578 | 2 | 2 | — | 0.498 |
| model | 0.836 | 0.689 | -0.147 | [0.773, 0.893] | [0.532, 0.803] | 0.807 | 0.752 | 155 | 69 | — | 0.849 |
| cascade | 0.851 | 0.723 | -0.128 | [0.787, 0.908] | [0.556, 0.836] | 0.817 | 0.761 | 155 | 69 | — | 0.866 |
| statcheck_r_fulltext_raw | 0.113 | 0.113 | 0.000 | [0.036, 0.257] | [0.036, 0.257] | 0.000 | 0.000 | 2 | 2 | — | — |
| statcheck_r_fulltext_repaired | 0.443 | 0.443 | 0.000 | [0.238, 0.646] | [0.238, 0.646] | 0.606 | 0.606 | 2 | 2 | — | — |
| statcheck_r_passages | 0.479 | 0.479 | 0.000 | [0.293, 0.654] | [0.293, 0.654] | 0.606 | 0.606 | 2 | 2 | — | — |
| statcheck_r_pdf | 0.237 | 0.237 | 0.000 | [0.055, 0.493] | [0.055, 0.493] | 0.000 | 0.000 | 2 | 2 | — | — |

Only before: model_passages. Only after: none.

## Gold, gold_train (human-coded, recall)

| system | recall before | recall after | delta | CI before | CI after | damaged before | damaged after | finds without results before | after | with fragments before | with fragments after |
|---|---|---|---|---|---|---|---|---|---|---|---|
| regex | 0.514 | 0.520 | 0.006 | [0.438, 0.591] | [0.444, 0.597] | 0.420 | 0.422 | 0 | 0 | — | 0.520 |
| model | 0.840 | 0.714 | -0.126 | [0.789, 0.883] | [0.653, 0.771] | 0.684 | 0.578 | 149 | 54 | — | 0.858 |
| cascade | 0.843 | 0.743 | -0.100 | [0.793, 0.885] | [0.683, 0.803] | 0.684 | 0.578 | 149 | 54 | — | 0.861 |
| statcheck_r_fulltext_raw | 0.285 | 0.285 | 0.000 | [0.196, 0.374] | [0.196, 0.374] | 0.034 | 0.034 | 0 | 0 | — | — |
| statcheck_r_fulltext_repaired | 0.506 | 0.506 | 0.000 | [0.419, 0.591] | [0.419, 0.591] | 0.459 | 0.459 | 0 | 0 | — | — |
| statcheck_r_passages | 0.557 | 0.557 | 0.000 | [0.479, 0.634] | [0.479, 0.634] | 0.459 | 0.459 | 0 | 0 | — | — |
| statcheck_r_pdf | 0.329 | 0.329 | 0.000 | [0.235, 0.424] | [0.235, 0.424] | 0.034 | 0.034 | 0 | 0 | — | — |

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

Lines labelled `reproduce` (re-reads of the committed state) are left out; `results/gold_eval_log.jsonl` keeps them.

| label | date | git_sha | system | recall | recall with fragments | finds without results |
|---|---|---|---|---|---|---|
| m1-before | 2026-10-04T17:33:19+00:00 | e51e018 | regex | 0.489 | — | 2 |
| m1-before | 2026-10-04T17:33:19+00:00 | e51e018 | model | 0.836 | — | 155 |
| m1-before | 2026-10-04T17:33:19+00:00 | e51e018 | cascade | 0.851 | — | 155 |
| m1-before | 2026-10-04T17:33:19+00:00 | e51e018 | model_passages | 0.849 | — | 150 |
| m1-before | 2026-10-04T17:33:19+00:00 | e51e018 | statcheck_r_fulltext_raw | 0.113 | — | 2 |
| m1-before | 2026-10-04T17:33:19+00:00 | e51e018 | statcheck_r_fulltext_repaired | 0.443 | — | 2 |
| m1-before | 2026-10-04T17:33:19+00:00 | e51e018 | statcheck_r_passages | 0.479 | — | 2 |
| m1-before | 2026-10-04T17:33:19+00:00 | e51e018 | statcheck_r_pdf | 0.237 | — | 2 |
| m2-dedup | 2026-10-04T17:51:44+00:00 | e51e018 | regex | 0.498 | — | 2 |
| m2-dedup | 2026-10-04T17:51:44+00:00 | e51e018 | model | 0.849 | — | 157 |
| m2-dedup | 2026-10-04T17:51:44+00:00 | e51e018 | cascade | 0.866 | — | 157 |
| m2-dedup | 2026-10-04T17:51:44+00:00 | e51e018 | model_passages | 0.849 | — | 150 |
| m2-dedup | 2026-10-04T17:51:44+00:00 | e51e018 | statcheck_r_fulltext_raw | 0.113 | — | 2 |
| m2-dedup | 2026-10-04T17:51:44+00:00 | e51e018 | statcheck_r_fulltext_repaired | 0.443 | — | 2 |
| m2-dedup | 2026-10-04T17:51:44+00:00 | e51e018 | statcheck_r_passages | 0.479 | — | 2 |
| m2-dedup | 2026-10-04T17:51:44+00:00 | e51e018 | statcheck_r_pdf | 0.237 | — | 2 |
| m2-fragments | 2026-10-04T18:02:27+00:00 | e51e018 | regex | 0.498 | 0.498 | 2 |
| m2-fragments | 2026-10-04T18:02:27+00:00 | e51e018 | model | 0.689 | 0.849 | 73 |
| m2-fragments | 2026-10-04T18:02:27+00:00 | e51e018 | cascade | 0.723 | 0.866 | 73 |
| m2-fragments | 2026-10-04T18:02:27+00:00 | e51e018 | model_passages | 0.689 | 0.849 | 69 |
| m2-fragments | 2026-10-04T18:02:27+00:00 | e51e018 | statcheck_r_fulltext_raw | 0.113 | — | 2 |
| m2-fragments | 2026-10-04T18:02:27+00:00 | e51e018 | statcheck_r_fulltext_repaired | 0.443 | — | 2 |
| m2-fragments | 2026-10-04T18:02:27+00:00 | e51e018 | statcheck_r_passages | 0.479 | — | 2 |
| m2-fragments | 2026-10-04T18:02:27+00:00 | e51e018 | statcheck_r_pdf | 0.237 | — | 2 |
| m2-passages | 2026-10-04T18:11:49+00:00 | e51e018 | regex | 0.498 | 0.498 | 2 |
| m2-passages | 2026-10-04T18:11:49+00:00 | e51e018 | model | 0.689 | 0.849 | 69 |
| m2-passages | 2026-10-04T18:11:49+00:00 | e51e018 | cascade | 0.723 | 0.866 | 69 |
| m2-passages | 2026-10-04T18:11:49+00:00 | e51e018 | statcheck_r_fulltext_raw | 0.113 | — | 2 |
| m2-passages | 2026-10-04T18:11:49+00:00 | e51e018 | statcheck_r_fulltext_repaired | 0.443 | — | 2 |
| m2-passages | 2026-10-04T18:11:49+00:00 | e51e018 | statcheck_r_passages | 0.479 | — | 2 |
| m2-passages | 2026-10-04T18:11:49+00:00 | e51e018 | statcheck_r_pdf | 0.237 | — | 2 |
| m2-after | 2026-10-04T18:25:44+00:00 | e51e018 | regex | 0.498 | 0.498 | 2 |
| m2-after | 2026-10-04T18:25:44+00:00 | e51e018 | model | 0.689 | 0.849 | 69 |
| m2-after | 2026-10-04T18:25:44+00:00 | e51e018 | cascade | 0.723 | 0.866 | 69 |
| m2-after | 2026-10-04T18:25:44+00:00 | e51e018 | statcheck_r_fulltext_raw | 0.113 | — | 2 |
| m2-after | 2026-10-04T18:25:44+00:00 | e51e018 | statcheck_r_fulltext_repaired | 0.443 | — | 2 |
| m2-after | 2026-10-04T18:25:44+00:00 | e51e018 | statcheck_r_passages | 0.479 | — | 2 |
| m2-after | 2026-10-04T18:25:44+00:00 | e51e018 | statcheck_r_pdf | 0.237 | — | 2 |
| m2-close | 2026-10-04T18:59:36+00:00 | 2042998 | regex | 0.498 | 0.498 | 2 |
| m2-close | 2026-10-04T18:59:36+00:00 | 2042998 | model | 0.689 | 0.849 | 69 |
| m2-close | 2026-10-04T18:59:36+00:00 | 2042998 | cascade | 0.723 | 0.866 | 69 |
| m2-close | 2026-10-04T18:59:36+00:00 | 2042998 | statcheck_r_fulltext_raw | 0.113 | — | 2 |
| m2-close | 2026-10-04T18:59:36+00:00 | 2042998 | statcheck_r_fulltext_repaired | 0.443 | — | 2 |
| m2-close | 2026-10-04T18:59:36+00:00 | 2042998 | statcheck_r_passages | 0.479 | — | 2 |
| m2-close | 2026-10-04T18:59:36+00:00 | 2042998 | statcheck_r_pdf | 0.237 | — | 2 |
| v4-final:old-gru-crf-s0 | 2026-10-05T10:56:37+00:00 | 0e0069a | model | 0.689 | 0.849 | 69 |
| v4-final:old-gru-crf-s0 | 2026-10-05T10:56:37+00:00 | 0e0069a | cascade | 0.723 | 0.866 | 69 |
| v4-final:old-gru-crf-s0 | 2026-10-05T10:56:37+00:00 | 0e0069a | regex | 0.498 | 0.498 | 2 |
| v4-final:old-gru-crf-s0 | 2026-10-05T10:56:37+00:00 | 0e0069a | statcheck_r_fulltext_raw | 0.113 | — | 2 |
| v4-final:old-gru-crf-s0 | 2026-10-05T10:56:37+00:00 | 0e0069a | statcheck_r_fulltext_repaired | 0.443 | — | 2 |
| v4-final:old-gru-crf-s0 | 2026-10-05T10:56:37+00:00 | 0e0069a | statcheck_r_passages | 0.479 | — | 2 |
| v4-final:old-gru-crf-s0 | 2026-10-05T10:56:37+00:00 | 0e0069a | statcheck_r_pdf | 0.237 | — | 2 |
| v4-final:gate-none | 2026-10-05T10:57:55+00:00 | 0e0069a | model | 0.731 | 0.889 | 69 |
| v4-final:gate-none | 2026-10-05T10:57:55+00:00 | 0e0069a | cascade | 0.763 | 0.897 | 69 |
| v4-final:gate-old | 2026-10-05T10:59:14+00:00 | 0e0069a | model | 0.718 | 0.880 | 66 |
| v4-final:gate-old | 2026-10-05T10:59:14+00:00 | 0e0069a | cascade | 0.750 | 0.887 | 66 |
| v4-final:gate-old+all | 2026-10-05T11:00:33+00:00 | 0e0069a | model | 0.712 | 0.893 | 66 |
| v4-final:gate-old+all | 2026-10-05T11:00:33+00:00 | 0e0069a | cascade | 0.750 | 0.897 | 66 |
| v4-final:gate-old+font_map | 2026-10-05T11:02:46+00:00 | 0e0069a | model | 0.706 | 0.855 | 66 |
| v4-final:gate-old+font_map | 2026-10-05T11:02:46+00:00 | 0e0069a | cascade | 0.739 | 0.866 | 66 |
| v4-final:gate-old+negative_damage | 2026-10-05T11:04:47+00:00 | 0e0069a | model | 0.697 | 0.851 | 68 |
| v4-final:gate-old+negative_damage | 2026-10-05T11:04:47+00:00 | 0e0069a | cascade | 0.729 | 0.859 | 68 |
| v4-final:gate-old+operator_drop | 2026-10-05T11:07:07+00:00 | 0e0069a | model | 0.750 | 0.905 | 75 |
| v4-final:gate-old+operator_drop | 2026-10-05T11:07:07+00:00 | 0e0069a | cascade | 0.784 | 0.914 | 75 |
| v4-final:gate-old+spaced_digits | 2026-10-05T11:08:53+00:00 | 0e0069a | model | 0.725 | 0.870 | 64 |
| v4-final:gate-old+spaced_digits | 2026-10-05T11:08:53+00:00 | 0e0069a | cascade | 0.761 | 0.878 | 64 |
| v4-final:gru-crf-all-s1 | 2026-10-05T11:10:55+00:00 | 0e0069a | model | 0.721 | 0.882 | 67 |
| v4-final:gru-crf-all-s1 | 2026-10-05T11:10:55+00:00 | 0e0069a | cascade | 0.752 | 0.889 | 67 |
| v4-final:gru-crf-all-s2 | 2026-10-05T11:13:48+00:00 | 0e0069a | model | 0.681 | 0.826 | 67 |
| v4-final:gru-crf-all-s2 | 2026-10-05T11:13:48+00:00 | 0e0069a | cascade | 0.710 | 0.828 | 67 |
| v4-final:gru-crf-all-s3 | 2026-10-05T11:16:18+00:00 | 0e0069a | model | 0.681 | 0.834 | 61 |
| v4-final:gru-crf-all-s3 | 2026-10-05T11:16:18+00:00 | 0e0069a | cascade | 0.725 | 0.849 | 61 |
| v4-final:gru-crf-all-s4 | 2026-10-05T11:18:03+00:00 | 0e0069a | model | 0.685 | 0.836 | 62 |
| v4-final:gru-crf-all-s4 | 2026-10-05T11:18:03+00:00 | 0e0069a | cascade | 0.721 | 0.847 | 62 |
| v4-final:gru-crf-none-s1 | 2026-10-05T11:19:49+00:00 | 0e0069a | model | 0.729 | 0.889 | 69 |
| v4-final:gru-crf-none-s1 | 2026-10-05T11:19:49+00:00 | 0e0069a | cascade | 0.763 | 0.897 | 69 |
| v4-final:gru-crf-none-s2 | 2026-10-05T11:22:09+00:00 | 0e0069a | model | 0.683 | 0.861 | 59 |
| v4-final:gru-crf-none-s2 | 2026-10-05T11:22:09+00:00 | 0e0069a | cascade | 0.714 | 0.863 | 59 |
| v4-final:gru-crf-none-s3 | 2026-10-05T11:24:14+00:00 | 0e0069a | model | 0.695 | 0.901 | 67 |
| v4-final:gru-crf-none-s3 | 2026-10-05T11:24:14+00:00 | 0e0069a | cascade | 0.729 | 0.910 | 67 |
| v4-final:gru-crf-none-s4 | 2026-10-05T11:26:30+00:00 | 0e0069a | model | 0.731 | 0.899 | 66 |
| v4-final:gru-crf-none-s4 | 2026-10-05T11:26:30+00:00 | 0e0069a | cascade | 0.758 | 0.901 | 66 |
| v4-final:gru-softmax-all-s0 | 2026-10-05T11:27:39+00:00 | 0e0069a | model | 0.710 | 0.880 | 75 |
| v4-final:gru-softmax-all-s0 | 2026-10-05T11:27:39+00:00 | 0e0069a | cascade | 0.744 | 0.889 | 75 |
| v4-final:gru-softmax-all-s1 | 2026-10-05T11:29:22+00:00 | 0e0069a | model | 0.718 | 0.899 | 82 |
| v4-final:gru-softmax-all-s1 | 2026-10-05T11:29:22+00:00 | 0e0069a | cascade | 0.756 | 0.910 | 82 |
| v4-verdicts | 2026-10-05T11:30:09+00:00 | 0e0069a | cascade | 0.723 | — | n/a |
| v4-verdicts | 2026-10-05T11:30:09+00:00 | 0e0069a | cascade_v4 | 0.763 | — | n/a |
| v4-verdicts | 2026-10-05T11:30:09+00:00 | 0e0069a | regex | 0.498 | — | n/a |
| v4-verdicts | 2026-10-05T11:30:09+00:00 | 0e0069a | statcheck_r_fulltext_raw | 0.113 | — | n/a |
| v4-verdicts | 2026-10-05T11:30:09+00:00 | 0e0069a | statcheck_r_fulltext_repaired | 0.443 | — | n/a |
| v4-verdicts | 2026-10-05T11:30:09+00:00 | 0e0069a | statcheck_r_passages | 0.479 | — | n/a |
| v4-verdicts | 2026-10-05T11:30:09+00:00 | 0e0069a | statcheck_r_pdf | 0.237 | — | n/a |
| v4-final:gru-softmax-all-s2 | 2026-10-05T11:30:36+00:00 | 0e0069a | model | 0.735 | 0.905 | 80 |
| v4-final:gru-softmax-all-s2 | 2026-10-05T11:30:36+00:00 | 0e0069a | cascade | 0.767 | 0.912 | 80 |
| v4-final:gru-softmax-all-s3 | 2026-10-05T11:31:36+00:00 | 0e0069a | model | 0.723 | 0.897 | 77 |
| v4-final:gru-softmax-all-s3 | 2026-10-05T11:31:36+00:00 | 0e0069a | cascade | 0.754 | 0.903 | 77 |
| v4-final:gru-softmax-all-s4 | 2026-10-05T11:32:34+00:00 | 0e0069a | model | 0.718 | 0.876 | 72 |
| v4-final:gru-softmax-all-s4 | 2026-10-05T11:32:34+00:00 | 0e0069a | cascade | 0.750 | 0.882 | 72 |
| v4-final:gru-softmax-none-s0 | 2026-10-05T11:33:32+00:00 | 0e0069a | model | 0.721 | 0.884 | 70 |
| v4-final:gru-softmax-none-s0 | 2026-10-05T11:33:32+00:00 | 0e0069a | cascade | 0.754 | 0.893 | 70 |
| v4-final:gru-softmax-none-s1 | 2026-10-05T11:34:31+00:00 | 0e0069a | model | 0.727 | 0.893 | 75 |
| v4-final:gru-softmax-none-s1 | 2026-10-05T11:34:31+00:00 | 0e0069a | cascade | 0.758 | 0.899 | 75 |
| v4-final:gru-softmax-none-s2 | 2026-10-05T11:35:29+00:00 | 0e0069a | model | 0.742 | 0.916 | 76 |
| v4-final:gru-softmax-none-s2 | 2026-10-05T11:35:29+00:00 | 0e0069a | cascade | 0.767 | 0.916 | 76 |
| v4-final:gru-softmax-none-s3 | 2026-10-05T11:36:27+00:00 | 0e0069a | model | 0.733 | 0.901 | 70 |
| v4-final:gru-softmax-none-s3 | 2026-10-05T11:36:27+00:00 | 0e0069a | cascade | 0.763 | 0.908 | 70 |
| v4-final:gru-softmax-none-s4 | 2026-10-05T11:37:26+00:00 | 0e0069a | model | 0.744 | 0.914 | 73 |
| v4-final:gru-softmax-none-s4 | 2026-10-05T11:37:26+00:00 | 0e0069a | cascade | 0.775 | 0.920 | 73 |
| v4-final:lstm-softmax-all-s0 | 2026-10-05T11:38:39+00:00 | 0e0069a | model | 0.733 | 0.884 | 70 |
| v4-final:lstm-softmax-all-s0 | 2026-10-05T11:38:39+00:00 | 0e0069a | cascade | 0.758 | 0.884 | 70 |
| v4-final:lstm-softmax-all-s1 | 2026-10-05T11:39:52+00:00 | 0e0069a | model | 0.687 | 0.855 | 72 |
| v4-final:lstm-softmax-all-s1 | 2026-10-05T11:39:52+00:00 | 0e0069a | cascade | 0.723 | 0.863 | 72 |
| v4-final:lstm-softmax-all-s2 | 2026-10-05T11:41:03+00:00 | 0e0069a | model | 0.687 | 0.828 | 73 |
| v4-final:lstm-softmax-all-s2 | 2026-10-05T11:41:03+00:00 | 0e0069a | cascade | 0.718 | 0.832 | 73 |
| v4-final:lstm-softmax-all-s3 | 2026-10-05T11:42:16+00:00 | 0e0069a | model | 0.702 | 0.859 | 68 |
| v4-final:lstm-softmax-all-s3 | 2026-10-05T11:42:16+00:00 | 0e0069a | cascade | 0.731 | 0.863 | 68 |
| v4-final:lstm-softmax-all-s4 | 2026-10-05T11:43:29+00:00 | 0e0069a | model | 0.672 | 0.834 | 84 |
| v4-final:lstm-softmax-all-s4 | 2026-10-05T11:43:29+00:00 | 0e0069a | cascade | 0.704 | 0.840 | 84 |
| v4-final:lstm-softmax-none-s0 | 2026-10-05T11:44:42+00:00 | 0e0069a | model | 0.731 | 0.880 | 77 |
| v4-final:lstm-softmax-none-s0 | 2026-10-05T11:44:42+00:00 | 0e0069a | cascade | 0.763 | 0.887 | 77 |
| v4-final:lstm-softmax-none-s1 | 2026-10-05T11:45:54+00:00 | 0e0069a | model | 0.744 | 0.899 | 77 |
| v4-final:lstm-softmax-none-s1 | 2026-10-05T11:45:54+00:00 | 0e0069a | cascade | 0.771 | 0.901 | 77 |
| v4-final:lstm-softmax-none-s2 | 2026-10-05T11:47:07+00:00 | 0e0069a | model | 0.687 | 0.853 | 74 |
| v4-final:lstm-softmax-none-s2 | 2026-10-05T11:47:07+00:00 | 0e0069a | cascade | 0.718 | 0.861 | 74 |
| v4-final:lstm-softmax-none-s3 | 2026-10-05T11:48:20+00:00 | 0e0069a | model | 0.748 | 0.910 | 75 |
| v4-final:lstm-softmax-none-s3 | 2026-10-05T11:48:20+00:00 | 0e0069a | cascade | 0.773 | 0.912 | 75 |
| v4-final:lstm-softmax-none-s4 | 2026-10-05T11:49:33+00:00 | 0e0069a | model | 0.765 | 0.933 | 69 |
| v4-final:lstm-softmax-none-s4 | 2026-10-05T11:49:33+00:00 | 0e0069a | cascade | 0.796 | 0.939 | 69 |
