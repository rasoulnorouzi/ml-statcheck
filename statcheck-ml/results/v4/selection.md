# Version 4 selection on the 36 dev papers

Rule system: `cascade`. Gold test and bronze holdout were not read.

| run | R_named | R_frag | FP | cascade R_named | cascade FP | parameters | export+score s |
|---|---|---|---|---|---|---|---|
| gate-none | 0.824 | 0.853 | 1 | 0.824 | 1 | 467912 | 29 |
| gate-old | 0.799 | 0.827 | 0 | 0.799 | 0 | 469000 | 24 |
| gate-old+font_map | 0.809 | 0.835 | 0 | 0.809 | 0 | 469000 | 25 |
| gate-old+negative_damage | 0.809 | 0.845 | 0 | 0.809 | 0 | 469320 | 25 |
| gate-old+spaced_digits | 0.809 | 0.853 | 0 | 0.809 | 0 | 469000 | 24 |
| gate-old+operator_drop | 0.845 | 0.878 | 2 | 0.845 | 2 | 469000 | 23 |
| gate-old+all | 0.806 | 0.838 | 0 | 0.806 | 0 | 469832 | 21 |
| gru-crf-none-s1 | 0.835 | 0.867 | 2 | 0.835 | 2 | 467912 | 22 |
| gru-crf-all-s1 | 0.820 | 0.856 | 1 | 0.820 | 1 | 469832 | 23 |
| gru-crf-none-s2 | 0.809 | 0.849 | 0 | 0.809 | 0 | 467912 | 21 |
| gru-crf-all-s2 | 0.795 | 0.835 | 1 | 0.795 | 1 | 469640 | 21 |
| gru-crf-none-s3 | 0.820 | 0.860 | 1 | 0.820 | 1 | 467912 | 21 |
| gru-crf-all-s3 | 0.795 | 0.842 | 1 | 0.795 | 1 | 469896 | 22 |
| gru-crf-none-s4 | 0.806 | 0.849 | 1 | 0.806 | 1 | 467912 | 21 |
| gru-crf-all-s4 | 0.824 | 0.853 | 0 | 0.824 | 0 | 469640 | 22 |
| gru-softmax-none-s0 | 0.831 | 0.867 | 1 | 0.831 | 1 | 466469 | 16 |
| gru-softmax-all-s0 | 0.824 | 0.856 | 4 | 0.824 | 4 | 468389 | 17 |
| gru-softmax-none-s1 | 0.838 | 0.874 | 2 | 0.838 | 2 | 466469 | 16 |
| gru-softmax-all-s1 | 0.795 | 0.824 | 8 | 0.795 | 8 | 468389 | 16 |
| gru-softmax-none-s2 | 0.849 | 0.881 | 3 | 0.849 | 3 | 466469 | 16 |
| gru-softmax-all-s2 | 0.820 | 0.849 | 3 | 0.820 | 3 | 468197 | 16 |
| gru-softmax-none-s3 | 0.824 | 0.860 | 1 | 0.824 | 1 | 466469 | 16 |
| gru-softmax-all-s3 | 0.820 | 0.849 | 3 | 0.820 | 3 | 468453 | 16 |
| gru-softmax-none-s4 | 0.827 | 0.860 | 1 | 0.827 | 1 | 466469 | 16 |
| gru-softmax-all-s4 | 0.806 | 0.838 | 3 | 0.806 | 3 | 468197 | 16 |
| lstm-softmax-none-s0 | 0.824 | 0.860 | 0 | 0.824 | 0 | 614949 | 19 |
| lstm-softmax-all-s0 | 0.817 | 0.853 | 2 | 0.817 | 2 | 616869 | 19 |
| lstm-softmax-none-s1 | 0.817 | 0.842 | 1 | 0.817 | 1 | 614949 | 19 |
| lstm-softmax-all-s1 | 0.835 | 0.863 | 4 | 0.835 | 4 | 616869 | 19 |
| lstm-softmax-none-s2 | 0.831 | 0.860 | 2 | 0.831 | 2 | 614949 | 18 |
| lstm-softmax-all-s2 | 0.831 | 0.860 | 3 | 0.831 | 3 | 616677 | 19 |
| lstm-softmax-none-s3 | 0.824 | 0.860 | 0 | 0.824 | 0 | 614949 | 19 |
| lstm-softmax-all-s3 | 0.813 | 0.849 | 3 | 0.813 | 3 | 616933 | 19 |
| lstm-softmax-none-s4 | 0.831 | 0.863 | 2 | 0.831 | 2 | 614949 | 18 |
| lstm-softmax-all-s4 | 0.809 | 0.838 | 1 | 0.809 | 1 | 616677 | 19 |
| gru-crf-s0 (shipped) | 0.827 | 0.867 | 0 | 0.827 | 0 | 466661 | 18 |

Rule 1: noise setting `none`. mean R_named differs by 0.01 or more; the higher wins. FP limit 3.0.
Rule 2: configuration `lstm-softmax`. means within 0.01 of the top for ['gru-softmax', 'lstm-softmax']; lowest mean FP, then fewest parameters.

| configuration | mean R_named | mean FP | parameters |
|---|---|---|---|
| gru-crf | 0.819 | 1.0 | 467912 |
| gru-softmax | 0.834 | 1.6 | 466469 |
| lstm-softmax | 0.825 | 1.0 | 614949 |

Rule 3: run `lstm-softmax-none-s0` R_named 0.824 against 0.827 for gru-crf-s0. The old model stays. candidate R_named is below the old R_named; the old model stays.
