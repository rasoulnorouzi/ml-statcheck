## Results

| System | P | R | F1 [CI] |
|---|---|---|---|
| statcheck_raw | 0.983 | 0.183 | 0.308 [0.222, 0.385] |
| statcheck_repaired | 0.993 | 0.467 | 0.636 [0.567, 0.698] |
| gru-crf-s0 | 0.952 | 0.861 | 0.904 [0.871, 0.934] |
| gru-softmax-s0 | 0.925 | 0.882 | 0.903 [0.868, 0.933] |
| lstm-softmax-s0 | 0.910 | 0.879 | 0.894 [0.851, 0.932] |
| cascade_gru-crf-s0 | 0.949 | 0.870 | 0.908 [0.875, 0.937] |

The cascade reaches holdout F1 0.908, against 0.636 for statcheck alone.
