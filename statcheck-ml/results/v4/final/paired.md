# Paired comparisons, version 4 gold test

Source: stored files under `results/v4/final/`. No gold paper and no model is read.
Seed 0, 2000 paper resamples.

Rows within a paper are not independent. The paper bootstrap is therefore the main
interval. The exact McNemar p treats rows as independent and is too small when results
cluster in a few papers. Read it as a secondary figure.

## Result found (verdict not null)

Population: 476 gold rows.

| A | B | A only | B only | both | neither | recall A | recall B | diff A-B [95% paper CI] | McNemar p |
|---|---|---|---|---|---|---|---|---|---|
| cascade | cascade_v4 | 4 | 23 | 340 | 109 | 0.723 | 0.763 | -0.040 [-0.123, +0.013] | 3.11e-04 |
| cascade | statcheck_r_passages | 128 | 12 | 216 | 120 | 0.723 | 0.479 | +0.244 [+0.092, +0.394] | 1.15e-25 |
| cascade | statcheck_r_pdf | 246 | 15 | 98 | 117 | 0.723 | 0.237 | +0.485 [+0.253, +0.684] | 5.18e-55 |
| cascade_v4 | statcheck_r_passages | 149 | 14 | 214 | 99 | 0.763 | 0.479 | +0.284 [+0.131, +0.419] | 1.14e-29 |

## Reference inconsistency flagged

Population: 73 gold rows with a reference verdict of inconsistent or decision_error.

| A | B | A only | B only | both | neither | recall A | recall B | diff A-B [95% paper CI] | McNemar p |
|---|---|---|---|---|---|---|---|---|---|
| cascade | cascade_v4 | 0 | 2 | 32 | 39 | 0.438 | 0.466 | -0.027 [-0.046, +0.000] | 0.500 |
| cascade | statcheck_r_passages | 7 | 1 | 25 | 40 | 0.438 | 0.356 | +0.082 [+0.000, +0.244] | 0.070 |
| cascade | statcheck_r_pdf | 22 | 0 | 10 | 41 | 0.438 | 0.137 | +0.301 [+0.065, +0.613] | 4.77e-07 |
| cascade_v4 | statcheck_r_passages | 9 | 1 | 25 | 38 | 0.466 | 0.356 | +0.110 [+0.022, +0.256] | 0.021 |

## Combined table

Gold columns come from the gold test; bronze columns come from the bronze holdout. The
bronze labels are machine annotations, not human validation. The two are never pooled.
Bronze figures for the two cascades are for the model alone, without the cascade.
`statcheck_raw` and `statcheck_repaired` exist only on the bronze windows.

| system | gold recall (named) [95% CI] | gold finds in no-result papers | bronze precision | bronze recall | bronze F1 |
|---|---|---|---|---|---|
| shipped cascade (gru-crf-s0) | 0.723 [0.556, 0.836] | 69 | 0.952 | 0.861 | 0.904 |
| candidate cascade (lstm-softmax-none-s0) | 0.763 [0.594, 0.871] | 77 | 0.899 | 0.854 | 0.876 |
| regex | 0.498 [0.299, 0.650] | 2 | — | — | — |
| statcheck_r_fulltext_raw | 0.113 [0.036, 0.257] | 2 | — | — | — |
| statcheck_r_fulltext_repaired | 0.443 [0.238, 0.646] | 2 | — | — | — |
| statcheck_r_passages | 0.479 [0.293, 0.654] | 2 | — | — | — |
| statcheck_r_pdf | 0.237 [0.055, 0.493] | 2 | — | — | — |
| statcheck_raw (bronze windows only) | — | — | 0.983 | 0.183 | 0.308 |
| statcheck_repaired (bronze windows only) | — | — | 0.993 | 0.467 | 0.636 |

## Provenance

Git sha: `c599c38025c71dc623a1cab89f9c717428b85f2b`

- `results/v4/final/verdicts.json`: `e70d0cc34b35495738ccc196fd4627f234b1602884d1220ff7213a4c0c9fb338`
- `results/v4/final/summary.json`: `75cda18e62aa6e0c1dab9f186ef16923433f6b37c2f1c3b5b10e6e4f49b34744`
- `results/v4/final/bronze.json`: `b685160f8506bfd54ac4ecc64c9c86b076cf9baa62f927b084b32a36fe91ea8e`
