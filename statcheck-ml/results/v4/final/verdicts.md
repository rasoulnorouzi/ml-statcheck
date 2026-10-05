# Verdicts on the gold rows (gold_test)

The reference is R statcheck 1.5.0 on one APA string per coded gold row. A system is scored on the rows that have a reference. "Flagged" means inconsistent or decision error.

## Reference

| rows | with reference | consistent | inconsistencies | decision errors |
|---|---|---|---|---|
| 476 | 465 | 392 | 73 | 20 |

Inconsistencies include the decision errors.

| no reference, cause | rows |
|---|---|
| no degrees of freedom coded | 2 |
| no p-value coded | 3 |
| no statistic coded | 6 |

Of 465 strings, 463 take the statistic and 453 take p as `raw` prints them; 4 are "ns". 2 write a coded ≤ or ≥ as < or >. In 9 strings `raw` does not print a coded value, so the coded value is used; 2 of the reference inconsistencies are such rows, and some of them are coding slips.

## pvalue.py against R

pvalue.py gives the R verdict on 465 of 465 rows. The largest relative difference in the computed p is 3.52e-13; 0 rows exceed 1e-06.

## Reference inconsistencies

| system | n | flagged | share [95% CI] | not found | found, judged consistent | found, not checked |
|---|---|---|---|---|---|---|
| cascade | 73 | 32 | 0.438 [0.163, 0.843] | 38 | 0 | 3 |
| cascade_v4 | 73 | 34 | 0.466 [0.191, 0.865] | 26 | 0 | 13 |
| regex | 73 | 28 | 0.384 [0.125, 0.750] | 45 | 0 | 0 |
| statcheck_r_fulltext_raw | 73 | 6 | 0.082 [0.0, 0.295] | 67 | 0 | 0 |
| statcheck_r_fulltext_repaired | 73 | 25 | 0.342 [0.095, 0.696] | 47 | 1 | 0 |
| statcheck_r_passages | 73 | 26 | 0.356 [0.106, 0.704] | 46 | 1 | 0 |
| statcheck_r_pdf | 73 | 10 | 0.137 [0.0, 0.423] | 63 | 0 | 0 |

## Reference decision errors

| system | n | flagged as decision error | flagged as inconsistent only | not found | found, judged consistent | found, not checked |
|---|---|---|---|---|---|---|
| cascade | 20 | 4 | 0 | 15 | 0 | 1 |
| cascade_v4 | 20 | 4 | 0 | 14 | 0 | 2 |
| regex | 20 | 2 | 0 | 18 | 0 | 0 |
| statcheck_r_fulltext_raw | 20 | 0 | 0 | 20 | 0 | 0 |
| statcheck_r_fulltext_repaired | 20 | 2 | 0 | 18 | 0 | 0 |
| statcheck_r_passages | 20 | 2 | 0 | 18 | 0 | 0 |
| statcheck_r_pdf | 20 | 0 | 0 | 20 | 0 | 0 |

## Flags where the reference has no error

A find outside gold is not always wrong: the coders followed statcheck's scope.

| system | consistent rows | found | flagged | of which decision errors | flags on finds outside gold, papers with results | the same, papers with no result | flags on gold rows with no reference |
|---|---|---|---|---|---|---|---|
| cascade | 392 | 307 | 1 | 0 | 0 | 0 | 0 |
| cascade_v4 | 392 | 314 | 0 | 0 | 2 | 0 | 0 |
| regex | 392 | 209 | 0 | 0 | 0 | 0 | 0 |
| statcheck_r_fulltext_raw | 392 | 48 | 0 | 0 | 0 | 0 | 0 |
| statcheck_r_fulltext_repaired | 392 | 185 | 0 | 0 | 0 | 0 | 0 |
| statcheck_r_passages | 392 | 201 | 0 | 0 | 0 | 0 | 0 |
| statcheck_r_pdf | 392 | 103 | 0 | 0 | 0 | 0 | 0 |

## Papers

| system | papers with a reference inconsistency | papers flagged correctly at least once | recall of the in-scope rows |
|---|---|---|---|
| cascade | 15 | 9 | 0.723 |
| cascade_v4 | 15 | 10 | 0.763 |
| regex | 15 | 8 | 0.498 |
| statcheck_r_fulltext_raw | 15 | 2 | 0.113 |
| statcheck_r_fulltext_repaired | 15 | 7 | 0.443 |
| statcheck_r_passages | 15 | 8 | 0.479 |
| statcheck_r_pdf | 15 | 3 | 0.237 |
