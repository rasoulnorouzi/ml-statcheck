# Version 4 status

Branch `v4-gold`. Team run `wf_963324ab-ab1`. Plan:
`docs/superpowers/plans/2026-10-04-v4-gold.md`. Ledger: `docs/v4/LEDGER.md`. Prompts:
`docs/v4/prompts/`. The session and the manager both update this list.

`[x]` done · `[~]` doing · `[ ]` to do

Last update: 2026-10-04, M3 interim commit; M3.2b verified but not accepted; M3c prompts written.

## Done

- [x] **S0 Setup** — session — opus. Sheet ignored by git; `.venv/`; 3100 PDFs converted;
  R statcheck 1.5.0; ports cloned; gold set (409 papers, 2704 results, 2664 located);
  first gold scores; the brief; the reporting rule in CLAUDE.md. Commit `54fc781`.
- [x] **M0 Plan** — manager — fable. The plan, the version 4 table in `PLAN.md`,
  the M1 prompts, the ledger. Tests before any change: 361 passed, 6 skipped.
- [x] **M1 Prepare** — six tasks, all accepted; the ledger holds one entry each.
  Suite at close: 384 passed, 6 skipped.
  - M1.1 audit (runtime-engineer, sonnet): `hybrid.py`, dead `repair.py` functions,
    `build/` removed; `reproduce.sh` picks the Linux interpreter.
  - M1.2 rename (eval-engineer, sonnet): the bronze holdout is `bronze` everywhere;
    provenance paths are POSIX; no bronze number moved.
  - M1.3 split (eval-engineer, sonnet): gold test 169 papers, 476 results; gold train
    240 papers, 2007 results; `dataset/gold/split.json` frozen, sha256 fbb0214f….
  - M1.4 gold stages (eval-engineer, sonnet): evaluation by split, tests, README,
    `reproduce.sh` stage `gold`, manifest, `results/gold_eval_log.jsonl`.
  - M1.5 original statcheck on gold (regex-porter, sonnet): four forms, R 4.5.0,
    statcheck 1.5.0; deterministic CSVs in `dataset/baseline/gold/`.
  - M1.6 statcheck scored on gold (eval-engineer, sonnet).

  The version 4 baseline on the gold test (169 papers, 476 in-scope results):

  | System | Recall | 95% CI | Finds in 120 no-result papers |
  |---|---|---|---|
  | statcheck R, raw full text | 0.113 | 0.036–0.257 | 2 |
  | statcheck R, repaired full text | 0.443 | 0.238–0.646 | 2 |
  | statcheck R, passages | 0.479 | 0.293–0.654 | 2 |
  | statcheck R, PDF (`checkPDF`) | 0.237 | 0.055–0.493 | 2 |
  | Python regex port | 0.489 | 0.292–0.640 | 2 |
  | shipped model alone | 0.836 | 0.773–0.893 | 155 |
  | shipped cascade | 0.851 | 0.787–0.908 | 155 |
  | shipped model on passages | 0.849 | 0.787–0.907 | 150 |

  Bronze holdout, unchanged: cascade P 0.949, R 0.870, F1 0.908.

- [x] **M2 Fixes without training** — seven tasks — commits `d446a67`, `2042998`; ports R
  `33faa2e`, web `32a74f4`. Dedup by statistic position; nameless finds become fragments;
  merged passages as the unit (spec version 2, 0.57x the text read); both ports follow.
- [x] **M2b Close-out of M2** — five tasks — commits `dcb3a3b`, `003a3ce`; ports R `1e4726d`,
  web `2ff741a`. `line` from the statistic position in all three ports; the import-path
  flake fixed; R CMD check OK; stale sentences (M2b.4 finished by the session after two
  rejections of the doc-writer, haiku); both recalls in the compare stage.
  Gold test after M2: cascade recall 0.866 with fragments, 0.723 named; finds in the 120
  no-result papers 155 → 69. Bronze holdout unchanged. Python 419, R 1280, web 385 passed.
  Detail per task: `docs/v4/LEDGER.md`.
- [x] **Session: pandoc 3.12** in `~/.local/bin`; R CMD check with the vignette built:
  Status OK.

## Doing

- [~] **M7 Training on the owner's A100** (Colab, `coab.ipynb`, kit 324bbced…, commit
  `c201cee`): the 23 pre-registered runs, 12 at a time, batch 64; models saved to
  `My Drive/statcheck_v4/`. The owner sends `v4_models.tar.gz` when it ends.

## Done since the session took over (2026-10-04 22:45 – 2026-10-05 00:30)

- [x] M3c sample and batches (`1097937`); M4 rating, 351 blind rater files, consensus
  1945 kept and 145 disputes (`c836bea`); M5 adjudication, final labels (1994 results),
  `train_v4.jsonl` (1927 bronze + 1165 passages) (`e0d289f`); pre-registration and its
  dated notes; selection tooling (`ed0054d`); GPU kit, scenarios, CRF speed-up and the
  progress table (`c201cee`).

## To do

- [ ] Unpack the models into `statcheck-ml/models/v4/`; `23_select_v4.py` (export, dev
  scores, rules 1 to 3)
- [ ] M8 One gold-test read and the bronze holdout; statcheck R verdicts against
  reference verdicts from the coders' values; the report
- [ ] M9 Ports: kits, R and web tests; M10 documentation and the double reproduce

## Gold discipline

The log `results/gold_eval_log.jsonl` holds five labelled evaluations of the gold test
under M1 and M2 (m1-before, m2-dedup, m2-fragments, m2-passages, m2-after), and M2b adds
m2-close. Each measured a rule change after the previous number was seen. The gold test
is therefore not a clean held-out score for the M2 rule changes (dedup, fragments,
passages). It stays clean for the training of version 4: no training input reads it,
and the M6 gates run on the gold-train dev papers only. Every report states this.

## Open points carried forward

- Done in M2.1: `results/gold_eval.json` holds no date and a relative model path.
- Done in M2b: `SAMPLING.md` line 27, the `raw` sentence in `dataset/README.md`, the
  fragment parity case, `ports.json` from a clean export.
- The R vignette is unchecked on this machine (no pandoc); `R CMD check` ran with
  `--ignore-vignettes`.
- `13_ports.py` runs only `test_parity_self.py` for the Python port and hard-codes
  `tests_failed: 0` for it; a later cleanup task parses the count.
- The `unparsed_units` count in `dataset/baseline/gold/versions.json` means "no result
  or error" for the text forms and "R error only" for the PDF form.
