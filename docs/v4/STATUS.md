# Version 4 status

Branch `v4-gold`. Team run `wf_963324ab-ab1`. Plan:
`docs/superpowers/plans/2026-10-04-v4-gold.md`. Ledger: `docs/v4/LEDGER.md`. Prompts:
`docs/v4/prompts/`. The session and the manager both update this list.

`[x]` done · `[~]` doing · `[ ]` to do

Last update: 2026-10-04, M1 closed and committed; M2 prompts written.

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

## Doing

- [~] **M2 Fixes without training** — seven tasks in a chain, prompts in
  `docs/v4/prompts/M2.md`: M2.1 measurement scaffold; M2.2 dedup by position; M2.3
  fragments; M2.4 passages as the unit (spec version 2); M2.5 R port; M2.6 web port;
  M2.7 after numbers and the compare table.

## To do

- [ ] M3 Annotation design and sampling: guideline 3.0 (label-architect, opus), passage
  sampler and batches (data-engineer, haiku)
- [ ] M4 Rating: three blind raters per batch, collect, agreement, four-vote consensus
- [ ] M5 Consensus and dataset: disputes, adjudicator, merge, dataset, alignment gate
- [ ] M6 Noise layer rebuilt and gated; the selection rule written before training
- [ ] M7 Training: ten runs, zoo export, the rule applied
- [ ] M8 Evaluation: gold test, bronze holdout, both combined; the report
- [ ] M9 Ports: kits, R and web tests, Python package tests
- [ ] M10 Documentation and the double reproduce

## Open points carried forward

- `results/gold_eval.json` holds the run date and an absolute model path; M2.1 moves
  the date to the log and makes the path relative, so two runs give the same bytes.
- `SAMPLING.md` line 27 still says "check the silver labels"; `dataset/README.md` says
  gold.jsonl holds no quotes, yet each result carries the owner's transcribed `raw`
  field. Both go to the M10 documentation task.
- The `unparsed_units` count in `dataset/baseline/gold/versions.json` means "no result
  or error" for the text forms and "R error only" for the PDF form.
