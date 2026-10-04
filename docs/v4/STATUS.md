# Version 4 status

Branch `v4-gold`. Team run `wf_963324ab-ab1`. Plan:
`docs/superpowers/plans/2026-10-04-v4-gold.md`. Ledger: `docs/v4/LEDGER.md`. Prompts:
`docs/v4/prompts/`. The session and the manager both update this list.

`[x]` done · `[~]` doing · `[ ]` to do

Last update: 2026-10-04, M2 tasks committed; M2b (close-out) prompts written.

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

- [x] **M2 Fixes without training** — prompts in `docs/v4/prompts/M2.md`. All seven tasks accepted and committed; the ledger holds one entry each. M2 closes when M2b lands (the brief's session notes).
  - [x] M2.1 Measurement scaffold: the before snapshot, the gold log, the compare stage — eval-engineer — sonnet. Accepted. Before snapshot at `e51e018` in `results/v4/before/`; `16_compare.py`; tests 387 passed.
  - [x] M2.2 Duplicate-statistic fault: dedup by position — runtime-engineer — sonnet. Accepted.
    Gold test recall before → after: cascade 0.851 → 0.866 [0.802–0.920]; model 0.836 → 0.849; regex 0.489 → 0.498. Finds in no-result papers: cascade 155 → 157. Parity cases 238 (pipeline section 4 → 6). Tests 394 passed, 6 skipped.
  - [x] M2.3 Finds with no test name reported as fragments — runtime-engineer — sonnet. Accepted.
    Gold test, cascade: finds in no-result papers 157 → 73 (84 more are now fragments). Recall of named results 0.866 → 0.723; recall with fragments counted stays 0.866. So about 14 points of gold results are found with no test name. Tests 403 passed, 6 skipped.
  - [x] M2.4 Merged passages as the unit, `prefilter.json` version 2 — regex-porter — sonnet. Accepted.
    Gold test: recall unchanged (cascade 0.723 named, 0.866 with fragments); finds in no-result papers 73 → 69; characters the model reads 11.4 M → 6.5 M (0.57×). Parity version 3, 243 cases. Tests 412 passed, 6 skipped.
  - [x] M2.5 R port reads the new spec; kit; testthat green — runtime-engineer — sonnet. Accepted.
    testthat: 268 tests, 1259 expectations, 0 failed. Sample PDF: 5 results, 4 consistent, 1 decision error. R CMD check not run. Regression found: `line` is now the first line of the passage (0 for all five sample rows, was 2, 2, 2, 2, 4).
  - [x] M2.6 Web port reads the new spec; kit; vitest green — runtime-engineer — sonnet. Accepted.
    vitest before: 6 failed, 364 passed (all 6 pipeline parity cases). After: 381 passed, 9 skipped (the browser test needs Chromium, not installed). Fragments in Markdown, JSON and the demo page, not in the CSV. Kit re-exported; rebuild it after the mother commit.
  - [x] M2.7 After numbers: bronze, gold, the compare table, the report — eval-engineer — sonnet. Accepted.
    Bronze: no number moved (the bronze scorer does not call run_text). Gold test, cascade: named recall 0.851 → 0.723, recall with fragments 0.851 → 0.866; no-result papers 155 finds → 69 finds + 82 fragments. Ports: Python 246, R 1259, web 381 parity checks, 0 failed.

- [~] **M2b Close-out of M2** — prompts in `docs/v4/prompts/M2b.md`. The three items the
  brief's session notes require before M2 closes, plus two small follow-ups.
  - [ ] M2b.1 `line` from the statistic position (Python, parity `line_rule`), the
    import-path flake, a parity case with a real fragment — runtime-engineer — sonnet
  - [ ] M2b.2 R port: `line` rule, kit, testthat, `R CMD check`, library reinstalled — runtime-engineer — sonnet
  - [ ] M2b.3 Web port: `line` rule, kit, vitest — runtime-engineer — sonnet
  - [ ] M2b.4 Stale sentences: report template lines 25, 352, 355, 362; `SAMPLING.md`
    line 27; `dataset/README.md` on the `raw` field — doc-writer — haiku
  - [ ] M2b.5 `recall_with_fragments` in `16_compare.py`; gold run `m2-close` (every
    number must equal `m2-after`); `ports.json`; report — eval-engineer — sonnet

## To do

- [ ] M3 Annotation design and sampling: guideline 3.0 (label-architect, opus), the
  gold-train dev split and the passage inventory, the sampler and batches of ten
  (data-engineer, haiku)
- [ ] M4 Rating: three blind raters per batch, collect, agreement, four-vote consensus
- [ ] M5 Consensus and dataset: disputes, adjudicator, merge, dataset, alignment gate
- [ ] M6 Noise layer rebuilt and gated; the selection rule written before training
- [ ] M7 Training: ten runs, zoo export, the rule applied
- [ ] M8 Evaluation: gold test, bronze holdout, both combined; the report
- [ ] M9 Ports: kits, R and web tests, Python package tests
- [ ] M10 Documentation and the double reproduce

## Gold discipline

The log `results/gold_eval_log.jsonl` holds five labelled evaluations of the gold test
under M1 and M2 (m1-before, m2-dedup, m2-fragments, m2-passages, m2-after), and M2b adds
m2-close. Each measured a rule change after the previous number was seen. The gold test
is therefore not a clean held-out score for the M2 rule changes (dedup, fragments,
passages). It stays clean for the training of version 4: no training input reads it,
and the M6 gates run on the gold-train dev papers only. Every report states this.

## Open points carried forward

- Done in M2.1: `results/gold_eval.json` holds no date and a relative model path.
- `SAMPLING.md` line 27 still says "check the silver labels"; `dataset/README.md` says
  gold.jsonl holds no quotes, yet each result carries the owner's transcribed `raw`
  field. Both go to M2b.4.
- The parity file has no pipeline case with a non-empty fragment list; M2b.1 adds one.
- `results/ports.json` names a kit exported from a dirty tree; M2b.5 regenerates it.
- The `unparsed_units` count in `dataset/baseline/gold/versions.json` means "no result
  or error" for the text forms and "R error only" for the PDF form.
