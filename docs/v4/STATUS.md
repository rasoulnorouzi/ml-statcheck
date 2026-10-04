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

- Team run `wf_963324ab-ab1` stopped in M3 (a worker with no report). A resume began to
  re-run M1 live and was stopped within two minutes; nothing changed on disk. The work
  continues in a new run, `wf_1ad266e8-2f6`, from M3 (`docs/v4/team-workflow-continue.js`,
  plan in `docs/v4/remaining-at-M3.json`).

- [~] **M3 Annotation design and sampling** — prompts in `docs/v4/prompts/M3.md`
  - [x] M3.1 Guideline 3.0 for passages, the four-vote rule, the agents, PROTOCOL.md
    version 3 — label-architect — opus. Accepted. Guideline sha256 961f3145…; embedding
    test 5 of 5. Rules added for a missing test letter, tables, passage edges, and "ns".
  - [x] M3.2 The gold-train dev split (15 % by paper) and the passage inventory of the
    other gold-train papers — data-engineer — haiku. Dev split accepted (36 papers). The
    first inventory was rejected. The retry fixed the script but left stale outputs and
    returned no report, which stopped the team run. The session regenerated the inventory
    with the fixed script. The manager verified it in the new run: 204 papers, 12592
    pieces, 585 passages with an in-scope coder row, 80 hard negatives, coder rows placed
    1725 of 1812, 0 span failures; rerun byte-identical (74d77061…). Ledger entry written.
  - [~] M3.2b Fix the inventory: piece hash as `text_sha256`, `paper_text_sha256`, the
    manifest, no `run_seconds` in the summary, dead code — data-engineer — haiku. The
    inventory part is verified (rerun byte-identical, 8f112191…; manifest verifies; 437
    tests pass) and committed. Rejected for the README paragraph and what-comments;
    the fix is M3.2c. The run `wf_1ad266e8-2f6` ended there; M3.3 and M3.4 were skipped.
  - [ ] M3.2c The README paragraph and the comments — data-engineer — haiku.
  - [ ] M3.3 The passage sampler (caps 900 / 400 / 200 / 300 kept: 585 + 80 + 200 + 300
    = 1165 passages); refuses a broken inventory — data-engineer — haiku. After M3.2c.
  - [ ] M3.4 Batches of ten, unpadded names, 117 batches; the stages wired into
    `reproduce.sh` and the pipeline README — data-engineer — haiku. After M3.3.
    Prompts: `docs/v4/prompts/M3c.md`.

## To do

- [ ] M4 Rating: 117 batches x 3 raters (haiku, sonnet, opus) = 351 blind rater runs, as
  general-purpose agents with guideline 3.0 in the prompt; collect with the guideline-hash
  check; agreement; the four-vote consensus script. Six agents at a time.
- [ ] M5 Consensus and dataset: disputes, adjudicator, merge, dataset, alignment gate
- [ ] M6 Noise layer rebuilt and gated; the selection rule written before training
- [ ] M7 Training: ten runs, zoo export, the rule applied
- [ ] M8 Evaluation: gold test, bronze holdout, both combined; the original statcheck R
  verdicts against reference verdicts from the coders' values; the report
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
- Done in M2b: `SAMPLING.md` line 27, the `raw` sentence in `dataset/README.md`, the
  fragment parity case, `ports.json` from a clean export.
- The R vignette is unchecked on this machine (no pandoc); `R CMD check` ran with
  `--ignore-vignettes`.
- `13_ports.py` runs only `test_parity_self.py` for the Python port and hard-codes
  `tests_failed: 0` for it; a later cleanup task parses the count.
- The `unparsed_units` count in `dataset/baseline/gold/versions.json` means "no result
  or error" for the text forms and "R error only" for the PDF form.
