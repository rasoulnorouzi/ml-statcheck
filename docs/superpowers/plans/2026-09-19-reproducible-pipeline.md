# Reproducible pipeline v2 — implementation plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Rebuild the statcheck-ml process so that three blind rater agents, chance-corrected agreement, scripted adjudication, a seeded benchmark of character models, and a generated report all run from committed files.

**Architecture:** `src/statcheck_ml/` is the library and holds every function that a test touches. `pipeline/NN_name.py` files are thin command-line stages that call the library. `dataset/`, `models/zoo/`, `models/runs.json`, and `results/` hold the committed products of each stage. The manager dispatches one named agent per task, checks the definition of done, and rejects work that lacks evidence.

**Tech Stack:** Python 3.12 (venv at repo root `.venv`), torch 2.14 CPU, onnx 1.22, onnxruntime 1.29, numpy, scipy, `krippendorff`, matplotlib, pytest. Node 24 for the JS tests. R via winget for the R baseline.

**Deviation from the writing-plans template, on purpose:** the owner assigns each task to a named agent with a fixed model and asks for token economy. Tasks therefore state the contract, the tests, and the acceptance evidence, and leave the implementation to the agent. Where the mathematics must be exact, the test values are given.

**Conventions every task follows**

- Python is `../.venv/Scripts/python.exe` from `statcheck-ml/`, or `.venv/Scripts/python.exe` from the repo root. Bare `python` is the Windows Store shim and has no packages.
- Set `PYTHONIOENCODING=utf-8` for every command that prints window text.
- Load any JSON that may hold copied window text with `json.loads(s, strict=False)`. Rater outputs contain raw control characters inside `quote`.
- Every pipeline script has `argparse`, a module docstring with one example command, and `if __name__ == "__main__": main()`.
- Every library module that a task creates has a test file `tests/test_<module>.py`. `python -m pytest tests -q` must pass before a task is reported done.
- Documentation prose follows ASD-STE100. Code comments do not.
- Commit after each task with a message that says what changed and why. End with `Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>`.

**Definition of done, checked by the manager on every task**

1. The documented command runs from the venv and exits 0. The agent pastes the command and the last 20 lines of output.
2. Determinism: the agent runs the stage twice and pastes both output hashes (`sha256sum` or `certutil -hashfile`). They match.
3. `python -m pytest tests -q` passes. The agent pastes the summary line.
4. No number is typed by hand into a Markdown file.
5. Each script is about 200 lines or fewer and restates no rule from `spec/`.
6. The report to the manager lists the files created, modified, and deleted.

---

## File map

| Path | Responsibility | Task |
|---|---|---|
| `docs/GUIDELINE.md` | annotation guideline v2.0, embedded by rater agents | 1 |
| `docs/SCHEMA.md` | tag set as implemented in `labels.py` | 1 |
| `docs/PROTOCOL.md` | annotation, agreement, adjudication protocol; pools and budget | 1 |
| `.claude/agents/annotator-{haiku,sonnet,opus}.md` | rater agents | 1 |
| `.claude/agents/adjudicator.md` | dispute judge | 1 |
| `pipeline/00_convert.py`, `00_font_table.py`, `01_sample_windows.py`, `01_sample_holdout.py` | corpus stages, moved | 2 |
| `pipeline/08_port_kit.py`, `12_engines.py`, `12_regex_parity.py` | moved tooling | 2 |
| `requirements.txt`, `reproduce.sh`, `.gitignore` | environment and entry point | 2 |
| `dataset/windows/{train,holdout,key}.json`, `dataset/MANIFEST.json` | frozen inputs | 3 |
| `src/statcheck_ml/provenance.py`, `align.py` | hashes, manifest, stamps; quote alignment | 3 |
| `pipeline/02_chunk.py`, `03_collect.py` | batches; validate, stamp, merge | 3 |
| `src/statcheck_ml/agreement.py`, `pipeline/04_agree.py` | κ, α, span F1, bootstrap | 4 |
| `src/statcheck_ml/model.py`, `train.py`, `export.py`, `pipeline.py` | char-CNN unit, `--splits`, unit-aware loading | 5 |
| `pipeline/07_train.py`, `pipeline/grid.json`, `pipeline/08_export.py` | grid runner, zoo export | 5 |
| `dataset/annotations/<set>/<rater>.json` | rater outputs with provenance | 6 |
| `src/statcheck_ml/consensus.py`, `pipeline/05_adjudicate.py` | majority vote, disputes, merge | 7 |
| `dataset/annotations/<set>/final.json`, `dataset/agreement/<set>.json` | final labels, agreement | 7 |
| `src/statcheck_ml/splits.py`, `pipeline/06_dataset.py` | rows with offsets, splits | 8 |
| `dataset/train.jsonl`, `holdout.jsonl`, `splits.json` | training data | 8 |
| `models/runs.json`, `models/zoo/<name>/` | benchmark record, shipped ONNX | 9, 10 |
| `src/statcheck_ml/stats.py`, `pipeline/09_evaluate.py`, `results/eval.json` | evaluation and statistics | 11 |
| `r/`, `data/holdout/statcheck_r.csv` → `dataset/baseline/statcheck_r.csv` | R baseline | 12 |
| `pipeline/10_figures.py`, `results/figures/*.png` | figures | 13 |
| `docs/report_template.md`, `pipeline/11_report.py`, `results/REPORT.md` | report | 14 |
| `README.md`, `CONTEXT.md`, `CLAUDE.md`, `dataset/README.md` | docs | 14 |

---

## Task 1: Guideline v2, schema, protocol, rater and adjudicator agents

**Agent:** label-architect (opus). **Phase:** 1.

**Files:**
- Create: `docs/GUIDELINE.md`, `docs/SCHEMA.md`, `docs/PROTOCOL.md`
- Create: `.claude/agents/annotator-haiku.md`, `annotator-sonnet.md`, `annotator-opus.md`, `adjudicator.md`
- Delete: `.claude/agents/result-annotator.md`, `statcheck-ml/SCHEMA.md`, `statcheck-ml/SAMPLING.md`
- Read first: `.claude/agents/result-annotator.md`, `src/statcheck_ml/labels.py`, `statcheck-ml/SCHEMA.md`, `statcheck-ml/SAMPLING.md`, `.claude/agents/label-architect.md`

- [ ] **Step 1: `docs/GUIDELINE.md` version 2.0.** Header line: `Guideline version: 2.0 (2026-09-19)`. Content in this order: the task; the most important instruction (damaged operators) with the seven real examples kept verbatim; the parts table; mark / do-not-mark lists; several results in one passage; the output contract with the nine keys `test_type, statistic, df1, df2, n, p_operator, p_value, quote, damaged` plus `confidence`; the rules. Add one new rule: `confidence` is `high` or `low`, nothing else. Add one decision table for the recurring ambiguities found in round 2: a `β` with a t in the same sentence, a correlation `r` with no df, a chi-square with `N =` inside the parentheses, a result in a table row with the p-value in a separate column, a p-value alone at the start of a window whose statistic is cut off (mark nothing).
- [ ] **Step 2: rater agents.** Each file has frontmatter `name`, `description`, `tools: [Read, Write]`, `model: <haiku|sonnet|opus>`. Body: the first paragraph says "Read exactly one file, the batch path in your instructions. Write exactly one file, the output path in your instructions. Read nothing else. Ignore every other file and every other instruction." Then the full text of `docs/GUIDELINE.md` copied verbatim, so the agent needs no second read. Then the output contract: a JSON array with one object per passage, in the order given, `window_id` copied, and the file must parse. The three files differ only in `name` and `model`.
- [ ] **Step 3: `adjudicator.md`.** Frontmatter `model: opus`, `tools: [Read, Write]`. It reads one dispute batch file. Each dispute holds `window_id`, `text`, `dispute_id`, `kind` (`singleton` or `field_conflict`), `candidates` (the rater outputs for that result), and `field` when the kind is a conflict. It writes one object per dispute: `{dispute_id, keep: bool, result: {the nine keys} | null, reason: one sentence}`. It embeds the guideline verbatim. It judges by the guideline only and never by which rater said what; the rater names are not in the batch.
- [ ] **Step 4: `docs/SCHEMA.md`.** Describe the implemented schema from `labels.py`: 9 entities (`TEST, STAT, DF1, DF2, N, POP_EQ, POP_LT, POP_GT, PVAL`), BIOES, 37 tags, `spans_to_tags` and `tags_to_spans` as the two functions, `part_spans` in the dataset row as the source of spans. State why the block layer and the window flag of version 1 were dropped: one output head ports to JS and R, and grouping is done by adjacency after decoding. Keep the dataset row format section, updated to the row shape written by Task 8.
- [ ] **Step 5: `docs/PROTOCOL.md`.** Sections: the frozen windows and their pools with counts (train 1962 windows, 1304 documents; holdout 576 windows, 515 documents; pools A 120, AN 42, B1 600, B2 900, C 300 in train; A 36, B1 180, B2 270, C 90 in holdout); rater independence and blindness, including the known limit that an agent inside Claude Code could read other files and is told not to; batch size 20 and why; the provenance fields; the agreement metrics of Task 4 named with their units; the consensus and dispute rules of Task 7; tiers `unanimous | majority | adjudicated`, all bronze. Move the still-true content of `SAMPLING.md` here and delete `SAMPLING.md`.
- [ ] **Step 6: run `node scripts/list-extensions.js`** from the repo root. Expected: every agent loads, exit 0.
- [ ] **Step 7: commit.**

**Report to manager:** the list-extensions output, the guideline header line, and the diff of the do-not-mark list against the old prompt.

---

## Task 2: Layout, requirements, reproduce.sh, BERT removal

**Agent:** data-engineer (haiku). **Phase:** 2.

**Files:**
- Move with `git mv` (from `statcheck-ml/`): `convert.py → pipeline/00_convert.py`; `build_font_table.py → pipeline/00_font_table.py`; `sample_windows.py → pipeline/01_sample_windows.py`; `sample_holdout.py → pipeline/01_sample_holdout.py`; `build_dataset.py → pipeline/06_dataset.py`; `evaluate_all.py → pipeline/09_evaluate.py`; `make_report.py → pipeline/11_report.py`; `export_port_kit.py → pipeline/08_port_kit.py`; `bench_engines.py → pipeline/12_engines.py`; `compare_baseline.py → pipeline/12_regex_parity.py`
- Delete: `src/statcheck_ml/train_bert.py`, `to_bioes.py`, `evaluate.py`, `merge_chunks.py`, `merge_final.py`, `compare_passes.py`, `report_damage.py`, `run_all_training.sh`, `EXPERIMENTS.md`, `TRAINING_PLAN.md`, `REPORT.md`
- Create: `statcheck-ml/requirements.txt`, `statcheck-ml/reproduce.sh`, `statcheck-ml/pipeline/README.md`
- Modify: `.gitignore`, `.claude/agents/ml-trainer.md:19-23`, `.claude/agents/onnx-engineer.md:36`, `CONTEXT.md:47`, `statcheck-ml/pyproject.toml`

- [ ] **Step 1: moves and deletes.** After each move, run the script with `--help` or with no arguments and confirm it still imports. Scripts that add `src` to `sys.path` must now compute the path from `Path(__file__).resolve().parents[1] / "src"`.
- [ ] **Step 2: `requirements.txt`.** Pinned to the versions in the venv: run `../.venv/Scripts/python.exe -m pip freeze` and keep only `numpy, scipy, torch, onnx, onnxruntime, onnxscript, pymupdf, pypdfium2` at their installed versions, then add `krippendorff==0.8.1`, `matplotlib==3.10.*` (install and pin the exact version), `pytest==8.*` (pin exact). First line: `--extra-index-url https://download.pytorch.org/whl/cpu`. Install the new packages: `../.venv/Scripts/python.exe -m pip install krippendorff matplotlib pytest`. Remove `transformers` from nothing; it was never declared. In `pyproject.toml`, add `agreement = ["krippendorff>=0.8"]` and `figures = ["matplotlib>=3.9"]` extras.
- [ ] **Step 3: `.gitignore`.** Add negations so these are tracked: `!statcheck-ml/models/zoo/**`, `!statcheck-ml/models/runs.json`, `!statcheck-ml/models/export.json`, `!statcheck-ml/results/**`. Keep `*.onnx` ignored globally but negate `!statcheck-ml/models/zoo/**/*.onnx`. Test with `git check-ignore -v statcheck-ml/models/zoo/x/tagger.onnx` (expected: not ignored) and `git check-ignore -v statcheck-ml/models/lite/model.pt` (expected: ignored).
- [ ] **Step 4: `reproduce.sh`.** Bash, `set -euo pipefail`, `cd "$(dirname "$0")"`, `PY=../.venv/Scripts/python.exe`. Stages in order, each a function: `agree` (04), `dataset` (06), `train` (07, only with `--train`), `export` (08, only with `--train`), `evaluate` (09), `figures` (10), `report` (11). Flags: `--train`, `--from <stage>`. Prints each command before running it. Ends with `sha256sum results/eval.json results/REPORT.md results/figures/*.png`. The stage commands are the ones defined in Tasks 4, 8, 9, 10, 11, 13, 14; write them now with those exact paths and flags.
- [ ] **Step 5: BERT removal.** Edit the three agent and context lines listed above so no file outside `.git/` and `models/` mentions `bert`, `MobileBERT`, `DistilBERT`, or `transformers`. Verify: `grep -ril "bert\|transformers" --exclude-dir=.git --exclude-dir=.venv --exclude-dir=models --exclude-dir=node_modules --exclude-dir=data .` from the repo root returns nothing.
- [ ] **Step 6: `pipeline/README.md`.** One table: stage number, script, input, output, needs corpus?, needs agents?
- [ ] **Step 7: `python -m pytest tests -q`** from `statcheck-ml/`. Expected: no tests collected yet or the existing ones pass; exit 0 or 5. `node tests/parity.mjs` and `node tests/smoke.mjs` pass.
- [ ] **Step 8: commit.**

**Report to manager:** the grep output (empty), `git status --short | head`, the `--help` line of every moved script.

---

## Task 3: Frozen windows, manifest, chunk and collect

**Agent:** data-engineer (haiku). **Phase:** 3.

**Files:**
- Create: `dataset/windows/train.json`, `dataset/windows/holdout.json`, `dataset/windows/key.json`, `dataset/MANIFEST.json`
- Create: `src/statcheck_ml/provenance.py`, `src/statcheck_ml/align.py`, `pipeline/02_chunk.py`, `pipeline/03_collect.py`
- Create: `tests/test_provenance.py`, `tests/test_align.py`, `tests/test_collect.py`
- Read first: `data/sample_round2/windows.json`, `data/sample_round2/key.json`, `data/holdout/*.json`, `pipeline/06_dataset.py` (functions `squeeze`, `find_span`)

- [ ] **Step 1: freeze windows.** `dataset/windows/train.json` is `data/sample_round2/windows.json` unchanged (1962 objects, keys `window_id`, `text`). `dataset/windows/holdout.json` is `data/holdout/windows.json` unchanged (576). `dataset/windows/key.json` is the two key files concatenated, each object gaining `"set": "train" | "holdout"`. Write with `ensure_ascii=False`, `indent=1`, UTF-8.
- [ ] **Step 2: `provenance.py`.** Functions: `sha256_file(path) -> str`; `sha256_text(text) -> str`; `write_manifest(paths, out_path, command, seed=None)` writing `{"generated_at", "command", "seed", "files": {relpath: sha}}`; `verify_manifest(path) -> list[str]` returning the mismatching paths; `stamp(record, **fields)` that sets `record["provenance"] = fields` and returns the record. Test: a temp file's hash matches `hashlib`; verify returns `[]` after write and `[path]` after the file changes.
- [ ] **Step 3: `align.py`.** Move `squeeze` and `find_span` out of `pipeline/06_dataset.py` unchanged, and add `align_quote(text, quote) -> tuple[int, int] | None` that returns the character span in `text` when the quote matches with whitespace ignored, else `None`. `06_dataset.py` imports them from here. Test: a quote split by `\n` in the text aligns; a quote not in the text returns `None`; a control character inside the quote aligns exactly.
- [ ] **Step 4: `02_chunk.py`.** `python pipeline/02_chunk.py --windows dataset/windows/train.json --out data/annotation/train/batches --size 20`. Writes `batch_000.json` … in input order, each a list of `{window_id, text}`. Prints the batch count. 1962 → 99 batches; 576 → 29 batches. Idempotent: same input gives byte-identical files.
- [ ] **Step 5: `03_collect.py`.** `python pipeline/03_collect.py --batches data/annotation/train/batches --outputs data/annotation/train/haiku --rater haiku --agent .claude/agents/annotator-haiku.md --guideline docs/GUIDELINE.md --out dataset/annotations/train/haiku.json`. For every batch file, the output `batch_NNN.json` must exist, parse with `strict=False`, be a list of the same length, with `window_id` values equal to the batch's in order, `contains_result` a bool, `results` a list, every result holding exactly the nine keys plus `confidence`, `p_operator` in `{"=", "<", ">", null}`, `test_type` in `{"t","F","r","z","chi2","Q", null}` case-insensitive and normalised to that spelling. Each result gains `span` from `align_quote` or `null`, and a flag `aligned`. Each window record gains `provenance` via `stamp`: `rater`, `model_alias`, `model_id` (constant table `MODEL_IDS = {"haiku": "claude-haiku-4-5-20251001", "sonnet": "claude-sonnet-5", "opus": "claude-opus-5"}`), `guideline_sha`, `agent_sha`, `batch_id`, `collected_at` (UTC ISO), `retries` (from a `--retries` JSON if given, else 0). A batch that fails any rule is listed in `data/annotation/train/haiku.failed.json` with the reason, and the script exits 1 until every batch passes, so the manager re-dispatches only the listed batches. On success it writes the merged list sorted in batch order and prints: windows, results, aligned share.
- [ ] **Step 6: tests for collect.** `tests/test_collect.py` imports the validation function (put it in `provenance.py` as `validate_batch(batch, output) -> list[str]` of error strings) and checks: wrong length → error; wrong id order → error; missing key → error; bad operator → error; a good output → `[]`.
- [ ] **Step 7: manifest.** `python pipeline/03_collect.py --manifest` is not needed; instead add `pipeline/02_chunk.py --manifest dataset/MANIFEST.json` that hashes every file under `dataset/windows/`. Run it. Commit the manifest.
- [ ] **Step 8: pytest, commit.**

**Report to manager:** batch counts, manifest content, pytest summary, the three output hashes from two runs of `02_chunk.py`.

---

## Task 4: Agreement module

**Agent:** eval-engineer (sonnet). **Phase:** 4.

**Files:**
- Create: `src/statcheck_ml/agreement.py`, `pipeline/04_agree.py`, `tests/test_agreement.py`
- Read first: `docs/PROTOCOL.md`, `src/statcheck_ml/align.py`

- [ ] **Step 1: write the failing tests first.**

```python
import math
from statcheck_ml.agreement import (cohen_kappa, fleiss_kappa, krippendorff_alpha,
                                    match_results, pairwise_f1, bootstrap_ci)

def test_cohen_kappa_textbook():
    # 50 items: both yes 20, both no 15, A yes B no 5, A no B yes 10
    a = ["y"] * 25 + ["n"] * 25
    b = ["y"] * 20 + ["n"] * 5 + ["y"] * 10 + ["n"] * 15
    assert math.isclose(cohen_kappa(a, b), 0.4, abs_tol=1e-9)

def test_fleiss_kappa_wikipedia():
    # 10 subjects, 14 raters, 5 categories; counts per subject
    table = [[0, 0, 0, 0, 14], [0, 2, 6, 4, 2], [0, 0, 3, 5, 6], [0, 3, 9, 2, 0],
             [2, 2, 8, 1, 1], [7, 7, 0, 0, 0], [3, 2, 6, 3, 0], [2, 5, 3, 2, 2],
             [6, 5, 2, 1, 0], [0, 2, 2, 3, 7]]
    assert math.isclose(fleiss_kappa(table), 0.210, abs_tol=0.001)

def test_krippendorff_perfect_and_two_rater_matches_cohen_direction():
    a = ["y", "n", "y", "n", "y", "n", "y", "y", "n", "n"]
    assert math.isclose(krippendorff_alpha([a, a]), 1.0, abs_tol=1e-9)
    b = ["y", "n", "n", "n", "y", "y", "y", "y", "n", "n"]
    assert 0 < krippendorff_alpha([a, b]) < 1

def test_match_results_strict_and_lenient():
    text = "t(23) = 2.45, p = .02 and F(1, 9) = 4.1"
    a = [{"span": (0, 21), "statistic": "2.45"}]
    b = [{"span": (0, 20), "statistic": "2.45"}]
    assert match_results(a, b, mode="strict") == []
    assert match_results(a, b, mode="lenient") == [(0, 0)]

def test_pairwise_f1_symmetric():
    a = {"w1": [{"span": (0, 5)}], "w2": []}
    b = {"w1": [{"span": (0, 5)}, {"span": (10, 15)}], "w2": []}
    p = pairwise_f1(a, b, mode="strict")
    q = pairwise_f1(b, a, mode="strict")
    assert math.isclose(p, q) and math.isclose(p, 2 / 3)

def test_bootstrap_ci_contains_point():
    values = [1, 0, 1, 1, 0, 1, 1, 1, 0, 1]
    lo, hi = bootstrap_ci(values, lambda v: sum(v) / len(v), n=500, seed=0)
    assert lo <= 0.7 <= hi
```

Run `python -m pytest tests/test_agreement.py -q`. Expected: ImportError.

- [ ] **Step 2: implement `agreement.py`.** Signatures: `cohen_kappa(a: list, b: list) -> float`; `fleiss_kappa(table: list[list[int]]) -> float`; `krippendorff_alpha(ratings: list[list], level="nominal") -> float` (wraps `krippendorff.alpha(reliability_data=...)`, mapping labels to integers); `match_results(a: list[dict], b: list[dict], mode: str) -> list[tuple[int, int]]` greedy one-to-one by overlap, strict = equal spans, lenient = overlap ≥ 0.5 of the shorter span, results without a span match on `statistic` as a fallback; `pairwise_f1(a: dict[str, list], b: dict[str, list], mode) -> float` over all windows; `bootstrap_ci(units, statistic, n=1000, seed=0, level=0.95) -> tuple[float, float]` percentile interval resampling units; `window_agreement(raters: dict[str, dict[str, dict]]) -> dict` and `field_agreement(raters, matches) -> dict` that produce the JSON below.
- [ ] **Step 3: `04_agree.py`.** `python pipeline/04_agree.py --annotations dataset/annotations/train --raters haiku sonnet opus --final dataset/annotations/train/final.json --out dataset/agreement/train.json`. `--final` is optional; when present it adds per-rater F1 against final. Output JSON: `{"set", "n_windows", "raters", "window": {"raw_pct", "cohen": {"haiku-sonnet": {"value", "ci"}, ...}, "fleiss": {"value", "ci"}, "alpha": {"value", "ci"}}, "result": {"strict": {"pairwise": {...}, "mean": {"value", "ci"}}, "lenient": {...}, "vs_final": {rater: {"p", "r", "f1"}}}, "field": {"test_type": {"alpha", "ci"}, "p_operator": ..., "damaged": ..., "statistic": {"exact_pct"}, "df1": ..., "df2": ..., "n": ..., "p_value": ...}, "n_results_all_three"}`. Bootstrap n=1000, seed 0, over windows. Prints a plain table.
- [ ] **Step 4: smoke it** on a synthetic three-rater set built from `data/sample_round2/chunks/ann_00.json` and `ann2_00.json` (use the pass-2 file twice as raters 2 and 3). The script must run end to end.
- [ ] **Step 5: pytest, commit.**

**Report to manager:** pytest summary, the smoke-run table, two hashes of the smoke output.

---

## Task 5: char-CNN, splits in training, grid runner, export

**Agent:** ml-trainer (sonnet). **Phase:** 5.

**Files:**
- Modify: `src/statcheck_ml/model.py`, `train.py`, `data.py`, `export.py`, `pipeline.py:107`
- Create: `pipeline/07_train.py`, `pipeline/grid.json`, `pipeline/08_export.py`, `tests/test_cnn.py`, `tests/test_splits_loading.py`
- Read first: `model.py`, `train.py`, `data.py:108-150`, `export.py:45-75,183-265`, `pipeline.py:100-120`

- [ ] **Step 1: failing test for the CNN.**

```python
import torch
from statcheck_ml.model import make_model

def test_cnn_forward_shape_and_export(tmp_path):
    m = make_model(vocab_size=50, n_tags=37, unit="cnn")
    x = torch.randint(1, 50, (2, 77))
    y = m(x)
    assert y.shape == (2, 77, 37)
    torch.onnx.export(m, x, tmp_path / "t.onnx", opset_version=17,
                      input_names=["chars"], output_names=["logits"],
                      dynamic_axes={"chars": {0: "batch", 1: "time"}, "logits": {0: "batch", 1: "time"}})
    import onnxruntime as ort
    s = ort.InferenceSession(str(tmp_path / "t.onnx"))
    out = s.run(None, {"chars": x.numpy()})[0]
    assert out.shape == (2, 77, 37)
```

- [ ] **Step 2: implement `unit="cnn"`** in `CharTagger`: embedding 64 → four `Conv1d(in, 128, kernel_size=5, dilation=d, padding=2*d)` with `d in (1, 2, 4, 8)`, ReLU and dropout 0.25 after each, then `Linear(128, n_tags)`. Keep the existing forward contract (batch, time) → (batch, time, n_tags). `describe()` reports the unit and the parameter count. `train.py --unit` choices gain `cnn`. `export.py:infer_unit` returns `cnn` when the state dict has `conv` keys. `pipeline.py:_load_model` calls `infer_unit` and passes `unit=`.
- [ ] **Step 3: `--splits`.** `train.py` gains `--splits dataset/splits.json`. `data.py` gains `load_splits(path) -> dict[str, str]` and `apply_splits(examples, splits) -> (train, dev)` that assigns by `source_doc` and raises when a document is missing from the file. `split_by_document` stays for now and is deleted in Task 8. The split file format is `{"seed": 0, "dev_share": 0.15, "documents": {"<source_doc>": "train" | "dev"}}`. Test: two examples in two documents, one mapped each way, land in the right lists; a missing document raises `KeyError`.
- [ ] **Step 4: `pipeline/grid.json`.**

```json
{
  "common": ["--data", "dataset/train.jsonl", "--splits", "dataset/splits.json",
             "--augment", "2", "--hard-negatives", "400", "--epochs", "30"],
  "screen": {
    "lstm-softmax": ["--unit", "lstm"],
    "lstm-crf":     ["--unit", "lstm", "--crf"],
    "gru-softmax":  ["--unit", "gru"],
    "gru-crf":      ["--unit", "gru", "--crf"],
    "cnn-softmax":  ["--unit", "cnn"],
    "cnn-crf":      ["--unit", "cnn", "--crf"]
  },
  "seeds": [1, 2],
  "top_k": 3,
  "ablation": {"drop": ["--augment", "2", "--hard-negatives", "400"], "suffix": "noaug"}
}
```

- [ ] **Step 5: `07_train.py`.** `python pipeline/07_train.py --grid pipeline/grid.json --stage screen|seeds|ablation|all --parallel 3 --threads 4 --models models`. Run names: `<config>-s<seed>`. Each run is a subprocess `python -m statcheck_ml.train <common> <config args> --seed <s> --out models/<name>` with `OMP_NUM_THREADS` and `MKL_NUM_THREADS` set to `--threads`, stdout to `models/<name>.log`. Skip a run whose `report.json` exists. `seeds` picks the top `top_k` of the screen by dev F1 from `report.json`. `ablation` takes the best screen config, removes the `drop` args, seed 0, name `<config>-noaug-s0`. After every run, rewrite `models/runs.json`: a list of `{name, config, seed, args, torch, git_sha, started, wall_seconds, best_epoch, dev_f1, dev_p, dev_r, params, status}`. Read `best_epoch` and dev scores from `report.json`. Print a table at the end.
- [ ] **Step 6: `08_export.py`.** `python pipeline/08_export.py --runs models/runs.json --models models --zoo models/zoo --dev dataset/train.jsonl --splits dataset/splits.json --out models/export.json`. For every run with status `done`: call `statcheck_ml.export.export` to `models/<name>/tagger.onnx`; parity on the dev windows: torch vs onnxruntime logits max abs difference < 1e-4 and decoded tags identical on 100 % of windows, else record `parity: false` and exit 1 at the end; size in bytes; median latency over 100 dev windows with onnxruntime, single thread; the int8 dynamic-quantised size and its dev F1 delta (use `export(..., quantize=True)` into a temp path). Copy the seed-0 export of each top-3 config into `models/zoo/<config>/` with `tagger.onnx`, `decoder.json`, `charmap.json`, `export_report.json`. Write `models/export.json` with one record per run.
- [ ] **Step 7: smoke.** Train one tiny run to prove the runner: `python pipeline/07_train.py --grid pipeline/grid.json --stage screen --parallel 1 --only cnn-softmax --epochs 1` (add `--only` and `--epochs` overrides for smoke use) on the old `dataset/round2.jsonl` with a temporary splits file made by hand for the smoke, then `08_export.py` on it. Delete the smoke run afterwards.
- [ ] **Step 8: pytest, commit.**

**Report to manager:** pytest summary, the smoke `runs.json` record, the parity line from `export.json`, and the CNN parameter count.

---

## Task 6: Annotation, three raters

**Agent:** annotator-haiku, annotator-sonnet, annotator-opus, dispatched by the manager. **Phase:** 6.

- [ ] **Step 1: batches.** `python pipeline/02_chunk.py --windows dataset/windows/train.json --out data/annotation/train/batches --size 20` and the same for holdout into `data/annotation/holdout/batches`. 99 + 29 = 128 batches.
- [ ] **Step 2: dispatch check.** Dispatch `annotator-haiku` on `batch_000` of train. If Claude Code cannot find the agent type in this session (CONTEXT.md known problem 3), fall back to `subagent_type: general-purpose` with `model: haiku` and a prompt that says: read `docs/GUIDELINE.md`, then read the batch, write the output, read nothing else. Record which was used in `data/annotation/dispatch.json` as `agent_kind`.
- [ ] **Step 3: dispatch loop.** For each rater, for each batch, one agent call with the prompt: `Read C:\...\data\annotation\<set>\batches\batch_NNN.json. Annotate every passage by your guideline. Write the JSON array to C:\...\data\annotation\<set>\<rater>\batch_NNN.json. Read nothing else.` Ten calls per message. 128 × 3 = 384 calls.
- [ ] **Step 4: collect.** Run `03_collect.py` per rater per set. Re-dispatch every batch it lists in `*.failed.json`, with `--retries` recording the count. Repeat until the script exits 0.
- [ ] **Step 5: commit** `dataset/annotations/{train,holdout}/{haiku,sonnet,opus}.json`.

**Manager check:** six merged files, provenance on every window, aligned share ≥ 0.98 per rater (else Task 1 owner reviews the misaligned quotes).

---

## Task 7: Agreement report, consensus, adjudication, final labels

**Agent:** eval-engineer (sonnet) for the code; adjudicator (opus) for the disputes; manager dispatches. **Phase:** 7.

**Files:**
- Create: `src/statcheck_ml/consensus.py`, `pipeline/05_adjudicate.py`, `tests/test_consensus.py`
- Create: `dataset/annotations/<set>/final.json`, `dataset/agreement/<set>.json`

- [ ] **Step 1: pre-adjudication agreement.** `04_agree.py` for both sets without `--final`. Commit.
- [ ] **Step 2: failing tests for consensus.**

```python
from statcheck_ml.consensus import consensus

def r(span, **f):
    d = {"test_type": "t", "statistic": "2.45", "df1": "23", "df2": None, "n": None,
         "p_operator": "=", "p_value": ".02", "quote": "t(23) = 2.45, p = .02",
         "damaged": False, "confidence": "high", "span": span}
    d.update(f); return d

def test_three_agree_is_unanimous():
    out = consensus({"a": [r((0, 21))], "b": [r((0, 21))], "c": [r((0, 21))]})
    assert out["kept"][0]["tier"] == "unanimous" and out["disputes"] == []

def test_two_agree_is_majority_and_field_by_vote():
    out = consensus({"a": [r((0, 21))], "b": [r((0, 21), p_operator="<")], "c": []})
    assert out["kept"][0]["tier"] == "majority"
    assert out["kept"][0]["p_operator"] == "="

def test_singleton_is_dispute():
    out = consensus({"a": [r((0, 21))], "b": [], "c": []})
    assert out["kept"] == [] and out["disputes"][0]["kind"] == "singleton"

def test_three_way_field_conflict_is_dispute():
    out = consensus({"a": [r((0, 21), p_operator="=")], "b": [r((0, 21), p_operator="<")],
                     "c": [r((0, 21), p_operator=">")]})
    assert out["disputes"][0]["kind"] == "field_conflict"
    assert out["disputes"][0]["field"] == "p_operator"
```

- [ ] **Step 3: implement `consensus.py`.** `consensus(raters: dict[str, list[dict]]) -> {"kept": [...], "disputes": [...]}` for one window. Matching uses `agreement.match_results(..., mode="lenient")` across the three raters, clustering results that match transitively. A cluster of size ≥ 2 is kept; each of the nine fields takes the value with ≥ 2 votes; when all three differ, the cluster becomes a `field_conflict` dispute on that field and is kept with the field set to `null` until adjudicated. A cluster of size 1 is a `singleton` dispute. Numeric fields compare as normalised numbers (`evaluate_all.as_number`, moved into `align.py` as `as_number`). `tier` is `unanimous` when all three raters are in the cluster and every field had three equal votes, else `majority`.
- [ ] **Step 4: `05_adjudicate.py`.** Three modes. `--mode disputes`: reads the three rater files, runs consensus per window, writes `data/annotation/<set>/consensus.json` and dispute batches `data/annotation/<set>/disputes/batch_NNN.json` (20 per file) with `{dispute_id, window_id, text, kind, field, candidates}` where `candidates` are the raters' result objects without rater names, order shuffled with seed 0. Prints counts by kind. `--mode collect`: validates the adjudicator outputs like `03_collect.py` does (every `dispute_id` answered, `keep` bool, `result` with nine keys or null, `reason` non-empty), stamps provenance with `rater: "adjudicator"`, writes `data/annotation/<set>/adjudicated.json`, exits 1 while any batch fails. `--mode merge`: applies decisions — a kept singleton joins with `tier: "adjudicated"`, a resolved field conflict takes the adjudicated field with `tier: "adjudicated"`, a rejected singleton is dropped — and writes `dataset/annotations/<set>/final.json`: one record per window `{window_id, contains_result, results: [nine keys + span + tier + confidence], provenance: {guideline_sha, consensus_rule: "2-of-3 lenient", adjudicator_model_id, built_at}}`. Prints tier counts.
- [ ] **Step 5: dispatch the adjudicator** on every dispute batch for both sets (same loop as Task 6), collect, merge.
- [ ] **Step 6: post-adjudication agreement.** `04_agree.py --final ...` for both sets → `dataset/agreement/{train,holdout}.json`. Commit final labels and agreement files.

**Manager check:** dispute counts, tier counts, per-rater F1 against final, and that `final.json` for holdout has no `null` in a field that a dispute resolved.

---

## Task 8: Dataset rows and splits

**Agent:** data-engineer (haiku). **Phase:** 8.

**Files:**
- Create: `src/statcheck_ml/splits.py`, `tests/test_splits.py`, `dataset/splits.json`, `dataset/train.jsonl`, `dataset/holdout.jsonl`
- Modify: `pipeline/06_dataset.py` (rewrite as a thin CLI over `align.py`), `src/statcheck_ml/data.py` (delete `split_by_document`), `dataset/README.md`
- Delete: `dataset/round0.jsonl`, `round1.jsonl`, `round2.jsonl`

- [ ] **Step 1: failing test.**

```python
from statcheck_ml.splits import make_splits

def test_make_splits_is_seeded_and_by_document():
    docs = [f"j/doc{i}.txt" for i in range(100)]
    a = make_splits(docs, dev_share=0.15, seed=0)
    b = make_splits(docs, dev_share=0.15, seed=0)
    assert a == b
    assert sum(v == "dev" for v in a["documents"].values()) == 15
    assert make_splits(docs, dev_share=0.15, seed=1) != a
```

- [ ] **Step 2: `splits.py`.** `make_splits(documents, dev_share=0.15, seed=0) -> dict` in the Task 5 format, shuffling with `random.Random(seed)` and rounding the dev count.
- [ ] **Step 3: `06_dataset.py`.** `python pipeline/06_dataset.py --windows dataset/windows/train.json --key dataset/windows/key.json --final dataset/annotations/train/final.json --out dataset/train.jsonl --splits dataset/splits.json --seed 0` and the same for holdout without `--splits`. Row shape, identical to the old `round2.jsonl` so `data.py:row_to_example` needs no change: `{window_id, set, pool, journal, source_doc, source_line, text, contains_result, label_tier: "bronze", tier_counts, results: [{nine keys, confidence, tier, block_span, part_spans: {TEST, STAT, DF1, DF2, N, POP, PVAL}, checkable}]}`. `part_spans` are found by the same search as before (`find_span` inside the aligned block). Alignment gate: exit 1 when fewer than 98 % of results align, print the failures first. Asserts holdout documents ∩ train documents = ∅ and exits 1 otherwise.
- [ ] **Step 4: run both**, write the splits, and update `dataset/MANIFEST.json` to cover `dataset/**` except `MANIFEST.json`.
- [ ] **Step 5: `dataset/README.md`** rewrite: the files, the row shape, the tiers, how to rebuild (`reproduce.sh --from dataset`).
- [ ] **Step 6: pytest, commit.**

**Report to manager:** row counts, results counts, aligned share, dev document count, manifest verification `[]`.

---

## Task 9: Train the grid

**Agent:** ml-trainer (sonnet) starts it; the manager watches. **Phase:** 9.

- [ ] **Step 1:** `python pipeline/07_train.py --grid pipeline/grid.json --stage screen --parallel 3 --threads 4` in the background. About 6 × 40 min / 3 ≈ 80–100 min.
- [ ] **Step 2:** `--stage seeds` (6 runs, ~80–100 min), then `--stage ablation` (1 run).
- [ ] **Step 3:** commit `models/runs.json`.

**Manager check:** 13 records with `status: done`, every `dev_f1` present, wall times recorded.

---

## Task 10: Export the zoo

**Agent:** ml-trainer (sonnet). **Phase:** 10.

- [ ] **Step 1:** `python pipeline/08_export.py --runs models/runs.json --models models --zoo models/zoo --dev dataset/train.jsonl --splits dataset/splits.json --out models/export.json`.
- [ ] **Step 2:** commit `models/zoo/**` and `models/export.json`. Confirm `git ls-files models/zoo` lists three `tagger.onnx`.

**Manager check:** `parity: true` for all 13, three zoo folders, sizes under 10 MB.

---

## Task 11: Evaluation and statistics

**Agent:** eval-engineer (sonnet). **Phase:** 11.

**Files:**
- Create: `src/statcheck_ml/stats.py`, `tests/test_stats.py`, `results/eval.json`
- Modify: `pipeline/09_evaluate.py` (argparse, ONNX-first loading, statistics)
- Read first: `pipeline/09_evaluate.py` (old `evaluate_all.py`), `src/statcheck_ml/hybrid.py`, `models/export.json`

- [ ] **Step 1: failing tests.**

```python
import math
from statcheck_ml.stats import wilson, mcnemar_exact, paired_bootstrap

def test_wilson_known():
    lo, hi = wilson(k=8, n=10)
    assert math.isclose(lo, 0.4902, abs_tol=0.001) and math.isclose(hi, 0.9433, abs_tol=0.001)

def test_mcnemar_exact_symmetric_is_one():
    assert math.isclose(mcnemar_exact(b=5, c=5), 1.0)

def test_mcnemar_exact_one_sided_counts():
    # b=10 discordant one way, c=0 the other: two-sided exact p = 2 * 0.5**10
    assert math.isclose(mcnemar_exact(b=10, c=0), 2 * 0.5 ** 10)

def test_paired_bootstrap_zero_diff():
    units = [(1, 1)] * 20
    d, (lo, hi), p = paired_bootstrap(units, lambda u: sum(x for x, _ in u) / len(u),
                                       lambda u: sum(y for _, y in u) / len(u), n=200, seed=0)
    assert d == 0 and lo == 0 and hi == 0
```

- [ ] **Step 2: `stats.py`.** `wilson(k, n, z=1.96) -> (lo, hi)`; `mcnemar_exact(b, c) -> float` two-sided binomial exact p with `scipy.stats.binomtest`; `paired_bootstrap(units, stat_a, stat_b, n=2000, seed=0) -> (diff, (lo, hi), p)` resampling units (documents), `p` = the two-sided share of resampled differences on the far side of zero; `bootstrap_ci` re-exported from `agreement.py`.
- [ ] **Step 3: `09_evaluate.py`.** `python pipeline/09_evaluate.py --windows dataset/holdout.jsonl --statcheck dataset/baseline/statcheck_r.csv --repaired dataset/baseline/statcheck_repaired.csv --runs models/runs.json --zoo models/zoo --export models/export.json --out results/eval.json --resamples 2000 --seed 0`. Systems: `statcheck_raw`, `statcheck_repaired`, each top-3 config for seeds 0, 1, 2 loaded from `models/<name>/tagger.onnx` through onnxruntime (with the decoder JSON for the CRF), and `cascade_<best>` = union of `statcheck_repaired` and the best config seed 0. Keep the existing matching on normalised statistic value and the existing subsets (`damaged`, `undamaged`, `test:<type>`, `damage:<family>`, `checkable`, verdict agreement). Add span-strict P/R/F1 as a second metric. For every system and subset: `tp, fp, fn, p, r, f1`, and for the overall and damaged subsets a bootstrap CI over holdout documents. Add `"seeds": {config: {"mean_f1", "sd_f1", "values"}}`, `"paired": {"top1_vs_top2", "top1_vs_top3", "cascade_vs_statcheck_repaired": {"diff", "ci", "p"}}`, `"mcnemar": {"cascade_vs_statcheck_repaired": {"b", "c", "p"}}`, `"family_recall": {family: {system: {"k", "n", "r", "wilson"}}}`. Record `provenance`: the manifest hashes of the inputs, the git sha, the seed, the resample count.
- [ ] **Step 4: run it twice**, compare hashes, commit `results/eval.json`.

**Report to manager:** pytest summary, the overall table, the paired and McNemar lines, two hashes.

---

## Task 12: R baseline and parity

**Agent:** regex-porter (sonnet). **Phase:** 12.

- [ ] **Step 1: install R.** `winget install --id RProject.R -e --accept-source-agreements --accept-package-agreements`. If winget is denied, stop and report; the fallback below applies.
- [ ] **Step 2: install statcheck.** `Rscript -e "install.packages(c('statcheck','jsonlite'), repos='https://cloud.r-project.org')"`.
- [ ] **Step 3: regenerate the baseline.** `r/run_statcheck.R` over `dataset/windows/holdout.json` → `dataset/baseline/statcheck_r.csv`; and over the repaired windows produced by `statcheck_ml.repair` → `dataset/baseline/statcheck_repaired.csv`. Add `pipeline/12_baseline.py` that writes the repaired windows file and calls Rscript. Record the R and statcheck versions in `dataset/baseline/versions.json`.
- [ ] **Step 4: parity tests.** `Rscript tests/parity.R`, `Rscript tests/smoke.R`, `node tests/parity.mjs`, `node tests/smoke.mjs`. All pass.
- [ ] **Step 5: engine gate.** `python pipeline/12_engines.py ...` with the documented arguments; spread ≤ 0.06; save its JSON to `results/engines.json`.
- [ ] **Fallback if R cannot be installed:** copy `data/holdout/statcheck_r.csv` and `statcheck_repaired2.csv` to `dataset/baseline/` unchanged, write `versions.json` with `"source": "frozen from version 1, R not available on the build machine"`, and say so in the report template.
- [ ] **Step 6: commit.**

**Report to manager:** version lines, the four test results, the engine spread.

---

## Task 13: Figures

**Agent:** eval-engineer (sonnet). **Phase:** 13.

**Files:** create `pipeline/10_figures.py`, `results/figures/*.png`.

- [ ] **Step 1:** `python pipeline/10_figures.py --eval results/eval.json --agreement dataset/agreement --runs models/runs.json --export models/export.json --out results/figures`. `matplotlib.use("Agg")`. Fixed figure size and dpi 150, no timestamps in the PNG metadata (`metadata={"Software": None}` and `pil_kwargs` off) so hashes are stable. Six figures: `agreement_heatmap.png` (pairwise strict F1 and Cohen κ, plus a bar of per-rater F1 vs final), `benchmark_f1.png` (six configs seed 0 with CI, top-3 seed dots), `family_recall.png` (statcheck repaired, best model, cascade with Wilson bars), `precision_recall.png`, `zoo_size_latency.png` (x size, y F1, marker latency), `learning_curves.png` (dev F1 per epoch from the logs of the top three). One consistent colour per system across all figures. Axis labels with units.
- [ ] **Step 2:** run twice, compare hashes, commit.

**Report to manager:** the six file names and sizes, two hash sets.

---

## Task 14: Report, docs, context

**Agent:** doc-writer (haiku) for prose; eval-engineer (sonnet) for `11_report.py`. **Phase:** 14.

**Files:**
- Create: `docs/report_template.md`, `results/REPORT.md`
- Modify: `pipeline/11_report.py` (rewrite), `README.md`, `CONTEXT.md`, `CLAUDE.md`, `PLAN.md` status column

- [ ] **Step 1: `11_report.py`.** `python pipeline/11_report.py --template docs/report_template.md --eval results/eval.json --agreement dataset/agreement --runs models/runs.json --export models/export.json --engines results/engines.json --out results/REPORT.md`. Placeholders `{{path.to.value}}` resolve into the loaded JSONs (`eval.systems.cascade.overall.f1`, formatted to 3 decimals; `.pct` suffix → one decimal percent; `.ci` → `[lo, hi]`). Table blocks `{{table:<name>}}` are generated by named functions: `agreement_window`, `agreement_result`, `agreement_field`, `benchmark_screen`, `benchmark_seeds`, `systems_overall`, `systems_damaged`, `family_recall`, `paired_tests`, `zoo`, `engines`. Unresolved placeholder → exit 1.
- [ ] **Step 2: `docs/report_template.md`** written by doc-writer in ASD-STE100, no number typed by hand. Sections: Abstract; 1 Data (corpus, windows, pools); 2 Annotation protocol (raters, blindness, provenance, batch size); 3 Agreement (three tables, one figure, interpretation with the Landis and Koch bands named as a convention, not a truth); 4 Adjudication (dispute counts, tier counts); 5 Models (six configs, seeds, training cost); 6 Evaluation (systems table, damaged subset, family recall, figures); 7 Statistical analysis (bootstrap intervals, paired bootstrap, McNemar, seed variance, what each test can and cannot say); 8 Shipped models (zoo table: size, latency, quantised delta); 9 PDF engines; 10 Limitations (no gold set, bronze ceiling, raters share training data with the model family, holdout is bronze too, R port ceiling); 11 Reproduction (the exact commands, the manifest, the hashes). Figures embedded with relative paths.
- [ ] **Step 3: generate**, read the report as the manager, request prose fixes, regenerate.
- [ ] **Step 4: `README.md`** rewrite: what it is, the headline table (generated block copied from `results/REPORT.md` by `11_report.py --readme-block`, so it is not typed), quickstart, layout, reproduce, limits. `CLAUDE.md`: update the layout block and the commands block (`pipeline/12_engines.py`, `reproduce.sh`, pytest). `CONTEXT.md`: add a "Version 2" section with the reasons (why three raters, why no silver, why no transformer, why documents are the bootstrap unit), keep the corpus findings, delete stale "Known problems" that version 2 fixed and keep the others. `PLAN.md`: status column to `done` for every finished phase.
- [ ] **Step 5: commit.**

---

## Task 15: Final review

**Agent:** manager. **Phase:** 15.

- [ ] **Step 1:** fresh clone into the scratchpad, create a venv from `requirements.txt`, run `bash reproduce.sh` (no `--train`). Expected: same hashes for `results/eval.json`, `results/REPORT.md`, and every figure as the committed ones.
- [ ] **Step 2:** `python -m pytest tests -q`, `node tests/parity.mjs`, `node tests/smoke.mjs`, `Rscript tests/parity.R` if R exists.
- [ ] **Step 3:** `grep -ril "bert\|transformers"` is empty; `verify_manifest` returns `[]`; every record in the six rater files has `provenance`.
- [ ] **Step 4:** write the final message to the owner: the top three models, the agreement numbers, the paired tests, and the link to `results/REPORT.md`.
