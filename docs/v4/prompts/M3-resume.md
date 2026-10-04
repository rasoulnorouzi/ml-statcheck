# Milestone M3, resumed — worker prompts, verbatim

Written by the manager before the resumed milestone ran (run `wf_1ad266e8-2f6`,
2026-10-04). The harness prepends one line to each prompt: the task id and title, the
branch, "do not commit", and the request for real commands and real output as
evidence. Nothing else is added.

The first run stopped inside M3. M3.1 and the dev split of M3.2 were accepted there.
The session regenerated the passage inventory with the fixed script. The manager
verified that inventory in this run (the ledger holds the numbers) and found four
faults that a small fix task repairs before the sampler runs: the record field
`text_sha256` holds the paper hash and not the piece hash; `dataset/MANIFEST.json`
is stale; `inventory_summary.json` holds `run_seconds`, so two runs differ in one key;
dead code and what-docstrings remain. The original M3.3 is split in two, because the
M3.2 worker on haiku failed twice on one large prompt: M3.3 draws the sample, M3.4
writes the batches.

Manager's decisions that the prompts carry:

- The caps stay at 900 / 400 / 200 / 300. The inventory holds 585 passages with an
  in-scope coder row and 80 hard negatives, so the gold and hard caps are not reached:
  every gold passage and every hard negative is rated. The draw is 585 + 80 + 200 +
  300 = 1165 passages, 117 batches of ten, 351 rater runs.
- `text_sha256` of an inventory record becomes the sha256 of the piece text, and the
  paper hash moves to `paper_text_sha256`. The sampler checks the piece hash.
- A committed summary file holds no timing. The script prints the seconds; it does not
  write them.
- The piece builder becomes a function `paper_pieces(text, max_chars)` that needs no
  gold record and no model, so the sampler rebuilds a drawn passage cheaply and the
  tests cover the builder alone.
- The rater output directories under `data/annotation/passages/` are created now, so
  a rater never has to create a directory.

## M3.2b — Fix the passage inventory: the piece hash, the manifest, the summary without a timing, dead code (data-engineer, haiku)

```
Repository: /home/rass/Desktop/projects/ml-statcheck, branch v4-gold. The Python project is /home/rass/Desktop/projects/ml-statcheck/statcheck-ml; run every script from that directory with /home/rass/Desktop/projects/ml-statcheck/.venv/bin/python. The corpus text is /home/rass/Desktop/projects/ml-statcheck/statcheck-ml/data/clean/<doc> (gitignored). Use absolute paths in every command, because the working directory resets between shell calls.

Rules:
- Read /home/rass/Desktop/projects/ml-statcheck/CLAUDE.md first and keep its rules. Prose you write follows ASD-STE100: active voice, articles kept, short sentences. No machine-written texture: comments say why, not what; no docstring that restates a signature.
- Do not run rm, mv, cp, or any git command that changes state. Delete, move, or copy a file from inside Python (os.remove, shutil). Never delete a file that git does not track. The manager commits.
- Never train on, tune on, or sample from a gold test paper. The script reads dataset/gold/gold.jsonl, which holds every gold paper; it selects the gold_train papers through dataset/gold/split.json, drops the dev papers of dataset/gold/dev.json, and calls statcheck_ml.gold.assert_not_gold_test on the papers it keeps before it reads any text. Keep that. Do not run pipeline/14_gold_eval.py.
- Do not edit a file that this task does not name as yours.
- Report every command you ran with its real output, every file you changed, and every number you measured. Do not report a number you did not measure.

Background. pipeline/17_passage_inventory.py (M3.2) builds dataset/passages/inventory.jsonl, one record per passage piece of the 204 gold-train papers outside dev.json, and dataset/passages/inventory_summary.json. The manager verified the current output (inventory.jsonl sha256 74d77061d25198056b1a96ce08c0e2fe13d2fa821745430795248b15aa7e73f2): 204 papers, 12592 pieces; units split 767, largest unit 10021; passages by flag has_gold 621, has_gold_in_scope 585, gold_damaged 141, cascade_fires 866; hard negatives 80; damaged_no_gold 1531; coder rows total_in_gold 1812, in_scope 1649, located_in_passage 1725, span_check_failures 0, on_blanked_line 7, not_inside_any_unit 80, unlocated 32. Those numbers are right and must not change. Four faults remain:
(1) The record field text_sha256 holds the sha256 of the whole repaired paper text: every piece of a paper carries the same value, equal to the paper's text_sha256 in gold.jsonl. The sampler of M3.3 must check the text of a drawn piece against the inventory, so it needs the hash of the piece text.
(2) dataset/MANIFEST.json holds stale hashes for passages/inventory.jsonl and passages/inventory_summary.json; verify_manifest('dataset/MANIFEST.json') returns those two paths.
(3) inventory_summary.json holds "run_seconds". Two runs of the script differed in that one key (72.35 and 101.63), so the committed file is not byte-reproducible, and reproduce.sh prints its hash.
(4) Dead code and what-docstrings: numbers() is defined and never called; the job tuple carries split_data and dev_set, which build_passages never reads; the docstrings of PassageRecord, numbers, check_span and build_passages restate what the code shows.
The piece building lives inside build_passages, which also reads the file and runs Pipeline().run_text. The M3.3 sampler must rebuild the pieces of a drawn paper without the gold record and without the cascade. Read the script, tests/test_passage_inventory.py (it loads the script by path with importlib), src/statcheck_ml/prefilter.py (Window, units: a unit's text is "\n".join of the reference-blanked lines, so a blanked line is empty and keeps its newline), src/statcheck_ml/provenance.py (write_manifest, verify_manifest) and the "Gold" section of dataset/README.md before you start.

YOUR TASK:
1. Move the piece building into a function paper_pieces(text, max_chars) in pipeline/17_passage_inventory.py. It takes the repaired text (after normalize() and repair_validated()) and returns one list per unit of Prefilter().units(text), in unit order, flattened into one list of dicts: {"unit": the index of the unit, "start_line", "end_line", "piece" (1-based), "pieces", "char_start", "text": the piece text, "\n".join of its lines of the unit text}. The splitting rule stays exactly as it is: walk the unit's lines; start a new piece when adding the next line would exceed max_chars; a single line longer than max_chars is a piece on its own. build_passages calls paper_pieces and keeps every other computation unchanged.
2. Each record: text_sha256 becomes the sha256 of the piece text (utf-8). The paper hash moves to a new field paper_text_sha256 with the value text_sha256 held before. Keep every other field and every other value, on_blanked_line in coder_results included.
3. The summary: remove run_seconds from the file. Print the seconds to stderr. The summary file must be byte-identical between two runs.
4. Remove numbers(). Remove split_data and dev_set from the job tuple. Replace the four what-docstrings with nothing, or with a one-line comment that says why (for example why a piece ends at a line end: a result that a line break splits stays in one piece).
5. Tests in tests/test_passage_inventory.py: add one test that paper_pieces covers every unit of a synthetic paper with no gap and no overlap, splits a unit over max_chars at a line end, and gives text whose sha256 equals the record's text_sha256; update any test that reads the old field. Keep the other tests.
6. Rerun the stage: python pipeline/17_passage_inventory.py data/clean/ --seed 0 (the defaults write dataset/passages/). Run it a second time with --out and --summary into the scratchpad and compare the sha256 of both files with the committed ones; the pairs must be equal. Regenerate dataset/MANIFEST.json with python pipeline/02_chunk.py --manifest-root dataset and confirm that verify_manifest('dataset/MANIFEST.json') returns [].
7. dataset/README.md: replace the paragraph under "Gold" that starts with "`gold/dev.json` splits" by one paragraph of at most five sentences, each under 25 words, in ASD-STE100, that says: dev.json holds the 36 gold-train dev papers (15 % by paper, seed 0, stratified by has_results); they never enter a rater sample or a training set, and they gate the noise families; the inventory lists every passage piece of the other 204 gold-train papers with its coordinates, the piece hash, the coder rows and the cascade counts, and no text; it is the sampling frame of the passage annotation, not a training file; the sample holds the texts.
8. Run the whole suite.

Report: the printed summary verbatim; the sha256 of inventory.jsonl and inventory_summary.json from both runs; the verify_manifest output; the pytest counts; git status --short. Files you own: pipeline/17_passage_inventory.py, tests/test_passage_inventory.py, dataset/passages/inventory.jsonl, dataset/passages/inventory_summary.json, dataset/MANIFEST.json, dataset/README.md (one paragraph). Do not touch src/, the spec files, gold.jsonl, split.json, dev.json, reproduce.sh or any result file.
```

## M3.3 — The passage sampler (data-engineer, haiku)

```
Repository: /home/rass/Desktop/projects/ml-statcheck, branch v4-gold. The Python project is /home/rass/Desktop/projects/ml-statcheck/statcheck-ml; run every script from that directory with /home/rass/Desktop/projects/ml-statcheck/.venv/bin/python. The corpus text is /home/rass/Desktop/projects/ml-statcheck/statcheck-ml/data/clean/<doc> (gitignored). Use absolute paths in every command, because the working directory resets between shell calls.

Rules:
- Read /home/rass/Desktop/projects/ml-statcheck/CLAUDE.md first and keep its rules. Prose you write follows ASD-STE100: active voice, articles kept, short sentences. No machine-written texture: comments say why, not what; no docstring that restates a signature.
- Do not run rm, mv, cp, or any git command that changes state. Delete, move, or copy a file from inside Python. Never delete a file that git does not track. The manager commits.
- Never sample from a gold test paper or a dev paper. The sampler reads the inventory, which holds neither, and still calls statcheck_ml.gold.assert_not_gold_test on the papers it draws from, and asserts that none is in dataset/gold/dev.json, before it reads any text. Do not run pipeline/14_gold_eval.py.
- Do not edit a file that this task does not name as yours.
- Report every command you ran with its real output, every file you changed, and every number you measured. Do not report a number you did not measure.

Background. dataset/passages/inventory.jsonl holds one record per passage piece of the 204 gold-train papers outside the dev set: passage_id, doc, journal, paper_has_results, paper_damaged, start_line, end_line, char_start, n_chars, n_lines, text_sha256 (the sha256 of the piece text, utf-8), paper_text_sha256, piece, pieces, coder_results, cascade {"results", "fragments"} and flags {"has_gold", "has_gold_in_scope", "gold_damaged", "cascade_fires"}. It holds no text. dataset/passages/inventory_summary.json holds the counts: 12592 pieces; has_gold_in_scope 585; hard negatives (paper_has_results false and cascade_fires true) 80; damaged_no_gold 1531; clean negatives by journal, 11677 in 61 journals. pipeline/17_passage_inventory.py holds paper_pieces(text, max_chars), which returns the pieces of one repaired text as dicts {"unit", "start_line", "end_line", "piece", "pieces", "char_start", "text"}; the repaired text is repair_validated(normalize(raw))[0] in two steps, as the script does. Import paper_pieces from the script by path with importlib, as tests/test_passage_inventory.py does. A batch is a JSON list of {"window_id": passage_id, "text": passage text}, the shape that pipeline/02_chunk.py writes and that the raters' output contract and pipeline/03_collect.py read. The version 2 sample files are the model for yours: dataset/windows/train.json is a list of {"window_id", "text"}; dataset/windows/key.json holds everything a rater must not see. Read pipeline/01_sample_windows.py (the journal stratification), pipeline/17_passage_inventory.py, src/statcheck_ml/gold.py and the section "The passages" of docs/PROTOCOL.md before you start.

YOUR TASK:
1. pipeline/18_sample_passages.py. Arguments: clean_text_dir (positional), --inventory dataset/passages/inventory.jsonl, --split dataset/gold/split.json, --dev dataset/gold/dev.json, --seed 0, --cap-gold 900, --cap-hard 400, --cap-damaged 200, --cap-clean 300, --max-chars 1500, --out dataset/passages/sample.json, --key dataset/passages/key.json, --summary dataset/passages/sample_summary.json. Strata, assigned in this order so that no passage is in two: "gold" = has_gold_in_scope true; "hard" = paper_has_results false and cascade_fires true; "damaged" = paper_damaged true and has_gold false, the passages with cascade_fires true taken first and then the rest; "clean" = has_gold false and cascade_fires false, drawn round-robin over the journals (journals sorted; inside each journal the passages sorted by passage_id and shuffled with random.Random(seed)) so that no journal dominates. Inside gold, hard and the two halves of damaged: sort by passage_id, shuffle with random.Random(seed), take up to the cap. Then call assert_not_gold_test on the docs of the drawn passages and assert that no drawn doc is in dev.json. Then rebuild each drawn paper's pieces once with paper_pieces, take the piece with the same start_line, end_line and piece, and assert that the sha256 of its text equals the record's text_sha256; raise on any difference. Output order: shuffle the drawn passages once with random.Random(seed), so that a batch mixes the strata and a rater cannot read the stratum from the position. sample.json: a list of {"window_id", "text"}. key.json: a list of {"window_id", "stratum", "doc", "journal", "start_line", "end_line", "piece", "pieces", "n_chars", "text_sha256", "flags", "n_coder_results", "cascade"} in the sample order; the coder rows stay in the inventory, not in the key. sample_summary.json: the counts per stratum, the pool size per stratum, which caps were hit, the passages drawn per journal, the number of passages, papers and journals drawn, and the seed and the caps; no timing. Print the summary.
2. Tests in tests/test_sample_passages.py with a synthetic inventory and a synthetic clean directory in tmp_path (the inventory records must hold the piece hashes of the synthetic texts, so build them with paper_pieces): the strata are disjoint and assigned in the stated order (one record that is both gold and hard lands in gold); a cap is respected; the clean stratum round-robins over journals (three journals, cap 4: no journal gives more than two); two runs give the same bytes; a record whose text_sha256 does not match the rebuilt piece raises; the key holds every window_id of the sample, in the same order, and a sample record holds no key other than window_id and text.
3. Run the sampler for real with the default caps. Run it a second time with --out, --key and --summary into the scratchpad and compare the sha256 of the three pairs; they must be equal. Add the stage to reproduce.sh directly after the 17_passage_inventory.py line, inside the same corpus gate, as "$PY pipeline/18_sample_passages.py data/clean/ --seed 0" with the echo line above it, and add dataset/passages/sample.json, dataset/passages/key.json and dataset/passages/sample_summary.json to the sha256sum line. Add the stage 18 row to pipeline/README.md, directly under the stage 17 row, in the same format. Regenerate dataset/MANIFEST.json with python pipeline/02_chunk.py --manifest-root dataset and confirm that verify_manifest returns []. Run the whole suite.

Report: the printed summary verbatim; the number of passages, papers and journals drawn; the sha256 of sample.json, key.json and sample_summary.json from both runs; the first five window_ids of sample.json; the pytest counts; git status --short. Files you own: pipeline/18_sample_passages.py, tests/test_sample_passages.py, dataset/passages/sample.json, dataset/passages/key.json, dataset/passages/sample_summary.json, dataset/MANIFEST.json, reproduce.sh (two places), pipeline/README.md (one row). Do not touch src/, the inventory, 17_passage_inventory.py, the spec files or any result file.
```

## M3.4 — Batches of ten with unpadded names (data-engineer, haiku)

```
Repository: /home/rass/Desktop/projects/ml-statcheck, branch v4-gold. The Python project is /home/rass/Desktop/projects/ml-statcheck/statcheck-ml; run every script from that directory with /home/rass/Desktop/projects/ml-statcheck/.venv/bin/python. Use absolute paths in every command, because the working directory resets between shell calls.

Rules:
- Read /home/rass/Desktop/projects/ml-statcheck/CLAUDE.md first and keep its rules. Prose you write follows ASD-STE100: active voice, articles kept, short sentences. No machine-written texture: comments say why, not what.
- Do not run rm, mv, cp, or any git command that changes state. Delete, move, or copy a file from inside Python. Never delete a file that git does not track. The manager commits.
- Do not read the corpus, dataset/gold/gold.jsonl or dataset/passages/inventory.jsonl. This task reads dataset/passages/sample.json only.
- Do not edit a file that this task does not name as yours.
- Report every command you ran with its real output, every file you changed, and every number you measured. Do not report a number you did not measure.

Background. dataset/passages/sample.json (M3.3) is a list of {"window_id", "text"}, the passages to rate, already in a shuffled order. pipeline/02_chunk.py writes rater batches: chunk(windows_file, out_dir, batch_size) writes batch_000.json, batch_001.json and so on, and the --manifest-root mode regenerates dataset/MANIFEST.json. The version 2 batches held 20 windows. A passage is up to 1500 characters, up to five times a window, and a rater that writes a long output loses records near its end (docs/PROTOCOL.md, "Batches of 20 windows"), so the passage batches hold ten. docs/PROTOCOL.md, section "The passages", already names the batch files dataset/passages/batches/batch_<i>.json with unpadded names, and the rater output path data/annotation/passages/<rater>/batch_<i>.json; pipeline/03_collect.py pairs an output with its batch by file name. statcheck_ml.provenance.write_manifest(paths, out_path, command, seed) writes a manifest of hashes. Read pipeline/02_chunk.py, src/statcheck_ml/provenance.py (write_manifest) and tests/test_collect.py (how a test builds a batch directory) before you start.

YOUR TASK:
1. pipeline/02_chunk.py: add --unpadded. With it, the batch name is batch_<i>.json with no zero padding; without it the name stays batch_NNN.json, so the version 2 names do not change. Add --manifest-out <path>: when given with --windows and --out, write a manifest of the batch files through write_manifest with the full command line as the command and --seed (new, default 0, unused by the chunking; say so in the help) as the seed. Put one line in the --size help that says why the passages use ten: a passage is up to five times a window, and a long output loses records.
2. Write the batches: python pipeline/02_chunk.py --windows dataset/passages/sample.json --out dataset/passages/batches --size 10 --unpadded --manifest-out dataset/passages/batches/MANIFEST.json --seed 0. Then check from Python: the directory holds ceil(n/10) files named batch_0.json to batch_<k-1>.json; every file but the last holds ten records; the records, read in batch order, equal sample.json in order; no record holds a key other than window_id and text. Print the batch count and the window_ids of batch_0.json.
3. Create the empty output directories data/annotation/passages/haiku, data/annotation/passages/sonnet, data/annotation/passages/opus and data/annotation/passages/adjudicator from Python (data/ is gitignored), so that a rater never creates a directory.
4. Tests: in tests/test_chunk.py (create it; no such file exists), with 23 synthetic records in tmp_path: --unpadded and size 10 give batch_0.json, batch_1.json, batch_2.json with 10, 10 and 3 records in the input order; the default naming still gives batch_000.json; --manifest-out writes a manifest whose files are the three batches.
5. Add the chunk command to reproduce.sh directly after the 18_sample_passages.py line, inside the same corpus gate, with its echo line, exactly as in step 2. Add one row to pipeline/README.md directly under the stage 18 row for the 02 call on the passages (input sample.json, output the batches and their manifest, needs nothing). Regenerate dataset/MANIFEST.json with python pipeline/02_chunk.py --manifest-root dataset and confirm that verify_manifest returns []. Run the whole suite.

Report: the batch count and the window_ids of batch_0.json; the check output of step 2; the pytest counts; git status --short (the batches directory appears as untracked). Files you own: pipeline/02_chunk.py, tests/test_chunk.py, dataset/passages/batches/ (new), dataset/MANIFEST.json, reproduce.sh (two lines), pipeline/README.md (one row). Do not touch src/, sample.json, key.json, the inventory, the spec files or any result file.
```
