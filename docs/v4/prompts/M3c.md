# Milestone M3c — worker prompts, verbatim

Written by the manager before the milestone ran (2026-10-04, after the resumed run
`wf_1ad266e8-2f6` ended with M3.2b rejected and M3.3, M3.4 skipped). The harness
prepends one line to each prompt: the task id and title, the branch, "do not commit",
and the request for real commands and real output as evidence. Nothing else is added.

What the resumed run left. M3.2b repaired the inventory: `text_sha256` is the piece
hash, `paper_text_sha256` the paper hash, the summary holds no timing, the manifest
verifies, `paper_pieces(text, max_chars)` is a function, the dead code is gone. The
manager verified every count and reproduced both files byte for byte. The task was
rejected for two small deliverables: the README paragraph does not say what item 7
asked, and what-comments remain in the script. The fix prompt was written but the run
ended before it ran. M3.3 and M3.4 were skipped because they depend on M3.2b.

Manager's decisions that the prompts carry:

- M3.2c is the fix prompt of M3.2b, unchanged in substance: the README paragraph is
  given verbatim, and the comments go.
- M3.3 writes the sampler and its tests and runs it twice. The wiring (`reproduce.sh`,
  the pipeline README rows, the manifest) moves to M3.4, so the sampler prompt is
  smaller. The M3.2 worker on haiku failed twice on one large prompt.
- M3.3 refuses an inventory whose summary shows zero cascade fires or more than five
  span-check failures (the brief, after the first M3.2 attempt).
- `char_start` of an inventory record is `Window.char_start`, an offset in the
  reference-blanked line text of `Prefilter._prepare`, not in the repaired text.
  `text[char_start:char_start + n_chars]` does not give the piece. The sampler rebuilds
  a drawn piece through `paper_pieces` by `start_line`, `end_line` and `piece`, and the
  prompt says so.
- The caps stay at 900 / 400 / 200 / 300: 585 + 80 + 200 + 300 = 1165 passages, 117
  batches of ten, 351 rater runs.
- The rater output directories under `data/annotation/passages/` are created in M3.4.

## M3.2c — The README paragraph and the what-comments of M3.2b (data-engineer, haiku)

```
Repository: /home/rass/Desktop/projects/ml-statcheck, branch v4-gold. Python project: /home/rass/Desktop/projects/ml-statcheck/statcheck-ml; run every command from that directory with /home/rass/Desktop/projects/ml-statcheck/.venv/bin/python. Use absolute paths. Do not run rm, mv, cp or any git command that changes state. Do not edit any file that this task does not name.

Two fixes remain from M3.2b.

1. dataset/README.md: replace the whole paragraph under "## Gold" that starts with "`gold/dev.json` holds 36 gold-train dev papers" by exactly these five sentences, verbatim, as one paragraph:
`gold/dev.json` holds the 36 gold-train dev papers: 15 % by paper, seed 0, stratified by `has_results`. They never enter a rater sample or a training set, and they gate the noise families. `passages/inventory.jsonl` lists every passage piece of the other 204 gold-train papers with its coordinates, the piece hash, the coder rows and the cascade counts, and no text. The inventory is the sampling frame of the passage annotation, not a training file. The sample holds the texts.
Change nothing else in the file.

2. pipeline/17_passage_inventory.py: delete the what-comments inside paper_pieces and build_passages ("# Split the unit into pieces if it exceeds max_chars", "# Start a new piece", "# Add the final piece", "# Create records for each piece", "# Calculate positions in repaired text", "# Calculate char_start: unit.char_start + offset within unit", "# Calculate piece text hash", "# Read and process text", "# Get units from prefilter and build pieces", "# Get units for line range checking", "# Build passage_id", the "# +1 for newline" tail, and any other comment that only restates the next line). Keep the one why-comment at the top of paper_pieces. Change no code.

Then rerun: python pipeline/17_passage_inventory.py data/clean/ --seed 0 --out <scratchpad>/inv.jsonl --summary <scratchpad>/sum.json and report sha256 of both; they must equal 8f1121911fbcc9ca999577c31953431bd4db669e749710c20ffdd4c9ce76a33b and 87ea5e5db760626e10194eb3c37499df012b82eab710b9d6a2e455fd93681158. Do not rewrite the committed inventory files or MANIFEST.json; they are unchanged. Run python -m pytest tests/test_passage_inventory.py -q and report the count. Report git status --short and the new README paragraph verbatim. Files you own: dataset/README.md (that paragraph only) and pipeline/17_passage_inventory.py (comments only).
```

## M3.3 — The passage sampler (data-engineer, haiku)

```
Repository: /home/rass/Desktop/projects/ml-statcheck, branch v4-gold. The Python project is /home/rass/Desktop/projects/ml-statcheck/statcheck-ml; run every script from that directory with /home/rass/Desktop/projects/ml-statcheck/.venv/bin/python. The corpus text is /home/rass/Desktop/projects/ml-statcheck/statcheck-ml/data/clean/<doc> (gitignored). Use absolute paths in every command, because the working directory resets between shell calls.

Rules:
- Read /home/rass/Desktop/projects/ml-statcheck/CLAUDE.md first and keep its rules. Prose you write follows ASD-STE100: active voice, articles kept, short sentences. No machine-written texture: comments say why, not what; no docstring that restates a signature.
- Do not run rm, mv, cp, or any git command that changes state. Delete, move, or copy a file from inside Python. Never delete a file that git does not track. The manager commits.
- Never sample from a gold test paper or a dev paper. The sampler reads the inventory, which holds neither, and still calls statcheck_ml.gold.assert_not_gold_test(docs, split) on the papers it draws from, and asserts that none is in dataset/gold/dev.json, before it reads any text. Do not run pipeline/14_gold_eval.py.
- Do not edit a file that this task does not name as yours. Do not edit reproduce.sh, pipeline/README.md or dataset/MANIFEST.json; the next task wires the stage in.
- Report every command you ran with its real output, every file you changed, and every number you measured. Do not report a number you did not measure.

Background. dataset/passages/inventory.jsonl holds one record per passage piece of the 204 gold-train papers outside the dev set: passage_id, doc, journal, paper_has_results, paper_damaged, start_line, end_line, char_start, n_chars, n_lines, text_sha256 (the sha256 of the piece text, utf-8), paper_text_sha256, piece, pieces, coder_results, cascade {"results", "fragments"} and flags {"has_gold", "has_gold_in_scope", "gold_damaged", "cascade_fires"}. It holds no text. dataset/passages/inventory_summary.json holds the counts: 12592 pieces; passages_by_flag.has_gold_in_scope 585; passages_by_flag.cascade_fires 866; coder_rows.span_check_failures 0; hard negatives (paper_has_results false and cascade_fires true) 80; damaged_no_gold 1531; clean negatives by journal, 11677 in 61 journals. pipeline/17_passage_inventory.py holds paper_pieces(text, max_chars), which returns the pieces of one repaired text as dicts {"unit", "start_line", "end_line", "piece", "pieces", "char_start", "text"}; the repaired text is repair_validated(normalize(raw))[0], in the two steps the script's build_passages uses. Import paper_pieces from the script by path with importlib, as tests/test_passage_inventory.py does. Warning: char_start is an offset in the prefilter's reference-blanked line text, not in the repaired text; text[char_start:char_start + n_chars] does not give the piece. Rebuild a piece only through paper_pieces, matched by start_line, end_line and piece. A batch is a JSON list of {"window_id": passage_id, "text": passage text}, the shape that pipeline/02_chunk.py writes and that pipeline/03_collect.py reads. The version 2 sample files are the model for yours: dataset/windows/train.json is a list of {"window_id", "text"}; dataset/windows/key.json holds everything a rater must not see. Read pipeline/01_sample_windows.py (the journal stratification), pipeline/17_passage_inventory.py, src/statcheck_ml/gold.py and the section "The passages" of docs/PROTOCOL.md before you start.

YOUR TASK:
1. pipeline/18_sample_passages.py. Arguments: clean_text_dir (positional), --inventory dataset/passages/inventory.jsonl, --inventory-summary dataset/passages/inventory_summary.json, --split dataset/gold/split.json, --dev dataset/gold/dev.json, --seed 0, --cap-gold 900, --cap-hard 400, --cap-damaged 200, --cap-clean 300, --max-chars 1500, --out dataset/passages/sample.json, --key dataset/passages/key.json, --summary dataset/passages/sample_summary.json.
   Gate: read the inventory summary first; raise SystemExit with a message if passages_by_flag.cascade_fires is 0 or coder_rows.span_check_failures is above 5. An inventory that shows either is broken, and a sample drawn from it would be wasted rater work.
   Strata, assigned in this order so that no passage is in two: "gold" = has_gold_in_scope true; "hard" = paper_has_results false and cascade_fires true; "damaged" = paper_damaged true and has_gold false, the passages with cascade_fires true taken first and then the rest; "clean" = has_gold false and cascade_fires false, drawn round-robin over the journals (journals sorted; inside each journal the passages sorted by passage_id and shuffled with random.Random(seed)) so that no journal dominates. Inside gold, hard and the two halves of damaged: sort by passage_id, shuffle with random.Random(seed), take up to the cap. Then call assert_not_gold_test on the docs of the drawn passages and assert that no drawn doc is in dev.json. Then rebuild each drawn paper's pieces once with paper_pieces(repaired_text, max_chars), take the piece with the same start_line, end_line and piece, and assert that the sha256 of its text equals the record's text_sha256; raise on any difference. Output order: shuffle the drawn passages once with random.Random(seed), so that a batch mixes the strata and a rater cannot read the stratum from the position. sample.json: a list of {"window_id", "text"}. key.json: a list of {"window_id", "stratum", "doc", "journal", "start_line", "end_line", "piece", "pieces", "n_chars", "text_sha256", "flags", "n_coder_results", "cascade"} in the sample order; the coder rows stay in the inventory, not in the key. sample_summary.json: the counts per stratum, the pool size per stratum, which caps were hit, the passages drawn per journal, the number of passages, papers and journals drawn, and the seed and the caps; no timing. Write both JSON files with indent=1 and sort_keys=True so two runs give the same bytes. Print the summary.
2. Tests in tests/test_sample_passages.py with a synthetic inventory, a synthetic inventory summary and a synthetic clean directory in tmp_path (the inventory records must hold the piece hashes of the synthetic texts, so build them with paper_pieces on the repaired synthetic text): the strata are disjoint and assigned in the stated order (one record that is both gold and hard lands in gold); a cap is respected; the clean stratum round-robins over journals (three journals, cap 4: no journal gives more than two); two runs give the same bytes; a record whose text_sha256 does not match the rebuilt piece raises; a summary with cascade_fires 0 raises SystemExit, and so does one with span_check_failures 6; the key holds every window_id of the sample, in the same order, and a sample record holds no key other than window_id and text.
3. Run the sampler for real with the default caps. Expected: gold 585 and hard 80 (below their caps, so every passage of both strata is drawn), damaged 200 (cap hit; cascade-firing first), clean 300 (cap hit), 1165 passages. If the counts differ, report them and stop; do not change a cap. Run it a second time with --out, --key and --summary into the scratchpad and compare the sha256 of the three pairs; they must be equal. Run the whole suite.

Report: the printed summary verbatim; the number of passages, papers and journals drawn; the sha256 of sample.json, key.json and sample_summary.json from both runs; the first five window_ids of sample.json; the pytest counts; git status --short. Files you own: pipeline/18_sample_passages.py, tests/test_sample_passages.py, dataset/passages/sample.json, dataset/passages/key.json, dataset/passages/sample_summary.json. Do not touch src/, the inventory, 17_passage_inventory.py, the spec files, reproduce.sh, pipeline/README.md, dataset/MANIFEST.json or any result file.
```

## M3.4 — Batches of ten with unpadded names; the stages wired in (data-engineer, haiku)

```
Repository: /home/rass/Desktop/projects/ml-statcheck, branch v4-gold. The Python project is /home/rass/Desktop/projects/ml-statcheck/statcheck-ml; run every script from that directory with /home/rass/Desktop/projects/ml-statcheck/.venv/bin/python. Use absolute paths in every command, because the working directory resets between shell calls.

Rules:
- Read /home/rass/Desktop/projects/ml-statcheck/CLAUDE.md first and keep its rules. Prose you write follows ASD-STE100: active voice, articles kept, short sentences. No machine-written texture: comments say why, not what.
- Do not run rm, mv, cp, or any git command that changes state. Delete, move, or copy a file from inside Python. Never delete a file that git does not track. The manager commits.
- Do not read the corpus, dataset/gold/gold.jsonl or dataset/passages/inventory.jsonl. This task reads dataset/passages/sample.json only.
- Do not edit a file that this task does not name as yours.
- Report every command you ran with its real output, every file you changed, and every number you measured. Do not report a number you did not measure.

Background. dataset/passages/sample.json (M3.3) is a list of {"window_id", "text"}, the passages to rate, already in a shuffled order; 1165 passages are expected. pipeline/18_sample_passages.py wrote it; its defaults are the committed paths and it takes clean_text_dir as its positional argument and --seed. pipeline/02_chunk.py writes rater batches: chunk(windows_file, out_dir, batch_size) writes batch_000.json, batch_001.json and so on; --manifest-root regenerates dataset/MANIFEST.json; --manifest exists already and writes a manifest of dataset/windows/, so do not reuse that name. The version 2 batches held 20 windows. A passage is up to 1500 characters, up to five times a window, and a rater that writes a long output loses records near its end (docs/PROTOCOL.md, "Batches of 20 windows"), so the passage batches hold ten. docs/PROTOCOL.md, section "The passages", names the batch files dataset/passages/batches/batch_<i>.json with unpadded names, and the rater output path data/annotation/passages/<rater>/batch_<i>.json; pipeline/03_collect.py pairs an output with its batch by file stem. statcheck_ml.provenance.write_manifest(paths, out_path, command, seed) writes a manifest of hashes. reproduce.sh runs 17_passage_inventory.py inside a corpus gate (the lines with "data/clean/"), and its last line is a sha256sum over the result files. pipeline/README.md holds one row per stage in a five-column table (stage, script, inputs, outputs, needs). Read pipeline/02_chunk.py, src/statcheck_ml/provenance.py (write_manifest), tests/test_collect.py (how a test builds a batch directory), reproduce.sh and pipeline/README.md before you start.

YOUR TASK:
1. pipeline/02_chunk.py: add --unpadded. With it, the batch name is batch_<i>.json with no zero padding; without it the name stays batch_NNN.json, so the version 2 names do not change. Add --manifest-out <path>: when given with --windows and --out, write a manifest of the batch files through write_manifest with the full command line as the command and --seed (new, default 0, unused by the chunking; say so in the help) as the seed. Put one line in the --size help that says why the passages use ten: a passage is up to five times a window, and a long output loses records.
2. Write the batches: python pipeline/02_chunk.py --windows dataset/passages/sample.json --out dataset/passages/batches --size 10 --unpadded --manifest-out dataset/passages/batches/MANIFEST.json --seed 0. Then check from Python: the directory holds ceil(n/10) files named batch_0.json to batch_<k-1>.json; every file but the last holds ten records; the records, read in numeric batch order, equal sample.json in order; no record holds a key other than window_id and text. Print the batch count and the window_ids of batch_0.json.
3. Create the empty output directories data/annotation/passages/haiku, data/annotation/passages/sonnet, data/annotation/passages/opus and data/annotation/passages/adjudicator from Python (data/ is gitignored), so that a rater never creates a directory.
4. Tests: in tests/test_chunk.py (create it; no such file exists), with 23 synthetic records in tmp_path: --unpadded and size 10 give batch_0.json, batch_1.json, batch_2.json with 10, 10 and 3 records in the input order; the default naming still gives batch_000.json; --manifest-out writes a manifest whose files are the three batches.
5. reproduce.sh: directly after the 17_passage_inventory.py line, inside the same corpus gate, add the sampler as "$PY pipeline/18_sample_passages.py data/clean/ --seed 0" with its echo line above it, and then the chunk command of step 2 with its echo line. Add dataset/passages/sample.json, dataset/passages/key.json and dataset/passages/sample_summary.json to the sha256sum line. Run bash -n reproduce.sh.
6. pipeline/README.md: add the stage 18 row directly under the stage 17 row (script 18_sample_passages.py; inputs the inventory, its summary, split.json, dev.json, clean text; outputs sample.json, key.json, sample_summary.json; needs corpus), and under it one row for the 02 call on the passages (inputs sample.json; outputs dataset/passages/batches/ and its manifest; needs nothing), both in the format of the rows around them.
7. Regenerate dataset/MANIFEST.json with python pipeline/02_chunk.py --manifest-root dataset and confirm from Python that statcheck_ml.provenance.verify_manifest('dataset/MANIFEST.json') returns []. Run the whole suite.

Report: the batch count and the window_ids of batch_0.json; the check output of step 2; the verify_manifest output; the pytest counts; git status --short (the batches directory appears as untracked). Files you own: pipeline/02_chunk.py, tests/test_chunk.py, dataset/passages/batches/ (new), dataset/MANIFEST.json, reproduce.sh (the two stage lines with their echo lines, and the sha256sum line), pipeline/README.md (two rows). Do not touch src/, sample.json, key.json, the inventory, the spec files or any result file.
```
