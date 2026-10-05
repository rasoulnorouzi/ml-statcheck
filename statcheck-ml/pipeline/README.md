# Pipeline stages

Each stage is one script with explicit inputs and outputs. `reproduce.sh` runs
stages 04 to 11 from the committed files. Stages that need the corpus or a
rater agent are marked; they were run once, and their outputs are committed
with a manifest of hashes.

| Stage | Script | Input | Output | Needs |
|---|---|---|---|---|
| 00 | `00_convert.py` | PDF archive | text files | corpus |
| 00 | `00_font_table.py` | PDF archive | `spec/font_table.json` | corpus |
| 01 | `01_sample_windows.py` | text files | `dataset/windows/train.json`, `key.json` | corpus |
| 01 | `01_sample_holdout.py` | text files, used key | `dataset/windows/holdout.json` | corpus |
| 02 | `02_chunk.py` | windows | rater batches of 20 windows, manifest | — |
| 03 | `03_collect.py` | rater output files | `annotations/<split>/<rater>.json`, stamped | rater agents |
| 04 | `04_agree.py` | three rater files, final labels | `agreement/<split>.json` | — |
| 05 | `05_adjudicate.py` | three rater files | `disputes.json`, `final.json` | adjudicator agent |
| 06 | `06_dataset.py` | windows, final labels | `train.jsonl`, `holdout.jsonl`, `splits.json` | — |
| 07 | `07_train.py` | `train.jsonl`, `splits.json`, `grid.json` | `models/<run>/`, `models/runs.json` | CPU hours |
| 08 | `08_export.py` | trained runs | `models/zoo/`, `models/export.json` | — |
| 08 | `08_port_kit.py` | spec, zoo | a port kit with its manifest | — |
| 09 | `09_evaluate.py` | holdout, baseline CSVs, zoo | `results/eval.json` | — |
| 10 | `10_figures.py` | eval, agreement, runs | `results/figures/*.png` | — |
| 11 | `11_report.py` | template, every result file | `results/REPORT.md`, `results/readme_block.md` | — |
| 12 | `12_baseline.py` | holdout windows | `dataset/baseline/*.csv` | R, statcheck |
| 12 | `12_engines.py` | PDFs, text from every engine | `results/engines.json` | corpus |
| 12 | `12_regex_parity.py` | windows, baseline CSV | parity of the Python port of the regex | — |
| 13 | `13_ports.py` | the two port repositories under `ports/` | `results/ports.json` | R, node |
| 14 | `14_gold.py` | the owner's sheet, clean text | `dataset/gold/gold.jsonl` | corpus, the sheet |
| 14 | `14_gold_split.py` | `gold.jsonl` | `dataset/gold/split.json` | — |
| 14 | `14_gold_dev_split.py` | `gold.jsonl`, `split.json` | `dataset/gold/dev.json` | — |
| 14 | `14_gold_eval.py` | `gold.jsonl`, `split.json`, `eval.json`, `dataset/baseline/gold/*`, the zoo | `results/gold_eval.json`, `results/gold_eval_log.jsonl` | corpus |
| 17 | `17_passage_inventory.py` | clean text, `gold.jsonl`, `split.json`, `dev.json` | `dataset/passages/inventory.jsonl`, `dataset/passages/inventory_summary.json` | corpus |
| 18 | `18_sample_passages.py` | clean text, the inventory, `split.json`, `dev.json` | `dataset/passages/sample.json`, `key.json`, `sample_summary.json` | corpus |
| 19 | `19_consensus.py` | rater outputs, the inventory, passage batches | `dataset/passages/consensus.json`, `dataset/passages/disputes` | — |
| 20 | `20_merge.py` | the consensus, adjudicator decisions | `dataset/annotations/passages/final.json` | — |
| 21 | `21_dataset_v4.py` | bronze training set, passage labels, the inventory, passage batches | `dataset/train_v4.jsonl`, `dataset/splits_v4.json` | — |
| 22 | `22_train_v4.py` | `dataset/train_v4.jsonl`, `dataset/splits_v4.json` | `models/v4/<run>/` | GPU |
| 23 | `23_select_v4.py` | `models/v4/<run>/`, the gold and dev sets | `results/v4/selection.json`, `selection.md`, `results/v4/dev/*.json` | the trained models |
| 02 | `02_chunk.py --unpadded --size 10` | `dataset/passages/sample.json` | `dataset/passages/batches/batch_<i>.json`, their `MANIFEST.json` | — |
| 16 | `16_compare.py` | `results/v4/before/*.json`, current `eval.json` and `gold_eval.json`, the gold log | `results/v4/compare.json`, `results/v4/compare.md` | — |
| 15 | `15_statcheck_gold.py` | gold papers, the clean text, the PDF archive | `dataset/baseline/gold/*.csv`, `versions.json` | R, statcheck, pdftools, corpus |
| 24 | `24_final_v4.py` | `results/v4/selection.json`, `models/v4/*/zoo`, the shipped zoo, `gold.jsonl`, `split.json`, the holdout | `results/v4/final/gold/*.json`, `bronze.json`, `summary.json`, `summary.md`, one line per model in `results/gold_eval_log.jsonl` | corpus |
| 25 | `25_verdicts_v4.py` | `gold.jsonl`, `split.json`, `dataset/baseline/gold/*`, the shipped zoo, `models/v4/lstm-softmax-none-s0/zoo` | `results/v4/final/verdicts.json`, `verdicts.md`, one line in `results/gold_eval_log.jsonl` | R, statcheck, corpus |
| 26 | `26_paired_v4.py` | `results/v4/final/verdicts.json`, `summary.json`, `bronze.json` | `results/v4/final/paired.json`, `paired.md` | — |
| 27 | `27_prf_v4.py` | `results/v4/final/gold/*.json`, `bronze.json` | `results/v4/final/prf.json`, `prf.md` | — |

Rules that every stage follows:

- Every stage that samples takes `--seed`, and the default is 0. A rerun with
  the same inputs gives the same bytes. `reproduce.sh` prints the hashes.
- `14_gold.py` takes `--seed` for symmetry and ignores it. It does not sample, and a rerun gives the same bytes.
- `14_gold_eval.py` appends one line to `results/gold_eval_log.jsonl` on every run. The log records each evaluation against the gold test.
- `25_verdicts_v4.py --papers gold_test` appends one line to the same log. `--papers gold_train` is for development; it needs `--out` and writes no log line.
- The files under `results/v4/before/` are frozen copies of the M1 outputs. No stage rewrites them.
- A stage reads the committed inputs only. No stage reads a scratch file.
- `--help` on any script prints its arguments and defaults.
