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
| 14 | `14_gold_eval.py` | `gold.jsonl`, `split.json`, `eval.json`, `dataset/baseline/gold/*`, the zoo | `results/gold_eval.json`, `results/gold_eval_log.jsonl` | corpus |
| 16 | `16_compare.py` | `results/v4/before/*.json`, current `eval.json` and `gold_eval.json`, the gold log | `results/v4/compare.json`, `results/v4/compare.md` | — |
| 15 | `15_statcheck_gold.py` | gold papers, the clean text, the PDF archive | `dataset/baseline/gold/*.csv`, `versions.json` | R, statcheck, pdftools, corpus |

Rules that every stage follows:

- Every stage that samples takes `--seed`, and the default is 0. A rerun with
  the same inputs gives the same bytes. `reproduce.sh` prints the hashes.
- `14_gold.py` takes `--seed` for symmetry and ignores it. It does not sample, and a rerun gives the same bytes.
- `14_gold_eval.py` appends one line to `results/gold_eval_log.jsonl` on every run. The log records each evaluation against the gold test.
- The files under `results/v4/before/` are frozen copies of the M1 outputs. No stage rewrites them.
- A stage reads the committed inputs only. No stage reads a scratch file.
- `--help` on any script prints its arguments and defaults.
