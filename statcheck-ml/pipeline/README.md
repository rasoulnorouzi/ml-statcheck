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

Rules that every stage follows:

- Every stage that samples takes `--seed`, and the default is 0. A rerun with
  the same inputs gives the same bytes. `reproduce.sh` prints the hashes.
- A stage reads the committed inputs only. No stage reads a scratch file.
- `--help` on any script prints its arguments and defaults.
