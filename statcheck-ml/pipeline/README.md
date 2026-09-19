# Pipeline Stages

| Stage | Script | Input | Output | Needs Corpus? | Needs Agents? |
|-------|--------|-------|--------|---------------|---------------|
| 00 | 00_convert.py | PDF archive | Text files | Yes | No |
| 00 | 00_font_table.py | PDF archive | Font mappings (JSON) | Yes | No |
| 01 | 01_sample_windows.py | Clean text directory | Window sample (JSON) | Yes | No |
| 01 | 01_sample_holdout.py | Clean text directory, used key | Holdout sample (JSON) | Yes | No |
| 02 | 02_chunk.py | Documents | Chunks | Yes | No |
| 03 | 03_collect.py | Chunks | Collected results | Yes | No |
| 04 | 04_agree.py | Annotations from all raters | Agreement metrics (JSON) | No | Yes |
| 05 | 05_adjudicate.py | Disagreements, rater annotations | Final adjudicated labels | No | Yes |
| 06 | 06_dataset.py | Windows, annotations, final labels | Training dataset (JSONL) | No | No |
| 07 | 07_train.py | Training dataset, config grid | Trained models (PyTorch) | No | No |
| 08 | 08_export.py | Models, training data | ONNX export, manifest | No | No |
| 08 | 08_port_kit.py | Source repository | Port kit with manifest | No | No |
| 09 | 09_evaluate.py | Holdout windows, baseline results, models | Evaluation metrics (JSON) | No | No |
| 10 | 10_figures.py | Evaluation results, agreement | Figures (PNG) | No | No |
| 11 | 11_report.py | All results and evaluation | Report (Markdown) | No | No |
| 12 | 12_engines.py | PDFs, text from all engines | Engine comparison report | Yes | No |
| 12 | 12_regex_parity.py | Windows, annotations, baseline CSV | Regex vs baseline comparison | No | No |
