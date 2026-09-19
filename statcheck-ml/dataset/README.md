# The dataset

This directory holds the annotated data. The repository commits this data.

The papers themselves are not here. Papers are copyrighted and large, so they remain outside version control.

## Files in this directory

| File | Purpose |
|---|---|
| `windows/train.json` | Window text for the training set |
| `windows/holdout.json` | Window text for the holdout set |
| `windows/key.json` | Mapping: window ID to pool, document, journal, line, set |
| `annotations/train/final.json` | Consensus labels for the training set |
| `annotations/holdout/final.json` | Consensus labels for the holdout set |
| `annotations/train/{haiku,sonnet,opus}.json` | Individual model annotations (training set) |
| `annotations/holdout/{haiku,sonnet,opus}.json` | Individual model annotations (holdout set) |
| `agreement/train.json` | Agreement data between raters (training set) |
| `agreement/holdout.json` | Agreement data between raters (holdout set) |
| `splits.json` | Document-level split: which documents go to train and dev |
| `train.jsonl` | Processed dataset with aligned spans, ready for training |
| `holdout.jsonl` | Processed dataset for final evaluation |
| `MANIFEST.json` | SHA256 hashes of all files for integrity checking |

## Row structure

One line of JSON represents one window. The training file `train.jsonl` contains 1962 windows. The holdout file `holdout.jsonl` contains 576 windows.

```json
{
  "window_id": "c75b46c65d",
  "round": "v2",
  "pool": "A",
  "journal": "Cognition",
  "source_doc": "Cognition/Amso_Cognition_2014_979Y.txt",
  "source_line": 412,
  "text": "... F(2, 30) = 8.72, p < .01 ...",
  "contains_result": true,
  "label_tier": "bronze",
  "set": "train",
  "tier_counts": {"unanimous": 5, "majority": 0, "adjudicated": 0},
  "results": [ ... ]
}
```

A window is one candidate line with the line before and after it. This shape is deliberate. 18.4% of results span a line break, so one line is not a safe unit.

`source_doc` and `source_line` record where the passage came from. A person with the same corpus can rebuild any passage and extend a window if a result was cut.

`set` is `"train"` or `"holdout"`.

`tier_counts` summarizes how many results reached each consensus tier (unanimous, majority, adjudicated).

## Result structure

```json
{
  "test_type": "F",
  "statistic": "8.72",
  "df1": "2",
  "df2": "30",
  "n": null,
  "p_operator": "<",
  "p_value": ".01",
  "quote": "F(2, 30) = 8.72, p < .01",
  "damaged": false,
  "confidence": "high",
  "block_span": [63, 87],
  "part_spans": {
    "TEST": [63, 64],
    "STAT": [72, 76],
    "DF1": [65, 66],
    "DF2": [68, 70],
    "POP": [80, 81],
    "PVAL": [82, 85]
  },
  "checkable": true,
  "tier": "unanimous",
  "raters": ["haiku", "sonnet", "opus"]
}
```

`block_span` covers the whole result. `part_spans` gives each extracted value. Every span is a character offset into `text`.

Spans are stored, not tag arrays. A per-character tag array is about ten times larger than the text and creates unreadable diffs. Training code expands spans into tags at model load time.

### checkable

`checkable` is true when arithmetic can verify the result. Three values are required: statistic, first degrees of freedom, and p-value.

A result can be found and still not be checkable. Example: `r = .73` with no degrees of freedom is real but not verifiable. Report the two counts separately. A model that raises detection without raising checkable detection has not improved the tool.

### damaged

`damaged` is true when format conversion destroyed a character. The common case is an operator that became a control character. In 198 of 3100 documents, conversion writes a control character in place of the operator, and 35.5% of results in that shape are affected.

### confidence

The rater assigned `"high"` or `"low"` confidence to the extraction. This field is for transparency and analysis, not training.

### tier

The consensus tier shows how many raters agreed.

- `"unanimous"`: All three raters found the result with the same quote.
- `"majority"`: Two of three raters found the result (and one found a different result, or found no result).
- `"adjudicated"`: The raters disagreed. A human reviewed and selected the best result.

All results in this dataset are `"bronze"` label tier: a language model produced them, and a human verified consensus only through adjudication. Gold labels require human extraction from the start.

### raters

`raters` is a list of the model names that extracted this result. The models are `haiku`, `sonnet`, and `opus`.

## Tiers (label_tier)

Every label is `"bronze"`. A language model produced it. No human extracted it from the papers themselves.

This matters. No measurement in this project can find a mistake that the annotator makes every time. More annotation does not remove this limit.

Read `CONTEXT.md` in the repository root for the decision.

## Pools

| Pool | Meaning |
|---|---|
| A | The regex matched a complete result |
| B1 | A p-value is present; no complete regex match |
| B2 | A test letter and digits are present; no p-value found |
| C | The density filter rejected the line |

Pool C shows what the prefilter rejects. It is the only pool drawn from rejected text, so it is the only place where a filter mistake appears.

## Provenance in final.json

The `final.json` files contain provenance metadata:

```json
{
  "provenance": {
    "guideline_sha": "9419abb1d150e014fdfef5f63fda166889d0c176a2db7f701243563f97bb1f3f",
    "consensus_rule": "2-of-3 lenient",
    "adjudicator_model_id": "claude-opus-5",
    "built_at": "2026-09-19T20:27:24.338220+00:00",
    "n_disputes": 53,
    "n_adjudicated": 36
  }
}
```

This metadata documents when and how consensus was reached.

## Rebuild the dataset

Run both commands:

```bash
python pipeline/06_dataset.py --windows dataset/windows/train.json --key dataset/windows/key.json --final dataset/annotations/train/final.json --set train --out dataset/train.jsonl --splits dataset/splits.json --seed 0 --dev-share 0.15 --holdout-windows dataset/windows/holdout.json

python pipeline/06_dataset.py --windows dataset/windows/holdout.json --key dataset/windows/key.json --final dataset/annotations/holdout/final.json --set holdout --out dataset/holdout.jsonl
```

Then regenerate the manifest:

```bash
python pipeline/02_chunk.py --manifest-root dataset
```

Verify integrity:

```bash
python -c "from statcheck_ml.provenance import verify_manifest; errors = verify_manifest('dataset/MANIFEST.json'); print('OK' if not errors else errors)"
```

## Why papers are not in the repository

Papers are copyrighted. Statcheck-ml handles papers only as a separate corpus held outside version control. The `source_doc` and `source_line` fields record the provenance so that a person with the corpus can rebuild any window and extend it if needed.

Statcheck-ml ships only this annotated data, the three model ports, and the tests. It does not ship papers.
