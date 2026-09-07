# The dataset

This directory holds the annotated data. It is committed to the repository.

It does not hold the papers. The papers are copyrighted and large, so they stay
outside version control.

## What a row contains

One file for each annotation round. One line of JSON for each window.

```
{
  "window_id":   "c75b46c65d",
  "round":       "round1",
  "pool":        "A",
  "journal":     "Cognition",
  "source_doc":  "Cognition/Amso_Cognition_2014_979Y.txt",
  "source_line": 412,
  "text":        "... F(2, 30) = 8.72, p < .01 ...",
  "contains_result": true,
  "label_tier":  "bronze",
  "results": [ ... ]
}
```

A window is one candidate line with the line before it and the line after it.
This shape is deliberate. 18.4% of results are separated from their p-value by a
line break, so one line is not a safe unit.

`source_doc` and `source_line` record where the passage came from. A person who
holds the same corpus can therefore rebuild any passage, and can extend a window
if a result was cut at the edge.

## What a result contains

```
{
  "test_type": "F", "statistic": "8.72",
  "df1": "2", "df2": "30", "n": null,
  "p_operator": "<", "p_value": ".01",
  "quote": "F(2, 30) = 8.72, p < .01",
  "damaged": false,
  "confidence": "high",
  "checkable": true,
  "block_span": [63, 87],
  "part_spans": {"TEST": [63, 64], "STAT": [72, 76], "DF1": [65, 66],
                 "DF2": [68, 70], "POP": [80, 81], "PVAL": [82, 85]}
}
```

`block_span` covers the whole result. `part_spans` gives each value. Every span
is a character offset into `text`.

Spans are stored, not tag arrays. A per-character tag array is about ten times
the size of the text, and it produces an unreadable difference in version
control. `to_bioes.py` expands the spans into tags when a model trains.

### checkable

`checkable` is true when the arithmetic has every value it needs: a statistic,
the first degrees of freedom, and a p-value.

A result can be found and still not be checked. `r = .73` with no degrees of
freedom is a real reported statistic that cannot be recomputed. Report the two
counts apart. A model that raises detection without raising checkable detection
has not improved the tool.

### damaged

`damaged` is true when the conversion destroyed a character in the result. The
common case is an operator that became a control character. In 198 documents of
3100 the conversion writes a control character where the operator belongs, and
35.5% of results of that shape are affected.

## label_tier

Every label in this directory is `bronze`. A language model produced it. No
human checked it.

This matters. No measurement in this project can find a mistake that the
annotator makes every time. More annotation does not remove this limit.

`CONTEXT.md` in the repository root explains what is done about it.

## Pools

| Pool | Meaning |
|---|---|
| A | the pattern matched a complete result |
| B1 | a p-value is present, no complete match |
| B2 | a test letter and digits are present, no p-value |
| C | the density filter rejected the line |

Pool C measures what the filter throws away. It is the only pool drawn from text
the filter rejected, so it is the only place where a filter mistake can be seen.

## How the files were made

```
python statcheck-ml/convert.py 02_pdfs.zip statcheck-ml/data/clean 14
python statcheck-ml/sample_windows.py statcheck-ml/data/clean <sample_dir> 15
# an annotator agent writes annotations_pass1.json into <sample_dir>
python statcheck-ml/build_dataset.py <sample_dir> statcheck-ml/dataset <round>
```

## Rounds

| File | Windows | Results | Note |
|---|---|---|---|
| `round0.jsonl` | 22 | 18 | first round; the sampler still drew reference entries |
| `round1.jsonl` | 55 | 42 | reference entries removed; four pools |

Do not train on a round that is reserved as a test set. No round is reserved
yet.
