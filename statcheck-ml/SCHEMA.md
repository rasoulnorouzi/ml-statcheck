# Annotation schema

Two formats are used. Do not confuse them. The annotator writes one. The model
trains on the other. A script converts the first into the second.

## Why two formats

A language model counts characters unreliably. If you ask it for a start offset and
an end offset, it returns numbers that are close but wrong.

So the annotator reports values and an exact quote. A script then finds the quote in
the passage, finds each value inside the quote, and produces the character offsets.
The offsets are therefore always correct, because a program made them.

## Format 1 — what the annotator writes

```json
{
  "window_id": "c75b46c65d",
  "contains_result": true,
  "results": [
    {"test_type": "t", "statistic": "1.28", "df1": "199", "df2": null,
     "n": null, "p_operator": "=", "p_value": ".203",
     "quote": "t(199) = 1.28, p = .203",
     "damaged": false, "confidence": "high"}
  ]
}
```

The quote must be copied character for character. The alignment depends on it.

## Format 2 — what the model trains on

Character-level BIOES tags. The model is a character model, so a character is the
unit. BIOES adds an End tag and a Single tag to BIO. Short spans such as `23` gain
from this, and most spans here are short.

Three outputs are produced for each window.

| Output | Shape | Purpose |
|---|---|---|
| part tags | BIOES over 7 labels | find each value |
| block tags | BIOES over 1 label | group the values of one result |
| window flag | one label for the window | say whether any result is present |
| operator class | 3 classes for each `POP` span | name the operator, even when the character is damaged |

### Why the operator needs its own output

In 6.4% of the documents the conversion writes a control character where the
operator belongs. A third of the results of that shape are affected. Marking the
position of the operator is therefore not enough, because the character itself
carries no meaning.

The model must choose between `=`, `<` and `>` from the surrounding characters. The
arithmetic depends on this choice, because `p < .05` and `p = .05` give different
verdicts.

Train this output on the results where the operator is undamaged. These give a large
supervised set at no extra cost. Then apply it to the damaged results.

The seven part labels are `TEST`, `STAT`, `DF1`, `DF2`, `N`, `POP`, and `PVAL`.

### Why a second tag layer

Part tags alone cannot say which statistic belongs to which degrees of freedom. One
window often holds several results. In the sample, one window held four.

The block layer marks each complete result as one span. Every part inside one block
belongs to one result. Grouping is therefore solved by a second tag layer, and the
project needs no separate linking model.

### Why a window flag

The flag gives a cheap answer for the common case, because most windows hold no
result. It also gives a useful signal during annotation. When the flag and the part
tags disagree, the window is difficult, and it should be annotated first.

## Alignment

The script searches for the quote, then for each value inside the quote.

The search must ignore differences in spaces and line breaks. A result is often split
by a line break, and a strict search fails on exactly those cases. In the first
sample, one quote in eighteen failed for this reason, and it was a split result.

Report the alignment rate for every batch. A batch below 98% has a problem in the
annotation or in the alignment, and it must not enter the training set.

## Detected is not the same as checkable

A result can be found and still not be checked. The arithmetic needs a test
statistic, the degrees of freedom, and a reported p-value.

In the first sample the annotator found `r = .73` with no degrees of freedom and no
p-value. It is a real reported statistic. It cannot be recomputed.

Record both states for every result:

- `detected`: the model found it
- `checkable`: the parts needed for the arithmetic are all present

Report the two counts apart. A model that raises detection but not checkable
detection has not improved the tool.
