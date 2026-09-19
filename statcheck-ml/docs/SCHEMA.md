# Label schema

This document describes the schema that `src/statcheck_ml/labels.py` implements. The
code is the authority. Change the code and this document together.

Two formats exist. Do not confuse them. A rater agent writes the first. The model
trains on the second. A script converts the first into the second.

A language model counts characters unreliably. A model that reports a start offset
and an end offset returns numbers that are close and wrong. So a rater reports values
and an exact quote. A script then finds the quote in the passage, finds each value
inside the quote, and produces the character offsets. A program makes every offset,
so every offset is correct.

## Entities and tags

The model is a character model, so one character is one unit. The tag scheme is
BIOES. Most spans here are one to four characters long, so the explicit End tag and
Single tag carry real information.

`labels.py` defines nine entities:

| Entity | Meaning |
|---|---|
| `TEST` | the test name, such as t or F |
| `STAT` | the value of the test statistic |
| `DF1` | the first degrees of freedom |
| `DF2` | the second degrees of freedom |
| `N` | the sample size written inside the parentheses |
| `PVAL` | the reported p-value |
| `POP_EQ` | the operator before the p-value, an equals sign |
| `POP_LT` | the operator, a less-than sign |
| `POP_GT` | the operator, a greater-than sign |

The tag set holds 37 tags: one `O` tag, and four tags (`B`, `I`, `E`, `S`) for each of
the nine entities. `TAGS`, `TAG_TO_ID`, and `ID_TO_TAG` expose the set and its order.

### Why the operator is part of the entity

The operator is not a separate output of the model. It is part of the tag itself.

The corpus forces this choice. In 198 documents of 3100 the PDF conversion writes a
control character where the operator belongs, and 35.5 % of the results of that shape
are affected. Marking the position of such a character says nothing, because the
character carries no meaning. Naming the operator in the tag makes the model choose
from the surrounding characters. That is the only way to recover the operator.

The model trains on the undamaged operators, which are free supervision, and applies
that to the damaged ones.

## The two functions

```python
spans_to_tags(length: int, spans: List[Tuple[int, int, str]]) -> List[str]
tags_to_spans(tags: List[str]) -> List[Tuple[int, int, str]]
```

`spans_to_tags` builds a BIOES tag sequence of the given length from
`(start, end, entity)` spans. A span of one character gets an `S` tag. A longer span
gets `B`, then `I`, then `E`. The function ignores an unknown entity and an empty
span. A later span wins where two spans overlap.

`tags_to_spans` reads the spans back out of a tag sequence. The decoding is forgiving
on purpose. A model can emit an `I` tag with no `B` before it. A strict decoder drops
such a span and hides a near miss during evaluation, so a stray continuation opens a
span instead.

`find_overlaps` returns every pair of spans that overlap. An overlap is always a bad
label, so the function reports it rather than hiding it.

## How a dataset row becomes spans

Each result in a dataset row carries `part_spans`, a map from a part name to a
`[start, end]` pair in the window text. The part names are `TEST`, `STAT`, `DF1`,
`DF2`, `N`, `POP`, and `PVAL`. Seven names produce nine entities.

`part_label(name, operator)` does the mapping:

- Six names pass through unchanged: `TEST`, `STAT`, `DF1`, `DF2`, `N`, `PVAL`.
- `POP` becomes `POP_EQ`, `POP_LT`, or `POP_GT`.

The operator decides which one. The operator comes from the `p_operator` field that
the rater wrote, not from the character in the text. The character in the text may be
damaged, and a damaged character names no operator. `OPERATOR_ENTITY` and
`ENTITY_OPERATOR` hold the map in both directions.

The conversion is therefore: read `part_spans`, map each name with `part_label`, then
call `spans_to_tags` with the window length.

## Grouping: adjacency, not a second layer

One window often holds several results. The schema must still say which statistic
belongs to which degrees of freedom.

Version 2 predicts the grouping by adjacency after decoding. The decoder reads the
spans in order of position. A new `TEST` span, or a repeat of an entity that the
current group already holds, starts a new group. Every span between two group starts
belongs to the earlier group.

The trade-off is explicit:

| Option | Cost | Decision |
|---|---|---|
| A second tag layer for the block | a second output head, ported to JS and R | rejected |
| A separate linking model | a second model, a second export, a second parity test | rejected |
| Adjacency after decoding | a rule in shared code, no extra parameters | chosen |

Adjacency fails when two results interleave in the text. That shape is rare. One
output head ports to JavaScript and to R with no extra ONNX output and no extra
runtime code, and that saving is larger than the loss.

`block_span` stays in the dataset row as the recorded truth of the grouping. The
evaluation uses it to measure how often adjacency recovers the correct group.

## Removed in version 2: the block layer and the window flag

Version 1 predicted three outputs: part tags, block tags, and a window flag. Version 2
predicts one.

**The block layer is removed.** It cost a second output head in the model, in the ONNX
graph, and in each of the three ports. Adjacency recovers almost the same grouping
from the part tags alone.

**The window flag is removed.** The flag is not independent of the spans. A window
holds a result when the model tags one, so the class follows from the spans. A second
head that can disagree with the first adds a failure mode and no information.

One head keeps the port thin. The rule in `CLAUDE.md` is one spec and three thin
ports, and every extra head is a rule written three times.

## The dataset row

One JSON object per line, in `dataset/train.jsonl` and `dataset/holdout.jsonl`.

| Key | Meaning |
|---|---|
| `window_id` | the identifier of the window |
| `set` | `train` or `holdout` |
| `round` | the annotation round that produced the labels |
| `pool` | the sampling pool: `A`, `AN`, `B1`, `B2`, or `C` |
| `journal` | the journal of the source document |
| `source_doc` | the path of the converted text file |
| `source_line` | the first line of the window in that file |
| `text` | the window text |
| `contains_result` | `true` when `results` holds one result or more |
| `label_tier` | the tier of the labels; `bronze` today |
| `tier_counts` | the count of results in the row by consensus tier |
| `results` | the list of results |

Each result holds:

| Key | Meaning |
|---|---|
| `test_type` | `t`, `F`, `r`, `z`, `chi2`, or `Q` |
| `statistic` | the value of the test statistic, as written |
| `df1` | the first degrees of freedom, or `null` |
| `df2` | the second degrees of freedom, or `null` |
| `n` | the sample size inside the parentheses, or `null` |
| `p_operator` | `=`, `<`, `>`, or `null` |
| `p_value` | the reported p-value, as written, or `null` |
| `quote` | the exact text of the result, copied from the window |
| `damaged` | `true` when a symbol of the result is not the expected character |
| `confidence` | `high` or `low` |
| `block_span` | `[start, end]` of the whole result in the window text |
| `part_spans` | a map from a part name to `[start, end]` |
| `checkable` | `true` when the parts needed for the arithmetic are all present |
| `tier` | `unanimous`, `majority`, or `adjudicated` |

`tier_counts` counts the results of the row by their `tier`, for example
`{"unanimous": 2, "majority": 1, "adjudicated": 0}`. The row-level count lets a
report filter a set by tier without reading every result.

### Detected is not the same as checkable

A script can find a result and still not check it. The arithmetic needs a test
statistic, the degrees of freedom, and a reported p-value.

A real example from the first sample is `r = .73` with no degrees of freedom and no
p-value. It is a real reported statistic. No script can recompute it.

Report the two counts apart. A model that raises detection but not checkable
detection has not improved the tool.

### Alignment

The script searches the window for the quote, then for each value inside the quote.

The search ignores differences in spaces and line breaks. A result is often split by a
line break, and a strict search fails on exactly those results. In the first sample,
one quote in eighteen failed for this reason, and it was a split result.

The alignment gate is 98 %. `pipeline/06_dataset.py` exits with an error below that
rate. A batch below the gate has a fault in the annotation or in the alignment, and it
must not enter the training set.

## Label tiers

| Tier | Source | Allowed use |
|---|---|---|
| Bronze | three rater agents, majority vote, scripted adjudication | training, development, and the holdout |
| Gold | human labels, supplied by the project owner | held-out test only, when it exists |

No human gold set exists today. Every label in the repository is bronze, and every
report says so. Never report an annotation that a language model produced as gold.
Never train on gold.

The `tier` field of a result records the consensus route, not the label tier. A result
with the tier `unanimous` is still bronze.

The silver tier of version 1 is retired. The regex has a recall near 0.2, so a window
it labels holds about four unlabeled true results for each labeled one. That is label
noise, not supervision.
