# Annotation protocol, version 2

This document describes how the project produces labels. The guideline is
`docs/GUIDELINE.md`. The schema is `docs/SCHEMA.md`. This document holds the process
around them.

Every number in a report comes from a script that reads a committed file. This
protocol is the rule set that those scripts apply.

## The frozen windows

The windows are frozen. The raters of version 2 read the same windows as the earlier
rounds, so the new numbers compare with the old ones.

| File | Windows | Documents |
|---|---|---|
| `dataset/windows/train.json` | 1962 | 1304 |
| `dataset/windows/holdout.json` | 576 | 515 |

`dataset/windows/key.json` holds the pool, the journal, the document, and the line of
every window. A rater never reads this file. `dataset/MANIFEST.json` holds the sha256
of every committed dataset file, the command that produced it, the seed, and the
timestamp.

### Pools

Each pool answers a different question. Do not merge them.

| Pool | Source | Purpose | Train | Holdout |
|---|---|---|---|---|
| A | the pattern matched a complete result | check the pattern | 120 | 36 |
| AN | the pattern matched, and the sample size is inside the parentheses | check the `N` part | 42 | 0 |
| B1 | a p-value is present, no complete match | find prose reports and bare statistics | 600 | 180 |
| B2 | a test letter and digits, no p-value | find damaged operators | 900 | 270 |
| C | the density filter rejected the line | measure what the filter drops | 300 | 90 |

Pool B holds the value of the project. These windows are where a result hides in a
form that the regex cannot match. The sampler enriches pool B before it draws:
it prefers a window with a test letter, a parenthesis, and a digit that failed the
full pattern. A near miss has a much higher yield than a random window.

Pool C is small, and the project cannot skip it. Every other pool comes from text that
the filter already kept, so pool C is the only measure of the recall of the filter.

The sample is stratified by field, by test type, and by journal, so that no journal
dominates a pool. The split assigns whole documents. No document puts windows in two
sets, because windows from one paper share an author, a template, and a font.

### Why the sampler strips the references

A bibliography has the same character density as a statistical result. In round 0,
six of eight pool B windows were reference list entries, so the sampler was wasting
the budget.

`pipeline/01_sample_windows.py` now removes the reference section and stray reference
lines.
Reference contamination fell to zero in round 1 and stayed there.

### Yields, as history

These numbers come from round 1, which annotated 55 windows. They guided the weights
of the round 2 sample. They are history, not a target.

| Pool | Yield in round 1 |
|---|---|
| A | 15 of 15 |
| B1 | 5 of 15 |
| B2 | 9 of 15 |
| C | 0 of 10 |

Pool B2 gave the better yield, which no one expected. It catches the results whose
operator became a control character, and a pattern can never read those. The round 2
sample therefore weights B2 above B1.

Round 1 also fixed the alignment. The matcher reached 100 % of quotes once it ignored
differences in whitespace. The strict matcher failed on exactly the results that a
line break had split.

## Three raters, independent and blind

Three agents annotate every window: `annotator-haiku`, `annotator-sonnet`, and
`annotator-opus`. The three agent files hold the same body and differ only in `name`
and `model`. Each agent has the tools `[Read, Write]`.

Three model families give three raters that do not share one training run. The design
also answers one extra question: which cheap model annotates best against the final
labels.

Each rater runs in its own dispatch. A rater sees one batch file and writes one output
file. A rater never sees another rater's output, never sees a vote count, and never
sees `key.json`. The agent body embeds the guideline verbatim, so a rater needs no
other file.

### The known limit of the blindness

An agent inside Claude Code can read a file that its prompt did not name. The prompt
tells it to read exactly one file and to ignore every other file, and the tool list
holds `Read`, which a rater needs for its batch.

The blindness is therefore a rule, not a sandbox. State this limit in every report
that quotes an agreement number. Two effects follow:

- An agreement number is an upper bound on true independence.
- The three raters share a training lineage, so agreement between them cannot find a
  mistake that all three make. Section "The limit of a model annotator" states the
  consequence.

## Batches of 20 windows

`pipeline/02_chunk.py` writes batches of 20 windows. It writes the same batches, in
the same order, for every rater.

Round 2 set the size. A larger batch loses windows: the agent shortens its output near
the end of a long list, and the collection script then rejects the batch. Round 2
started from 1962 frozen windows and returned 1955 annotated windows, and the loss
came from the long batches. Twenty windows fit in one output without truncation.

The manager dispatches one agent for each batch and each rater. The output path is
`data/annotation/<set>/<rater>/batch_NNN.json`.

## Collection and provenance

`pipeline/03_collect.py` validates each rater output, then stamps it, then merges it
into `dataset/annotations/<set>/<rater>.json`.

The validation rejects an output that fails any of these checks:

- The file parses as JSON.
- The list holds exactly as many objects as the batch holds windows.
- Every `window_id` matches the batch, in order.
- Every result holds the nine keys and `confidence`.
- Every `quote` aligns to the window text when the matcher ignores whitespace.

The script lists every failed batch for a new dispatch. It never repairs an output.

Every annotation record carries these provenance fields:

| Field | Meaning |
|---|---|
| `rater` | the agent name, such as `annotator-sonnet` |
| `model_alias` | `haiku`, `sonnet`, or `opus` |
| `model_id` | the exact model identifier that served the call |
| `guideline_sha` | the sha256 of `docs/GUIDELINE.md` |
| `agent_sha` | the sha256 of the agent file |
| `batch_id` | the batch file name |
| `collected_at` | the UTC timestamp of the collection |
| `retries` | the number of dispatches the batch needed |

The script keeps every individual rater file. A consensus file cannot reproduce an
agreement number, so the individual annotations are the record.

## Agreement

`pipeline/04_agree.py` reads the three rater files and writes
`dataset/agreement/<set>.json`. It reports agreement per label type, not only overall.

| Level | Unit | Metrics |
|---|---|---|
| Window | `contains_result` | raw %, pairwise Cohen κ, Fleiss κ, Krippendorff α nominal |
| Result | an aligned character span | strict pairwise span F1 (exact span), lenient pairwise span F1 (overlap of 50 % or more), mean pairwise F1, per-rater F1 against the final labels |
| Field | a result that all three raters found | Krippendorff α nominal for `test_type`, `p_operator`, and `damaged`; exact-match % for `statistic`, `df1`, `df2`, `n`, and `p_value` as normalised numbers |

Every metric carries a bootstrap 95 % interval. The bootstrap resamples windows, 1000
times, with seed 0. `src/statcheck_ml/agreement.py` implements Cohen κ and Fleiss κ,
with unit tests against published examples. Krippendorff α comes from the pinned
`krippendorff` package.

## Consensus and disputes

`pipeline/05_adjudicate.py` builds the final labels.

**Consensus.** The script keeps a result when at least two raters found it. Two raters
found the same result when their spans overlap by 50 % or more. Each field then takes
the majority value of the raters that found the result.

**Disputes.** The script sends two cases to the adjudicator:

| Kind | Trigger |
|---|---|
| `singleton` | one rater alone found the result |
| `field_conflict` | the three raters wrote three different values in one field |

A dispute file holds 20 disputes. Each dispute holds `dispute_id`, `window_id`,
`text`, `kind`, `field` when the kind is `field_conflict`, and `candidates`. The
candidates are result objects as the raters wrote them.

**The adjudicator is blind to the votes.** The dispute file holds no rater name and no
vote count. The `adjudicator` agent (opus, `[Read, Write]`) judges by the guideline and
the passage only. It returns `{"dispute_id", "keep", "result", "reason"}` for each
dispute. The output is collected and stamped like an annotation.

For a `singleton`, `keep` says whether the passage really reports the result. For a
`field_conflict`, `keep` is always true, and the judged value replaces the field.

The script merges the outcome into `dataset/annotations/<set>/final.json`.

## Tiers

Each result in the final labels carries a consensus tier:

| Tier | Meaning |
|---|---|
| `unanimous` | all three raters found the result, and every field agreed |
| `majority` | two raters found the result, or a field took a majority value |
| `adjudicated` | the adjudicator decided the result or the field |

**All three tiers are bronze.** The tier records the consensus route. It does not
promote a label. No label in this repository is gold, because no human has labeled a
window. Every report states this.

## The limit of a model annotator

No measurement in this project finds a mistake that every annotator makes. If all
three raters miss one way of reporting a result, the trained model learns to miss it
too, and every score stays high.

More annotation does not remove this limit. More agreement does not remove it either,
because the raters share their training.

One independent check remains. The arithmetic does not come from an annotator. A wrong
degrees of freedom or a misread statistic usually gives a p-value far from the reported
one. Published work puts the true rate of inconsistency near 10 %. A batch far from
that rate has an extraction fault, and no annotator was asked.

The arithmetic checks the results that the raters found. It says nothing about the
results that they missed.
