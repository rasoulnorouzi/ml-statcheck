# Annotation protocol, version 3

This document describes how the project produces labels. The guideline is
`docs/GUIDELINE.md`. The schema is `docs/SCHEMA.md`. This document holds the process
around them.

Every number in a report comes from a script that reads a committed file. This
protocol is the rule set that those scripts apply.

Version 3 adds the passage annotation of project version 4: guideline 3.0, passages
as the unit, and the coders as a fourth vote. The sections marked "Version 2 record"
describe how the bronze window set was made. They stay unchanged as the record of that
set. The other sections hold for the passages, and some hold for both sets.

## The passages

The unit of version 4 is the passage. A passage is a run of lines of one paper that
the prefilter keeps (`spec/prefilter.json` version 2, `"unit": "passage"`). A passage
longer than 1500 characters is split at line ends into pieces of at most 1500
characters, and each piece is one passage. `pipeline/17_passage_inventory.py` (M3.2)
builds the pieces of every gold-train paper outside the dev papers. It writes one
record per piece to `dataset/passages/inventory.jsonl`. A record holds the coordinates,
the text sha256, the coder rows inside the piece and the cascade counts, but no text.
`pipeline/18_sample_passages.py` (M3.3) draws the sample from the inventory in four
strata. The strata are passages with an in-scope coder row, hard negatives, passages of
damaged papers, and clean negatives. It writes the drawn texts to `dataset/passages/sample.json`
and everything that a rater must not see to `dataset/passages/key.json`.
`pipeline/02_chunk.py --size 10 --unpadded` writes the batches to
`dataset/passages/batches/batch_<i>.json`, with unpadded names such as `batch_0.json`.
A batch holds ten passages, not 20, because a passage is up to five times as long as
a window. A long output loses records, as "Batches of 20 windows" states. A rater
writes its output to `data/annotation/passages/<rater>/batch_<i>.json`, with the name
of its batch, because `pipeline/03_collect.py` pairs an output with its batch by name.

The passages come from gold-train papers only. Every sampler calls
`statcheck_ml.gold.assert_not_gold_test` on the papers it draws from. No passage of a
gold test paper is ever rated, and no passage label ever enters the gold test.

A batch holds the same shape as a version 2 batch: a list of `{"window_id", "text"}`.
The key `window_id` holds the passage identifier. The rater output keeps the ten keys
of guideline 2.0, so `03_collect.py`, `validate_batch` and `align_quote` read it
unchanged. `validate_batch` accepts a `null` test type, which guideline 3.0 allows.

## The gold-train dev papers

`dataset/gold/dev.json` holds the dev papers. `pipeline/14_gold_dev_split.py` draws
them from the gold-train papers: 15 % by paper, seed 0, stratified by `has_results`.
The inventory skips the dev papers, and the sampler draws only from the inventory. So
a dev paper never enters a rater sample. A dev paper never enters a training set.

The dev papers are the gate set of the noise families (M6). A family stays only when
it raises recall on its own real damage type in the dev papers. It must also not raise
the finds in the dev papers that report no result.

## Three raters, independent and blind

This section holds for the window set of version 2 and for the passages of version 4.

Three agents annotate every window: `annotator-haiku`, `annotator-sonnet`, and
`annotator-opus`. The three agent files hold the same body and differ only in `name`
and `model`. Each agent has the tools `[Read, Write]`.

Three model families give three raters that do not share one training run. The design
also answers one extra question: which cheap model annotates best against the final
labels.

Each rater runs in its own dispatch. A rater sees one batch file and writes one output
file. A rater never sees another rater's output, never sees a vote count, and never
sees `key.json`. The agent body embeds the guideline verbatim, so a rater needs no
other file. `tests/test_guideline_embedding.py` fails when an agent body and
`docs/GUIDELINE.md` differ.

A rater of version 4 does not know that the coders exist. It sees the passage and the
guideline only. The coder rows stay in the inventory, which no rater reads.

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

The same limit holds for the passages. A rater of version 4 can read
`dataset/passages/inventory.jsonl` or `key.json` if it disobeys its prompt, and the
inventory holds the coder rows. The blindness to the coders is a rule too.

The adjudicator of version 4 has one more limit. A coder candidate always carries `n`
as `null` and `confidence` as `high`, because the coders recorded neither. A careful
reader can use that pattern to guess that a candidate came from the coders.

## The four-vote consensus

This section is the rule set of the consensus script of M4.6. It holds for the
passages only.

Four voters vote: the three raters and the coders. A coder row votes only when its
test type is one of the six names. A located coder row votes by span. A coder row that
is not located votes by statistic value only. Two results are the same when their
spans overlap by 50 % or more of the shorter span. A result stays with two votes of
four. A result with one vote, from any voter, is a singleton for the adjudicator. A
field takes the value with two votes or more and more votes than any other value,
among the voters that found the result. The coders never vote on `n`, `quote` or
`confidence`, and the final quote comes from the raters. A field with no such value is
a field conflict for the adjudicator, and a two-two split is a conflict. Every tier is
bronze. The subsections below give the details.

### The voters

Four voters vote on each passage: `annotator-haiku`, `annotator-sonnet`,
`annotator-opus`, and `coders`. The three human coders MS, ME and MN count as one
voter, named `coders`, whichever of them coded a row. Each voter has the same weight.

### Which coder rows vote

The coder rows of a passage are the `coder_results` of its record in
`dataset/passages/inventory.jsonl`. A row belongs to the passage whose line range holds
the line of its statistic.

- A row whose `test_type` is not one of `t`, `F`, `r`, `z`, `chi2`, `Q` does not vote.
  It is never a candidate. The coders code CI, Wald and `b` rows, which the guideline
  excludes.
- A row with a `span_in_passage` is located. It votes by span, like a rater result.
- A row with `span_in_passage` `null` is not located. It votes by statistic value only.
  It matches a result of another voter when the two statistics are equal in magnitude
  as numbers. `match_results` in `src/statcheck_ml/agreement.py` already does this
  when one side has no span.
- A row that is not located cannot open a result on its own. When it matches no
  result of another voter, it is dropped and counted. It is never a singleton, because
  the adjudicator cannot see it in the passage.
- A gold row is in no passage when `pipeline/14_gold.py` did not locate it in the
  paper. A gold row on a line that the prefilter dropped is in no passage too. Such a
  row does not vote.

### When two results are the same

Two results of two different voters are the same result when their spans overlap by
50 % or more of the shorter span. This is the lenient match of version 2. The match
between two voters is one to one, and the larger overlap wins first. The matches join
into clusters transitively. A cluster holds at most one result of each voter. A match
that would put two results of one voter in one cluster is skipped.

A rater result whose quote does not align has no span. It matches by statistic value,
as in version 2.

### When a result stays

A cluster with two votes or more stays. Any two voters are enough: two raters, or one
rater and the coders. A cluster with one vote is a singleton, whichever voter cast it.
A singleton goes to the adjudicator.

### How a field takes its value

The voters of a field are the voters of the cluster, with one exception. The coders
vote on `test_type`, `statistic`, `df1`, `df2`, `p_operator`, `p_value` and
`damaged`. They never vote on `n`, `quote` or `confidence`.

The values compare as in version 2. `statistic`, `df1`, `df2`, `n` and `p_value`
compare as numbers. `test_type` and `p_operator` compare after `normalise_test_type`
and `normalise_operator` in `src/statcheck_ml/provenance.py`. `damaged` compares as
written. A `null` is a value, so a `null` vote and a number vote differ.

A field takes the value V when V has two votes or more and more votes than any other
value. When only one voter of the cluster votes on the field, that value stands. In
every other case the field is in conflict.

| Votes on one field | Outcome |
|---|---|
| 4, 3–1, 3, 2–1–1, 2–1, 2 | the value with the most votes |
| 2–2, 1–1–1–1, 1–1–1, 1–1 | field conflict |
| one voter only | the value of that voter |

With four voters, a two-two split is a conflict. A one-one split in a cluster of two is
a conflict too. In version 2 the first rater won that split. Version 3 drops that rule,
because the order of the raters is not evidence.

Each field in conflict raises its own `field_conflict` dispute.

Two fields follow other rules, and neither raises a conflict.

- `confidence`: the value that more than half of the raters of the cluster wrote.
  With no such value, the value is `low`.
- `quote` and the span: they come from the raters. The span is the span that two or
  more raters of the cluster share exactly. With no shared span, the span is the span
  of the first rater of the cluster in the order `annotator-haiku`, `annotator-sonnet`,
  `annotator-opus`. The quote is the quote of the rater whose span is chosen. A cluster
  of one rater and the coders takes the span and the quote of that rater.

### What the adjudicator sees

A dispute holds `dispute_id`, `window_id`, `text`, `kind`, `field` when the kind is
`field_conflict`, and `candidates`. The text is the passage. The candidates are result
objects with the ten keys. The script shuffles them with seed 0. A dispute holds no
voter name and no vote count.

A singleton dispute holds one candidate. A field conflict dispute holds every result
of the cluster, the coder result included.

A coder candidate has the same shape as a rater candidate, so that the adjudicator
cannot tell it apart:

- `quote` is the passage text at the `span_in_passage` of the row. It is never the
  coder's transcription.
- `statistic`, `df1`, `df2` and `p_value` take the written form of the passage text
  at the span, when the span holds a number equal to the value. Otherwise they take
  the value of the inventory.
- `test_type`, `p_operator` and `damaged` take the values of the inventory.
- `n` is `null`, and `confidence` is `high`.

A dispute batch holds 20 disputes. The output for one dispute is one result object, so
the output length does not grow with the passage. The input grows to at most 30 000
characters of passage text for each batch.

The `adjudicator` agent judges by the guideline and the passage only, as in version 2.
For a `singleton`, `keep` says whether the passage really reports the result. For a
`field_conflict`, `keep` is always true, and the judged value replaces the field.

### Tiers

Each result in the final labels carries a consensus tier:

| Tier | Meaning |
|---|---|
| `unanimous` | the three raters found the result, every voter of each field wrote the same value, and the adjudicator decided nothing |
| `majority` | two votes or more kept the result, and it is not unanimous, and the adjudicator decided nothing |
| `adjudicated` | the adjudicator kept a singleton, or decided one field or more |

**All three tiers are bronze.** A coder vote does not promote a label. The coders are
one voice of four, and the owner states that the coders are not always right.

Each result records `voters`, the list of the voters that found it, in the order
`annotator-haiku`, `annotator-sonnet`, `annotator-opus`, `coders`. Each result also
records `coder_vote`: `span`, `value`, or `null` when the coders did not vote. The
report gives these numbers for each tier and in total:

- the share of the kept results with a coder vote;
- the count of coder votes by span and by value;
- the count of in-scope coder rows that matched no result and were dropped;
- the count of coder rows out of scope.

### The limit of the coder vote

The coders are one voice. A coder error that two raters repeat survives the vote. A
coder error that one rater repeats survives too, because two votes keep a result. The
coders also do not vote on a correlation without degrees of freedom, because they did
not code it. Those results rest on the raters alone.

So the coder vote does not make the labels independent of the raters. The arithmetic
check remains the only independent signal. Section "The limit of a model annotator"
describes it.

## Collection, agreement and merge for the passages

`pipeline/03_collect.py` collects the passage outputs with the same validation and the
same provenance fields as version 2. The section "Collection and provenance" lists
them. The script keeps every individual rater file.

Agreement is measured among the three raters, as in version 2, per label type. The
coders are not part of the rater agreement numbers, because their scope differs from
the guideline.

## The frozen windows

Version 2 record. This section describes the window set of version 2.

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

## Batches of 20 windows

Version 2 record. The passages use batches of ten, as the section "The passages"
states.

`pipeline/02_chunk.py` writes batches of 20 windows. It writes the same batches, in
the same order, for every rater.

Round 2 set the size. A larger batch loses windows: the agent shortens its output near
the end of a long list, and the collection script then rejects the batch. Round 2
started from 1962 frozen windows and returned 1955 annotated windows, and the loss
came from the long batches. Twenty windows fit in one output without truncation.

The manager dispatches one agent for each batch and each rater. The output path is
`data/annotation/<set>/<rater>/batch_NNN.json`.

## Collection and provenance

Version 2 record. The passages use the same stage and the same fields.

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

Version 2 record. The passages use the same metrics among the three raters.

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

Version 2 record. The passages use the four-vote consensus instead.

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

Version 2 record. The passages use the tiers of the section "The four-vote consensus".

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

This section holds for the windows and for the passages.

No measurement in this project finds a mistake that every annotator makes. If all
three raters miss one way of reporting a result, the trained model learns to miss it
too, and every score stays high.

More annotation does not remove this limit. More agreement does not remove it either,
because the raters share their training.

The coder vote of the passages narrows the limit and does not remove it. A coder row
can catch a result that all three raters miss. It reaches the labels only as a
singleton that the adjudicator keeps, and the adjudicator is also a model. A result that the coders did not code, such as a
correlation without degrees of freedom, has no human vote.

One independent check remains. The arithmetic does not come from an annotator. A wrong
degrees of freedom or a misread statistic usually gives a p-value far from the reported
one. Published work puts the true rate of inconsistency near 10 %. A batch far from
that rate has an extraction fault, and no annotator was asked.

The arithmetic checks the results that the raters found. It says nothing about the
results that they missed.
