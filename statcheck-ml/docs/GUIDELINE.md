# Annotation guideline — reported statistical test results, version 3.0 (2026-10-04)

This guideline is the written artifact of the annotation protocol. Each rater agent
embeds this text verbatim. A collection script records the sha256 of this file on
every annotation record. Change the version number when you change the text.

## The task

You read passages of text. The passages come from academic articles. You mark each
reported statistical hypothesis test result in each passage.

A passage is a run of lines from one article. It holds up to 1500 characters. It
often holds several sentences, or several rows of a table. One passage can hold many
results. Mark each one.

A statistical hypothesis test result reports a test statistic together with the
values that belong to it. A complete report has a test name, a test statistic value,
one or two degrees of freedom, and a p-value.

Read only the passage. Judge only what the passage contains. Report what the passage
says, and nothing more.

Many passages contain no test result. That is a normal and expected outcome. Do not
try to find something in every passage.

## The most important instruction

**These passages come from PDF files, and the conversion damaged many symbols.**

An operator is often not the character you expect. The equals sign, the less-than
sign and the greater-than sign are frequently replaced. A result is still a result
when its operator is damaged, and these are the results that matter most.

Look at these real examples. Every one is a valid result that must be marked:

```
c2 (3, N ¼ 96) ¼ 2.19, p ¼ .535      the fraction sign replaced the equals sign
F(1,184) \x02 7.64, p \x03 .01        control characters replaced both operators
t(60.81) \x02 \x070.23                a control character also replaced the minus sign
Q(1) b 0.01, p = 0.972                the letter b replaced the less-than sign
F(5, 953) = 1.65, p N 0.05            the letter N replaced the greater-than sign
2(1, N = 223) = 8.69                  the chi-square symbol was deleted entirely
v2 (1, N = 26) = 18.61                the letter v replaced the chi-square symbol
```

So:

- A character you cannot read, sitting where an operator belongs, **is** an operator.
- Record the operator you infer from context in `p_operator`, and set `damaged` to
  true. A p-value near zero almost always follows a less-than sign.
- Copy the damaged character into `quote` exactly as it appears. Never repair it.
- A bare `2`, or `c2`, or `v2`, or `X2` before parentheses is a chi-square test.

If you find yourself reporting that a passage holds nothing, read it once more for a
damaged operator before you decide.

## The parts of a result

For each result, record these parts when they are present:

| Part | Meaning |
|---|---|
| `test_type` | the test name: t, F, r, z, chi2, or Q; `null` when no rule below gives a name |
| `statistic` | the value of the test statistic |
| `df1` | the first degrees of freedom |
| `df2` | the second degrees of freedom, if the test has one |
| `n` | the sample size, when it is written inside the parentheses |
| `p_operator` | the operator before the p-value: `=`, `<` or `>` |
| `p_value` | the reported p-value |

A part that is absent is `null`. Never supply a value that is not written. Never
calculate a missing value. The test name is the one exception, and the next section
gives its rules.

## When the test letter is missing

Sometimes a passage holds a result but not its test letter. The conversion can delete
the letter. The author can also name the test in words. Apply these rules in order,
and stop at the first rule that gives a name.

1. **Words name the test.** The sentence that holds the result names the test in
   words. Or the header of the table column or row names it. Write that test.
   Examples: `t test` gives `t`, `chi-square test` gives `chi2`, `analysis of
   variance` or `ANOVA` gives `F`, `correlation` gives `r`.
2. **The shape names the test.** Write the test that the shape identifies.
   - Two degrees of freedom with a statistic value, such as `(2, 89) = 6.14, p < .01`,
     give `F`. A p-value can follow or not.
   - A sample size inside the parentheses, such as `(1, N = 223) = 8.69`, gives `chi2`.
3. **Nothing names the test.** Write `test_type` as `null`. Mark the result all the
   same, with every other part that the passage holds. Example: `(48) = 2.31, p = .02`
   with no test named in the sentence.

Never take a name from the size of a value or from the topic of the article.

A test letter that is missing directly before the parentheses or the value is a
damaged symbol. Set `damaged` to `true`. A test that the author named only in words is
not damaged.

A name in a table header stays outside `quote`. The quote holds the row.

## What is a result, and what is not

**Mark these:**

- Any of t, F, r, z, chi-square or Q with a value, whether or not a p-value follows.
- A test named in words: "the chi-square statistic ... is 32.9182, p = 0.3261".
- A result reported in a table row. Mark every row of the table that reports a result.
- A result split by a line break inside the passage. The statistic can end one line
  and the p-value begin the next. It is one result.
- A test statistic with no degrees of freedom, such as `t = 7.70, p < .0001`.
- A result whose operator is a damaged character. This is the most valuable case.
- A result whose test letter is missing. The section "When the test letter is
  missing" says what to write in `test_type`.
- A chi-square written as `2`, `c2`, `v2`, `X2`, or `χ2` before the parentheses.
- A chi-square that carries a sample size inside the parentheses, such as
  `chi2(1, N = 223) = 8.69`.
- A negative statistic, including one whose minus sign became a damaged character.
- Each member of a series, even when the members repeat one shape.
- A statistic at the end of the passage whose p-value lies after the passage. Record
  the parts that the passage holds.

**Do not mark these:**

- A regression coefficient on its own: `b`, `B`, `β`, `γ`. These are not test
  statistics, even when a p-value follows them.
- An odds ratio, a confidence interval, a mean, or a standard deviation.
- A significance legend such as `* p < .05, ** p < .01`.
- A bare p-value with no statistic anywhere near it.
- A row of numbers in a correlation matrix with no test reported.
- A page range, a year in parentheses, or a numbered reference.
- An effect size or a fit index: `R2`, `η2`, `g2`, `ω2`, Cohen `d`, `AIC`, `BIC`,
  `RMSEA`, `CFI`, or Cronbach `α`.
- A sample size on its own, such as `N = 223` outside the parentheses of a test.
- A degrees of freedom value with no statistic value.
- A p-value at the start of a passage whose statistic lies before the passage. The
  passage does not hold that result.

## Several results in one passage

A passage often holds several results. A table of twenty result rows gives twenty
results. Keep the parts of each result together, and never mix parts from two
different results. Mark every one, not only the first.

Order the results as the passage writes them, from the first character to the last.

A result can appear twice in one passage, for example once in the text and once in a
table. Mark two results, one for each place. The quote of each result holds the text
of its own place.

## Tables

Mark every row of a table that reports a result. Each such row is one result.

The statistic and the p-value of one row can stand far apart, with other columns
between them. Mark one result. Copy the whole row into `quote`, with every column
between the statistic and the p-value.

The conversion can break a table, so that a p-value no longer stands on the line of
its statistic. Pair a statistic with a p-value only when the passage shows that they
belong together. When you cannot pair them, mark the statistic alone. Set `p_operator`
and `p_value` to `null` and `confidence` to `low`.

## Output

Write a JSON array. One object for each passage, in the order given. The array must
have exactly as many objects as the input has passages. The key `window_id` names the
passage. Copy it exactly from the input.

```json
[
  {
    "window_id": "<copy from the input>",
    "contains_result": true,
    "results": [
      {
        "test_type": "t", "statistic": "2.45",
        "df1": "23", "df2": null, "n": null,
        "p_operator": "=", "p_value": ".02",
        "quote": "t(23) = 2.45, p = .02",
        "damaged": false,
        "confidence": "high"
      }
    ]
  },
  { "window_id": "<...>", "contains_result": false, "results": [] }
]
```

Each result object holds exactly ten keys: the nine keys `test_type`, `statistic`,
`df1`, `df2`, `n`, `p_operator`, `p_value`, `quote`, `damaged`, and the key
`confidence`. Write every key, even when the value is `null`.

`quote` is the exact text of the result, copied character for character from the
passage, including any line break or damaged character inside it.

## Rules

- Copy values exactly as written. Keep a leading decimal point. Do not add a zero.
- Report a result even when it looks wrong. Judging correctness is not your task.
- `confidence` is `high` or `low`, and nothing else. Use no other word.
- `p_operator` is `=`, `<`, or `>`, and nothing else. Write `<` for `≤` and `>` for
  `≥`. Write the operator you infer for a damaged character, never the character.
- `ns` or `n.s.` in the place of a p-value is not a p-value. Write `p_operator` and
  `p_value` as `null`.
- `test_type` is `t`, `F`, `r`, `z`, `chi2`, `Q`, or `null`. Write `null` only when
  the section "When the test letter is missing" gives no name.
- Use `low` confidence when a passage is ambiguous. Do not guess silently.
- `damaged` is `true` or `false`. Set it to `true` when any operator or symbol of the
  result is not the character you expect.
- `contains_result` is `true` when the `results` list holds one result or more.
- Return only the JSON array. Write no other text.

## Decision table for recurring cases

These cases return in almost every batch. Decide them the same way every time.

| The passage holds | The decision | Why |
|---|---|---|
| A `β` and a `t` in one sentence, such as `β = .34, t(98) = 2.11, p = .04` | Mark the `t` result. Do not mark the `β`. | The `t` is the test statistic. The `β` is a coefficient. |
| A correlation `r` with no degrees of freedom, such as `r = .73, p < .01` | Mark it. Set `df1` and `df2` to `null`. | A reported statistic stays a result. A script decides later whether it is checkable. |
| A chi-square with a sample size inside the parentheses, such as `c2(1, N = 96) = 4.2` | Mark it. Put `1` in `df1` and `96` in `n`. | The sample size is a part of the report, not a second degrees of freedom. |
| A table row with the statistic in one column and the p-value in another column | Mark one result that holds both. Copy the whole row into `quote`. | The columns report one result. The layout split it. |
| A p-value at the start of a passage, with its statistic before the passage | Do not mark it. Mark the other results of the passage as usual. | The passage does not hold the result. Never guess a statistic you cannot see. |
| A test name in any other spelling, such as `X2`, `chi-square`, or `χ2` | Write `test_type` as exactly one of `t`, `F`, `r`, `z`, `chi2`, `Q`. | One spelling per test keeps the field comparable between raters. |
| A result with no test letter, such as `(2, 89) = 6.14, p < .01` or `(48) = 2.31, p = .02` | Mark it. Write the test that words in the sentence or a table header name. Else write the test that the shape names: two degrees of freedom give `F`, and `N` inside the parentheses gives `chi2`. Else write `null`. | The name that the text implies is part of the report. A `null` name keeps a result that the passage does not name. |
| A table row whose statistic column and p-value column stand far apart, with other columns between them | Mark one result. Copy the whole row into `quote`, with every column between them. Pair a statistic and a p-value only when the passage shows that they belong together. | The row reports one result. A wrong pair is worse than a missing p-value. |
| The same result twice in one passage, such as once in the text and once in a table | Mark two results, one for each place. Each quote holds the text of its own place. | Each place reports the result. Mark text, not facts. |
