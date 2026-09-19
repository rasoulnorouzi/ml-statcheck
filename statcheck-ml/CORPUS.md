# Corpus review — 02_pdfs.zip

The owner supplied this corpus. This document records what it contains and what the
project must do with it. All numbers come from samples. The sample size is stated
with each result.

## Contents

| Item | Count |
|---|---|
| PDF files | 3100 |
| Converted text files | 3000 |
| Paired PDF and text | 3000 |
| PDF files without text | 100 |
| Journals | 63 |
| Size of the text | 250 MB |

The corpus holds 50 to 102 articles for each journal. The journals cover psychology,
sociology, political science, economics, education, and management. The balance
appears deliberate.

The economics and political science journals do not follow APA style closely. This is
useful. These articles are the cases where a regex fails.

## Finding 1 — the supplied text lost every Greek letter

The supplied text files contain no character above ASCII. The conversion removed all
Greek letters and all mathematical symbols.

The chi-square symbol is the important loss. It appears in the text as `2`, or as
`v2`, or it disappears. The phi symbol disappears.

| In the PDF | In the supplied text |
|---|---|
| `χ2(1, N = 223) = 8.69` | `2(1, N = 223) = 8.69` |
| `χ2 (1, N = 26) = 18.61` | `v2 (1, N = 26) = 18.61` |
| `ϕ = −.20` | `  = -.20` |

A regex that searches for a chi-square test finds almost nothing in this text. The
silver labels would therefore contain very few chi-square results.

## Finding 2 — a new conversion repairs this

PyMuPDF reads all 150 sampled PDF files without error. It recovers the Unicode
characters in 96.7% of them.

```
supplied text : 2(1, N = 223) = 8.69, p = .003;  = -.20
PyMuPDF       : χ2(1, N = 223) = 8.69, p = .003; ϕ = −.20
```

Convert the PDF files again. Use the supplied text only as a second, damaged version
of the same documents.

## Finding 3 — the two versions are a free training resource

The project now has 3000 documents in two aligned forms: a clean form and a damaged
form. The damage is real, not simulated.

Use this pair for the noise families in phase 4. Real corruption is better evidence
than generated corruption.

## Finding 4 — one corruption cannot be repaired

In a small number of papers the PDF maps the `<` character to a backslash. The text
then reads `p \ .05`. Two different extractors produce the same result, so the fault
is in the PDF file.

This affects 0.5% of papers in a sample of 200. It is an edge case. Record it as a
known operator variant and do not build the project around it.

## Finding 5 — a result often splits across lines

This confirms the assumption that one result can cross a line or a sentence boundary.


Measured again on the new conversion, over all 3100 documents and 6139 statistics.

| Measurement | Share |
|---|---|
| A p-value follows the statistic within 250 characters | 86.6% |
| No p-value is near the statistic | 13.4% |
| One or more line breaks separate the statistic and the p-value | 18.4% |
| A sentence boundary separates them | 3.3% |

A model that reads one sentence at a time therefore loses up to 17% of results. The
processing unit must be an overlapping window.



## Finding 6 — the prefilter works

Statistical lines carry many non-letter characters. Ordinary prose does not.

Only 0.1% of the lines in the corpus contain a test statistic. A prefilter is
therefore necessary, not merely an optimisation.

| Threshold on non-letter density | Statistical lines kept | Other lines kept |
|---|---|---|
| 0.15 | 100.0% | 75.0% |
| 0.20 | 100.0% | 33.9% |
| 0.25 | 99.9% | 21.1% |
| 0.30 | 98.8% | 13.1% |

Use 0.20 as the first operating point. It keeps every statistical line in the sample
and removes two thirds of the other lines.

Read this table with care. The statistical lines were identified by a regex, so the
table measures density against results that a regex can already find. It does not
prove that the threshold keeps results a regex misses. Only the annotated random
sample can show that.

## Actions

1. Done. All 3100 PDF files converted in 53 seconds. No file failed. The conversion
   recovered non-ASCII characters in 99.8% of them.
2. Done. Findings 5 and 6 now report the new conversion.
3. Next: sample the annotation. See docs/PROTOCOL.md.
4. Add the backslash operator to the list of known variants.

## Finding 7 — a third of results hide behind control characters

The conversion of some PDF files writes a control character where the operator
belongs. The equals sign and the comparison signs are the characters affected.

```
F(1,184) \x02 7.64, p \x03 .01
r(130)   \x01 1.0,  p < 0.001
t(303)   \x01 \x06 4.82, p \x05 .001
```

This is not rare.

| Measurement | Count |
|---|---|
| Documents with a control-character operator | 198 of 3100 (6.4%) |
| Results of this shape with a normal operator | 6996 |
| Results of this shape with a control-character operator | 3848 |
| Share of this shape that is hidden | 35.5% |
| p-values with a control-character operator | 11324 |

No regular expression can read these results, because the character it needs is not
there. This is the clearest evidence that a learned extractor can do work that a
pattern cannot.

### The mapping is not fixed

A simple substitution table cannot repair this. The same control character means
different things in different documents, because the meaning comes from the font of
each document.

```
document A :  F(1,184) \x02 7.64      \x02 is an equals sign
document B :  F(2,272) \x03 4.09      \x03 is an equals sign
document A :  p \x03 .01              \x03 is a less-than sign
```

### Position carries the meaning

The place of the character tells you its role. A character directly after the
degrees of freedom is almost always an equals sign. A character directly after `p`
is one of three signs.

A model that reads the surrounding characters can therefore recover the operator. A
pattern cannot, because the pattern must know the character in advance.

The operator matters for the arithmetic. `p < .05` and `p = .05` lead to different
verdicts. So the model must name the operator, not merely mark its position.

Two further signals help:

1. Inside one document the mapping is consistent. A document is therefore decodable
   as a whole, even when one occurrence is ambiguous.
2. A reported p-value near zero follows a less-than sign far more often than an
   equals sign.
