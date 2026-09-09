# statcheck-ml — method, measurements and results

This report records what the project built, how it measured the result, and what
each number means. Every number comes from a measurement in this repository. The
command that produces each one is named, so a reader can repeat it.

The prose follows ASD-STE100 Simplified Technical English.

---

## 1. The problem

The R package `statcheck` reads statistical results from a paper and checks
them. It does two different jobs:

1. It **finds** reported results in the text, with regular expressions.
2. It **recomputes** the p-value and compares it to the reported one.

Job 2 is mathematics and it is correct. Job 1 is a regular expression, and it
fails when the PDF conversion damages the text. This project replaces job 1 with
a learned model. It does not touch job 2.

**The rule that shapes everything: machine learning finds results, and never
judges them.** No model output reaches a verdict. The comparison stays
closed-form mathematics in all three ports.

---

## 2. The corpus

The owner supplied 3100 PDF files from 63 journals, in psychology, sociology,
political science, economics, education and management. 3000 files also had a
text version.

| Item | Count |
|---|---|
| PDF files | 3100 |
| Journals | 63 |
| Size of the text | 250 MB |
| Lines longer than 20 characters | 3,661,147 |
| Lines that hold a statistic | 4,989 |

**About 1 line in 700 holds a result.** That ratio decides the architecture: a
filter must run before the model, or the work is not affordable in a browser.

### 2.1 The supplied text was unusable

The supplied text files contained no character above ASCII. The conversion had
removed every Greek letter and every mathematical symbol.

```
supplied text : 2(1, N = 223) = 8.69, p = .003;   = -.20
PyMuPDF       : χ2(1, N = 223) = 8.69, p = .003; ϕ = −.20
```

The project therefore converted the PDF files again with PyMuPDF. Both versions
were kept, because the damaged version is real corruption and is useful for
training.

### 2.2 Symbol fonts destroy the operator

A publisher sets an operator such as `=` or `<` in a font that carries no
ToUnicode map. The character then has no Unicode value, and the converter writes
whatever is left. PyMuPDF writes a control character. Poppler writes a letter in
the Greek and Coptic block.

```
F(1, 17) <U+0003> 35.72, p <U+0004> .0005
```

The information is absent from the file. PyMuPDF and pdftotext produce the same
loss, so no converter can recover it by reading harder.

**More than half of the holdout results sit in text damaged in this way.**

---

## 3. Definitions

This section defines every term the report uses later.

### 3.1 Window

A **window** is the unit the model reads. It is a candidate line plus the lines
around it. It is not a sentence, because **18.4% of results are separated from
their p-value by at least one line break**.

### 3.2 Prefilter

The **prefilter** selects the lines that could hold a result. It answers "could
this hold a result", never "does this look like a result".

**The prefilter caps the recall of the whole system.** Text it discards can
never be recovered by the model. It is therefore tuned for recall alone.

Rules, in `spec/prefilter.json`:

| Rule | Value |
|---|---|
| Minimum share of characters that are not letters | 0.20 |
| Minimum line length | 20 characters |
| Lines of context each side | 2, then grown |
| Text a window must hold | 250 characters |
| Maximum lines of context | 8 |

Measured on the clean corpus:

| Density threshold | Recall | Share of other lines kept |
|---|---|---|
| 0.15 | 1.000 | 0.750 |
| **0.20** | **1.000** | **0.339** |
| 0.25 | 0.999 | 0.211 |
| 0.30 | 0.988 | 0.131 |

### 3.3 Label schema

The model tags characters, not words. The tag set is **BIOES** over 9 entities,
which gives **37 tags**.

BIOES marks each character as the **B**eginning, **I**nside or **E**nd of an
entity, as a **S**ingle-character entity, or as **O**utside any entity. It
separates two entities that touch, which a simpler scheme cannot.

The 9 entities:

| Entity | Meaning |
|---|---|
| `TEST` | the name of the test, such as `F` or `t` |
| `STAT` | the value of the statistic |
| `DF1` | the first degrees of freedom |
| `DF2` | the second degrees of freedom |
| `N` | the sample size |
| `PVAL` | the reported p-value |
| `POP_EQ` | the p-operator, when it is `=` |
| `POP_LT` | the p-operator, when it is `<` |
| `POP_GT` | the p-operator, when it is `>` |

**The operator is folded into the tag set.** The model has no second head. A
separate head would have to be ported to JavaScript and to R, and the operator
is often a damaged character that only the surrounding text explains.

### 3.4 Label tiers

| Tier | Source | Use |
|---|---|---|
| silver | the regular expression | training only |
| bronze | language model annotators | training and holdout |
| gold | a human | never obtained |

**No machine annotation is ever called gold.** The models never train on the
holdout.

### 3.5 Damage families

A damaged result is put in one family, by the first pattern that matches. The
order matters, because one result can carry two kinds of damage, and the first
entry names the one that does the most harm.

| Family | What happened |
|---|---|
| decimal point lost | the value itself is wrong; no extractor can read it |
| control character | the operator became a control character |
| fraction sign for = | `=` became `¼` |
| backslash for < | `<` became `\` |
| letter b for < | `<` became `b` |
| letter N for > | `>` became `N` |
| chi-square symbol lost | `χ` disappeared or became `v2`, `c2`, `x2` |
| letter p or ! for operator | the operator became `p` or `!` |

---

## 4. Annotation

No human labels were available. Language model agents produced every label.

### 4.1 The protocol

1. **Two agents annotate each window, blind.** Neither sees the project, the
   other agent, or any memory of earlier rounds. They see the passage and the
   guidelines only.
2. **Agreement becomes a label.** Where both agents agree, the label stands.
3. **An opus agent adjudicates the disputes only.** It sees both answers.

Blindness matters: an annotator that knows the project will find what the
project wants.

### 4.2 What was annotated

| Set | Windows |
|---|---|
| round 0 | 22 |
| round 1 | 55 |
| round 2 | 1955 |
| **training total** | **2032** |
| holdout | 576 |

The holdout holds **315 results**, of which **164 are damaged** and **201 are
checkable**.

### 4.3 The annotation ceiling

The annotators miss results. Error analysis showed that **70.5% of the model's
apparent errors were annotator misses, not model errors**, and almost all of
them on damaged text.

The measured ceiling is **84.1% on the training set and 90.5% on the holdout**.
No model can score above the labels it is measured against. **Read every number
in this report against that ceiling.**

---

## 5. The model

### 5.1 Why characters

A word-piece tokenizer splits statistical notation unpredictably, and it would
have to be ported to JavaScript and to R. A character map is a small JSON file
that all three ports read.

This was later confirmed by measurement, not left as an assumption. See
section 9.4.

### 5.2 Architecture

```
characters -> embedding (64) -> BiLSTM or BiGRU (2 layers, 128 hidden each way)
           -> dropout (0.25) -> linear (256 -> 37 tags) -> CRF (optional)
```

| Item | Value |
|---|---|
| Vocabulary | 175 characters, from `spec/charmap.json` |
| Index 0 | PAD; its embedding stays at zero and never trains |
| Index 1 | UNK; a character seen fewer than 3 times becomes UNK |
| Embedding | 64 |
| Hidden | 128 each direction, 2 layers |
| Dropout | 0.25 |
| Output | 37 tags |

### 5.3 The CRF

A **conditional random field** scores the whole tag sequence, not each character
alone. It therefore cannot produce a sequence that BIOES forbids, such as an
`I-STAT` that follows an `O`.

**A transition mask blocks 71% of transitions** before training starts. The CRF
decodes outside the ONNX graph, so the exported model stays simple.

### 5.4 Cost

| Model | Unit | CRF | Parameters | Checkpoint |
|---|---|---|---|---|
| crf-aug | LSTM | yes | 616,072 | 2.5 MB |
| lstm-crf | LSTM | yes | 615,176 | 2.5 MB |
| gru-aug | GRU | no | 467,592 | 1.9 MB |
| **gru-crf** | GRU | yes | **466,696** | **1.9 MB** |

---

## 6. Training

### 6.1 Splits

The split is by **document**, and a further set of **journals is held out
entirely**. A split by window would put two windows of one paper on both sides,
and the score would be optimistic.

### 6.2 Class imbalance

Most characters are `O`. Class weights are inverse frequency, **capped at 40**,
so a rare tag cannot dominate the loss.

### 6.3 The four noise families

The generator makes perturbed copies of annotated windows. It moves every label
with the text it changes. **850 examples were produced with 0 alignment
failures.**

| Family | What it simulates |
|---|---|
| 1. OCR confusions | `l`/`1`, `O`/`0`, `rn`/`m`, and similar |
| 2. Unicode variants | minus against en dash, thin spaces, `χ` degrading |
| 3. Operator damage | every replacement measured in this corpus |
| 4. Engine alphabets | the same damage as another PDF engine writes it |
| (hard negatives) | text that looks like a result and is not |

Hard negatives carry no labels. The regular expression fires on them, so they
are the cheapest way to teach the model what a result is not.

Family 4 was added after the engine study in section 9. **No model in this
report was trained with it.** It takes effect only on a retrain.

### 6.4 What was trained

Eleven character models, and two transformers. Every run is in
`run_all_training.sh` and can be repeated.

| Run | Configuration |
|---|---|
| final-softmax | no CRF |
| final-crf | CRF |
| final-aug | augmentation, no CRF |
| final-gru | GRU, no CRF |
| final-gru-crf | GRU with CRF |
| final-crf-aug | CRF with augmentation |
| final-gru-aug | GRU with augmentation |
| abl-hardneg-only | CRF, hard negatives only |
| abl-perturb-only | CRF, perturbation only |

The last two are ablations. They separate the two augmentation families.

---

## 7. The pipeline

A PDF goes in, and checked results come out. Six stages, and the order is the
design.

| Stage | What it does |
|---|---|
| 1. extract | A named PDF engine turns the PDF into text. |
| 2. normalize | Line width and damaged operator characters are made engine independent. |
| 3. repair | Damaged operators are restored, and arithmetic decides the mapping. |
| 4. prefilter | Only candidate passages continue. |
| 5. find | The pattern reads what it can, and the model reads the rest. |
| 6. check | The p-value is recomputed. This stage is mathematics alone. |

Every stage records what it did, so a result traces back to the page and the
character it came from.

### 7.1 Why the pattern runs first

The pattern has precision near 1.000. It runs first so that an existing
statcheck user sees no regression. The model then reads only what the pattern
could not.

### 7.2 The repair stage

Publishers destroy the operator, but not the numbers. The repair stage infers
the mapping for a whole document, and **arithmetic decides which mapping is
right**: it tries each candidate operator and keeps the one whose recomputed
p-value agrees with the reported one.

The test is annotator independent. On the holdout, **repair alone lifts the R
package from 59 results to 158**.

**This must be stated plainly: the repair is not machine learning.** It is the
single largest improvement to the regular expression baseline, and it is
deterministic.

### 7.3 Repair and the model do not stack

The model was also tested on repaired text. It performed **worse**, not better.

| System | Input | P | R | F1 |
|---|---|---|---|---|
| lstm-crf | raw | 0.934 | 0.898 | **0.916** |
| lstm-crf | repaired | 0.905 | 0.905 | 0.905 |
| lstm-crf + statcheck | raw | 0.933 | 0.927 | **0.930** |
| lstm-crf + statcheck | repaired | 0.905 | 0.933 | 0.919 |

The repair is not free of error. When it infers an operator wrongly it creates a
plausible result where none exists, and the model then tags it with confidence.

**The final design gives each part the input it reads best: the model reads raw
text, and the repair serves the pattern branch alone.**

### 7.4 The repair is the only reason the pattern branch exists

If the model reads raw text, and the repair only feeds the pattern, then it is
fair to ask what the pattern branch adds. The answer is measured.

| Cascade | Results found | Recall | F2 |
|---|---|---|---|
| crf-aug alone | 287 | 0.911 | 0.911 |
| crf-aug + statcheck on **raw** text | 287 | 0.911 | 0.911 |
| crf-aug + statcheck on **repaired** text | **295** | **0.937** | **0.931** |

**Statcheck reading raw text adds nothing.** It finds 287 results against 287,
and one extra false positive. The model already finds everything the raw
regular expression finds.

Reading repaired text it adds 8 results, for 1 false positive.

The choice is therefore not "repair or no repair". It is "repair and a second
branch, or the model alone":

- **with the repair**: recall 0.937, damaged recall 0.926
- **without it**: recall 0.911, damaged recall 0.877, and the pattern branch
  should be removed as well

### 7.5 The repair works on every engine

The repair looks for a suspect character, and its list of suspects was written
from PyMuPDF text. Poppler writes a destroyed operator as a letter in the Greek
and Coptic block, which is not in that list.

**The normalisation stage saves it.** Stage 2 renames those characters to the
canonical alphabet before stage 3 looks for them. That was an assumption until
it was measured:

| Engine | Operator readable, raw | after normalising | after repair |
|---|---|---|---|
| poppler | 0.134 | 0.134 | **0.899** |
| PyMuPDF | 0.167 | 0.167 | 0.849 |
| PDF.js | 0.160 | 0.160 | 0.840 |
| R pdftools | 0.146 | 0.146 | 0.764 |
| PDFium | 0.163 | 0.163 | **0.602** |

**The repair helps poppler most, which is the engine the R port uses.** It helps
PDFium least, and that is not yet explained.

**Every port therefore carries the repair.** The rules are in `spec/repair.json`.

---

## 8. Metrics

### 8.1 Definitions

A result matches when the value of its statistic matches. Let TP be a matched
result, FP a result the system reported that the labels do not hold, and FN a
labelled result the system did not report.

```
Precision = TP / (TP + FP)      of what the system reported, how much was right
Recall    = TP / (TP + FN)      of what the labels hold, how much was found
F1        = 2PR / (P + R)       precision and recall weighted equally
F2        = 5PR / (4P + R)      recall weighted four times inside the mean
Coverage  = share of windows in which the system reported anything
```

### 8.2 Why F2 leads

**statcheck-ml is a screening tool.** A missed result is invisible to the
reader. A false positive costs a few seconds of a reader's attention. Recall is
therefore worth more than precision, and F2 says so.

F1 is also reported, so a reader who disagrees can use it.

### 8.3 Two warnings about these numbers

**Subset precision is not meaningful.** The scorer adds a false positive to the
overall bucket only. Precision inside the damaged, undamaged and checkable
subsets is therefore 1.000 by construction, and F1 and F2 there are inflated.
**Only recall is real inside a subset**, and only recall is reported.

**The Python port of the regular expression is not the baseline.** It agrees
with the R package on 74% of cases only. Every baseline number in this report
comes from the R package itself.

---

## 9. Results

Holdout: 576 windows, 315 results, 164 damaged, 201 checkable.

Reproduce with:

```
python evaluate_all.py data/holdout/windows.json \
    data/holdout/annotations_final.json data/holdout/statcheck_r.csv \
    models/eval_final_fixed.json <name>=<model.pt>[:crf] ...
```

### 9.1 Overall

| System | P | R | F1 | F2 | Coverage |
|---|---|---|---|---|---|
| **crf-aug + statcheck** | 0.910 | **0.937** | 0.923 | **0.931** | 0.372 |
| lstm-crf + statcheck | 0.933 | 0.927 | **0.930** | 0.928 | 0.356 |
| **gru-crf + statcheck** | 0.947 | 0.911 | 0.929 | 0.918 | 0.351 |
| gru-aug + statcheck | 0.938 | 0.917 | 0.928 | 0.922 | 0.344 |
| abl-perturb + statcheck | 0.953 | 0.908 | 0.930 | 0.917 | 0.342 |
| abl-hardneg + statcheck | 0.971 | 0.848 | 0.905 | 0.870 | 0.318 |
| crf-aug alone | 0.911 | 0.911 | 0.911 | 0.911 | 0.366 |
| gru-crf alone | 0.949 | 0.883 | 0.914 | 0.895 | 0.344 |
| softmax alone | 0.916 | 0.832 | 0.872 | 0.847 | 0.337 |
| **statcheck, repaired text** | 0.994 | 0.502 | 0.667 | 0.557 | 0.184 |
| **statcheck, raw text** | 0.983 | 0.187 | 0.315 | 0.223 | 0.076 |

### 9.2 Recall by subset

Recall only. Section 8.3 explains why.

| System | damaged | undamaged | checkable |
|---|---|---|---|
| crf-aug + statcheck | **0.926** | **0.947** | **0.960** |
| gru-crf + statcheck | 0.914 | 0.908 | **0.960** |
| statcheck, repaired | 0.620 | 0.375 | 0.761 |
| **statcheck, raw** | **0.012** | 0.375 | 0.284 |

**On damaged text the regular expression finds 2 results of 164. The cascade
finds 152.**

### 9.3 Recall by damage family

| Family | n | crf-aug | gru-crf | statcheck repaired | statcheck raw |
|---|---|---|---|---|---|
| control character | 80 | 0.850 | 0.850 | 0.812 | 0.000 |
| fraction sign for = | 36 | **1.000** | **1.000** | 0.611 | 0.000 |
| letter b for < | 18 | **1.000** | **1.000** | 0.500 | 0.000 |
| letter p or ! for operator | 10 | **1.000** | 0.900 | 0.000 | 0.000 |
| other damage | 8 | **1.000** | **1.000** | 0.000 | 0.000 |
| decimal point lost | 5 | **1.000** | 0.800 | 0.000 | 0.000 |
| backslash for < | 4 | **1.000** | **1.000** | 0.750 | 0.000 |
| chi-square symbol lost | 2 | **1.000** | **1.000** | **1.000** | **1.000** |

The families with few members carry little weight. The control character family
is the largest and the hardest, and it is where the remaining loss sits.

### 9.4 Recall by kind of test

| Test | n | crf-aug | gru-crf | statcheck repaired | statcheck raw |
|---|---|---|---|---|---|
| t | 106 | 0.896 | 0.915 | 0.613 | 0.245 |
| F | 104 | 0.981 | 0.981 | 0.683 | 0.240 |
| r | 64 | 0.906 | 0.812 | 0.078 | 0.031 |
| chi-square | 33 | **1.000** | 0.939 | 0.424 | 0.152 |
| z | 5 | **1.000** | 0.800 | 0.600 | 0.200 |

**The correlation is where the regular expression fails hardest**: it finds 0.031
of them, against 0.906 for the model. A correlation is often written without
degrees of freedom, and the pattern needs them.

### 9.5 The transformers

MobileBERT and DistilBERT reached dev F1 0.333. This is not overfitting and not
a training fault. It is the opposite: the loss fell to 0.04 while precision
stayed at 0.26.

**A word-piece tokenizer deletes control characters before it splits anything.**
In this corpus a control character is a destroyed operator.

```
F(1, 17) <U+0003> 35.72   ->   'F' '(' '1' ',' '17' ')' '35' '.' '72'
```

The operator is covered by no token, cased or uncased. The transformer therefore
read `F(1,17) < 35.72` and `F(1,17) = 35.72` as the same string, on half of the
holdout.

`train_bert.py` now swaps each destroyed operator for a visible character that
both tokenizers keep, and uses a cased DistilBERT. **No transformer in this
report was retrained with that fix.** Treat the 0.333 as a measurement of
word-piece tokenization on damaged text, not as the ceiling of a transformer.

This result supports the choice of a character model with evidence rather than
assertion.

---

## 9.6 Where a result is lost, on real PDF files

Section 9 scores the model on windows prepared in advance. That measurement
cannot see a result the prefilter never turned into a window, because such a
result is not in the file the scorer reads.

This ran the whole pipeline on 60 real PDF files holding 97 labelled results.
No document failed.

| Stage | Results | Share |
|---|---|---|
| lost by the engine | 7 | 0.072 |
| **lost by the prefilter** | **0** | **0.000** |
| lost by the finder | 3 | 0.031 |
| **found, but not checkable** | **49** | **0.505** |
| found and checked | 38 | 0.392 |

**The pipeline reports 89.7% of the labelled results and can check 39.2%.**

Two things follow, and both changed what the project worked on next.

**The prefilter loses nothing.** A window that cuts a result in half is a real
failure mode, and it appears in the tutorial, but it did not occur once in 60
real papers. Work planned on it was stopped.

**Finding is close to solved. Checking is not.** The gap is not a defect in the
extractor. It is that the result as printed does not carry enough information
to recompute the p-value.

### The literature, not the tool

Over all 323 labelled results:

| | Results | Share |
|---|---|---|
| can be checked | 210 | 0.650 |
| **no p-value is reported** | **73** | **0.226** |
| no degrees of freedom are reported | 40 | 0.124 |

**More than one labelled result in five carries no p-value beside it.** The
annotators read the passage and recorded no p-value, so this is a statement
about the text as the passage presents it.

Two cautions belong with that number. The labels are machine labels, so the
count carries the annotation ceiling of section 4.3. And the passage is what
the annotator saw: a p-value printed in a table elsewhere in the paper is not
in the passage, and is not counted here.

**The tool never claims the author omitted anything.** At run time it reports
what it observed: `no p-value found beside this result`. Whether the author
omitted the number, the font destroyed it, or the extraction missed it is a
separate question, and the quote beside each result is what answers it.

A correlation is the tempting case. statcheck recomputes one with df = N - 2,
and 26 of the 40 documents state a sample size somewhere. They usually state
several:

```
Cohen_OrgSci_2016      chi2  307        N candidates [18, 31293]
Bogaert_JournManage    Q     1422.527   N candidates [53, 68]
```

**The project does not guess.** Choosing the wrong sample size produces a
confident wrong verdict, and that is worse than reporting nothing. The result
is reported as found and marked `undecidable`.

---

## 10. Portability across PDF engines

Each port gets a different PDF engine, and the engines do not return the same
text. **No engine serves all three ports.**

| Engine | Python | R | Browser | Licence |
|---|---|---|---|---|
| PyMuPDF | yes | no | through WebAssembly | AGPL-3.0 or commercial |
| PDFium | yes, `pypdfium2` | no | through WebAssembly | BSD-3-Clause |
| poppler | yes | yes, `pdftools` | no | GPL-2 |
| PDF.js | no | no | yes, and free | Apache-2.0 |

R is the constraint. `pdftools` bundles poppler, measured as version 26.01.0.
`pdf_text()` is **not** the same as the `pdftotext` command line: it reads more,
0.916 against 0.902 on the 131 documents both read. Measure the R port with
`pdf_text()`.

### 10.1 The chosen engine for each port

| Port | Engine | Reason |
|---|---|---|
| Python | PyMuPDF | Reads the most, and the models were trained on it. |
| R | pdftools | R has no other maintained reader. |
| Browser | PDF.js | Needs no second binary to download. |

**PDFium is the permissive alternative to PyMuPDF**, at 0.916 against 0.920. Use
it when the AGPL licence does not suit distribution.

### 10.2 The normalisation stage

The stage runs directly after the engine. It adds and removes no text.

1. **Reflow.** A line longer than 180 characters is cut near 90 characters.
   Poppler returns a whole column as one line, and every prefilter rule measures
   one line.
2. **Canonicalise.** A character is renamed only when **the model cannot read
   it** and it stands where an operator belongs.

The second rule took three attempts to get right. See section 12.2.

### 10.3 Prefilter recall by engine

198 documents, 323 results. Reproduce with `bench_engines.py`.

| Engine | Holds the result | Before the stage | After |
|---|---|---|---|
| PyMuPDF | 0.929 | 0.929 | **0.929** |
| PDF.js | 0.926 | 0.907 | **0.923** |
| PDFium | 0.920 | 0.907 | **0.916** |
| R pdftools | 0.916 | — | **0.912** |
| poppler | 0.889 | 0.793 | **0.885** |
| **spread** | | **0.136** | **0.043** |

### 10.4 What the model finds, by engine

55 documents, document level recall. This is a different measurement from the
table above, and the two must not be compared directly.

| Engine | Raw text | Normalised |
|---|---|---|
| PDF.js | 0.901 | **0.901** |
| PDFium | 0.890 | **0.890** |
| PyMuPDF | 0.879 | **0.879** |
| poppler | 0.791 | **0.846** |
| **spread** | **0.110** | **0.055** |

**No engine loses anything, and poppler gains 0.055.**

### 10.5 A warning about the spread

`bench_engines.py` gates the ports on the spread between the best engine and the
worst. **A smaller spread is not always better.** An earlier and faulty version
of the stage showed a smaller spread, 0.044, only because it pulled PyMuPDF and
PDF.js down to meet poppler.

**Read the spread beside the absolute numbers, never alone.**

### 10.6 The window must hold text, not lines

A fixed count of lines is not a fixed amount of context.

| Engine | Mean characters in a window at 2 lines |
|---|---|
| PyMuPDF | 298 |
| poppler | 330 |
| PDFium | 202 |
| **PDF.js** | **175** |

PDF.js breaks a page into short lines, so two lines each side cut the result in
half. A window now grows until it holds **250 characters**.

The density rule was not the cause. **Removing it entirely recovers one result,
and costs about 8000 extra windows for it.**

---

## 11. Reproducing this work

```
# validate every agent, skill and command
node scripts/list-extensions.js

# the three ports agree
cd statcheck-ml
node tests/parity.mjs
Rscript tests/parity.R

# each port loads and runs
node tests/smoke.mjs
Rscript tests/smoke.R

# every engine, and the spread between them
python bench_engines.py <pdf_dir> <key.json> <labels.json> --text-dir <dir>

# the whole training chain
bash run_all_training.sh

# the comparison tables in section 9
python evaluate_all.py <windows.json> <labels.json> <statcheck.csv> <out.json> \
    <name>=<model.pt>[:crf] ...
```

**The three ports read one definition of every shared rule**, in
`src/statcheck_ml/spec/`. Never restate a rule in a port.

The parity tests read committed cases, so they need no corpus and no Python.
They earned their place at once: they caught three real port faults, described
in section 12.3.

---

## 12. What went wrong, and what it teaches

This section exists because a reader who repeats the work needs the mistakes as
much as the results.

### 12.1 A reported number was wrong

An early report said that 90% of operators were damaged by a backslash. The true
value was **0.5%**. The number came from a shell escaping artifact in a
measurement script.

### 12.2 Three wrong diagnoses of one regression

The normalisation stage cost PyMuPDF 0.011 at the model, while the prefilter
showed no loss at all.

1. The first guess was the character encoding. A test recovered nothing.
2. The second guess was the reflow. A gate on it changed nothing.
3. The answer came from **listing every character the stage changed**. The rule
   renamed by position alone, so it caught the copyright sign, the
   multiplication sign, the curly quotes and the significance star of
   `* p < .05`. **91% of the renames on PyMuPDF were of characters the model
   already knew.**

The gate added during attempt 2 was later measured, found to change nothing, and
**removed**.

Two lessons:

- **A stage measured at one point in the pipeline can cost recall at the next
  one.** Measure the model, not only the filter.
- **When two guesses in a row are wrong, stop guessing.** List what the code
  changed.

### 12.3 The port faults the parity test caught

- JavaScript searched one character wider than Python for the cut position.
- R kept the space it cut on, because R counts from one and Python from zero.
- R broke a tie with locale collation, so the same input gave different results
  on different machines.

**An unverified port is worse than no port.**

### 12.4 The labels, not the model, set the limit

A window grown to 250 characters made every model look worse: F1 fell by 0.034
on average. **Recall did not move at all.** 18 extra detections explain the whole
change, and most of them are real results that the labels do not cover, such as
`F(2, 23) = 60.73, p < .001`.

The labels stop at the edge of the window the annotators saw.

---

## 13. Limitations

1. **There is no gold set.** Every label is a machine label. The ceiling is
   84.1% on training and 90.5% on the holdout.
2. **poppler reads 4 points less of the corpus than PyMuPDF**, and no
   normalisation recovers text the engine never returned. **The R port has a
   lower ceiling than the Python port.** State this in the R documentation.
3. **`pdftools` hangs on some documents**, twice in 198. The R port gives each
   document a time limit. `R.utils` supplies it, and without that package the
   limit does not apply.
4. **The models are trained on PyMuPDF text alone.** Family 4 of the generator
   holds the other engines' alphabets, but no model has been retrained with it.
5. **The repair does not help two damage families.** A lost decimal point and a
   `p` or `!` operator have no anchor for the arithmetic to use.
6. **Critical value tables are a source of false positives.** They should be
   added as hard negatives.
7. **A window can still cut a result in half** when the result is longer than
   the grown window.

---

## 14. What is not built

The models are trained, frozen and measured. **The packages are not.**

| Item | Status |
|---|---|
| Python | Usable. `Pipeline.run_pdf()` runs end to end. |
| ONNX export | **Done and verified.** |
| p-value core in JavaScript and R | **Not started.** A parity suite is required. |
| Browser port | Text and model files only. |
| R port | Text and model files only. |

### 14.1 The ONNX export

| Model | File | Opset | Agreement with PyTorch |
|---|---|---|---|
| crf-aug | 2.47 MB | 17 | every tag, at every length |
| gru-crf | **1.87 MB** | 17 | every tag, at every length |
| lstm-crf | 2.47 MB | 17 | every tag, at every length |

The export is checked at lengths 16, 64, 283, 512 and 1024, and at batch 1 and
batch 3. The largest difference in the scores is 3.2e-05, and no tag changes.

Two faults were found and fixed while exporting:

1. **PyTorch wrote the weights in a separate file**, as `tagger.onnx` beside
   `tagger.onnx.data`. Python loads that pair without trouble. The browser
   runtime must be told about the second file, and a port that copies only the
   `.onnx` loads a model with no weights. The exporter now writes one file, and
   `export_port_kit.py` refuses to copy a split pair.
2. **The CRF is not in the graph**, so a port with only the `.onnx` cannot
   decode at all. The exporter now writes `decoder.json` beside it, holding the
   masked transitions, the start and end scores, and the tag list. A port runs
   a Viterbi pass over those numbers and adds no rule of its own.

Quantisation is off by default. The model is under 3 MB, and one graph that
behaves identically in three runtimes is worth more than a smaller one.

The text layer is the part this work completed. Everything below it, in both
ports, remains.

The p-value core is the right next piece. It unblocks both ports, and it is the
place where a silent mistake would do the most harm, because it decides the
verdict.

---

## 14.2 Two ideas from error analysis, measured and rejected

Reading all 28 false positives and all 28 false negatives suggested two changes.
Both were built, trained and measured. **Neither ships.**

| Run | What changed | R | F2 | FP |
|---|---|---|---|---|
| **`final-crf-aug`** | the baseline | **0.937** | **0.931** | 29 |
| `v2-crf-aug` | both ideas, 400 negatives | 0.873 | 0.892 | 7 |
| `v3-crf-hn150` | both ideas, 150 negatives | 0.879 | 0.894 | 12 |
| `v4-crf-nohn` | the number damage alone | 0.876 | 0.894 | 8 |

The first reading blamed the hard negatives, because suppression is what they
do. `v4-crf-nohn` carries none of them and still loses recall, so the cause is
`damage_number`. It failed at its own purpose: damaged recall fell from 0.877 to
0.816, which is the number it was written to raise.

Both are switched off by default, so a retrain from this repository reproduces
the model that ships. Neither is deleted.

**The hard negatives did work at what they were built for.** False positives
fell from 28 to 6, and `v2-gru-crf-aug` reaches precision 0.986 with 4 false
positives against 16. That is a documented high-precision option, not the
default.

`EXPERIMENTS.md` holds every run, its configuration, and its log.

---

## 15. Summary

| | crf-aug + statcheck | gru-crf + statcheck | statcheck, raw |
|---|---|---|---|
| Precision | 0.910 | 0.947 | 0.983 |
| **Recall** | **0.937** | 0.911 | 0.187 |
| F1 | 0.923 | 0.929 | 0.315 |
| **F2** | **0.931** | 0.918 | 0.223 |
| Damaged recall | **0.926** | 0.914 | **0.012** |
| Parameters | 616,072 | **466,696** | — |

Use **crf-aug** where recall matters, which is the usual case for a screening
tool. Use **gru-crf** in the browser: it is 24% smaller and 0.013 F2 behind.

**Both find about five times as many results as the R package, and 76 times as
many in damaged text.**

### The decision

| Role | Model | Why |
|---|---|---|
| **default** | `final-crf-aug` + statcheck | the highest F2 of every run, 0.931 |
| **browser** | `final-crf-aug` + statcheck | the same model as Python. Measured at 20.4 ms for one window against 17.4 ms for the GRU, which is 0.6 s over a whole document. 600 kB and 0.6 s do not buy back 0.013 of F2. |
| optional | `v2-gru-crf-aug` + statcheck | precision 0.986 and 4 false positives, for a first pass over many papers |

Four later runs tried to improve on the default and none did. Section 14.2 and
`EXPERIMENTS.md` record what was tried and what it measured.

### Read every number beside two facts

1. **The ceiling is the labels.** Every label is a machine label, and the
   measured ceiling is 90.5% on the holdout.
2. **Not every result can be checked, and that is the literature.** More than
   one published result in five carries no p-value. The tool now names the part
   the paper omitted rather than reporting a bare `undecidable`, so an
   incomplete paper is not confused with a tool that failed.
