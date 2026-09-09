"""Write a small paper that the tutorial can use.

The tutorial needs a PDF it can ship. A real article is copyrighted and cannot
go in the repository, so this writes one.

The numbers are chosen, not invented. Every result below was passed through
`check` first, so the notebook shows a known answer:

  six results agree with the recomputed p-value
  one is a decision error: t(46) = 1.80 is reported as p = .04, and the
    recomputed value is .078, so the two disagree about significance
  one cannot be checked at all, because a correlation is reported with no
    degrees of freedom

The paper also carries a reference list, so the prefilter has something to
discard, and a table, so a long line has to be handled.

Usage: python examples/make_sample_paper.py
"""
from __future__ import annotations

from pathlib import Path

TITLE = "Attention and recall under time pressure"
AUTHORS = "A. Example, B. Sample, and C. Fictional"
AFFILIATION = "Department of Nothing in Particular"

BODY = [
    ("Abstract", [
        "We tested whether time pressure changes recall. Ninety-six people took",
        "part. Recall fell under pressure, and the effect held across two",
        "measures. We report every test in full, so that the analysis can be",
        "checked by a reader.",
    ]),
    ("Method", [
        "Ninety-six adults took part, and each of them completed both",
        "conditions. We measured recall as the number of items reported, and",
        "we measured confidence on a seven-point scale. The study was not",
        "preregistered.",
    ]),
    # Each result is written on one line, with its p-value beside it. A result
    # split across a line break is common in real papers, and the window can
    # then cut it in half. That is a real limit of the tool, and it is shown in
    # the notebook on purpose, but it must not be the whole example.
    ("Results", [
        "Reaction times differed between the groups, t(23) = 2.45, p = .022.",
        "The effect of condition on recall was reliable, F(2, 30) = 5.10, p = .012.",
        "The effect did not depend on the order of the blocks.",
        "",
        "Accuracy differed between the conditions, χ2(1, N = 223) = 8.69, p = .003.",
        "Confidence rose with accuracy, r = .42, p = .02, although the two",
        "measures were collected on different days.",
        "",
        "Recall under pressure was lower than recall at rest, t(46) = 1.80, p = .04.",
        "The same pattern appeared in the second sample, F(1, 118) = 9.20, p = .003.",
        "",
        "The manipulation check succeeded, z = 1.96, p = .05. Participants who",
        "noticed the manipulation reported more effort, t(19) = 4.15, p < .001.",
        "",
        # This one is split on purpose, so the notebook can show the limit.
        "Self-reported tiredness rose over the session, F(3, 92) = 2.71,",
        "p = .049, although the measure was collected only twice.",
    ]),
    ("Discussion", [
        "Time pressure lowered recall in both samples. The effect is small and",
        "the design is not conclusive. We report it so that others can test it",
        "in a larger sample.",
    ]),
    ("References", [
        "Example, A., & Sample, B. (2019). Attention under load. Journal of",
        "    Invented Findings, 12(3), 145-162.",
        "Fictional, C. (2021). Recall and time. Review of Nothing, 8(1), 3-19.",
        "Sample, B., Example, A., & Fictional, C. (2020). Measuring effort.",
        "    Bulletin of Made-Up Work, 44(2), 201-218.",
    ]),
]

TABLE = [
    "Table 1. Recall by condition",
    "Condition          N     M      SD     t       p",
    "Rest              48   14.2    3.1    2.45   .022",
    "Pressure          48   11.8    3.6    1.80   .040",
]


def main(out_path: str | None = None) -> str:
    import pymupdf

    out = Path(out_path or Path(__file__).parent / "sample_paper.pdf")

    doc = pymupdf.open()
    page = doc.new_page()
    text = pymupdf.TextWriter(page.rect)

    serif = pymupdf.Font("tiro")
    bold = pymupdf.Font("tibo")

    y = 72.0
    left = 72.0

    text.append((left, y), TITLE, font=bold, fontsize=14)
    y += 22
    text.append((left, y), AUTHORS, font=serif, fontsize=10)
    y += 14
    text.append((left, y), AFFILIATION, font=serif, fontsize=10)
    y += 28

    for heading, lines in BODY:
        if y > 700:
            text.write_text(page)
            page = doc.new_page()
            text = pymupdf.TextWriter(page.rect)
            y = 72.0
        text.append((left, y), heading, font=bold, fontsize=11)
        y += 16
        for line in lines:
            if y > 720:
                text.write_text(page)
                page = doc.new_page()
                text = pymupdf.TextWriter(page.rect)
                y = 72.0
            if line:
                text.append((left, y), line, font=serif, fontsize=10)
            y += 13
        y += 10

        if heading == "Results":
            y += 6
            for line in TABLE:
                text.append((left, y), line, font=serif, fontsize=9)
                y += 12
            y += 10

    text.write_text(page)
    doc.save(str(out))
    doc.close()
    print(f"wrote {out}  ({out.stat().st_size:,} bytes)")
    return str(out)


#: The same results, with the operator destroyed the way a symbol font destroys
#: it. PyMuPDF writes a control character back as a character outside the model
#: vocabulary, which is what a real damaged paper gives, so this exercises the
#: normalisation and the repair.
DAMAGED_LINES = [
    "Reaction times differed between the groups, t(23) \x03 2.45, p \x03 .022.",
    "The effect of condition was reliable, F(2, 30) \x03 5.10, p \x03 .012.",
    "Accuracy differed between conditions, F(1, 118) \x03 9.20, p \x03 .003.",
    "Recall was lower under pressure, t(46) \x03 1.80, p \x03 .04.",
    "Effort was higher in the noticing group, t(19) \x03 4.15, p \x03 .001.",
]


def make_damaged(out_path: str | None = None) -> str:
    """Write the same results with their operators destroyed.

    A clean paper hides what the model is for, because the pattern reads every
    well-formed result and the model only sees what is left. This file is the
    other half of the demonstration.
    """
    import pymupdf

    out = Path(out_path or Path(__file__).parent / "sample_paper_damaged.pdf")
    doc = pymupdf.open()
    page = doc.new_page()
    text = pymupdf.TextWriter(page.rect)
    serif = pymupdf.Font("tiro")
    bold = pymupdf.Font("tibo")

    text.append((72, 72), "A paper whose operators the conversion destroyed",
                font=bold, fontsize=12)
    y = 104.0
    for line in DAMAGED_LINES:
        text.append((72, y), line, font=serif, fontsize=10)
        y += 16
    text.write_text(page)
    doc.save(str(out))
    doc.close()
    print(f"wrote {out}  ({out.stat().st_size:,} bytes)")
    return str(out)


if __name__ == "__main__":
    main()
    make_damaged()
