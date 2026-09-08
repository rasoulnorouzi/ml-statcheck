"""The whole pipeline: a PDF goes in, checked results come out.

Five stages, in this order, and the order is the design:

  1. extract   PyMuPDF turns the PDF into text. Character offsets from this
               step are the coordinate system every later stage uses.
  2. repair    Damaged operators are restored. Publishers set symbols in fonts
               with no ToUnicode map, so the operator can be absent from the
               text. On the holdout this stage alone lifts the pattern
               extractor from 59 results to 158.
  3. prefilter Only about 1 line in 700 holds a result. Everything else is
               dropped, which is what makes the rest affordable in a browser.
  4. find      The pattern reads what it can, and the model reads the rest.
               The pattern goes first because its precision is near 1.000, so
               an existing statcheck user sees no regression.
  5. check     The p-value is recomputed and compared. This step is closed-form
               mathematics and no model output reaches the verdict.

Every stage records what it did, so a result can be traced back to the page
and the character it came from.

Usage:
    from statcheck_ml.pipeline import Pipeline
    report = Pipeline(model_path="models/final-crf/model.pt", use_crf=True).run_pdf("paper.pdf")
"""
from __future__ import annotations

import json
import time
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Dict, List, Optional

from .labels import ENTITY_OPERATOR, ID_TO_TAG, TAG_TO_ID, tags_to_spans
from .prefilter import Prefilter
from .pvalue import CONSISTENT, Check, Result, check


@dataclass
class Found:
    """One result, with where it came from and how it was found."""

    test_type: str
    statistic: Optional[float]
    df1: Optional[float] = None
    df2: Optional[float] = None
    p_operator: Optional[str] = None
    p_value: Optional[float] = None
    quote: str = ""
    source: str = "model"          # "pattern" or "model"
    line: int = -1
    repaired: bool = False
    verdict: Optional[str] = None
    computed_p: Optional[float] = None
    reason: str = ""


def _num(text):
    if text in (None, ""):
        return None
    s = str(text).strip().replace("−", "-").replace("–", "-").lstrip("<>=").strip()
    if s.startswith("."):
        s = "0" + s
    elif s.startswith("-."):
        s = "-0" + s[1:]
    try:
        return float(s)
    except ValueError:
        return None


class Pipeline:
    """Read a document and check every statistical result in it."""

    def __init__(self, model_path: Optional[str] = None, use_crf: bool = False,
                 repair_text: bool = True, use_pattern: bool = True,
                 alpha: float = 0.05):
        self.repair_text = repair_text
        self.use_pattern = use_pattern
        self.alpha = alpha
        self.prefilter = Prefilter()
        self.model = None
        self.vocab = None
        self.use_crf = use_crf
        if model_path:
            self._load_model(model_path)

    def _load_model(self, path: str):
        import torch
        from .model import CharTagger

        ckpt = torch.load(path, weights_only=False)
        self.vocab = ckpt["vocab"]
        self.model = CharTagger(len(self.vocab), len(TAG_TO_ID),
                                tags=list(TAG_TO_ID) if self.use_crf else None)
        self.model.load_state_dict(ckpt["state_dict"])
        self.model.eval()

    # ---------------- stage 1 ----------------

    @staticmethod
    def extract_text(pdf_path: str) -> str:
        """Convert a PDF to text. This defines every later offset."""
        import pymupdf

        doc = pymupdf.open(pdf_path)
        text = "".join(page.get_text() for page in doc)
        doc.close()
        return text

    # ---------------- stage 2 ----------------

    def repair(self, text: str) -> tuple:
        """Restore operators the conversion destroyed."""
        if not self.repair_text:
            return text, {}
        from .repair_validated import repair_validated

        fixed, info = repair_validated(text)
        return fixed, info

    # ---------------- stage 4a ----------------

    @staticmethod
    def find_with_pattern(window_text: str, line: int) -> List[Found]:
        from .extract import extract

        out = []
        for e in extract(window_text):
            out.append(Found(
                test_type=e.test_type, statistic=_num(e.statistic),
                df1=_num(e.df1), df2=_num(e.df2),
                p_operator=e.p_operator, p_value=_num(e.p_value),
                quote=e.raw, source="pattern", line=line))
        return out

    # ---------------- stage 4b ----------------

    def find_with_model(self, window_text: str, line: int) -> List[Found]:
        if self.model is None:
            return []
        import torch

        from .data import encode, normalise

        text = normalise(window_text)
        ids = torch.tensor([encode(text, self.vocab)], dtype=torch.long)
        with torch.no_grad():
            logits = self.model(ids)
            if self.use_crf and self.model.crf is not None:
                mask = torch.ones(1, ids.size(1))
                tags = [ID_TO_TAG[int(t)] for t in self.model.crf.decode(logits, mask)[0]]
            else:
                tags = [ID_TO_TAG[int(t)] for t in logits.argmax(-1)[0]]

        out, current = [], None
        groups = []
        for start, end, label in sorted(tags_to_spans(tags[:len(text)])):
            if label == "TEST" or (label == "STAT" and current and "STAT" in current):
                if current:
                    groups.append(current)
                current = {}
            if current is None:
                current = {}
            current.setdefault(label, (text[start:end], start, end))
        if current:
            groups.append(current)

        for g in groups:
            stat = _num(g.get("STAT", ("",))[0])
            if stat is None:
                continue
            operator = next((ENTITY_OPERATOR[k] for k in g if k.startswith("POP_")), None)
            spans = [v[1] for v in g.values()]
            ends = [v[2] for v in g.values()]
            out.append(Found(
                test_type=(g.get("TEST", ("",))[0] or "t").strip().lower(),
                statistic=stat,
                df1=_num(g.get("DF1", ("",))[0]),
                df2=_num(g.get("DF2", ("",))[0]),
                p_operator=operator,
                p_value=_num(g.get("PVAL", ("",))[0]),
                quote=text[min(spans):max(ends)] if spans else "",
                source="model", line=line))
        return out

    # ---------------- stage 5 ----------------

    def check_one(self, found: Found) -> Found:
        result = Result(test_type=found.test_type, statistic=found.statistic,
                        df1=found.df1, df2=found.df2,
                        p_operator=found.p_operator, p_value=found.p_value)
        p_text = None if found.p_value is None else f"{found.p_value}"
        outcome: Check = check(result, alpha=self.alpha, reported_p_text=p_text)
        found.verdict = outcome.verdict
        found.computed_p = outcome.computed_p
        found.reason = outcome.reason
        return found

    # ---------------- the whole thing ----------------

    def run_text(self, text: str) -> dict:
        started = time.time()
        stages = {}

        fixed, repair_info = self.repair(text)
        stages["repair"] = repair_info

        windows = list(self.prefilter.windows(fixed))
        stages["prefilter"] = {
            "lines": len(fixed.split("\n")),
            "windows_kept": len(windows),
        }

        found: List[Found] = []
        seen = set()
        n_pattern = n_model = 0
        for w in windows:
            hits = self.find_with_pattern(w.text, w.line) if self.use_pattern else []
            for f in hits:
                key = round(f.statistic, 3) if f.statistic is not None else None
                if key in seen:
                    continue
                seen.add(key)
                found.append(f)
                n_pattern += 1
            for f in self.find_with_model(w.text, w.line):
                key = round(f.statistic, 3) if f.statistic is not None else None
                if key is None or key in seen:
                    continue
                seen.add(key)
                found.append(f)
                n_model += 1

        stages["find"] = {"by_pattern": n_pattern, "by_model": n_model}

        checked = [self.check_one(f) for f in found]
        verdicts: Dict[str, int] = {}
        for f in checked:
            verdicts[f.verdict or "unknown"] = verdicts.get(f.verdict or "unknown", 0) + 1
        stages["check"] = verdicts

        return {
            "results": [asdict(f) for f in checked],
            "stages": stages,
            "seconds": round(time.time() - started, 2),
        }

    def run_pdf(self, pdf_path: str) -> dict:
        text = self.extract_text(pdf_path)
        report = self.run_text(text)
        report["source"] = str(pdf_path)
        report["characters"] = len(text)
        return report


def summarise(report: dict) -> str:
    """A short human-readable summary of one document."""
    s = report["stages"]
    lines = [
        f"{report.get('source', 'text')}",
        f"  characters        : {report.get('characters', 0):,}",
        f"  windows kept      : {s['prefilter']['windows_kept']} of "
        f"{s['prefilter']['lines']} lines",
        f"  found by pattern  : {s['find']['by_pattern']}",
        f"  found by model    : {s['find']['by_model']}",
        f"  verdicts          : " + ", ".join(f"{k}={v}" for k, v in s["check"].items()),
        f"  seconds           : {report['seconds']}",
    ]
    if s.get("repair", {}).get("replacements"):
        lines.insert(2, f"  operators repaired: {s['repair']['replacements']}")
    return "\n".join(lines)
