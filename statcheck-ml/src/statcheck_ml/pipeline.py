"""The whole pipeline: a PDF goes in, checked results come out.

Six stages, in this order, and the order is the design:

  1. extract   A PDF engine turns the PDF into text. Character offsets from
               this step are the coordinate system every later stage uses.
               The engine is named, not assumed, because each port gets a
               different one and they do not read the same text.
  2. normalize The line width and the damaged operator characters are made
               engine independent. Without this step the same PDF gives a
               different answer in Python, in R and in a browser. Measured on
               the holdout, it lifts poppler prefilter recall from 0.793 to
               0.879 and leaves PyMuPDF unchanged.
  3. repair    Damaged operators are restored. Publishers set symbols in fonts
               with no ToUnicode map, so the operator can be absent from the
               text. On the holdout this stage alone lifts the pattern
               extractor from 59 results to 158.
  4. prefilter Only about 1 line in 700 holds a result. Everything else is
               dropped, which is what makes the rest affordable in a browser.
  5. find      The pattern reads what it can, and the model reads the rest.
               The pattern goes first because its precision is near 1.000, so
               an existing statcheck user sees no regression.
  6. check     The p-value is recomputed and compared. This step is closed-form
               mathematics and no model output reaches the verdict.

Every stage records what it did, so a result can be traced back to the page
and the character it came from.

Usage:
    from statcheck_ml import Pipeline
    report = Pipeline().run_pdf("paper.pdf")

The model is the one packaged with the install, so no path is needed and no
checkout is needed. A checkout that compares configurations names a zoo
directory instead, such as "models/zoo/gru-softmax". `model_path=None` loads
no model at all, which is how the pattern-only baseline is measured.
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
    #: The statistic and the p-value exactly as the paper printed them. The
    #: comparison allows for the rounding of both, and the decimals live in
    #: the text: a statistic printed 5.10 leaves a tenth of the room one
    #: printed 5.1 does.
    statistic_text: Optional[str] = None
    p_text: Optional[str] = None
    quote: str = ""
    source: str = "model"          # "pattern" or "model"
    line: int = -1
    repaired: bool = False
    verdict: Optional[str] = None
    computed_p: Optional[float] = None
    reason: str = ""
    #: The parts the paper did not report, such as ("p_value",). Empty when the
    #: result was reported in full. A reader uses this to tell an incomplete
    #: paper apart from a tool that failed.
    missing: tuple = ()
    #: [start, end) of the statistic value. The find_* methods give offsets in
    #: the window text; `run_text` shifts them to document offsets.
    statistic_span: Optional[tuple] = None


def bundled_model_path() -> Path:
    """The zoo directory packaged inside this install.

    `pipeline/08_export.py --update-spec` puts the recommended config there,
    so an install carries exactly one model and needs no checkout to find it.
    """
    zoo = Path(__file__).parent / "zoo"
    configs = sorted(d for d in zoo.iterdir()
                     if (d / "tagger.onnx").exists()) if zoo.is_dir() else []
    if len(configs) != 1:
        raise FileNotFoundError(
            f"expected one packaged model under {zoo}, found {len(configs)}")
    return configs[0]


#: `model_path` was not given. The packaged model is then used. An explicit
#: None still loads no model, so the pattern-only baseline stays reachable.
_BUNDLED = "<bundled>"


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


#: A fragment is never checked, so it carries none of the verdict keys.
_CHECK_KEYS = ("verdict", "computed_p", "reason", "missing")


def _as_row(f: Found) -> dict:
    row = asdict(f)
    if row["statistic_span"] is not None:
        row["statistic_span"] = list(row["statistic_span"])
    return row


def _as_fragment(f: Found) -> dict:
    row = _as_row(f)
    for key in _CHECK_KEYS:
        del row[key]
    return row


class Pipeline:
    """Read a document and check every statistical result in it.

    `Pipeline()` uses the model packaged with the install. Pass `model_path`
    to name another one, or None to run the pattern alone.
    """

    def __init__(self, model_path=_BUNDLED, use_crf: bool = False,
                 repair_text: bool = True, use_pattern: bool = True,
                 alpha: float = 0.05, engine: str = "pymupdf",
                 normalize_text: bool = True):
        self.repair_text = repair_text
        self.use_pattern = use_pattern
        self.alpha = alpha
        self.engine = engine
        self.normalize_text = normalize_text
        self.prefilter = Prefilter()
        self.model = None
        self.tagger = None
        self.vocab = None
        #: Which model actually loaded. The caller asks for it by a path that
        #: may be left out, so the answer is worth keeping.
        self.model_path = None
        self.use_crf = use_crf
        if model_path is _BUNDLED:
            model_path = bundled_model_path()
        if model_path:
            self._load_model(str(model_path))

    def _load_model(self, path: str):
        # A zoo directory (tagger.onnx, decoder.json, charmap.json) is what a
        # clean checkout holds; the torch checkpoint is for the training side.
        p = Path(path)
        if not p.exists():
            raise FileNotFoundError(
                f"no model at {path!r}. A relative path such as "
                f"'models/zoo/gru-crf' resolves only inside a checkout of the "
                f"repository. Pipeline() with no model_path uses the model "
                f"packaged with this install")
        self.model_path = str(p)
        run_dir = p if p.is_dir() else (p.parent if p.suffix == ".onnx" else None)
        if run_dir is not None:
            from .onnx_runtime import OnnxTagger
            self.tagger = OnnxTagger(run_dir)
            self.use_crf = self.tagger.has_crf
            self.model, self.vocab = None, self.tagger.vocab
            return

        import torch
        from .data import load_charmap
        from .export import infer_unit
        from .model import CharTagger

        ckpt = torch.load(path, weights_only=False)
        vocab = ckpt.get("vocab")
        if vocab is None:
            # A checkpoint normally embeds the vocabulary it trained with. A
            # checkpoint that does not still needs one, and the run's own
            # charmap.json (beside the checkpoint) is closer to correct than
            # the shared spec file, which may belong to a different run.
            vocab = load_charmap(Path(path).parent)["chars"]
        self.vocab = vocab
        state = ckpt["state_dict"]
        unit = ckpt.get("unit") or infer_unit(state)
        self.model = CharTagger(len(self.vocab), len(TAG_TO_ID), unit=unit,
                                tags=list(TAG_TO_ID) if self.use_crf else None)
        self.model.load_state_dict(state)
        self.model.eval()

    # ---------------- stage 1 ----------------

    #: The engines the ports can use. No engine covers all three ports, so the
    #: engine is named here rather than assumed. Measured recall for each is in
    #: `spec/normalize.json`.
    ENGINES = ("pymupdf", "pdfium", "poppler")

    @staticmethod
    def extract_text(pdf_path: str, engine: str = "pymupdf") -> str:
        """Convert a PDF to text. This defines every later offset.

        PyMuPDF is the default because the models were trained on its output.
        PDFium reads almost as much and has a Python and a JavaScript build,
        which makes it the engine to use when one engine must serve two ports.
        Poppler is what R gets through pdftools, and it reads the least.
        """
        if engine == "pymupdf":
            import pymupdf

            doc = pymupdf.open(pdf_path)
            text = "".join(page.get_text() for page in doc)
            doc.close()
            return text
        if engine == "pdfium":
            import pypdfium2

            doc = pypdfium2.PdfDocument(pdf_path)
            parts = []
            for page in doc:
                page_text = page.get_textpage()
                parts.append(page_text.get_text_range())
                page_text.close()
                page.close()
            doc.close()
            return "".join(parts)
        if engine == "poppler":
            import subprocess

            done = subprocess.run(
                ["pdftotext", "-enc", "UTF-8", str(pdf_path), "-"],
                capture_output=True, timeout=180)
            return done.stdout.decode("utf-8", "replace")
        raise ValueError(f"unknown engine {engine!r}, expected one of "
                         f"{Pipeline.ENGINES}")

    # ---------------- stage 3 ----------------

    def repair(self, text: str) -> tuple:
        """Restore operators the conversion destroyed."""
        if not self.repair_text:
            return text, {}
        from .repair_validated import repair_validated

        fixed, info = repair_validated(text)
        return fixed, info

    # ---------------- stage 5a ----------------

    @staticmethod
    def find_with_pattern(window_text: str, line: int) -> List[Found]:
        from .extract import extract

        out = []
        for e in extract(window_text):
            out.append(Found(
                test_type=e.test_type, statistic=_num(e.statistic),
                df1=_num(e.df1), df2=_num(e.df2),
                p_operator=e.p_operator, p_value=_num(e.p_value),
                statistic_text=e.statistic, p_text=e.p_value,
                quote=e.raw, source="pattern", line=line,
                statistic_span=(e.stat_start, e.stat_end)))
        return out

    # ---------------- stage 5b ----------------

    def find_with_model(self, window_text: str, line: int) -> List[Found]:
        if self.model is None and self.tagger is None:
            return []
        from .data import normalise

        text = normalise(window_text)
        # The statistic offsets are taken in the normalised text and reused in
        # the window text, so normalise must not change the length.
        assert len(text) == len(window_text)
        if self.tagger is not None:
            tags = self.tagger.tag_text(text)
        else:
            import torch

            from .data import encode

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
                # A span with no test name stays nameless. Assuming t turned
                # correct chi-squares into reported errors.
                test_type=(g.get("TEST", ("",))[0] or "").strip().lower() or None,
                statistic=stat,
                df1=_num(g.get("DF1", ("",))[0]),
                df2=_num(g.get("DF2", ("",))[0]),
                p_operator=operator,
                p_value=_num(g.get("PVAL", ("",))[0]),
                statistic_text=g.get("STAT", ("",))[0] or None,
                p_text=g.get("PVAL", ("",))[0] or None,
                quote=text[min(spans):max(ends)] if spans else "",
                source="model", line=line,
                statistic_span=g["STAT"][1:3]))
        return out

    # ---------------- stage 6 ----------------

    def check_one(self, found: Found) -> Found:
        result = Result(test_type=found.test_type, statistic=found.statistic,
                        df1=found.df1, df2=found.df2,
                        p_operator=found.p_operator, p_value=found.p_value)
        p_text = found.p_text
        if p_text is None and found.p_value is not None:
            p_text = f"{found.p_value}"
        outcome: Check = check(result, alpha=self.alpha, reported_p_text=p_text,
                               statistic_text=found.statistic_text)
        found.verdict = outcome.verdict
        found.computed_p = outcome.computed_p
        found.reason = outcome.reason
        found.missing = outcome.missing
        return found

    # ---------------- the whole thing ----------------

    def run_text(self, text: str) -> dict:
        """Run stages 2 to 6 on text and return the checked results.

        Deduplication rule. Every find carries the interval [start, end) of its
        statistic value, counted in characters over the repaired document with
        reference lines blanked: the unit's first-character offset
        plus the position of the value inside the unit (the stat group
        of the pattern, the STAT span of the model). Visit units in
        document order, the pattern before the model in each unit. A
        find is a duplicate when its interval overlaps the interval of a
        find already emitted, and the first find stays. Two results with
        the same value at different places are both kept. A find with no
        interval falls back to its value rounded to three decimals, once
        per document.

        Output. "results" holds the finds with a test name, checked, each
        with a verdict. "fragments" holds the finds with no test name: a
        statistic, maybe a p-value, nothing to recompute a p-value from. A
        fragment has the keys of a result except verdict, computed_p, reason
        and missing, and check_one never sees it. The deduplication above
        runs over both lists together in emission order, so a named find
        that comes first removes a nameless one at the same statistic.
        stages["find"]["fragments"] is the count of the second list.

        The line of a find is the 0-based index of the line that holds
        the first character of its statistic value, counted over the
        line split of the repaired document (the split the prefilter
        uses; a blanked reference line keeps its newline, so the count
        is the same). It applies to results and to fragments. A find
        with no statistic interval keeps the first line of its unit.
        """
        started = time.time()
        stages = {}

        if self.normalize_text:
            from .normalize import normalize

            text, normalize_info = normalize(text)
            stages["normalize"] = normalize_info

        fixed, repair_info = self.repair(text)
        stages["repair"] = repair_info

        windows = list(self.prefilter.windows(fixed))
        units = list(self.prefilter.units(fixed))
        stages["prefilter"] = {
            "lines": len(fixed.split("\n")),
            "windows_kept": len(windows),
            "units_kept": len(units),
            "characters_read": sum(len(u.text) for u in units),
        }

        found: List[Found] = []
        taken: List[tuple] = []     # statistic intervals already emitted
        seen_values = set()         # fallback key, used when no interval exists
        n_pattern = n_model = n_fallback = 0

        def is_duplicate(f: Found) -> bool:
            if f.statistic_span is not None:
                s, e = f.statistic_span
                return any(s < te and ts < e for ts, te in taken)
            return round(f.statistic, 3) in seen_values if f.statistic is not None \
                else None in seen_values

        def remember(f: Found):
            nonlocal n_fallback
            if f.statistic_span is not None:
                taken.append(tuple(f.statistic_span))
            else:
                n_fallback += 1
                seen_values.add(round(f.statistic, 3) if f.statistic is not None
                                else None)

        def place(f: Found, w):
            """Shift the span to document offsets and set the line."""
            if f.statistic_span is None:
                return
            start, end = f.statistic_span
            # The unit text is the lines joined by "\n", so counting newlines
            # before the value gives its line without a lookup table.
            f.line = w.start_line + w.text.count("\n", 0, start)
            f.statistic_span = (start + w.char_start, end + w.char_start)

        for w in units:
            hits = self.find_with_pattern(w.text, w.line) if self.use_pattern else []
            for f in hits:
                place(f, w)
                if is_duplicate(f):
                    continue
                remember(f)
                found.append(f)
                n_pattern += 1
            for f in self.find_with_model(w.text, w.line):
                place(f, w)
                if is_duplicate(f):
                    continue
                remember(f)
                found.append(f)
                n_model += 1

        named = [f for f in found if f.test_type]
        nameless = [f for f in found if not f.test_type]
        stages["find"] = {"by_pattern": n_pattern, "by_model": n_model,
                          "value_key_fallback": n_fallback,
                          "fragments": len(nameless)}

        checked = [self.check_one(f) for f in named]
        verdicts: Dict[str, int] = {}
        incomplete: Dict[str, int] = {}
        for f in checked:
            verdicts[f.verdict or "unknown"] = verdicts.get(f.verdict or "unknown", 0) + 1
            for part in f.missing:
                incomplete[part] = incomplete.get(part, 0) + 1
        stages["check"] = verdicts
        # The parts the tool did not find beside a result. It cannot know why:
        # the author may have omitted the number, the font may have destroyed
        # it, or the extraction may have missed it. The quote beside each result
        # is what tells a reader which.
        stages["not_found"] = incomplete

        return {
            "results": [_as_row(f) for f in checked],
            "fragments": [_as_fragment(f) for f in nameless],
            "stages": stages,
            "seconds": round(time.time() - started, 2),
        }

    def run_pdf(self, pdf_path: str) -> dict:
        text = self.extract_text(pdf_path, self.engine)
        report = self.run_text(text)
        report["source"] = str(pdf_path)
        report["engine"] = self.engine
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
        f"  fragments         : {s['find'].get('fragments', 0)} (no test name)",
        f"  verdicts          : " + ", ".join(f"{k}={v}" for k, v in s["check"].items()),
        f"  parts not found   : " + (", ".join(f"{k}={v}" for k, v in
                                              s.get("not_found", {}).items())
                                     or "none"),
        f"  seconds           : {report['seconds']}",
    ]
    if s.get("repair", {}).get("replacements"):
        lines.insert(2, f"  operators repaired: {s['repair']['replacements']}")
    return "\n".join(lines)
