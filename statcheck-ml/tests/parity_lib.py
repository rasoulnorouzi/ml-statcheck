"""Shared functions for `make_parity_cases.py` and `test_parity_self.py`.

Both files call the Python reference through these wrappers and nothing else,
so the generator and the self-test cannot drift apart: if a wrapper changes,
both callers see the same new behaviour on their next run.

Nothing here invents behaviour. Every function is a thin call into
`statcheck_ml`, reshaped only enough to fit the JSON case format documented in
`tests/parity_cases.json`.
"""
from __future__ import annotations

import math
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

ROOT = Path(__file__).resolve().parent.parent
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from statcheck_ml.data import normalise as data_normalise  # noqa: E402
from statcheck_ml.evalutil import build_result, group_spans  # noqa: E402
from statcheck_ml.extract import extract  # noqa: E402
from statcheck_ml.labels import tags_to_spans  # noqa: E402
from statcheck_ml.normalize import normalize  # noqa: E402
from statcheck_ml.onnx_runtime import OnnxTagger  # noqa: E402
from statcheck_ml.pipeline import Pipeline  # noqa: E402
from statcheck_ml.prefilter import Prefilter, Window  # noqa: E402
from statcheck_ml.pvalue import Check, Result, check  # noqa: E402
from statcheck_ml.repair_validated import repair_validated  # noqa: E402

MODEL_DIR = ROOT / "models" / "zoo" / "gru-crf"

#: Copied from `pvalue.check`'s own docstring and the comment above the "="
#: branch, so a port author sees the exact rounding rule without reading the
#: Python source. "reported_p_text is the p-value exactly as written, for
#: example ".03". It is used to learn how many decimals were reported, so the
#: comparison allows for the rounding the author applied." and: "The author
#: rounded. A reported .03 stands for any value that rounds to .03 at the
#: same number of decimals."
PVALUE_ROUNDING_RULE = (
    "reported_p_text is the p-value exactly as written, for example \".03\". "
    "It is used to learn how many decimals were reported, so the comparison "
    "allows for the rounding the author applied. The author rounded. A "
    "reported .03 stands for any value that rounds to .03 at the same "
    "number of decimals."
)

LOGIT_TOLERANCE = 1e-3
LOGIT_NOTE = "a port matches to 1e-3 absolute and every argmax tag exactly"

# ---------------------------------------------------------------- normalise

def normalise_case(text: str) -> str:
    """The text `normalize()` produces, the same function the pipeline uses."""
    return normalize(text)[0]


# ------------------------------------------------------------------- repair

def repair_case(text: str) -> dict:
    """`repair_validated`, reduced to the two fields a port must reproduce."""
    fixed, info = repair_validated(text)
    return {"text": fixed, "replacements": int(info.get("replacements", 0))}


# ------------------------------------------------------------------ extract

def extract_case(text: str) -> List[dict]:
    """The pattern extractor, reduced to the six fields a port must reproduce.

    `extract.Extraction` also carries `n`, `quote` and `block_span`; those are
    left out because the task names only these six.
    """
    out = []
    for e in extract(text):
        out.append({
            "test_type": e.test_type,
            "statistic": e.statistic,
            "df1": e.df1,
            "df2": e.df2,
            "p_operator": e.p_operator,
            "p_value": e.p_value,
        })
    return out


# ---------------------------------------------------------------- prefilter

def window_char_span(text: str, w: Window) -> Tuple[int, int]:
    """The character offsets a `Window` covers in `text`.

    `Prefilter.windows` only carries line numbers (`start_line`, `end_line`),
    because that is all the rest of the pipeline needs. A port's own prefilter
    test wants a character offset instead, so this rebuilds it from the same
    line split the prefilter itself uses (`text.split("\\n")`).
    """
    lines = text.split("\n")
    offsets = [0]
    for ln in lines:
        offsets.append(offsets[-1] + len(ln) + 1)  # +1 for the '\n' cut away
    start = offsets[w.start_line]
    end = offsets[w.end_line] + len(lines[w.end_line])
    return start, end


def prefilter_case(text: str) -> List[dict]:
    pf = Prefilter()
    out = []
    for w in pf.windows(text):
        start, end = window_char_span(text, w)
        out.append({"start": start, "end": end, "line": w.line})
    return out


# -------------------------------------------------------------------- model

_TAGGER: Optional[OnnxTagger] = None


def get_tagger() -> OnnxTagger:
    global _TAGGER
    if _TAGGER is None:
        _TAGGER = OnnxTagger(MODEL_DIR)
    return _TAGGER


def model_input_text(text: str) -> str:
    """The text the model actually sees.

    `evalutil.model_predictions` (used by the phase-11 evaluator) calls
    `data.normalise` -- the whitespace-folding pass, not `normalize.normalize`
    -- on the window text before tagging. This mirrors that exactly, so the
    tags a port computes from the same input match.
    """
    return data_normalise(text)


def model_case(text: str) -> dict:
    """Tags for one window, plus the text the model saw (already normalised)."""
    tagger = get_tagger()
    seen = model_input_text(text)
    tags = tagger.tag_text(seen)
    return {"text": seen, "tags": tags}


def model_logits_case(text: str) -> dict:
    """Raw ONNX logits for one short window, rounded to 5 decimals."""
    tagger = get_tagger()
    seen = model_input_text(text)
    if not seen:
        return {"text": seen, "logits": []}
    ids = tagger.encode(seen)
    logits = tagger.session.run(["logits"], {"ids": ids})[0][0]
    rounded = [[round(float(x), 5) for x in row] for row in logits]
    return {"text": seen, "logits": rounded}


# -------------------------------------------------------------------- group

def group_case(text: str, tags: List[str]) -> List[dict]:
    """`evalutil.group_spans` + `build_result`, applied to a stored tag list.

    `text` must be the text the tags were produced over (the normalised text
    a `model` case stores), because the spans are character offsets into it.
    """
    out = []
    for parts, spans in group_spans(text, tags_to_spans(tags)):
        res, _ptext = build_result(parts)
        if res is None:
            continue
        starts = [s for s, _ in spans]
        ends = [e for _, e in spans]
        out.append({
            "test_type": res.test_type,
            "statistic": res.statistic,
            "df1": res.df1,
            "df2": res.df2,
            "p_operator": res.p_operator,
            "p_value": res.p_value,
            "span": [min(starts), max(ends)],
        })
    return out


# ------------------------------------------------------------------- pvalue

def parse_number(text) -> Optional[float]:
    """Text such as ".03" or "<.001" to a float, or None.

    Mirrors `pipeline._num`: strip a leading comparison operator, restore a
    missing leading zero, accept the Unicode minus and en dash statcheck's
    corpus shows. Kept here rather than imported because `pipeline._num` is a
    private helper, and the case generator must not depend on one module's
    internals to build inputs for a different module's test.
    """
    if text in (None, ""):
        return None
    s = str(text).strip().replace("−", "-").replace("–", "-")
    s = s.lstrip("<>=").strip()
    if s.startswith("."):
        s = "0" + s
    elif s.startswith("-."):
        s = "-0" + s[1:]
    try:
        return float(s)
    except ValueError:
        return None


def pvalue_case(test_type: str, statistic: Optional[float], df1: Optional[float],
                df2: Optional[float], p_operator: Optional[str],
                p_text: Optional[str]) -> Check:
    """Build a `Result` the way a port would from parsed fields, and check it."""
    p_value = parse_number(p_text)
    res = Result(test_type=test_type, statistic=statistic, df1=df1, df2=df2,
                p_operator=p_operator, p_value=p_value)
    return check(res, reported_p_text=p_text)


def pvalue_expected(outcome: Check) -> Optional[dict]:
    """`{"computed_p", "verdict"}`, or None when `computed_p` is NaN.

    `compute_p` can return `nan` rather than `None` -- `scipy.stats.chi2.sf`
    with `df1=0` is the case this file exercises -- and `nan` cannot round-trip
    through JSON. A case whose reference answer is NaN is skipped rather than
    written, per the task instructions.
    """
    if outcome.computed_p is not None and math.isnan(outcome.computed_p):
        return None
    return {"computed_p": outcome.computed_p, "verdict": outcome.verdict}


# ------------------------------------------------------------------ pipeline

_PIPELINE: Optional[Pipeline] = None

PIPELINE_RESULT_KEYS = (
    "source", "test_type", "statistic", "df1", "df2", "p_operator",
    "p_value", "computed_p", "verdict", "line",
)


def get_pipeline() -> Pipeline:
    global _PIPELINE
    if _PIPELINE is None:
        _PIPELINE = Pipeline(model_path=str(MODEL_DIR))
    return _PIPELINE


def pipeline_case(text: str) -> List[dict]:
    report = get_pipeline().run_text(text)
    return [{k: r.get(k) for k in PIPELINE_RESULT_KEYS} for r in report["results"]]
