"""statcheck-ml: find reported statistical results, then check them.

The package has two halves that must not be confused.

  Finding  is learned. A pattern cannot read a result whose operator was
           destroyed by the conversion from PDF, and 35.5% of results of that
           shape are affected in this corpus.
  Checking is mathematics. It is never learned, and no model output reaches a
           verdict.

`extract` holds the pattern-based baseline. It exists so that improvement can
be measured, and it is never used as a fallback for the model.
"""

from .prefilter import Prefilter, Window
from .pvalue import (Result, Check, compute_p, check,
                     CONSISTENT, INCONSISTENT, DECISION_ERROR, UNDECIDABLE)

__version__ = "0.1.0"

__all__ = [
    "Prefilter", "Window",
    "Result", "Check", "compute_p", "check",
    "CONSISTENT", "INCONSISTENT", "DECISION_ERROR", "UNDECIDABLE",
]
