"""The command line port: `statcheck-ml check paper.pdf`.

A thin wrapper over `Pipeline`. It never imports torch: the shipped zoo
loads through `OnnxTagger`, and PyMuPDF is imported only when a PDF is read.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import List, Optional

from .pipeline import Pipeline, bundled_model_path

class _UsageError(Exception):
    pass

def default_model_dir() -> Path:
    """The one model packaged inside the wheel, as a usage error when absent."""
    try:
        return bundled_model_path()
    except FileNotFoundError as err:
        raise _UsageError(str(err)) from err

def _fmt_num(x) -> str:
    if x is None:
        return "NA"
    x = float(x)
    return str(int(x)) if x.is_integer() else f"{x:g}"

def _fmt_result_line(r: dict) -> str:
    df1, df2 = r.get("df1"), r.get("df2")
    if df1 is None:
        test = r["test_type"]
    elif df2 is None:
        test = f"{r['test_type']}({_fmt_num(df1)})"
    else:
        test = f"{r['test_type']}({_fmt_num(df1)}, {_fmt_num(df2)})"
    computed = "NA" if r.get("computed_p") is None else f"{r['computed_p']:.4f}"
    return (f"{r.get('line', -1):>5}  {test} = {_fmt_num(r.get('statistic'))}, "
            f"p {r.get('p_operator') or '='} {_fmt_num(r.get('p_value'))}  {computed}  "
            f"{r.get('verdict') or 'NA'}  [{r.get('source') or '?'}]")

def _header(report: dict) -> str:
    find = report.get("stages", {}).get("find", {})
    counts = ", ".join(f"{k}={v}" for k, v in
                        report.get("stages", {}).get("check", {}).items()) or "none"
    return (f"{len(report['results'])} results found "
            f"({find.get('by_pattern', 0)} pattern, {find.get('by_model', 0)} model) - {counts}")

def _run(path: Path, model: Path, engine: str) -> dict:
    pipe = Pipeline(model_path=str(model), engine=engine)
    suffix = path.suffix.lower()
    if suffix == ".pdf":
        if engine == "pymupdf":
            try:
                import pymupdf  # noqa: F401
            except ImportError as exc:
                raise _UsageError("reading a PDF needs the 'pdf' extra: "
                                   "pip install 'statcheck-ml[pdf]'") from exc
        return pipe.run_pdf(str(path))
    if suffix == ".txt":
        report = pipe.run_text(path.read_text(encoding="utf-8"))
        report["source"] = str(path)
        return report
    raise _UsageError(f"unsupported file type {suffix!r}: expected .pdf or .txt")

def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(prog="statcheck-ml")
    sub = parser.add_subparsers(dest="command", required=True)
    check = sub.add_parser("check", help="find and check statistical results in a file")
    check.add_argument("file", help="a .pdf or a .txt file")
    check.add_argument("--model", default=None,
                        help="a zoo directory (default: the model packaged with this install)")
    check.add_argument("--json", action="store_true", help="print the full report as JSON")
    check.add_argument("--engine", default="pymupdf", choices=list(Pipeline.ENGINES),
                        help="the PDF engine to use (default: pymupdf)")
    args = parser.parse_args(argv)
    # A Windows console defaults to its code page; a chi-square sign in a
    # result line would end the run with an encoding error.
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")

    path = Path(args.file)
    if not path.exists():
        print(f"error: file not found: {path}", file=sys.stderr)
        return 2
    try:
        model = Path(args.model) if args.model else default_model_dir()
        if not model.exists():
            raise _UsageError(f"model path not found: {model}")
        report = _run(path, model, args.engine)
    except _UsageError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2

    if args.json:
        print(json.dumps(report, ensure_ascii=False, indent=1))
        return 0
    print(_header(report))
    for r in report["results"]:
        print(_fmt_result_line(r))
    return 0

if __name__ == "__main__":
    sys.exit(main())
