"""Measure the three ports from their own test suites and one timed run.

Each port proves itself in its own repository against the kit's parity file.
This stage collects the evidence into results/ports.json so that the report
can state it: test counts, the mother commit each kit was built from, and the
seconds one port needs for the damaged sample PDF, warm (second of two runs).

Needs R and node, so reproduce.sh does not run it; the JSON is committed.

Usage:
    python pipeline/13_ports.py --ports ../ports --out results/ports.json \
        --rscript "<path to Rscript.exe>"
"""
from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

SAMPLE = REPO / "examples" / "sample_paper_damaged.pdf"
WINDOW = ("The main effect was significant, F(2, 87) = 4.11, p = .03, and the "
          "interaction was not, F(2, 87) = 0.42, p = .66. Follow-up tests gave "
          "t(28) = 2.87, p = .006 for the first group and t(28) = 0.91, p = .37 "
          "for the second. The correlation was r(38) = .42, p = .007, and chi2(1) = 8.69.")[:300]
N_WINDOWS = 100


def run(cmd, cwd, **env):
    import os
    full = {**os.environ, "PYTHONIOENCODING": "utf-8", **env}
    r = subprocess.run(cmd, cwd=cwd, capture_output=True, text=True, encoding="utf-8",
                       errors="replace", env=full)
    if r.returncode != 0:
        raise SystemExit(f"{' '.join(map(str, cmd))} failed in {cwd}:\n{r.stdout[-2000:]}\n{r.stderr[-2000:]}")
    return r.stdout


def kit_commit(manifest_path: Path) -> str:
    return json.loads(manifest_path.read_text(encoding="utf-8"))["mother_commit"]


def python_port() -> dict:
    from statcheck_ml.pipeline import Pipeline
    out = run([sys.executable, "-m", "pytest", "tests/test_parity_self.py", "-q", "-p", "no:warnings"], REPO)
    passed = int(re.search(r"(\d+) passed", out).group(1))
    zoo = REPO / "src" / "statcheck_ml" / "zoo" / "gru-crf"
    pipe = Pipeline(model_path=str(zoo))
    pipe.run_pdf(str(SAMPLE))
    started = time.perf_counter()
    report = pipe.run_pdf(str(SAMPLE))
    seconds_pdf = time.perf_counter() - started
    from statcheck_ml.onnx_runtime import OnnxTagger
    tagger = OnnxTagger(zoo)
    tagger.tag_text(WINDOW)
    started = time.perf_counter()
    for _ in range(N_WINDOWS):
        tagger.tag_text(WINDOW)
    return {"tests_passed": passed, "tests_failed": 0, "engine": "pymupdf",
            "seconds_sample_pdf": round(seconds_pdf, 3),
            "seconds_100_windows": round(time.perf_counter() - started, 2),
            "sample_results": len(report["results"])}


def r_port(port: Path, rscript: str) -> dict:
    script = (
        'r <- testthat::test_local(reporter = "silent"); d <- as.data.frame(r);'
        'library(statcheckml); kit <- sc_kit(); model <- sc_load_model(kit);'
        f'pdf <- "{SAMPLE.as_posix()}"; invisible(sc_check(pdf, kit, model));'
        'tm <- system.time(res <- sc_check(pdf, kit, model));'
        f'w <- rep("{WINDOW}", {N_WINDOWS}); invisible(sc_tag(w[1], kit, model));'
        'tw <- system.time(sc_tag(w, kit, model));'
        'cat(jsonlite::toJSON(list(tests_passed = sum(d$passed), tests_failed = sum(d$failed),'
        ' seconds_sample_pdf = unname(tm[["elapsed"]]), seconds_100_windows = unname(tw[["elapsed"]]),'
        ' sample_results = nrow(res)), auto_unbox = TRUE))'
    )
    out = run([rscript, "-e", script], port)
    data = json.loads(out.strip().splitlines()[-1])
    data["engine"] = "pdftools (poppler)"
    data["kit_mother_commit"] = kit_commit(port / "inst" / "kit" / "manifest.json")
    return data


def web_port(port: Path) -> dict:
    report = port / "vitest-report.json"
    run(["npx.cmd" if sys.platform == "win32" else "npx", "vitest", "run",
         "--reporter=json", f"--outputFile={report}"], port)
    summary = json.loads(report.read_text(encoding="utf-8"))
    report.unlink()
    script = (
        "import('./src/index.js').then(async m => {"
        " const fs = await import('node:fs');"
        " const pdfjs = await import('pdfjs-dist/legacy/build/pdf.mjs');"
        " const kit = await m.loadKit('./kit');"
        " const model = await m.loadModel(kit);"
        f" const data = fs.readFileSync('{SAMPLE.as_posix()}');"
        " await m.checkPdf(data, kit, model, { pdfjs });"
        " const t = process.hrtime.bigint(); const r = await m.checkPdf(data, kit, model, { pdfjs });"
        " const seconds = Number(process.hrtime.bigint() - t) / 1e9;"
        f" const w = {json.dumps(WINDOW)}; await m.tag(w, model);"
        f" const t2 = process.hrtime.bigint(); for (let i = 0; i < {N_WINDOWS}; i++) await m.tag(w, model);"
        " console.log(JSON.stringify({seconds, windows: Number(process.hrtime.bigint() - t2) / 1e9, n: r.results.length})); })"
    )
    timing = json.loads(run(["node", "-e", script], port).strip().splitlines()[-1])
    return {"tests_passed": summary["numPassedTests"], "tests_failed": summary["numFailedTests"],
            "engine": "pdf.js", "seconds_sample_pdf": round(timing["seconds"], 3),
            "seconds_100_windows": round(timing["windows"], 2),
            "sample_results": timing["n"],
            "kit_mother_commit": kit_commit(port / "kit" / "manifest.json")}


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--ports", default="../ports")
    ap.add_argument("--out", default="results/ports.json")
    ap.add_argument("--rscript", required=True)
    args = ap.parse_args()
    ports = (REPO / args.ports).resolve()
    cases = json.loads((REPO / "tests" / "parity_cases.json").read_text(encoding="utf-8"))
    sections = {k: len(v) for k, v in cases["sections"].items()}

    result = {
        "parity_sections": sections,
        "sample_pdf": SAMPLE.name,
        "ports": {
            "python": python_port(),
            "r": r_port(ports / "statcheck-ml-r", args.rscript),
            "web": web_port(ports / "statcheck-ml-web"),
        },
    }
    out = REPO / args.out
    out.write_text(json.dumps(result, indent=1, sort_keys=True), encoding="utf-8")
    for name, p in result["ports"].items():
        print(f"{name:7s} tests {p['tests_passed']} passed, {p['tests_failed']} failed; "
              f"sample PDF {p['seconds_sample_pdf']} s, {p['sample_results']} results; "
              f"100 windows {p['seconds_100_windows']} s")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
