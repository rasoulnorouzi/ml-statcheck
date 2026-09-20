"""The CLI must agree with `Pipeline` and must never load torch for a zoo model."""
import json
import subprocess
import sys
from pathlib import Path

import pytest

from statcheck_ml.cli import default_model_dir, main
from statcheck_ml.pipeline import Pipeline

ZOO = Path(__file__).resolve().parents[1] / "models" / "zoo" / "gru-crf"
PDF = Path(__file__).resolve().parents[1] / "examples" / "sample_paper_damaged.pdf"


def test_default_model_dir_is_packaged_with_the_wheel():
    packaged = default_model_dir()
    assert packaged == Path(__file__).resolve().parents[1] / "src" / "statcheck_ml" / "zoo" / "gru-crf"
    assert packaged.exists()
    for name in ("tagger.onnx", "decoder.json", "charmap.json"):
        assert (packaged / name).exists()


def test_check_with_no_model_flag_uses_the_packaged_model(tmp_path, capsys):
    txt = tmp_path / "paper.txt"
    txt.write_text("t(34) = 2.45, p = .020", encoding="utf-8")

    rc = main(["check", str(txt)])
    assert rc == 0
    lines = capsys.readouterr().out.splitlines()
    assert lines[0].startswith("1 results found")
    assert "consistent" in lines[1]


@pytest.mark.skipif(not ZOO.exists() or not PDF.exists(), reason="zoo or example missing")
def test_check_json_matches_pipeline_directly(capsys):
    rc = main(["check", str(PDF), "--model", str(ZOO), "--json"])
    assert rc == 0
    report = json.loads(capsys.readouterr().out)

    expected = Pipeline(model_path=str(ZOO)).run_pdf(str(PDF))
    # `--json` round-trips through `json.dumps`, which turns a `missing` tuple
    # into a list; round-trip the expectation the same way before comparing.
    expected_results = json.loads(json.dumps(expected["results"], ensure_ascii=False))
    assert report["results"] == expected_results


def test_check_text_output_reports_a_consistent_result(tmp_path, capsys):
    # Matches the corpus statement examples/sample_paper.pdf uses for t(34):
    # the exact reported p that the statistic implies, so the verdict is
    # "consistent" rather than a rounding-driven "inconsistent".
    txt = tmp_path / "paper.txt"
    txt.write_text("t(34) = 2.45, p = .020", encoding="utf-8")

    rc = main(["check", str(txt), "--model", str(ZOO)])
    assert rc == 0
    lines = capsys.readouterr().out.splitlines()
    assert lines[0].startswith("1 results found")
    result_lines = lines[1:]
    assert len(result_lines) == 1
    assert "consistent" in result_lines[0]
    assert "t(34)" in result_lines[0]


def test_cli_never_imports_torch_for_a_zoo_model(tmp_path):
    txt = tmp_path / "paper.txt"
    txt.write_text("t(34) = 2.45, p = .020", encoding="utf-8")

    code = (
        "import statcheck_ml.cli as c, sys; "
        f"c.main(['check', {str(txt)!r}, '--model', {str(ZOO)!r}]); "
        "print('torch' in sys.modules)"
    )
    done = subprocess.run([sys.executable, "-c", code],
                          capture_output=True, text=True, timeout=60)
    assert done.returncode == 0, done.stderr
    assert done.stdout.strip().splitlines()[-1] == "False"
