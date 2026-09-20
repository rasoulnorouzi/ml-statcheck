"""The Python port runs from a zoo directory, which is what a checkout holds."""
from pathlib import Path

import pytest

from statcheck_ml.pipeline import Pipeline

ZOO = Path(__file__).resolve().parents[1] / "models" / "zoo" / "gru-crf"
PDF = Path(__file__).resolve().parents[1] / "examples" / "sample_paper.pdf"


@pytest.mark.skipif(not ZOO.exists() or not PDF.exists(), reason="zoo or example missing")
def test_pipeline_loads_zoo_and_finds_results():
    pipe = Pipeline(model_path=str(ZOO))
    assert pipe.tagger is not None and pipe.model is None
    assert pipe.use_crf is True                      # read from decoder.json
    report = pipe.run_pdf(str(PDF))
    results = report["results"] if isinstance(report, dict) else report.results
    assert len(results) > 0
    assert any(r["source"] == "model" for r in results)
