"""The Python port runs from a zoo directory, which is what a checkout holds."""
from pathlib import Path

import pytest

from statcheck_ml.pipeline import Pipeline, bundled_model_path

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


def test_pipeline_defaults_to_the_packaged_model():
    """An install has no checkout, so the default must not need a path."""
    packaged = bundled_model_path()
    assert (packaged / "tagger.onnx").exists()

    pipe = Pipeline()
    assert pipe.tagger is not None
    assert Path(pipe.model_path) == packaged


def test_pipeline_without_a_model_runs_the_pattern_alone():
    pipe = Pipeline(model_path=None)
    assert pipe.tagger is None and pipe.model is None


def test_a_missing_model_path_names_the_packaged_one():
    with pytest.raises(FileNotFoundError, match="packaged with this install"):
        Pipeline(model_path="models/zoo/no-such-config")
