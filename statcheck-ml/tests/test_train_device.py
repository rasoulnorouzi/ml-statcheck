"""Device and noise-family switches of the training command."""
import importlib.util
import json
from pathlib import Path

import pytest

pytest.importorskip("torch")

from statcheck_ml.augment import Augmenter
from statcheck_ml.data import load_jsonl
from statcheck_ml.splits import make_splits
from statcheck_ml.train import (FAMILIES, make_augmenter, parse_families,
                                resolve_device, train)

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="module")
def tiny(tmp_path_factory):
    d = tmp_path_factory.mktemp("tiny")
    rows = [r for r in load_jsonl(str(ROOT / "dataset" / "train.jsonl")) if r["results"]][:40]
    (d / "rows.jsonl").write_text(
        "\n".join(json.dumps(r, ensure_ascii=False) for r in rows) + "\n", encoding="utf-8")
    (d / "splits.json").write_text(json.dumps(
        make_splits([r["source_doc"] for r in rows], dev_share=0.25, seed=0)), encoding="utf-8")
    return d


def run(tiny, out, **kw):
    return train([str(tiny / "rows.jsonl")], str(tiny / out), epochs=1, batch_size=8,
                 seed=0, unit="cnn", splits_path=str(tiny / "splits.json"), **kw)


def test_cpu_one_epoch_records_device(tiny):
    report = run(tiny, "cpu", device="cpu")
    assert report["device"] == "cpu"
    assert report["torch"]
    assert report["seed"] == 0
    assert (tiny / "cpu" / "model.pt").exists()


def test_cpu_loss_is_repeatable_and_unchanged_by_the_default(tiny):
    a = run(tiny, "a", device="cpu", augment=1, hard_negatives=3)
    b = run(tiny, "b", augment=1, hard_negatives=3)       # default device
    if not __import__("torch").cuda.is_available():
        assert a["history"][0]["loss"] == b["history"][0]["loss"]
    c = run(tiny, "c", device="cpu", augment=1, hard_negatives=3)
    assert a["history"][0]["loss"] == c["history"][0]["loss"]


def test_checkpoint_loads_on_cpu(tiny):
    import torch
    run(tiny, "ck", device="cpu")
    ck = torch.load(tiny / "ck" / "model.pt", map_location="cpu", weights_only=False)
    assert all(v.device.type == "cpu" for v in ck["state_dict"].values())


def test_families_parse():
    assert parse_families("ocr, unicode") == ["ocr", "unicode"]
    assert parse_families("") == []
    assert parse_families(None) == []


def test_unknown_family_lists_valid_names():
    with pytest.raises(ValueError) as err:
        parse_families("ocr,nonsense")
    assert "nonsense" in str(err.value)
    for name in FAMILIES:
        assert name in str(err.value)


def test_default_matches_plain_augmenter():
    plain, made = Augmenter(seed=3), make_augmenter(3)
    for name in FAMILIES:
        assert getattr(made, name, None) == getattr(plain, name, None)


def test_no_default_families_then_families():
    aug = make_augmenter(0, ["number_damage"], no_default_families=True)
    assert aug.number_damage is True
    for name in ("ocr", "unicode", "operator", "hard_negative", "extra_negatives"):
        assert getattr(aug, name) is False


def test_new_family_is_ignored_before_it_exists(capsys):
    aug = make_augmenter(0, ["font_map"])
    assert getattr(aug, "font_map", True) is True
    if not hasattr(Augmenter(), "font_map"):
        assert "ignored" in capsys.readouterr().out


def test_resolve_device_cpu():
    assert resolve_device("cpu").type == "cpu"
    assert resolve_device("auto").type in ("cpu", "cuda")


def test_grid_only_subset():
    spec = importlib.util.spec_from_file_location("train_grid", ROOT / "pipeline" / "07_train.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    grid = json.loads((ROOT / "pipeline" / "grid.json").read_text(encoding="utf-8"))
    common = mod.override_common(grid["common"], device="cuda")
    assert common[common.index("--device") + 1] == "cuda"
    runs = mod.stage_runs("screen", grid, common, ROOT / "models", {"lstm-crf", "gru-crf"})
    assert [r["config"] for r in runs] == ["lstm-crf", "gru-crf"]
    assert all("--device" in r["args"] for r in runs)
