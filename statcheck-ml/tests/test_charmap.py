import json

from statcheck_ml.data import load_charmap, save_vocab


def test_save_vocab_writes_pad_unk_and_chars(tmp_path):
    vocab = {"\x00": 0, "\x01": 1, "a": 2, "b": 3}
    out = tmp_path / "charmap.json"
    save_vocab(vocab, out)
    payload = json.loads(out.read_text(encoding="utf-8"))
    assert payload["pad_id"] == 0
    assert payload["unk_id"] == 1
    assert payload["chars"] == {"\x00": 0, "\x01": 1, "a": 2, "b": 3}


def test_load_charmap_prefers_model_dir_over_spec(tmp_path, monkeypatch):
    spec_charmap = tmp_path / "spec" / "charmap.json"
    save_vocab({"\x00": 0, "\x01": 1, "x": 2}, spec_charmap)
    monkeypatch.setattr("statcheck_ml.data.SPEC_CHARMAP_PATH", spec_charmap)

    model_dir = tmp_path / "models" / "run-a"
    save_vocab({"\x00": 0, "\x01": 1, "y": 2}, model_dir / "charmap.json")

    # A model directory with its own charmap.json wins.
    preferred = load_charmap(model_dir)
    assert preferred["chars"] == {"\x00": 0, "\x01": 1, "y": 2}


def test_load_charmap_falls_back_to_spec(tmp_path, monkeypatch):
    spec_charmap = tmp_path / "spec" / "charmap.json"
    save_vocab({"\x00": 0, "\x01": 1, "x": 2}, spec_charmap)
    monkeypatch.setattr("statcheck_ml.data.SPEC_CHARMAP_PATH", spec_charmap)

    # A model directory with no charmap.json of its own falls back to spec.
    no_charmap_dir = tmp_path / "models" / "run-b"
    no_charmap_dir.mkdir(parents=True)
    fallback = load_charmap(no_charmap_dir)
    assert fallback["chars"] == {"\x00": 0, "\x01": 1, "x": 2}

    # No model directory at all also falls back to spec.
    default = load_charmap(None)
    assert default["chars"] == {"\x00": 0, "\x01": 1, "x": 2}
