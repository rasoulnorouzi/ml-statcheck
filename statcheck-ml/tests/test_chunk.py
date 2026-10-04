import importlib.util
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def load_chunk():
    spec = importlib.util.spec_from_file_location("chunk_stage", str(ROOT / "pipeline" / "02_chunk.py"))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def write_records(tmp_path, n):
    records = [{"window_id": f"w{i}", "text": f"t({i}) = {i}.0, p = .05"} for i in range(n)]
    path = tmp_path / "sample.json"
    path.write_text(json.dumps(records), encoding="utf-8")
    return path, records


def test_unpadded_batches_of_ten_keep_order(tmp_path):
    path, records = write_records(tmp_path, 23)
    count, total = load_chunk().chunk(path, tmp_path / "b", 10, unpadded=True)
    assert (count, total) == (3, 23)
    got = []
    for i, size in enumerate([10, 10, 3]):
        batch = json.loads((tmp_path / "b" / f"batch_{i}.json").read_text(encoding="utf-8"))
        assert len(batch) == size
        got += batch
    assert got == records


def test_default_names_stay_padded(tmp_path):
    path, _ = write_records(tmp_path, 5)
    load_chunk().chunk(path, tmp_path / "b", 10)
    assert (tmp_path / "b" / "batch_000.json").exists()
    assert not (tmp_path / "b" / "batch_0.json").exists()
