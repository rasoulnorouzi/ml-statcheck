import pytest
from statcheck_ml.data import apply_splits


def test_apply_splits_by_document():
    splits = {"seed": 0, "dev_share": 0.5, "documents": {"a.txt": "train", "b.txt": "dev"}}
    ex = [{"source_doc": "a.txt", "id": 1}, {"source_doc": "b.txt", "id": 2}]
    train, dev = apply_splits(ex, splits)
    assert [e["id"] for e in train] == [1] and [e["id"] for e in dev] == [2]


def test_apply_splits_missing_document_raises():
    with pytest.raises(KeyError):
        apply_splits([{"source_doc": "zzz.txt"}], {"documents": {}})
