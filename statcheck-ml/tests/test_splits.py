from statcheck_ml.splits import make_splits


def test_make_splits_is_seeded_and_by_document():
    docs = [f"j/doc{i}.txt" for i in range(100)]
    a = make_splits(docs, dev_share=0.15, seed=0)
    b = make_splits(docs, dev_share=0.15, seed=0)
    assert a == b
    assert sum(v == "dev" for v in a["documents"].values()) == 15
    assert make_splits(docs, dev_share=0.15, seed=1) != a
    assert a["seed"] == 0 and a["dev_share"] == 0.15
