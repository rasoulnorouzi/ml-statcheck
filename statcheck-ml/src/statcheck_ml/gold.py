import json


def load_split(path):
    with open(path) as f:
        return json.load(f)


def _pid(doc):
    stem = str(doc).rsplit("/", 1)[-1]
    if stem.endswith(".txt"):
        stem = stem[:-4]
    return stem.rsplit("_", 1)[-1]


def assert_not_gold_test(docs, split):
    full = set(split["gold_test"])
    ids = {_pid(d) for d in full}
    for d in docs:
        if str(d) in full or _pid(d) in ids:
            raise ValueError(f"gold test paper in training input: {d}")
