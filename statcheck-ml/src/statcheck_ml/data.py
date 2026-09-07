"""Load the dataset, build the character vocabulary, and make batches.

The model reads characters, not word pieces. A word-piece tokenizer splits
statistical notation unpredictably, and it would have to be ported to
JavaScript and to R. A character map is a small JSON file that all three ports
can read, which is why the character model is the one that ships.

The vocabulary is written to `spec/charmap.json` for exactly that reason.
"""
from __future__ import annotations

import json
import unicodedata
from collections import Counter
from pathlib import Path
from typing import Dict, Iterable, List, Sequence

from .labels import ENTITIES, OUTSIDE, TAG_TO_ID, part_label, spans_to_tags

PAD = "\x00"
UNK = "\x01"

# Characters seen fewer times than this are mapped to UNK, so the vocabulary
# stays small enough to ship.
MIN_COUNT = 3


def load_jsonl(*paths: str) -> List[dict]:
    """Read one or more dataset files."""
    rows: List[dict] = []
    for path in paths:
        with open(path, encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if line:
                    # strict=False keeps raw control characters, which are
                    # damaged operators in this corpus and must survive.
                    rows.append(json.loads(line, strict=False))
    return rows


def normalise(text: str) -> str:
    """Fold characters that mean the same thing, and keep everything else.

    Only differences that carry no information are removed, such as the several
    space characters used by typesetters. A control character is kept exactly
    as it is, because it is often a damaged operator and the model must learn
    to read it.
    """
    out = []
    for ch in text:
        if ch in (" ", " ", " ", " ", " "):
            out.append(" ")
        elif unicodedata.category(ch) == "Zs":
            out.append(" ")
        else:
            out.append(ch)
    return "".join(out)


def row_to_example(row: dict) -> dict:
    """Turn one dataset row into text plus a BIOES tag sequence."""
    text = normalise(row["text"])
    spans = []
    for res in row.get("results") or []:
        operator = res.get("p_operator")
        for name, span in (res.get("part_spans") or {}).items():
            entity = part_label(name, operator)
            if entity:
                spans.append((span[0], span[1], entity))
    return {
        "window_id": row["window_id"],
        "text": text,
        "tags": spans_to_tags(len(text), spans),
        "pool": row.get("pool"),
        "journal": row.get("journal"),
        "source_doc": row.get("source_doc"),
    }


def build_vocab(examples: Iterable[dict], min_count: int = MIN_COUNT) -> Dict[str, int]:
    """Build the character map. PAD is 0 and UNK is 1, always."""
    counts = Counter(ch for ex in examples for ch in ex["text"])
    keep = sorted(ch for ch, n in counts.items() if n >= min_count)
    vocab = {PAD: 0, UNK: 1}
    for ch in keep:
        if ch not in vocab:
            vocab[ch] = len(vocab)
    return vocab


def save_vocab(vocab: Dict[str, int], path: str | Path) -> None:
    """Write the character map, so every port reads the same one."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "_comment": "Character map. Python, JavaScript and R must read this file.",
        "pad_id": 0, "unk_id": 1,
        "chars": {ch: i for ch, i in sorted(vocab.items(), key=lambda kv: kv[1])},
    }
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=1), encoding="utf-8")


def encode(text: str, vocab: Dict[str, int]) -> List[int]:
    return [vocab.get(ch, 1) for ch in text]


def split_by_document(examples: Sequence[dict], dev_share: float = 0.15,
                      test_share: float = 0.15,
                      test_journals: Sequence[str] = ()) -> dict:
    """Split into training, development and test parts.

    Whole documents move together. Two windows from one paper share an author,
    a template and a font, so splitting by window leaks the answers and every
    score comes out too high.

    The development part chooses when to stop training. It is therefore not a
    fair test, because the model was selected on it. The test part is touched
    once, at the end, and never guides a decision.

    Named journals are held out on top of that. That part measures whether the
    model reads a venue it has never seen, which a random split cannot show.
    """
    test_journals = set(test_journals)
    held, rest = [], []
    for ex in examples:
        (held if ex.get("journal") in test_journals else rest).append(ex)

    # A stable hash, so a rerun gives the same split.
    import hashlib

    def bucket(doc: str) -> float:
        h = hashlib.sha1(doc.encode()).hexdigest()[:8]
        return int(h, 16) / 0xFFFFFFFF

    def key(ex):
        return ex.get("source_doc") or ex["window_id"]

    train, dev, test = [], [], []
    for ex in rest:
        b = bucket(key(ex))
        if b < dev_share:
            dev.append(ex)
        elif b < dev_share + test_share:
            test.append(ex)
        else:
            train.append(ex)
    return {"train": train, "dev": dev, "test": test,
            "test_unseen_journals": held}


def tag_frequencies(examples: Iterable[dict]) -> Counter:
    """Count how often each tag appears. Used to weight the loss."""
    return Counter(t for ex in examples for t in ex["tags"])


def class_weights(examples: Iterable[dict], cap: float = 40.0) -> List[float]:
    """Weight each tag by the inverse of its frequency.

    Only about 4.5% of characters carry a label, so an unweighted loss is
    minimised by predicting O everywhere. The weight is capped, because a tag
    seen twice would otherwise dominate the gradient.
    """
    freq = tag_frequencies(examples)
    total = sum(freq.values()) or 1
    weights = []
    for tag_id in range(len(TAG_TO_ID)):
        tag = [t for t, i in TAG_TO_ID.items() if i == tag_id][0]
        n = freq.get(tag, 0)
        if n == 0:
            weights.append(1.0)
        else:
            weights.append(min(cap, (total / (len(TAG_TO_ID) * n)) ** 0.5))
    if OUTSIDE in freq:
        weights[TAG_TO_ID[OUTSIDE]] = 1.0
    return weights
