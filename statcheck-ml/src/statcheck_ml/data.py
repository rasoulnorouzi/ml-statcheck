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


#: The one charmap the ports (JavaScript, R) read. Training never writes here
#: any more, because every run has its own vocabulary and two runs training at
#: once would race to overwrite this file. It changes only through
#: `pipeline/08_export.py --update-spec`, once a run is chosen to ship.
SPEC_CHARMAP_PATH = Path(__file__).parent / "spec" / "charmap.json"


def load_charmap(model_dir: str | Path | None = None) -> dict:
    """Read a character map, preferring a run's own over the shared spec.

    `<model_dir>/charmap.json` is what `save_vocab` writes beside a checkpoint,
    so a model reads exactly the vocabulary it was trained with. When no run
    directory is given, or it has no charmap of its own, this falls back to
    `spec/charmap.json`, the copy the ports read.
    """
    if model_dir is not None:
        candidate = Path(model_dir) / "charmap.json"
        if candidate.exists():
            return json.loads(candidate.read_text(encoding="utf-8"))
    return json.loads(SPEC_CHARMAP_PATH.read_text(encoding="utf-8"))


def load_splits(path: str | Path) -> dict:
    """Read a committed document-level split.

    The file assigns every source document to "train" or "dev" once, so a
    rerun uses exactly the same split rather than a rehash. The holdout is a
    separate file and is never read here.
    """
    return json.loads(Path(path).read_text(encoding="utf-8"))


def apply_splits(examples: Sequence[dict], splits: dict) -> tuple:
    """Assign each example to train or dev by its `source_doc`.

    Every document in `examples` must appear in `splits["documents"]`. One
    missing document means the split file is stale, and training on it
    silently would measure the wrong thing, so this raises instead.
    """
    documents = splits.get("documents", {})
    train, dev = [], []
    for ex in examples:
        doc = ex.get("source_doc")
        if doc not in documents:
            raise KeyError(doc)
        bucket = train if documents[doc] == "train" else dev
        bucket.append(ex)
    return train, dev


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
