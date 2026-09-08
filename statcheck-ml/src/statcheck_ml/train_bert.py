"""Fine-tune DistilBERT for the same tagging task, as a comparison.

This is a research baseline, not a shipping tier. A transformer cannot run
natively in R, and its tokenizer would have to be ported to JavaScript and to
R as well, which breaks the rule that the prefilter, the vocabulary and the
p-value constants live in one shared specification.

It exists to answer one question: how much accuracy does the small character
model give up?

The comparison is fair only if both models are scored the same way, so the
sub-word predictions are mapped back to character spans before scoring, and
the same span-level metric is used.

Usage:
    python -m statcheck_ml.train_bert --data statcheck-ml/dataset/*.jsonl
"""
from __future__ import annotations

import argparse
import glob
import json
import random
from pathlib import Path
from typing import Dict, List, Sequence

import torch
import torch.nn as nn

from .data import load_jsonl, row_to_example, split_by_document
from .labels import TAG_TO_ID, ID_TO_TAG, tags_to_spans
from .train import score

# Two transformers are compared. MobileBERT is the only one small enough to
# consider shipping; DistilBERT is the accuracy ceiling and nothing more.
MODELS = {
    "distilbert": "distilbert-base-uncased",
    "mobilebert": "google/mobilebert-uncased",
}
MODEL_NAME = MODELS["distilbert"]


def encode_example(ex: dict, tokenizer, max_length: int = 512):
    """Tokenize, and give every sub-word token the tag of its first character.

    `offset_mapping` gives the character range of each token, so the character
    tags produced by the dataset can be carried onto tokens and back again.
    A token that covers several tags takes the tag of its first character,
    which is what the BIOES prefix already encodes.
    """
    enc = tokenizer(ex["text"], return_offsets_mapping=True, truncation=True,
                    max_length=max_length, return_tensors=None)
    labels = []
    for start, end in enc["offset_mapping"]:
        if start == end:                      # a special token
            labels.append(-100)
        else:
            labels.append(TAG_TO_ID.get(ex["tags"][start], 0))
    enc["labels"] = labels
    return enc


def decode_spans(ex: dict, enc, predicted_ids) -> list:
    """Map token predictions back onto character spans."""
    char_tags = ["O"] * len(ex["text"])
    for (start, end), tag_id in zip(enc["offset_mapping"], predicted_ids):
        if start == end:
            continue
        tag = ID_TO_TAG[int(tag_id)]
        if tag == "O":
            continue
        entity = tag.split("-", 1)[1]
        # A token carries one tag, so its characters are rebuilt as a span of
        # that entity. tags_to_spans then merges neighbours.
        for i in range(start, min(end, len(char_tags))):
            char_tags[i] = ("B-" if i == start else "I-") + entity
    return tags_to_spans(char_tags)


def collate(batch, pad_id):
    width = max(len(b["input_ids"]) for b in batch)
    ids = torch.full((len(batch), width), pad_id, dtype=torch.long)
    att = torch.zeros((len(batch), width), dtype=torch.long)
    lab = torch.full((len(batch), width), -100, dtype=torch.long)
    for i, b in enumerate(batch):
        n = len(b["input_ids"])
        ids[i, :n] = torch.tensor(b["input_ids"])
        att[i, :n] = torch.tensor(b["attention_mask"])
        lab[i, :n] = torch.tensor(b["labels"])
    return ids, att, lab


def run(data_globs: Sequence[str], out_dir: str, epochs: int = 6,
        batch_size: int = 8, lr: float = 3e-5, seed: int = 0,
        model_name: str = MODEL_NAME, augment: int = 0,
        hard_negatives: int = 0) -> dict:
    from transformers import AutoModelForTokenClassification, AutoTokenizer

    random.seed(seed)
    torch.manual_seed(seed)

    paths: List[str] = []
    for pattern in data_globs:
        paths.extend(sorted(glob.glob(pattern)))
    examples = [row_to_example(r) for r in load_jsonl(*paths)]
    parts = split_by_document(examples)
    train_set, dev_set = parts["train"], parts["dev"]

    # The same augmentation the character model uses, so the comparison is
    # between architectures and not between training sets.
    if augment or hard_negatives:
        from .augment import Augmenter, check_alignment
        aug = Augmenter(seed=seed)
        extra = aug.expand(train_set, per_example=augment, negatives=hard_negatives)
        broken = check_alignment(extra)
        if broken:
            raise SystemExit(f"augmentation broke {len(broken)} labels")
        print(f"augmentation: {len(extra)} extra windows")
        train_set = list(train_set) + extra

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForTokenClassification.from_pretrained(
        model_name, num_labels=len(TAG_TO_ID))

    n_params = sum(p.numel() for p in model.parameters())
    print(f"windows: train {len(train_set)}, dev {len(dev_set)}")
    print(f"{model_name}: {n_params:,} parameters, {n_params*4/1e6:.0f} MB as float32")

    train_enc = [encode_example(e, tokenizer) for e in train_set]
    dev_enc = [encode_example(e, tokenizer) for e in dev_set]
    pad_id = tokenizer.pad_token_id

    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    best_f1, best_epoch, history = -1.0, -1, []
    for epoch in range(1, epochs + 1):
        model.train()
        order = list(range(len(train_enc)))
        random.shuffle(order)
        total = 0.0
        for i in range(0, len(order), batch_size):
            batch = [train_enc[j] for j in order[i:i + batch_size]]
            ids, att, lab = collate(batch, pad_id)
            opt.zero_grad()
            loss = model(input_ids=ids, attention_mask=att, labels=lab).loss
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            total += loss.detach().item() * len(batch)

        model.eval()
        predicted: Dict[str, list] = {}
        with torch.no_grad():
            for i in range(0, len(dev_enc), batch_size):
                chunk = dev_enc[i:i + batch_size]
                rows = dev_set[i:i + batch_size]
                ids, att, _ = collate(chunk, pad_id)
                logits = model(input_ids=ids, attention_mask=att).logits
                best = logits.argmax(-1)
                for j, (enc, ex) in enumerate(zip(chunk, rows)):
                    predicted[ex["window_id"]] = decode_spans(
                        ex, enc, best[j, :len(enc["input_ids"])].tolist())

        metrics = score(dev_set, predicted)
        f1 = metrics["overall"]["f1"]
        history.append({"epoch": epoch, "loss": total / max(len(order), 1), **metrics["overall"]})
        print(f"  epoch {epoch}  loss {total/max(len(order),1):.4f}  "
              f"dev P {metrics['overall']['precision']:.3f} "
              f"R {metrics['overall']['recall']:.3f} F1 {f1:.3f}")
        if f1 > best_f1:
            best_f1, best_epoch, best_metrics = f1, epoch, metrics

    report = {"model": model_name, "parameters": n_params,
              "size_mb_float32": n_params * 4 / 1e6,
              "best_epoch": best_epoch, "dev": best_metrics, "history": history}
    (out / "report.json").write_text(json.dumps(report, indent=1), encoding="utf-8")
    print(f"\nbest epoch {best_epoch}, dev F1 {best_f1:.3f}")
    return report


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", nargs="+", required=True)
    ap.add_argument("--out", default="statcheck-ml/models/distilbert")
    ap.add_argument("--epochs", type=int, default=6)
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--arch", choices=list(MODELS), default="distilbert")
    ap.add_argument("--augment", type=int, default=0)
    ap.add_argument("--hard-negatives", type=int, default=0)
    args = ap.parse_args()
    run(args.data, args.out, epochs=args.epochs, batch_size=args.batch_size,
        model_name=MODELS[args.arch], augment=args.augment,
        hard_negatives=args.hard_negatives)


if __name__ == "__main__":
    main()
