"""Train the character tagger, and measure it honestly.

Two measurement rules matter here.

Character accuracy is not reported. About 95% of characters carry no label, so
a model that predicts nothing scores 95%. Every number below is span level: a
span counts as correct only when its start, its end and its entity all match.

Scores are reported for each entity separately. A model can find every test
name and still miss every p-value, and one overall figure hides that.

Usage:
    python -m statcheck_ml.train --data statcheck-ml/dataset/*.jsonl --epochs 30
"""
from __future__ import annotations

import argparse
import glob
import json
import random
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, List, Sequence

import torch
import torch.nn as nn

from .data import (build_vocab, class_weights, encode, load_jsonl,
                   row_to_example, save_vocab, split_by_document)
from .labels import ENTITIES, TAG_TO_ID, ID_TO_TAG, tags_to_spans
from .model import CharTagger


def make_batches(examples: Sequence[dict], vocab: Dict[str, int],
                 batch_size: int, shuffle: bool = True):
    """Group examples of a similar length, then pad each group."""
    order = sorted(range(len(examples)), key=lambda i: len(examples[i]["text"]))
    groups = [order[i:i + batch_size] for i in range(0, len(order), batch_size)]
    if shuffle:
        random.shuffle(groups)
    for group in groups:
        rows = [examples[i] for i in group]
        width = max(len(r["text"]) for r in rows)
        ids = torch.zeros(len(rows), width, dtype=torch.long)
        tags = torch.full((len(rows), width), -100, dtype=torch.long)
        for j, r in enumerate(rows):
            seq = encode(r["text"], vocab)
            ids[j, :len(seq)] = torch.tensor(seq, dtype=torch.long)
            tags[j, :len(seq)] = torch.tensor([TAG_TO_ID[t] for t in r["tags"]],
                                              dtype=torch.long)
        yield ids, tags, rows


def predict_spans(model: CharTagger, examples: Sequence[dict],
                  vocab: Dict[str, int], batch_size: int = 32) -> Dict[str, list]:
    """Return the spans the model predicts for each window."""
    model.eval()
    out: Dict[str, list] = {}
    with torch.no_grad():
        for ids, _, rows in make_batches(examples, vocab, batch_size, shuffle=False):
            logits = model(ids)
            if model.crf is not None:
                mask = (ids != 0).float()
                mask[:, 0] = 1.0                      # the first step always counts
                paths = model.crf.decode(logits, mask)
                for j, r in enumerate(rows):
                    n = len(r["text"])
                    tags = [ID_TO_TAG[int(t)] for t in paths[j][:n]]
                    out[r["window_id"]] = tags_to_spans(tags)
            else:
                best = logits.argmax(-1)
                for j, r in enumerate(rows):
                    n = len(r["text"])
                    tags = [ID_TO_TAG[int(t)] for t in best[j, :n]]
                    out[r["window_id"]] = tags_to_spans(tags)
    return out


def score(examples: Sequence[dict], predicted: Dict[str, list]) -> dict:
    """Span-level precision, recall and F1, overall and for each entity."""
    tp = Counter(); fp = Counter(); fn = Counter()
    for ex in examples:
        gold = set(tags_to_spans(ex["tags"]))
        pred = set(predicted.get(ex["window_id"], []))
        for s in pred & gold:
            tp[s[2]] += 1
        for s in pred - gold:
            fp[s[2]] += 1
        for s in gold - pred:
            fn[s[2]] += 1

    def prf(t, p, f):
        prec = t / (t + p) if t + p else 0.0
        rec = t / (t + f) if t + f else 0.0
        f1 = 2 * prec * rec / (prec + rec) if prec + rec else 0.0
        return prec, rec, f1

    per_entity = {}
    for e in ENTITIES:
        per_entity[e] = dict(zip(("precision", "recall", "f1"), prf(tp[e], fp[e], fn[e])))
        per_entity[e]["support"] = tp[e] + fn[e]

    overall = dict(zip(("precision", "recall", "f1"),
                       prf(sum(tp.values()), sum(fp.values()), sum(fn.values()))))
    overall["support"] = sum(tp.values()) + sum(fn.values())
    return {"overall": overall, "per_entity": per_entity}


def train(data_globs: Sequence[str], out_dir: str, epochs: int = 30,
          batch_size: int = 16, lr: float = 2e-3, seed: int = 0,
          test_journals: Sequence[str] = (), patience: int = 6,
          use_crf: bool = False, augment: int = 0,
          hard_negatives: int = 0, unit: str = "lstm") -> dict:
    random.seed(seed)
    torch.manual_seed(seed)

    paths: List[str] = []
    for pattern in data_globs:
        paths.extend(sorted(glob.glob(pattern)))
    rows = load_jsonl(*paths)
    examples = [row_to_example(r) for r in rows]

    parts = split_by_document(examples, test_journals=test_journals)
    train_set, dev_set = parts["train"], parts["dev"]
    if not dev_set:
        raise SystemExit("the development split is empty; add more documents")

    # Augmentation is applied to the training part only. Perturbing the
    # development or test parts would measure the generator, not the model.
    if augment or hard_negatives:
        from .augment import Augmenter, check_alignment
        aug = Augmenter(seed=seed)
        extra = aug.expand(train_set, per_example=augment, negatives=hard_negatives)
        broken = check_alignment(extra)
        if broken:
            raise SystemExit(f"augmentation broke {len(broken)} labels; first: {broken[0]}")
        print(f"augmentation: {len(extra)} extra windows "
              f"({hard_negatives} of them hard negatives)")
        train_set = list(train_set) + extra

    vocab = build_vocab(train_set)
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    save_vocab(vocab, Path(__file__).parent / "spec" / "charmap.json")

    tag_list = list(TAG_TO_ID) if use_crf else None
    model = CharTagger(len(vocab), len(TAG_TO_ID), tags=tag_list, unit=unit)
    weights = torch.tensor(class_weights(train_set), dtype=torch.float)
    loss_fn = nn.CrossEntropyLoss(weight=weights, ignore_index=-100)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, mode="max", factor=0.5, patience=2)

    print(f"windows: train {len(train_set)}, dev {len(dev_set)}, "
          f"held-out journals {len(parts['test_unseen_journals'])}")
    print(f"vocabulary: {len(vocab)} characters")
    print(f"model: {model.size_report()}")

    history, best_f1, best_epoch, bad = [], -1.0, -1, 0
    for epoch in range(1, epochs + 1):
        model.train()
        started, total, seen = time.time(), 0.0, 0
        for ids, tags, _ in make_batches(train_set, vocab, batch_size):
            opt.zero_grad()
            logits = model(ids)
            if model.crf is not None:
                mask = (tags != -100).float()
                mask[:, 0] = 1.0
                loss = model.crf(logits, tags, mask)
            else:
                loss = loss_fn(logits.reshape(-1, logits.size(-1)), tags.reshape(-1))
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            opt.step()
            total += loss.detach().item() * ids.size(0)
            seen += ids.size(0)

        metrics = score(dev_set, predict_spans(model, dev_set, vocab))
        f1 = metrics["overall"]["f1"]
        sched.step(f1)
        history.append({"epoch": epoch, "loss": total / max(seen, 1), **metrics["overall"]})
        print(f"  epoch {epoch:3d}  loss {total/max(seen,1):.4f}  "
              f"dev P {metrics['overall']['precision']:.3f} "
              f"R {metrics['overall']['recall']:.3f} "
              f"F1 {f1:.3f}  ({time.time()-started:.0f}s)")

        if f1 > best_f1:
            best_f1, best_epoch, bad = f1, epoch, 0
            torch.save({"state_dict": model.state_dict(), "vocab": vocab,
                        "tags": list(TAG_TO_ID)}, out / "model.pt")
        else:
            bad += 1
            if bad >= patience:
                print(f"  no improvement for {patience} epochs; stopping")
                break

    best = torch.load(out / "model.pt", weights_only=False)
    model.load_state_dict(best["state_dict"])
    final = score(dev_set, predict_spans(model, dev_set, vocab))

    report = {"best_epoch": best_epoch, "crf": bool(model.crf),
              "augment": augment, "hard_negatives": hard_negatives, "unit": unit, "dev": final, "history": history,
              "vocab_size": len(vocab), "parameters": model.n_parameters(),
              "train_windows": len(train_set), "dev_windows": len(dev_set)}
    if parts["test_unseen_journals"]:
        report["unseen_journals"] = score(parts["test_unseen_journals"],
                                          predict_spans(model, parts["test_unseen_journals"], vocab))
    (out / "report.json").write_text(json.dumps(report, indent=1), encoding="utf-8")

    print(f"\nbest epoch {best_epoch}, dev F1 {final['overall']['f1']:.3f}")
    print(f"{'entity':10s} {'P':>6s} {'R':>6s} {'F1':>6s} {'n':>6s}")
    for e, m in final["per_entity"].items():
        print(f"{e:10s} {m['precision']:6.3f} {m['recall']:6.3f} {m['f1']:6.3f} {m['support']:6d}")
    return report


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", nargs="+", required=True)
    ap.add_argument("--out", default="statcheck-ml/models/lite")
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--lr", type=float, default=2e-3)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--hold-out-journals", nargs="*", default=[])
    ap.add_argument("--crf", action="store_true", help="add a CRF above the emissions")
    ap.add_argument("--augment", type=int, default=0,
                    help="perturbed copies to make of each training window")
    ap.add_argument("--hard-negatives", type=int, default=0,
                    help="generated passages that look like results and are not")
    ap.add_argument("--unit", choices=["lstm", "gru"], default="lstm")
    args = ap.parse_args()
    train(args.data, args.out, epochs=args.epochs, batch_size=args.batch_size,
          lr=args.lr, seed=args.seed, test_journals=args.hold_out_journals,
          use_crf=args.crf, augment=args.augment,
          hard_negatives=args.hard_negatives, unit=args.unit)


if __name__ == "__main__":
    main()
