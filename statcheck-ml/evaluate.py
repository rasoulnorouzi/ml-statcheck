"""Head-to-head evaluation: the model against the real statcheck package.

Both systems are scored on the same held-out passages, against the same
adjudicated annotations, with the same metric. Two questions are answered
separately, because a tool can be good at one and useless at the other.

  FINDING   Does the system locate the reported result at all?
            Scored on the value of the test statistic, because statcheck
            reports values and not character offsets.

  CHECKING  Given a result, does the system reach the right verdict about
            whether the reported p-value matches the statistic?
            Scored only on results both systems found, so the comparison is
            about the arithmetic and not about extraction.

The test part is used once. The development part chose when to stop training,
so a score on it is optimistic and is reported separately.

Usage:
  python evaluate.py <dataset glob> <statcheck.csv> <model.pt> [--crf]
"""
from __future__ import annotations

import csv
import glob as globmod
import json
import sys
from collections import Counter, defaultdict

import torch

from statcheck_ml.data import load_jsonl, row_to_example, split_by_document, encode
from statcheck_ml.labels import TAG_TO_ID, ID_TO_TAG, ENTITY_OPERATOR, tags_to_spans
from statcheck_ml.model import CharTagger
from statcheck_ml.pvalue import Result, check, compute_p, CONSISTENT


def as_number(text):
    if text is None:
        return None
    s = str(text).strip().replace("−", "-").replace("–", "-").lstrip("<>=").strip()
    if s.startswith("."):
        s = "0" + s
    elif s.startswith("-."):
        s = "-0" + s[1:]
    try:
        return round(float(s), 4)
    except ValueError:
        return None


def group_spans(text, spans):
    """Turn a flat list of spans into results.

    A new result starts at each TEST span, or at each STAT span when no test
    name was tagged. This is the grouping rule the block layer would replace.
    """
    spans = sorted(spans)
    results, current = [], None
    for start, end, label in spans:
        piece = text[start:end]
        if label == "TEST" or (label == "STAT" and current and "STAT" in current):
            if current:
                results.append(current)
            current = {}
        if current is None:
            current = {}
        current.setdefault(label, piece)
    if current:
        results.append(current)
    return results


def result_from_parts(parts):
    """Build a Result from tagged pieces, or None when it cannot be built."""
    stat = as_number(parts.get("STAT"))
    if stat is None:
        return None, None
    operator = next((ENTITY_OPERATOR[k] for k in parts if k.startswith("POP_")), None)
    return Result(
        test_type=(parts.get("TEST") or "").strip().lower() or "t",
        statistic=stat,
        df1=as_number(parts.get("DF1")),
        df2=as_number(parts.get("DF2")),
        p_operator=operator,
        p_value=as_number(parts.get("PVAL")),
    ), parts.get("PVAL")


def predict(model, examples, vocab, use_crf):
    out = {}
    model.eval()
    with torch.no_grad():
        for ex in examples:
            ids = torch.tensor([encode(ex["text"], vocab)], dtype=torch.long)
            logits = model(ids)
            if use_crf and model.crf is not None:
                mask = torch.ones(1, ids.size(1))
                tags = [ID_TO_TAG[int(t)] for t in model.crf.decode(logits, mask)[0]]
            else:
                tags = [ID_TO_TAG[int(t)] for t in logits.argmax(-1)[0]]
            out[ex["window_id"]] = tags_to_spans(tags[:len(ex["text"])])
    return out


def prf(tp, fp, fn):
    p = tp / (tp + fp) if tp + fp else 0.0
    r = tp / (tp + fn) if tp + fn else 0.0
    f = 2 * p * r / (p + r) if p + r else 0.0
    return p, r, f


def main():
    data_glob, csv_path, model_path = sys.argv[1], sys.argv[2], sys.argv[3]
    use_crf = "--crf" in sys.argv

    rows = load_jsonl(*sorted(globmod.glob(data_glob)))
    by_id = {r["window_id"]: r for r in rows}
    examples = [row_to_example(r) for r in rows]
    parts = split_by_document(examples)

    ck = torch.load(model_path, weights_only=False)
    vocab = ck["vocab"]
    model = CharTagger(len(vocab), len(TAG_TO_ID),
                       tags=list(TAG_TO_ID) if use_crf else None)
    model.load_state_dict(ck["state_dict"])

    # statcheck output, keyed by window
    sc = defaultdict(list)
    with open(csv_path, encoding="utf-8") as fh:
        for row in csv.DictReader(fh):
            sc[row["window_id"]].append(row)

    report = {}
    for split_name in ("dev", "test"):
        split = parts[split_name]
        if not split:
            continue
        predicted = predict(model, split, vocab, use_crf)

        m_tp = m_fp = m_fn = 0
        s_tp = s_fp = s_fn = 0
        gold_total = 0
        model_results, statcheck_results, gold_results = {}, {}, {}

        for ex in split:
            wid = ex["window_id"]
            gold = {as_number(r.get("statistic"))
                    for r in (by_id[wid].get("results") or [])} - {None}
            gold_total += len(gold)
            gold_results[wid] = by_id[wid].get("results") or []

            grouped = group_spans(ex["text"], predicted.get(wid, []))
            mvals = set()
            built = []
            for g in grouped:
                res, ptext = result_from_parts(g)
                if res is not None:
                    mvals.add(round(res.statistic, 4))
                    built.append((res, ptext))
            model_results[wid] = built
            m_tp += len(mvals & gold)
            m_fp += len(mvals - gold)
            m_fn += len(gold - mvals)

            svals = {as_number(r["test_value"]) for r in sc.get(wid, [])} - {None}
            statcheck_results[wid] = sc.get(wid, [])
            s_tp += len(svals & gold)
            s_fp += len(svals - gold)
            s_fn += len(gold - svals)

        mp, mr, mf = prf(m_tp, m_fp, m_fn)
        sp, sr, sf = prf(s_tp, s_fp, s_fn)

        # ---- CHECKING: only where both found the same result ----
        agree = disagree = undecidable = 0
        for wid, built in model_results.items():
            srows = {as_number(r["test_value"]): r for r in statcheck_results.get(wid, [])}
            for res, ptext in built:
                key = round(res.statistic, 4)
                if key not in srows:
                    continue
                ours = check(res, reported_p_text=ptext)
                theirs = srows[key]
                if ours.computed_p is None:
                    undecidable += 1
                    continue
                sc_error = str(theirs.get("error", "")).strip().upper() in ("TRUE", "1")
                ours_error = ours.verdict != CONSISTENT
                if ours_error == sc_error:
                    agree += 1
                else:
                    disagree += 1

        report[split_name] = {
            "windows": len(split), "gold_results": gold_total,
            "model": {"precision": mp, "recall": mr, "f1": mf,
                      "tp": m_tp, "fp": m_fp, "fn": m_fn},
            "statcheck": {"precision": sp, "recall": sr, "f1": sf,
                          "tp": s_tp, "fp": s_fp, "fn": s_fn},
            "verdict_agreement": {"agree": agree, "disagree": disagree,
                                  "undecidable": undecidable},
        }

    print(json.dumps(report, indent=1))

    for name, r in report.items():
        note = "used for early stopping, so optimistic" if name == "dev" else "touched once"
        print(f"\n=== {name.upper()} ({r['windows']} windows, {r['gold_results']} annotated results) "
              f"[{note}] ===")
        print(f"{'FINDING':12s} {'P':>7s} {'R':>7s} {'F1':>7s} {'TP':>6s} {'FP':>6s} {'FN':>6s}")
        for who in ("model", "statcheck"):
            m = r[who]
            print(f"{who:12s} {m['precision']:7.3f} {m['recall']:7.3f} {m['f1']:7.3f} "
                  f"{m['tp']:6d} {m['fp']:6d} {m['fn']:6d}")
        v = r["verdict_agreement"]
        total = v["agree"] + v["disagree"]
        rate = 100 * v["agree"] / total if total else 0.0
        print(f"CHECKING     verdict agreement on shared results: "
              f"{v['agree']}/{total} ({rate:.1f}%), undecidable {v['undecidable']}")


if __name__ == "__main__":
    main()
