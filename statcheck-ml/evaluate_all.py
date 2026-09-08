"""Compare every model, alone and inside the cascade, against the real statcheck.

This is the measurement the project exists to produce. It answers one question
for each system: how many of the reported results does it find, and how often
does it reach the right verdict about them.

Metrics, and why each one is here.

  precision, recall, F1   the standard three.
  F2                      recall counted twice as heavily as precision. This
                          tool screens papers for a human to check, so a missed
                          result costs more than one to reject by eye.
  damaged and undamaged   reported apart. More than half the results in the
                          holdout are text whose operator the conversion
                          destroyed, and the whole case for a learned extractor
                          rests on that part.
  by test type            a system can read t tests and miss chi-square.
  checkable               results carrying everything the arithmetic needs. A
                          result that is found but cannot be recomputed does
                          not help a user.
  coverage                the share of passages where a system finds anything.

Usage:
  python evaluate_all.py <windows.json> <labels.json> <statcheck.csv> <out.json> \
      name=path/to/model.pt[:crf] [name=... ...]
"""
from __future__ import annotations

import csv
import json
import sys
from collections import Counter, defaultdict

import torch

from statcheck_ml.data import encode, normalise
from statcheck_ml.labels import TAG_TO_ID, ID_TO_TAG, ENTITY_OPERATOR, tags_to_spans
from statcheck_ml.model import CharTagger
from statcheck_ml.pvalue import Result, check, CONSISTENT


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


def read_json(path):
    with open(path, encoding="utf-8") as fh:
        return json.loads(fh.read(), strict=False)


def group_spans(text, spans):
    """Turn a flat span list into results, one per test name."""
    results, current = [], None
    for start, end, label in sorted(spans):
        if label == "TEST" or (label == "STAT" and current and "STAT" in current):
            if current:
                results.append(current)
            current = {}
        if current is None:
            current = {}
        current.setdefault(label, text[start:end])
    if current:
        results.append(current)
    return results


def build_result(parts):
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


def load_model(spec):
    path, _, flag = spec.partition(":")
    use_crf = flag == "crf"
    ck = torch.load(path, weights_only=False)
    vocab = ck["vocab"]
    model = CharTagger(len(vocab), len(TAG_TO_ID),
                       tags=list(TAG_TO_ID) if use_crf else None)
    model.load_state_dict(ck["state_dict"])
    model.eval()
    return model, vocab, use_crf


def model_predictions(model, vocab, use_crf, windows):
    """Return {window_id: [(Result, p_text), ...]}."""
    out = {}
    with torch.no_grad():
        for w in windows:
            text = normalise(w["text"])
            ids = torch.tensor([encode(text, vocab)], dtype=torch.long)
            logits = model(ids)
            if use_crf and model.crf is not None:
                mask = torch.ones(1, ids.size(1))
                tags = [ID_TO_TAG[int(t)] for t in model.crf.decode(logits, mask)[0]]
            else:
                tags = [ID_TO_TAG[int(t)] for t in logits.argmax(-1)[0]]
            built = []
            for parts in group_spans(text, tags_to_spans(tags[:len(text)])):
                res, ptext = build_result(parts)
                if res is not None:
                    built.append((res, ptext))
            out[w["window_id"]] = built
    return out


def prf(tp, fp, fn):
    p = tp / (tp + fp) if tp + fp else 0.0
    r = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * p * r / (p + r) if p + r else 0.0
    # F2 weights recall four times as heavily inside the harmonic mean, which
    # suits a tool that screens for a human reader.
    f2 = 5 * p * r / (4 * p + r) if (4 * p + r) else 0.0
    return {"precision": p, "recall": r, "f1": f1, "f2": f2,
            "tp": tp, "fp": fp, "fn": fn}


def score_system(found_by_window, gold_by_window, gold_meta, window_ids):
    """Score one system, overall and by subset."""
    buckets = defaultdict(lambda: [0, 0, 0])     # tp, fp, fn
    windows_with_hit = 0

    for wid in window_ids:
        gold = gold_by_window.get(wid, set())
        got = found_by_window.get(wid, set())
        if got:
            windows_with_hit += 1

        for value in got & gold:
            meta = gold_meta.get((wid, value), {})
            buckets["overall"][0] += 1
            buckets["damaged" if meta.get("damaged") else "undamaged"][0] += 1
            buckets[f"test:{meta.get('test_type') or 'unknown'}"][0] += 1
            if meta.get("checkable"):
                buckets["checkable"][0] += 1
        for value in got - gold:
            buckets["overall"][1] += 1
        for value in gold - got:
            meta = gold_meta.get((wid, value), {})
            buckets["overall"][2] += 1
            buckets["damaged" if meta.get("damaged") else "undamaged"][2] += 1
            buckets[f"test:{meta.get('test_type') or 'unknown'}"][2] += 1
            if meta.get("checkable"):
                buckets["checkable"][2] += 1

    out = {k: prf(*v) for k, v in buckets.items()}
    out["coverage"] = windows_with_hit / max(len(window_ids), 1)
    return out


def main():
    windows_path, labels_path, csv_path, out_path = sys.argv[1:5]
    specs = [a for a in sys.argv[5:] if not a.startswith("--repaired=")]
    repaired_csv = next((a.split("=", 1)[1] for a in sys.argv[5:]
                         if a.startswith("--repaired=")), None)

    windows = read_json(windows_path)
    window_ids = [w["window_id"] for w in windows]
    labels = {r["window_id"]: r for r in read_json(labels_path)}

    # --- the labels ---
    gold_by_window, gold_meta = {}, {}
    n_gold = n_damaged = n_checkable = 0
    for wid in window_ids:
        values = set()
        for res in (labels.get(wid, {}).get("results") or []):
            v = as_number(res.get("statistic"))
            if v is None:
                continue
            values.add(v)
            checkable = bool(res.get("statistic")) and bool(res.get("df1")) and bool(res.get("p_value"))
            gold_meta[(wid, v)] = {"damaged": bool(res.get("damaged")),
                                   "test_type": (res.get("test_type") or "").lower(),
                                   "checkable": checkable}
            n_damaged += bool(res.get("damaged"))
            n_checkable += checkable
        gold_by_window[wid] = values
        n_gold += len(values)

    # --- statcheck ---
    sc_by_window = defaultdict(set)
    sc_rows = defaultdict(dict)
    with open(csv_path, encoding="utf-8") as fh:
        for row in csv.DictReader(fh):
            v = as_number(row["test_value"])
            if v is not None:
                sc_by_window[row["window_id"]].add(v)
                sc_rows[row["window_id"]][v] = row

    report = {"windows": len(window_ids), "gold_results": n_gold,
              "gold_damaged": n_damaged, "gold_checkable": n_checkable,
              "systems": {}}
    report["systems"]["statcheck (raw text)"] = score_system(
        sc_by_window, gold_by_window, gold_meta, window_ids)

    # The same package, reading text whose damaged operators were repaired.
    sc_fixed = defaultdict(set)
    if repaired_csv:
        with open(repaired_csv, encoding="utf-8") as fh:
            for row in csv.DictReader(fh):
                v = as_number(row["test_value"])
                if v is not None:
                    sc_fixed[row["window_id"]].add(v)
        report["systems"]["statcheck (repaired)"] = score_system(
            sc_fixed, gold_by_window, gold_meta, window_ids)

    # --- each model, alone and in the cascade ---
    for spec in specs:
        name, _, path = spec.partition("=")
        model, vocab, use_crf = load_model(path)
        preds = model_predictions(model, vocab, use_crf, windows)

        alone = {wid: {round(r.statistic, 4) for r, _ in v} for wid, v in preds.items()}
        report["systems"][name] = score_system(alone, gold_by_window, gold_meta, window_ids)

        # The cascade: statcheck first, the model adds only what it missed.
        # The repaired reading is used when it is available, because that is
        # the cascade a user would actually run.
        base = sc_fixed if sc_fixed else sc_by_window
        hybrid = {}
        from_sc = from_model = 0
        for wid in window_ids:
            s = base.get(wid, set())
            m = alone.get(wid, set())
            hybrid[wid] = s | m
            from_sc += len(s)
            from_model += len(m - s)
        h = score_system(hybrid, gold_by_window, gold_meta, window_ids)
        h["from_statcheck"] = from_sc
        h["from_model"] = from_model
        report["systems"][f"{name}+statcheck"] = h

        # --- verdicts, on results both systems found ---
        agree = disagree = undecidable = 0
        for wid in window_ids:
            rows = sc_rows.get(wid, {})
            for res, ptext in preds.get(wid, []):
                key = round(res.statistic, 4)
                if key not in rows:
                    continue
                ours = check(res, reported_p_text=ptext)
                if ours.computed_p is None:
                    undecidable += 1
                    continue
                theirs = str(rows[key].get("error", "")).strip().upper() in ("TRUE", "1")
                if (ours.verdict != CONSISTENT) == theirs:
                    agree += 1
                else:
                    disagree += 1
        report["systems"][name]["verdicts"] = {
            "agree": agree, "disagree": disagree, "undecidable": undecidable}

    with open(out_path, "w", encoding="utf-8") as fh:
        json.dump(report, fh, indent=1)

    # --- printed tables ---
    print(f"{report['windows']} windows, {n_gold} annotated results "
          f"({n_damaged} damaged, {n_checkable} checkable)\n")
    header = f"{'system':22s} {'P':>6s} {'R':>6s} {'F1':>6s} {'F2':>6s} {'TP':>5s} {'FP':>5s} {'FN':>5s} {'cover':>6s}"
    print("OVERALL"); print(header)
    for name, s in report["systems"].items():
        o = s["overall"]
        print(f"{name:22s} {o['precision']:6.3f} {o['recall']:6.3f} {o['f1']:6.3f} "
              f"{o['f2']:6.3f} {o['tp']:5d} {o['fp']:5d} {o['fn']:5d} {s['coverage']:6.3f}")

    for subset in ("damaged", "undamaged", "checkable"):
        print(f"\n{subset.upper()}")
        print(f"{'system':22s} {'P':>6s} {'R':>6s} {'F1':>6s} {'TP':>5s} {'FN':>5s}")
        for name, s in report["systems"].items():
            if subset not in s:
                continue
            o = s[subset]
            print(f"{name:22s} {o['precision']:6.3f} {o['recall']:6.3f} {o['f1']:6.3f} "
                  f"{o['tp']:5d} {o['fn']:5d}")

    print(f"\nwrote {out_path}")


if __name__ == "__main__":
    main()
