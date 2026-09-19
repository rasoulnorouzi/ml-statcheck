"""Export every finished grid run to ONNX, and prove each export on real text.

`statcheck_ml.export.export` already checks synthetic ids at several lengths.
This adds the check that matters for the zoo: real dev windows, run through
the torch model and through the ONNX graph, with the tag sequence decoded the
same way a port would decode it (the CRF in `crf.py` when the run has one).

The top three seed-0 configs by dev F1, excluding the augmentation ablation,
are copied into `models/zoo/<config>/` as the four files a port needs:
tagger.onnx, decoder.json, charmap.json, export_report.json.

Usage:
    python pipeline/08_export.py --runs models/runs.json --models models \
        --zoo models/zoo --dev dataset/train.jsonl --splits dataset/splits.json
"""
from __future__ import annotations

import argparse
import glob
import json
import shutil
import statistics
import sys
import tempfile
import time
from pathlib import Path

import numpy as np
import torch

REPO = Path(__file__).resolve().parents[1]     # statcheck-ml
sys.path.insert(0, str(REPO / "src"))

from statcheck_ml.data import (SPEC_CHARMAP_PATH, apply_splits, encode, load_jsonl,
                               load_splits, row_to_example, save_vocab)
from statcheck_ml.export import export as export_model
from statcheck_ml.export import load as load_checkpoint
from statcheck_ml.labels import ID_TO_TAG, tags_to_spans
from statcheck_ml.train import make_batches
from statcheck_ml.train import score as span_score


def resolve(p: str) -> Path:
    path = Path(p)
    return path if path.is_absolute() else REPO / path


def open_session(path, intra_threads=None):
    import onnxruntime as ort

    opts = ort.SessionOptions()
    if intra_threads is not None:
        opts.intra_op_num_threads = intra_threads
    return ort.InferenceSession(str(path), sess_options=opts, providers=["CPUExecutionProvider"])


def dev_parity(model, vocab, session, has_crf, dev_examples, batch_size=32):
    """Compare torch and ONNX on real windows: emission diff and decoded tags.

    The CRF decode is `crf.CRF.decode`, the same one `export.write_decoder`
    reads its numbers from, so this exercises the identical Viterbi pass a
    port runs over `decoder.json` without a second implementation of it.
    """
    worst, total, agree = 0.0, 0, 0
    for ids, _, rows in make_batches(dev_examples, vocab, batch_size, shuffle=False):
        with torch.no_grad():
            torch_logits = model(ids)
        onnx_logits = session.run(["logits"], {"ids": ids.numpy()})[0]
        worst = max(worst, float(np.abs(torch_logits.numpy() - onnx_logits).max()))
        if has_crf and model.crf is not None:
            mask = (ids != 0).float()
            mask[:, 0] = 1.0
            torch_paths = model.crf.decode(torch_logits, mask)
            onnx_paths = model.crf.decode(torch.from_numpy(onnx_logits), mask)
        else:
            torch_paths = torch_logits.argmax(-1).tolist()
            onnx_paths = onnx_logits.argmax(-1).tolist()
        for j, r in enumerate(rows):
            n = len(r["text"])
            total += 1
            if list(torch_paths[j][:n]) == list(onnx_paths[j][:n]):
                agree += 1
    return worst, (agree / total if total else 1.0)


def predict_spans_onnx(session, dev_examples, vocab, has_crf, crf_module, batch_size=32):
    out = {}
    for ids, _, rows in make_batches(dev_examples, vocab, batch_size, shuffle=False):
        logits = session.run(["logits"], {"ids": ids.numpy()})[0]
        if has_crf and crf_module is not None:
            mask = (ids != 0).float()
            mask[:, 0] = 1.0
            paths = crf_module.decode(torch.from_numpy(logits), mask)
        else:
            paths = logits.argmax(-1)
        for j, r in enumerate(rows):
            n = len(r["text"])
            tags = [ID_TO_TAG[int(t)] for t in paths[j][:n]]
            out[r["window_id"]] = tags_to_spans(tags)
    return out


def measure_latency(onnx_path, vocab, dev_examples, n_windows):
    session = open_session(onnx_path, intra_threads=1)
    times = []
    for ex in dev_examples[:n_windows]:
        ids = np.array([encode(ex["text"], vocab)], dtype=np.int64)
        started = time.perf_counter()
        session.run(["logits"], {"ids": ids})
        times.append((time.perf_counter() - started) * 1000)
    return statistics.median(times) if times else None


def process_run(record: dict, models_dir: Path, dev_examples, latency_windows: int) -> dict:
    name = record["name"]
    run_dir = models_dir / name
    model_path = run_dir / "model.pt"
    onnx_path = run_dir / "tagger.onnx"

    model, vocab, unit, has_crf = load_checkpoint(str(model_path))
    export_model(str(model_path), str(onnx_path), quantize=False)
    save_vocab(vocab, run_dir / "charmap.json")

    fp32_session = open_session(onnx_path)
    max_diff, tag_agreement = dev_parity(model, vocab, fp32_session, has_crf, dev_examples)
    parity = bool(max_diff < 1e-4 and tag_agreement == 1.0)

    fp32_spans = predict_spans_onnx(fp32_session, dev_examples, vocab, has_crf, model.crf)
    dev_f1 = span_score(dev_examples, fp32_spans)["overall"]["f1"]

    with tempfile.TemporaryDirectory() as tmp:
        tmp_onnx = Path(tmp) / "tagger.onnx"
        q_report = export_model(str(model_path), str(tmp_onnx), quantize=True)
        quant = q_report["quantized"]
        quant_bytes = int(round(quant["size_mb"] * 1e6))
        qsession = open_session(Path(quant["path"]))
        quant_spans = predict_spans_onnx(qsession, dev_examples, vocab, has_crf, model.crf)
        dev_f1_quant = span_score(dev_examples, quant_spans)["overall"]["f1"]

    latency = measure_latency(onnx_path, vocab, dev_examples, latency_windows)

    return {"name": name, "onnx_bytes": onnx_path.stat().st_size, "quant_bytes": quant_bytes,
            "parity": parity, "max_abs_diff": max_diff, "tag_agreement": tag_agreement,
            "latency_ms_median": latency, "dev_f1": dev_f1, "dev_f1_quant": dev_f1_quant,
            "in_zoo": False}


def check_charmaps_consistent(done: list, models_dir: Path) -> None:
    """Refuse to promote a charmap when the runs disagree about one.

    Every run trains its own vocabulary. `--update-spec` only makes sense when
    every done run happened to see the same characters; if one training
    corpus differed, promoting any single run's charmap would silently change
    what characters the shipped normalisation stage treats as known.
    """
    reference, reference_name, differing = None, None, []
    for record in done:
        path = models_dir / record["name"] / "charmap.json"
        chars = json.loads(path.read_text(encoding="utf-8"))["chars"]
        if reference is None:
            reference, reference_name = chars, record["name"]
        elif chars != reference:
            differing.append(record["name"])
    if differing:
        raise SystemExit(
            f"--update-spec: charmap of {', '.join(differing)} differs from "
            f"{reference_name}; refusing to promote one over the others")


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--runs", default="models/runs.json")
    ap.add_argument("--models", default="models")
    ap.add_argument("--zoo", default="models/zoo")
    ap.add_argument("--dev", default="dataset/train.jsonl")
    ap.add_argument("--splits", default="dataset/splits.json")
    ap.add_argument("--out", default="models/export.json")
    ap.add_argument("--only", default=None)
    ap.add_argument("--latency-windows", type=int, default=100)
    ap.add_argument("--update-spec", action="store_true",
                    help="promote the top-1 zoo config's charmap.json to "
                         "src/statcheck_ml/spec/charmap.json. Off by default: "
                         "this is the one path that changes the shared spec.")
    args = ap.parse_args()

    models_dir = resolve(args.models)
    zoo_dir = resolve(args.zoo)
    out_path = resolve(args.out)

    all_records = json.loads(resolve(args.runs).read_text(encoding="utf-8"))
    done = [r for r in all_records if r.get("status") == "done"]
    if args.only:
        done = [r for r in done if r["name"] == args.only]
    if not done:
        raise SystemExit("no done runs to export")

    paths = sorted(glob.glob(str(resolve(args.dev))))
    rows = load_jsonl(*paths)
    examples = [row_to_example(r) for r in rows]
    splits = load_splits(str(resolve(args.splits)))
    _, dev_examples = apply_splits(examples, splits)
    print(f"dev windows: {len(dev_examples)}")

    reports = []
    by_name = {}
    any_bad = False
    for record in done:
        print(f"exporting {record['name']} ...")
        report = process_run(record, models_dir, dev_examples, args.latency_windows)
        reports.append(report)
        by_name[record["name"]] = report
        if not report["parity"]:
            any_bad = True
        print(f"  parity {report['parity']}  max diff {report['max_abs_diff']:.2e}  "
              f"tag agreement {report['tag_agreement']:.4f}  dev F1 {report['dev_f1']:.3f}  "
              f"quant dev F1 {report['dev_f1_quant']:.3f}  "
              f"latency {report['latency_ms_median']:.2f} ms")

    # The ablation run's config name carries the grid's "noaug" suffix, which
    # keeps it out of the zoo without needing grid.json here.
    seed0 = [r for r in done if r.get("seed") == 0 and "noaug" not in r.get("config", "")]
    top3 = sorted(seed0, key=lambda r: r["dev_f1"] if r["dev_f1"] is not None else -1,
                  reverse=True)[:3]
    zoo_dir.mkdir(parents=True, exist_ok=True)
    for record in top3:
        run_dir = models_dir / record["name"]
        dest = zoo_dir / record["config"]
        dest.mkdir(parents=True, exist_ok=True)
        for fname in ("tagger.onnx", "decoder.json", "charmap.json", "export_report.json"):
            shutil.copy2(run_dir / fname, dest / fname)
        by_name[record["name"]]["in_zoo"] = True
        print(f"zoo: {record['config']} <- {record['name']}")

    if args.update_spec:
        if not top3:
            raise SystemExit("--update-spec: no zoo configs to promote")
        # The no-augmentation ablation sees fewer characters by design, so it is
        # not a candidate and must not veto the promotion.
        check_charmaps_consistent([r for r in done if "noaug" not in r.get("config", "")],
                                  models_dir)
        top1 = top3[0]
        src_charmap = zoo_dir / top1["config"] / "charmap.json"
        SPEC_CHARMAP_PATH.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src_charmap, SPEC_CHARMAP_PATH)
        print(f"updated spec: {SPEC_CHARMAP_PATH} <- {top1['config']} ({top1['name']})")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(reports, indent=1), encoding="utf-8")
    print(f"wrote {out_path}")
    if any_bad:
        sys.exit(1)


if __name__ == "__main__":
    main()
