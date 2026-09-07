"""Export the trained tagger to ONNX, and prove the export still works.

An export that loads is not an export that is correct. So this module always
runs the same inputs through the PyTorch model and through the ONNX graph, and
it fails when the two disagree.

The sequence length is a dynamic axis. A model frozen to the calibration length
looks correct in a test and then truncates real documents.

Usage:
    python -m statcheck_ml.export --model statcheck-ml/models/lite/model.pt
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from .labels import TAG_TO_ID
from .model import CharTagger

OPSET = 17


def export(model_path: str, out_path: str, quantize: bool = True,
           check_lengths=(64, 283, 512)) -> dict:
    ckpt = torch.load(model_path, weights_only=False)
    vocab = ckpt["vocab"]
    model = CharTagger(len(vocab), len(TAG_TO_ID))
    model.load_state_dict(ckpt["state_dict"])
    model.eval()

    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)

    example = torch.randint(2, len(vocab), (1, 283), dtype=torch.long)
    torch.onnx.export(
        model, (example,), str(out),
        input_names=["ids"], output_names=["logits"],
        dynamic_axes={"ids": {0: "batch", 1: "time"},
                      "logits": {0: "batch", 1: "time"}},
        opset_version=OPSET,
    )

    import onnxruntime as ort

    session = ort.InferenceSession(str(out), providers=["CPUExecutionProvider"])
    report = {"path": str(out), "opset": OPSET,
              "size_mb": out.stat().st_size / 1e6, "checks": []}

    # The export is only trusted when it matches the source model at several
    # lengths, including one it never saw.
    worst = 0.0
    for n in check_lengths:
        ids = torch.randint(2, len(vocab), (2, n), dtype=torch.long)
        with torch.no_grad():
            expected = model(ids).numpy()
        got = session.run(["logits"], {"ids": ids.numpy()})[0]
        diff = float(np.abs(expected - got).max())
        agree = float((expected.argmax(-1) == got.argmax(-1)).mean())
        worst = max(worst, diff)
        report["checks"].append({"length": n, "max_difference": diff,
                                 "same_tag_share": agree})
        if agree < 1.0:
            raise SystemExit(f"the ONNX graph disagrees with the model at length {n}")
    report["max_difference"] = worst

    if quantize:
        from onnxruntime.quantization import QuantType, quantize_dynamic

        qpath = out.with_name(out.stem + "-int8.onnx")
        quantize_dynamic(str(out), str(qpath), weight_type=QuantType.QInt8)
        qsession = ort.InferenceSession(str(qpath), providers=["CPUExecutionProvider"])
        ids = torch.randint(2, len(vocab), (2, 283), dtype=torch.long)
        with torch.no_grad():
            expected = model(ids).numpy()
        got = qsession.run(["logits"], {"ids": ids.numpy()})[0]
        report["quantized"] = {
            "path": str(qpath), "size_mb": qpath.stat().st_size / 1e6,
            # Quantisation changes the numbers. What matters is how often the
            # chosen tag changes, not how far the scores moved.
            "same_tag_share": float((expected.argmax(-1) == got.argmax(-1)).mean()),
        }

    (out.parent / "export_report.json").write_text(json.dumps(report, indent=1), encoding="utf-8")
    return report


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="statcheck-ml/models/lite/model.pt")
    ap.add_argument("--out", default="statcheck-ml/models/lite/tagger.onnx")
    ap.add_argument("--no-quantize", action="store_true")
    args = ap.parse_args()
    r = export(args.model, args.out, quantize=not args.no_quantize)
    print(f"exported {r['path']}  {r['size_mb']:.2f} MB  opset {r['opset']}")
    for c in r["checks"]:
        print(f"  length {c['length']:5d}: max difference {c['max_difference']:.2e}, "
              f"same tag {100*c['same_tag_share']:.2f}%")
    if "quantized" in r:
        q = r["quantized"]
        print(f"int8: {q['path']}  {q['size_mb']:.2f} MB  "
              f"same tag {100*q['same_tag_share']:.2f}%")


if __name__ == "__main__":
    main()
