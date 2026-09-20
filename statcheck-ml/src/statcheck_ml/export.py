"""Export the trained tagger to ONNX, and prove the export still works.

An export that loads is not an export that is correct. This module always runs
the same inputs through the PyTorch model and through the ONNX graph, and it
fails when the two disagree.

Three things travel to a port, not one:

  tagger.onnx     the graph, which turns character ids into tag scores
  decoder.json    the CRF, which turns tag scores into a tag sequence
  charmap.json    the characters, which turn text into character ids

**The CRF is not in the graph.** It is a small matrix and a Viterbi loop, and
every port implements the loop over the numbers in `decoder.json`. Putting it in
the graph would need loop operators that the browser runtime handles poorly.

The sequence length and the batch are dynamic axes. A model frozen to the
calibration length looks correct in a test and then truncates real documents.

Usage:
    python -m statcheck_ml.export --model models/final-crf-aug/model.pt \
        --out models/final-crf-aug/tagger.onnx
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from .labels import ID_TO_TAG, TAG_TO_ID
from .model import CharTagger

#: Opset 17 is understood by onnxruntime-web, by onnxruntime in Python, and by
#: the R binding. Raising it buys nothing here and can break the browser.
OPSET = 17

#: Lengths used to check the export. 283 is the calibration length, and the
#: others are shorter and longer, so a graph frozen to one length fails here.
CHECK_LENGTHS = (16, 64, 283, 512, 1024)


def infer_unit(state_dict) -> str:
    """Recover the unit from the checkpoint's own weight names and shapes.

    A CNN has no recurrent weight at all, only `convs.*` entries, so that is
    checked first. Otherwise a GRU has three gates and an LSTM four, so the
    recurrent weight of the first layer has 3*hidden rows against 4*hidden.
    Older checkpoints do not record the unit, and this makes them loadable
    anyway.
    """
    if any(k.startswith("convs.") for k in state_dict):
        return "cnn"
    weight = state_dict.get("lstm.weight_hh_l0")
    if weight is None:
        return "lstm"
    rows, hidden = weight.shape
    return "gru" if rows == 3 * hidden else "lstm"


def load(model_path: str):
    """Load a checkpoint, whatever unit it uses and whether it has a CRF."""
    ckpt = torch.load(model_path, weights_only=False)
    vocab = ckpt.get("vocab")
    if vocab is None:
        # A checkpoint normally embeds its own vocabulary. When it does not,
        # the run's own charmap.json (beside the checkpoint) is closer to
        # correct than the shared spec file, which may belong to another run.
        from .data import load_charmap

        vocab = load_charmap(Path(model_path).parent)["chars"]
    state = ckpt["state_dict"]
    unit = ckpt.get("unit") or infer_unit(state)
    has_crf = ckpt.get("crf")
    if has_crf is None:
        has_crf = any(k.startswith("crf.") for k in state)
    model = CharTagger(len(vocab), len(TAG_TO_ID), unit=unit,
                       tags=list(TAG_TO_ID) if has_crf else None)
    model.load_state_dict(state)
    model.eval()
    return model, vocab, unit, bool(has_crf)


def write_decoder(model, out_dir: Path, unit: str, has_crf: bool) -> dict:
    """Write the numbers a port needs to turn tag scores into tags.

    The transitions written here already carry the mask, so a port adds nothing
    and only reads. A masked transition is a large negative number, which makes
    a forbidden tag sequence impossible without a rule in the port.
    """
    decoder = {
        "version": 1,
        "_comment": (
            "Written by statcheck_ml.export. The CRF decodes outside the ONNX "
            "graph, so every port runs a Viterbi pass over these numbers. The "
            "transitions already carry the mask: entry [i][j] is the score of "
            "tag j following tag i, and a large negative value forbids it."),
        "unit": unit,
        "tags": [ID_TO_TAG[i] for i in range(len(ID_TO_TAG))],
        "has_crf": has_crf,
    }
    if has_crf and model.crf is not None:
        with torch.no_grad():
            transitions = model.crf._transitions()
            decoder["transitions"] = transitions.tolist()
            decoder["start"] = model.crf.start.tolist()
            decoder["end"] = model.crf.end.tolist()
    path = out_dir / "decoder.json"
    path.write_text(json.dumps(decoder, indent=1), encoding="utf-8")
    return decoder


def inline_external_data(path: Path) -> bool:
    """Put the weights back inside the .onnx file.

    Recent PyTorch writes the graph and the weights apart, as `tagger.onnx` and
    `tagger.onnx.data`. Python loads that pair without trouble. The browser
    runtime does not: it must be told about the second file, and a port that
    copies only the `.onnx` loads a model with no weights in it.

    One file cannot be separated from its weights by accident, so this rewrites
    the pair as one file and removes the loose one.
    """
    import onnx

    data = path.with_name(path.name + ".data")
    if not data.exists():
        return False
    model = onnx.load(str(path), load_external_data=True)
    onnx.save_model(model, str(path), save_as_external_data=False)
    data.unlink()
    return True


def make_time_axis_dynamic(path: Path) -> bool:
    """Give the output a symbolic time axis, whatever the exporter wrote.

    PyTorch writes the input as (batch, time) and the output as
    (batch, 283, 37), where 283 is only the length of the calibration input.
    Python's runtime tolerates the mismatch and prints a warning for every call.
    A stricter runtime does not, and the browser runtime is stricter.

    The graph itself is correct: the values match PyTorch at every length. Only
    the declared shape is wrong, so this rewrites the declaration.
    """
    import onnx

    model = onnx.load(str(path))
    changed = False
    for output in model.graph.output:
        dims = output.type.tensor_type.shape.dim
        if len(dims) < 2:
            continue
        # dim 0 is the batch, dim 1 is the time axis, dim 2 is the tag.
        if dims[1].HasField("dim_value"):
            dims[1].ClearField("dim_value")
            dims[1].dim_param = "time"
            changed = True
    if changed:
        onnx.save_model(model, str(path), save_as_external_data=False)
    return changed


def check_declared_shapes(path: Path) -> None:
    """Fail when an axis that must be free is frozen to a number.

    The values were already checked against PyTorch at several lengths. That
    test passes even when the declared shape is wrong, because the runtime used
    for the test ignores the declaration. This checks the declaration itself.
    """
    import onnx

    model = onnx.load(str(path), load_external_data=False)

    def axes(value):
        return [d.dim_param if d.HasField("dim_param") else d.dim_value
                for d in value.type.tensor_type.shape.dim]

    for value in list(model.graph.input) + list(model.graph.output):
        found = axes(value)
        if len(found) < 2:
            continue
        for position in (0, 1):          # the batch and the time axis
            if isinstance(found[position], int):
                raise SystemExit(
                    f"{path.name}: {value.name} declares axis {position} as "
                    f"{found[position]}, and it must be free. A port that "
                    f"trusts the declaration will refuse a document of any "
                    f"other length.")


def export(model_path: str, out_path: str, quantize: bool = False,
           check_lengths=CHECK_LENGTHS) -> dict:
    model, vocab, unit, has_crf = load(model_path)

    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)

    example = torch.randint(2, len(vocab), (1, 283), dtype=torch.long)
    # The TorchScript exporter, not the dynamo one that torch 2.9+ selects by
    # default. The dynamo exporter traces the recurrent units with a static
    # time axis (283) and refuses the dynamic one every port needs; the
    # TorchScript exporter keeps the axis free for all three units.
    torch.onnx.export(
        model, (example,), str(out),
        input_names=["ids"], output_names=["logits"],
        dynamic_axes={"ids": {0: "batch", 1: "time"},
                      "logits": {0: "batch", 1: "time"}},
        opset_version=OPSET,
        dynamo=False,
    )

    inline_external_data(out)
    make_time_axis_dynamic(out)
    check_declared_shapes(out)

    import onnxruntime as ort

    session = ort.InferenceSession(str(out), providers=["CPUExecutionProvider"])
    report = {
        "model": str(model_path), "path": str(out), "opset": OPSET,
        "unit": unit, "has_crf": has_crf, "vocabulary": len(vocab),
        "tags": len(TAG_TO_ID),
        "parameters": sum(p.numel() for p in model.parameters()),
        "size_mb": round(out.stat().st_size / 1e6, 3),
        "checks": [],
    }

    # The export is trusted only when it matches the source model at several
    # lengths and at more than one batch size, including lengths it never saw.
    worst = 0.0
    for n in check_lengths:
        for batch in (1, 3):
            ids = torch.randint(2, len(vocab), (batch, n), dtype=torch.long)
            with torch.no_grad():
                expected = model(ids).numpy()
            got = session.run(["logits"], {"ids": ids.numpy()})[0]
            diff = float(np.abs(expected - got).max())
            agree = float((expected.argmax(-1) == got.argmax(-1)).mean())
            worst = max(worst, diff)
            report["checks"].append({
                "length": n, "batch": batch,
                "max_difference": diff, "same_tag_share": agree})
            if agree < 1.0:
                raise SystemExit(
                    f"the ONNX graph disagrees with the model at length {n}, "
                    f"batch {batch}: only {100 * agree:.2f}% of tags match")
    report["max_difference"] = worst

    decoder = write_decoder(model, out.parent, unit, has_crf)
    report["decoder"] = {
        "path": str(out.parent / "decoder.json"),
        "has_crf": decoder["has_crf"], "tags": len(decoder["tags"]),
    }

    if quantize:
        from onnxruntime.quantization import QuantType, quantize_dynamic

        qpath = out.with_name(out.stem + "-int8.onnx")
        quantize_dynamic(str(out), str(qpath), weight_type=QuantType.QInt8)
        qsession = ort.InferenceSession(str(qpath),
                                        providers=["CPUExecutionProvider"])
        ids = torch.randint(2, len(vocab), (2, 283), dtype=torch.long)
        with torch.no_grad():
            expected = model(ids).numpy()
        got = qsession.run(["logits"], {"ids": ids.numpy()})[0]
        report["quantized"] = {
            "path": str(qpath),
            "size_mb": round(qpath.stat().st_size / 1e6, 3),
            # Quantisation changes the numbers. What matters is how often the
            # chosen tag changes, not how far the scores moved.
            "same_tag_share": float(
                (expected.argmax(-1) == got.argmax(-1)).mean()),
        }

    (out.parent / "export_report.json").write_text(
        json.dumps(report, indent=1), encoding="utf-8")
    return report


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", required=True)
    ap.add_argument("--out", default=None,
                    help="default: tagger.onnx beside the checkpoint")
    ap.add_argument("--quantize", action="store_true",
                    help="also write an int8 copy. Off by default: the model "
                         "is already under 3 MB, and one graph in every port "
                         "is worth more than a smaller one.")
    args = ap.parse_args()
    out = args.out or str(Path(args.model).parent / "tagger.onnx")
    r = export(args.model, out, quantize=args.quantize)

    print(f"exported {r['path']}")
    print(f"  {r['size_mb']:.2f} MB, opset {r['opset']}, {r['unit']}, "
          f"{r['parameters']:,} parameters")
    print(f"  decoder: {r['decoder']['path']} "
          f"({'with CRF' if r['decoder']['has_crf'] else 'no CRF'})")
    print(f"  largest disagreement with PyTorch: {r['max_difference']:.2e}")
    for c in r["checks"]:
        print(f"    length {c['length']:5d} batch {c['batch']}: "
              f"same tag {100 * c['same_tag_share']:.2f}%")
    if "quantized" in r:
        q = r["quantized"]
        print(f"  int8: {q['path']}  {q['size_mb']:.2f} MB  "
              f"same tag {100 * q['same_tag_share']:.2f}%")


if __name__ == "__main__":
    main()
