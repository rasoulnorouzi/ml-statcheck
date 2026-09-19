import torch
from statcheck_ml.model import make_model


def test_cnn_forward_shape_and_export(tmp_path):
    m = make_model(vocab_size=50, n_tags=37, unit="cnn")
    m.eval()
    x = torch.randint(1, 50, (2, 77))
    y = m(x)
    assert y.shape == (2, 77, 37)
    torch.onnx.export(m, x, tmp_path / "t.onnx", opset_version=17,
                      input_names=["chars"], output_names=["logits"],
                      dynamic_axes={"chars": {0: "batch", 1: "time"}, "logits": {0: "batch", 1: "time"}})
    import onnxruntime as ort
    s = ort.InferenceSession(str(tmp_path / "t.onnx"))
    out = s.run(None, {"chars": x.numpy()})[0]
    assert out.shape == (2, 77, 37)
