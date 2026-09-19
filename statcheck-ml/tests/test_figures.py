from statcheck_ml.figutil import COLORS, parse_log

LOG = """\
splits: dataset/splits.json (seed 0, dev_share 0.15)
augmentation: 1944 extra windows (400 of them hard negatives)
windows: train 3626, dev 280
vocabulary: 183 characters
model: lstm model, 615,141 parameters, 2.5 MB as float32, about 0.6 MB as int8
  epoch   1  loss 0.4079  dev P 0.684 R 0.863 F1 0.763  (68s)
  epoch   2  loss 0.0687  dev P 0.715 R 0.917 F1 0.803  (71s)
  epoch  10  loss 0.0079  dev P 0.825 R 0.895 F1 0.859  (71s)
"""


def test_parse_log_reads_every_epoch_line():
    epochs = parse_log(LOG)
    assert [e["epoch"] for e in epochs] == [1, 2, 10]
    assert epochs[0] == {"epoch": 1, "loss": 0.4079, "p": 0.684, "r": 0.863,
                         "f1": 0.763, "seconds": 68.0}
    assert epochs[2]["f1"] == 0.859


def test_parse_log_skips_banner_lines():
    epochs = parse_log(LOG)
    assert len(epochs) == 3  # not the 8 lines in the log


def test_color_map_has_every_required_system():
    for name in ("statcheck_raw", "statcheck_repaired", "cascade", "lstm", "gru", "cnn"):
        assert name in COLORS
        assert isinstance(COLORS[name], str) and COLORS[name].startswith("#")
