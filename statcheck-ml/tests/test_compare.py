import importlib.util
import json
import os

spec = importlib.util.spec_from_file_location(
    "compare", os.path.join(os.path.dirname(__file__), "..", "pipeline", "16_compare.py"))
compare = importlib.util.module_from_spec(spec)
spec.loader.exec_module(compare)


def sysrow(recall, finds, ci=(0.1, 0.9), dmg=0.5):
    return {"recall": recall, "recall_ci": list(ci), "recall_damaged": dmg,
            "finds_in_papers_without_results": finds}


def write(path, obj):
    path.write_text(json.dumps(obj))
    return str(path)


def setup(tmp_path):
    before = {t: {"papers": 3, "regex": sysrow(0.5, 4), "model": sysrow(0.8, 6), "old": sysrow(0.1, 0)}
              for t in ("gold_test", "gold_train")}
    after = {t: {"papers": 3, "regex": sysrow(0.5, 4), "model": dict(sysrow(0.85, 5), recall_with_fragments=0.9),
                 "new": sysrow(0.2, 1)}
             for t in ("gold_test", "gold_train")}
    bb = {"systems": {"a": {"overall": {"p": 0.9, "r": 0.8, "f1": 0.85}},
                      "gone": {"overall": {"p": 0.1, "r": 0.1, "f1": 0.1}}}}
    ab = {"systems": {"a": {"overall": {"p": 0.95, "r": 0.8, "f1": 0.87}}}}
    log = tmp_path / "log.jsonl"
    log.write_text("\n".join(json.dumps(x) for x in [
        {"date": "d1", "git_sha": "aaa", "label": "m1", "systems": ["model"],
         "gold_test_recall": {"model": 0.8}},
        {"date": "d2", "git_sha": "bbb", "label": "", "systems": ["model"], "gold_test_recall": {"model": 0.1}},
        {"date": "d3", "git_sha": "ccc", "label": "m2", "systems": ["model"],
         "gold_test": {"model": dict(sysrow(0.85, 5), recall_with_fragments=0.9)}},
        {"date": "d4", "git_sha": "ddd", "label": "reproduce", "systems": ["model"],
         "gold_test_recall": {"model": 0.85}},
    ]) + "\n")
    return [
        "--before-gold", write(tmp_path / "bg.json", before), "--after-gold", write(tmp_path / "ag.json", after),
        "--before-bronze", write(tmp_path / "bb.json", bb), "--after-bronze", write(tmp_path / "ab.json", ab),
        "--log", str(log), "--out", str(tmp_path / "o.json"), "--md", str(tmp_path / "o.md")]


def run(argv, monkeypatch):
    monkeypatch.setattr("sys.argv", ["16_compare.py"] + argv)
    compare.main()


def test_deltas_and_missing_systems(tmp_path, monkeypatch):
    run(setup(tmp_path), monkeypatch)
    out = json.load(open(tmp_path / "o.json"))
    t = out["gold"]["gold_test"]
    by = {r["system"]: r for r in t["rows"]}
    assert by["model"]["recall_delta"] == 0.05
    assert by["regex"]["recall_delta"] == 0.0
    assert by["model"]["finds_in_papers_without_results_before"] == 6
    assert by["model"]["finds_in_papers_without_results_after"] == 5
    assert t["only_before"] == ["old"] and t["only_after"] == ["new"]
    b = out["bronze_holdout"]
    assert b["rows"][0]["p_delta"] == 0.05 and b["rows"][0]["r_delta"] == 0.0
    assert b["only_before"] == ["gone"]
    assert "old" in (tmp_path / "o.md").read_text()


def test_reproduce_reads_leave_the_output_unchanged(tmp_path, monkeypatch):
    args = setup(tmp_path)
    run(args, monkeypatch)
    first = (tmp_path / "o.json").read_bytes()
    with open(tmp_path / "log.jsonl", "a") as fh:
        fh.write(json.dumps({"date": "d5", "git_sha": "eee", "label": "reproduce",
                             "systems": ["model"], "gold_test_recall": {"model": 0.85}}) + "\n")
    run(args, monkeypatch)
    assert (tmp_path / "o.json").read_bytes() == first
    assert "reproduce" not in [s["label"] for s in json.load(open(tmp_path / "o.json"))["steps"]]


def test_steps_keep_order_and_skip_unlabelled(tmp_path, monkeypatch):
    run(setup(tmp_path), monkeypatch)
    steps = json.load(open(tmp_path / "o.json"))["steps"]
    assert [s["label"] for s in steps] == ["m1", "m2"]
    assert steps[0]["systems"]["model"]["recall"] == 0.8
    assert steps[1]["systems"]["model"]["finds_in_papers_without_results"] == 5


def test_two_runs_same_bytes(tmp_path, monkeypatch):
    argv = setup(tmp_path)
    run(argv, monkeypatch)
    first = ((tmp_path / "o.json").read_bytes(), (tmp_path / "o.md").read_bytes())
    run(argv, monkeypatch)
    assert first == ((tmp_path / "o.json").read_bytes(), (tmp_path / "o.md").read_bytes())


def test_fragment_recall_dash_when_a_side_lacks_it(tmp_path, monkeypatch):
    run(setup(tmp_path), monkeypatch)
    out = json.load(open(tmp_path / "o.json"))
    by = {r["system"]: r for r in out["gold"]["gold_test"]["rows"]}
    assert by["model"]["recall_with_fragments_before"] is None
    assert by["model"]["recall_with_fragments_after"] == 0.9
    md = (tmp_path / "o.md").read_text().splitlines()
    gold = [x for x in md if x.startswith("| model |") and x.rstrip().endswith("\u2014 | 0.900 |")]
    assert gold
    regex = [x for x in md if x.startswith("| regex |") and x.rstrip().endswith("\u2014 | \u2014 |")]
    assert regex


def test_steps_table_dash_for_a_line_without_the_key(tmp_path, monkeypatch):
    run(setup(tmp_path), monkeypatch)
    steps = json.load(open(tmp_path / "o.json"))["steps"]
    assert steps[0]["systems"]["model"]["recall_with_fragments"] is None
    assert steps[1]["systems"]["model"]["recall_with_fragments"] == 0.9
    md = (tmp_path / "o.md").read_text()
    assert "| m1 | d1 | aaa | model | 0.800 | \u2014 |" in md
    assert "| m2 | d3 | ccc | model | 0.850 | 0.900 | 5 |" in md
