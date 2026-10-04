"""Rule logic of pipeline/23_select_v4.py on hand-made metric tables, then the
export and dev scoring end to end on a smoke run when one is available."""
import importlib.util
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("select_v4", ROOT / "pipeline/23_select_v4.py")
sel = importlib.util.module_from_spec(spec)
spec.loader.exec_module(sel)

NONE, ALL = "gate-none", "gate-old+all"


def table(none=(0.70, 3), new=(0.70, 3), grid=None):
    """Metrics for the gate pair and, from grid, for the other grid runs."""
    m = {NONE: {"R_named": none[0], "FP": none[1]}, ALL: {"R_named": new[0], "FP": new[1]}}
    for (cfg, noise, seed), (r, fp) in (grid or {}).items():
        m[sel.run_name(cfg, noise, seed)] = {"R_named": r, "FP": fp}
    return m


def grid(noise, means, fps=None, spread=0.0):
    """Three seeds per configuration whose mean R_named is means[cfg]."""
    out = {}
    for cfg, mean in means.items():
        for seed, delta in zip(sel.SEEDS, (-spread, 0.0, spread)):
            out[(cfg, noise, seed)] = (mean + delta, (fps or {}).get(cfg, 3))
    return out


def test_rule1_ineligible_when_fp_exceeds_none_plus_two():
    r = sel.rule1(table(none=(0.60, 3), new=(0.90, 6)))
    assert r["eligible"] == {"none": True, "all": False}
    assert r["chosen"] == "none"


def test_rule1_fp_limit_is_inclusive():
    r = sel.rule1(table(none=(0.60, 3), new=(0.90, 5)))
    assert r["eligible"]["all"] and r["chosen"] == "all"


def test_rule1_higher_recall_wins_at_the_gap():
    assert sel.rule1(table(none=(0.70, 3), new=(0.71, 5)))["chosen"] == "all"
    assert sel.rule1(table(none=(0.72, 3), new=(0.70, 3)))["chosen"] == "none"


def test_rule1_tie_below_gap_takes_lower_fp():
    assert sel.rule1(table(none=(0.700, 4), new=(0.705, 3)))["chosen"] == "all"
    assert sel.rule1(table(none=(0.705, 3), new=(0.700, 4)))["chosen"] == "none"


def test_rule1_tie_on_recall_and_fp_takes_none():
    r = sel.rule1(table(none=(0.70, 3), new=(0.704, 3)))
    assert r["chosen"] == "none" and "'none'" in r["reason"]


def test_rule1_ineligible_all_loses_even_when_recall_tied_or_better():
    assert sel.rule1(table(none=(0.70, 0), new=(0.70, 3)))["chosen"] == "none"


PARAMS = {"gru-crf": 100, "gru-softmax": 90, "lstm-softmax": 120}


def params_for(noise):
    return {sel.run_name(c, noise, 0): n for c, n in PARAMS.items()}


def test_rule2_highest_mean_wins_over_a_lucky_seed():
    # lstm has the best single seed (0.85) but the lower mean.
    g = grid("none", {"gru-crf": 0.70, "gru-softmax": 0.60}, spread=0.0)
    for seed, r in zip(sel.SEEDS, (0.85, 0.55, 0.55)):
        g[("lstm-softmax", "none", seed)] = (r, 3)
    r = sel.rule2(table(grid=g), params_for("none"), "none")
    assert r["chosen"] == "gru-crf"
    assert r["table"]["lstm-softmax"]["mean_R_named"] == pytest.approx(0.65)


def test_rule2_tie_takes_lower_mean_fp_then_fewer_parameters():
    means = {"gru-crf": 0.700, "gru-softmax": 0.705, "lstm-softmax": 0.50}
    m = table(grid=grid("none", means, fps={"gru-crf": 2, "gru-softmax": 5, "lstm-softmax": 0}))
    r = sel.rule2(m, params_for("none"), "none")
    assert r["chosen"] == "gru-crf"            # tied with gru-softmax, lower FP
    m = table(grid=grid("none", means, fps={"gru-crf": 4, "gru-softmax": 4, "lstm-softmax": 0}))
    assert sel.rule2(m, params_for("none"), "none")["chosen"] == "gru-softmax"  # 90 < 100


def test_rule2_gap_of_exactly_one_hundredth_is_not_a_tie():
    means = {"gru-crf": 0.70, "gru-softmax": 0.71, "lstm-softmax": 0.50}
    m = table(grid=grid("none", means, fps={"gru-crf": 0, "gru-softmax": 9, "lstm-softmax": 0}))
    assert sel.rule2(m, params_for("none"), "none")["chosen"] == "gru-softmax"


def test_rule2_reads_the_noise_setting_it_is_given():
    g = grid("none", {"gru-crf": 0.9, "gru-softmax": 0.1, "lstm-softmax": 0.1})
    g.update(grid("all", {"gru-crf": 0.1, "gru-softmax": 0.9, "lstm-softmax": 0.1}))
    m = table(grid=g)
    assert sel.rule2(m, params_for("none"), "none")["chosen"] == "gru-crf"
    assert sel.rule2(m, params_for("all"), "all")["chosen"] == "gru-softmax"


def test_rule3_keeps_the_old_model_when_candidate_is_lower():
    m = table(grid=grid("none", {"gru-crf": 0.5, "gru-softmax": 0.5, "lstm-softmax": 0.5}))
    r = sel.rule3(m, {"R_named": 0.80}, "gru-softmax", "none")
    assert r["candidate_run"] == "gru-softmax-none-s0"
    assert not r["replaces_gru_crf_s0"] and "old model stays" in r["reason"]


def test_rule3_equal_recall_replaces():
    m = table(none=(0.80, 3), new=(0.80, 3))
    assert sel.rule3(m, {"R_named": 0.80}, "gru-crf", "none")["replaces_gru_crf_s0"]
    assert sel.rule3(m, {"R_named": 0.8001}, "gru-crf", "none")["replaces_gru_crf_s0"] is False


def test_select_chains_the_rules_and_ships_the_seed_zero_run():
    g = grid("all", {"gru-crf": 0.60, "gru-softmax": 0.80, "lstm-softmax": 0.70})
    m = table(none=(0.50, 3), new=(0.60, 4), grid=g)
    out = sel.select(m, {n: 1 for n in m} | params_for("all"), {"R_named": 0.55})
    assert out["rule1"]["chosen"] == "all"
    assert out["rule2"]["chosen"] == "gru-softmax"
    assert out["chosen_run"] == "gru-softmax-all-s0" and out["replaces_gru_crf_s0"]


def test_run_names_are_the_ones_the_trainer_lists():
    listed = set(sel.expected_runs())
    used = {sel.run_name(c, n, s) for c in sel.CONFIGS for n in ("none", "all") for s in sel.SEEDS}
    assert used <= listed and len(listed) == 23  # the 5 other gate runs are reported only


def test_missing_run_stops_with_a_message(tmp_path):
    r = subprocess.run([sys.executable, str(ROOT / "pipeline/23_select_v4.py"), "--models", str(tmp_path),
                        "--out-dir", str(tmp_path / "out")], capture_output=True, text=True)
    assert r.returncode != 0
    assert "23 of 23 runs are missing" in r.stderr and "gate-none" in r.stderr
    assert not (tmp_path / "out" / "selection.json").exists()


# End to end on a one-epoch smoke run. Set STATCHECK_SMOKE_RUN to its directory.
SMOKE = os.environ.get("STATCHECK_SMOKE_RUN")


@pytest.mark.skipif(not SMOKE or not (Path(SMOKE) / "model.pt").exists(), reason="no smoke run")
def test_smoke_run_end_to_end(tmp_path):
    import shutil

    models = tmp_path / "models"
    shutil.copytree(SMOKE, models / ALL)
    log = ROOT / "results/gold_eval_log.jsonl"
    before = log.read_bytes()
    r = subprocess.run([sys.executable, str(ROOT / "pipeline/23_select_v4.py"), "--models", str(models),
                        "--out-dir", str(tmp_path / "out"), "--allow-partial",
                        "--parity-windows", "20"],
                       capture_output=True, text=True, cwd=ROOT)
    assert r.returncode == 0, r.stderr
    assert log.read_bytes() == before            # a dev score never writes the gold-test log
    res = json.loads((tmp_path / "out/selection.json").read_text())
    run = res["runs"][ALL]
    assert res["partial"] and res["selection"] is None
    assert run["parity"] and run["tag_agreement"] == 1.0
    assert 0 <= run["R_named"] <= run["R_frag"] <= 1 and run["FP"] >= 0
    assert (models / ALL / "zoo/tagger.onnx").exists()
    assert (tmp_path / "out/selection.md").read_text().count("|") > 20
    dev = json.loads((tmp_path / f"out/dev/{ALL}.json").read_text())
    assert set(dev) == {"note", "papers", "dev", "provenance"}     # no gold_test, no bronze
    assert dev["dev"]["papers"] == 36


def test_dev_scoring_opens_no_gold_test_paper(tmp_path):
    # The clean dir holds the dev papers only, so opening any other paper fails.
    dev = json.loads((ROOT / "dataset/gold/dev.json").read_text())["dev"]
    clean = tmp_path / "clean"
    for doc in dev:
        (clean / doc).parent.mkdir(parents=True, exist_ok=True)
        os.symlink(ROOT / "data/clean" / doc, clean / doc)
    log = ROOT / "results/gold_eval_log.jsonl"
    before = log.read_bytes()
    out = tmp_path / "dev.json"
    r = subprocess.run([sys.executable, str(ROOT / "pipeline/14_gold_eval.py"), "--papers", "dev",
                        "--systems", "regex", "--clean-dir", str(clean), "--out", str(out),
                        "--resamples", "5", "--bronze", str(tmp_path / "no_such_bronze.json")],
                       capture_output=True, text=True, cwd=ROOT)
    assert r.returncode == 0, r.stderr
    assert log.read_bytes() == before
    report = json.loads(out.read_text())
    assert report["dev"]["papers"] == 36 and "gold_test" not in report


def test_dev_papers_lie_inside_gold_train():
    dev = set(json.loads((ROOT / "dataset/gold/dev.json").read_text())["dev"])
    split = json.loads((ROOT / "dataset/gold/split.json").read_text())
    assert dev <= set(split["gold_train"]) and not dev & set(split["gold_test"])
