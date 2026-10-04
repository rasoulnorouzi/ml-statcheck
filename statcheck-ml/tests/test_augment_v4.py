"""The four paper families of the augmenter, and the output with no new flag."""
from __future__ import annotations

import hashlib
import json

import pytest

from statcheck_ml.augment import (CONTROL_SLOTS, OPERATOR_DAMAGE, AlignmentError,
                                  Augmenter, check_alignment, font_glyphs)
from statcheck_ml.labels import spans_to_tags, tags_to_spans

NEW_FLAGS = ("font_map", "negative_damage", "spaced_digits", "operator_drop")


def example(wid, text, parts):
    """Build an example from (substring, label, occurrence) triples."""
    spans = []
    for sub, label, nth in parts:
        start = -1
        for _ in range(nth + 1):
            start = text.index(sub, start + 1)
        spans.append((start, start + len(sub), label))
    return {"window_id": wid, "text": text, "tags": spans_to_tags(len(text), spans),
            "pool": "A", "journal": "j", "source_doc": "d.txt"}


EXAMPLES = [
    example("w1", "main effect, F(1, 73) = 3.74, p = .057, and", [
        ("F", "TEST", 0), ("1", "DF1", 0), ("73", "DF2", 0), ("3.74", "STAT", 0),
        ("=", "POP_EQ", 1), (".057", "PVAL", 0)]),
    example("w2", "the slope, t(8) = -4.40, p < .001. M = 3.2", [
        ("t", "TEST", 1), ("8", "DF1", 0), ("-4.40", "STAT", 0), ("<", "POP_LT", 0),
        (".001", "PVAL", 0)]),
    example("w3", "χ2(1, N = 223) = 8.69, p > 0.05 in Study 2-3", [
        ("χ2", "TEST", 0), ("1", "DF1", 0), ("223", "N", 0), ("8.69", "STAT", 0),
        (">", "POP_GT", 0), ("0.05", "PVAL", 0)]),
    example("w4", "see pp. 10-20 and r = .41 in Table 2", []),
]

# Two results, both operators, a negative statistic and decimal points.
TWO = example("two", "the slope, t(28) = -2.43, p = .021, and F(1, 73) = 3.74, p < .001", [
    ("t", "TEST", 1), ("28", "DF1", 0), ("-2.43", "STAT", 0), ("=", "POP_EQ", 1),
    (".021", "PVAL", 0), ("F", "TEST", 0), ("1", "DF1", 1), ("73", "DF2", 0),
    ("3.74", "STAT", 0), ("<", "POP_LT", 0), (".001", "PVAL", 0)])
TWO_SPANS = tags_to_spans(TWO["tags"])
TWO_VALUES = [(lab, TWO["text"][s:e]) for s, e, lab in TWO_SPANS]


def digest(out):
    return hashlib.sha256(json.dumps(out, sort_keys=True,
                                     ensure_ascii=False).encode()).hexdigest()


def run(**flags):
    aug = Augmenter(seed=7)
    for name, value in flags.items():
        setattr(aug, name, value)
    return aug.expand(EXAMPLES, per_example=4, negatives=6)


# -- no new flag: the same bytes as before the paper families existed ----------

# Produced by augment.py at commit 1097937, before the paper families were added.
BEFORE = {
    "default": "b76950269716972e2012485c6e2aa286ec49f3f633f043d7a54a30da45bd34b1",
    "number_extra": "2fa9275e2d6ea4d2eed74a2c37f28d30c794e8730c1bf59da1cb581b5715e6a2",
    "all_off": "7bd5a69b0fbb51e50ebd086bec998bcccaa9f1d4a922054cf31fc03178c1820f",
}


def test_no_new_flag_gives_the_same_bytes_as_before():
    assert digest(run()) == BEFORE["default"]
    assert digest(run(number_damage=True, extra_negatives=True)) == BEFORE["number_extra"]
    assert digest(run(ocr=False, unicode=False, operator=False,
                      hard_negative=False)) == BEFORE["all_off"]


def test_new_flags_default_to_off():
    aug = Augmenter()
    assert not any(getattr(aug, name) for name in NEW_FLAGS)
    assert digest(run(**{name: False for name in NEW_FLAGS})) == BEFORE["default"]


# -- font_map -------------------------------------------------------------------

def invert(text, fmap):
    back = {v: ("-" if k == "-" else k) for k, v in fmap.items()}
    return "".join(back.get(ch, ch) for ch in text)


def test_font_map_with_a_fixed_map():
    fmap = {"=": "\x02", "<": "\x03", "-": "\x07", ".": "\x05"}
    text, spans, used = Augmenter().apply_font_map(TWO["text"], TWO_SPANS, fmap)
    assert used == fmap
    assert text == ("the slope, t(28) \x02 \x072\x0543, p \x02 \x05021, and "
                    "F(1, 73) \x02 3\x0574, p \x03 \x05001")
    assert [(lab, text[s:e]) for s, e, lab in spans] == [
        ("TEST", "t"), ("DF1", "28"), ("STAT", "\x072\x0543"), ("POP_EQ", "\x02"),
        ("PVAL", "\x05021"), ("TEST", "F"), ("DF1", "1"), ("DF2", "73"),
        ("STAT", "3\x0574"), ("POP_LT", "\x03"), ("PVAL", "\x05001")]


@pytest.mark.parametrize("seed", range(200))
def test_font_map_is_one_consistent_map_and_keeps_every_value(seed):
    aug = Augmenter(seed=seed)
    text, spans, fmap = aug.apply_font_map(TWO["text"], TWO_SPANS)
    assert fmap, "a clean window always gets a map"
    # Same length: every replacement here is one character, so positions line up.
    assert len(text) == len(TWO["text"])
    seen = {}
    for old, new in zip(TWO["text"], text):
        if old != new:
            assert seen.setdefault(old, new) == new, f"{old!r} damaged two ways"
    for op in "=<":
        if op in fmap:
            assert op not in text
    # Distinct glyphs keep distinct codes.
    assert len(set(fmap.values())) == len(fmap)
    # Each label still covers its true value once the map is undone.
    assert [(lab, invert(text[s:e], fmap)) for s, e, lab in spans] == TWO_VALUES


def test_font_map_draws_from_the_font_table_and_the_operator_list():
    allowed = {op: set(OPERATOR_DAMAGE[op]) | {c for c, _ in font_glyphs().get(op, [])}
               | set(CONTROL_SLOTS) for op in "=<>"}
    assert "¼" in {c for c, _ in font_glyphs()["="]}
    drawn = {op: set() for op in "=<>-."}
    for seed in range(300):
        for symbol, char in Augmenter(seed=seed).draw_font_map().items():
            drawn[symbol].add(char)
            if symbol in allowed:
                assert char in allowed[symbol]
            else:
                assert char in CONTROL_SLOTS
    assert "¼" in drawn["="] and "b" in drawn["<"] and "N" in drawn[">"]
    assert "!" in drawn["<"]          # MathTechnicalP01 code 33, only in the font table
    assert drawn["-"] and drawn["."]


def test_font_map_leaves_a_window_with_real_damage_alone():
    text = "t(8) \x01 \x034.40, p \x04 .001"
    spans = [(0, 1, "TEST"), (7, 12, "STAT")]
    assert Augmenter(seed=0).apply_font_map(text, spans) == (text, spans, {})


def test_font_map_skips_hyphens_that_are_not_minus_signs():
    fmap = {"-": "\x07"}
    text, _, _ = Augmenter().apply_font_map("pp. 10-20, pre-1990, b = -0.3, (-.2)", [], fmap)
    assert text == "pp. 10-20, pre-1990, b = \x070.3, (\x07.2)"


# -- negative_damage --------------------------------------------------------------

NEGATIVE = example("neg", "the model was M = 3.41 (SD = 0.82) and b = -0.27 here", [])


def test_negative_damage_off_leaves_negatives_out():
    assert Augmenter(seed=0).perturb(NEGATIVE) is None


@pytest.mark.parametrize("seed", range(50))
def test_negative_damage_damages_and_leaves_no_label(seed):
    made = Augmenter(seed=seed, negative_damage=True).perturb(NEGATIVE)
    assert made is not None and made["text"] != NEGATIVE["text"]
    assert made["window_id"] == "neg-aug"
    assert set(made["tags"]) == {"O"} and len(made["tags"]) == len(made["text"])
    assert NEGATIVE["text"] == "the model was M = 3.41 (SD = 0.82) and b = -0.27 here"


def test_negative_damage_uses_the_same_map_as_a_result():
    """The same seed damages a negative and a result with the same characters."""
    text = "x = 1.5, y < 2.5, b = -0.27"
    neg = Augmenter(seed=11).apply_font_map(text, [])
    pos = Augmenter(seed=11).apply_font_map(text, [(4, 7, "STAT")])
    assert neg[0] == pos[0] and neg[2] == pos[2]


def test_negative_damage_twins_every_hard_negative():
    out = Augmenter(seed=5, negative_damage=True).expand([], negatives=40)
    clean = {o["window_id"]: o for o in out if not o["window_id"].endswith("-aug")}
    damaged = [o for o in out if o["window_id"].endswith("-aug")]
    assert len(clean) == 40 and damaged
    for o in damaged:
        twin = clean[o["window_id"][:-4]]
        assert o["text"] != twin["text"]
        assert set(o["tags"]) == {"O"} and len(o["tags"]) == len(o["text"])
    assert not check_alignment(out)


# -- spaced_digits ------------------------------------------------------------------

def test_spaced_digits_at_full_rate():
    ex = example("sp", "F1(1, 73) = 3.74, p = .057", [
        ("F1", "TEST", 0), ("1", "DF1", 1), ("73", "DF2", 0), ("3.74", "STAT", 0),
        ("=", "POP_EQ", 1), (".057", "PVAL", 0)])
    text, spans = Augmenter().space_digits(ex["text"], tags_to_spans(ex["tags"]), rate=1.0)
    assert text == "F1(1 , 7 3 ) = 3 . 7 4 , p = . 0 5 7"
    assert [(lab, text[s:e]) for s, e, lab in spans] == [
        ("TEST", "F1"), ("DF1", "1"), ("DF2", "7 3"), ("STAT", "3 . 7 4"),
        ("POP_EQ", "="), ("PVAL", ". 0 5 7")]


@pytest.mark.parametrize("seed", range(100))
def test_spaced_digits_keeps_every_value(seed):
    text, spans = Augmenter(seed=seed).space_digits(TWO["text"], TWO_SPANS)
    assert [(lab, "".join(text[s:e].split())) for s, e, lab in spans] == TWO_VALUES
    for s, e, _ in spans:
        assert not text[s].isspace() and not text[e - 1].isspace()
    assert "".join(text.split()) == "".join(TWO["text"].split())


# -- operator_drop ------------------------------------------------------------------

def test_operator_drop_removes_only_the_operator_span():
    ex = example("z", "z = 3.29, p = .01", [
        ("z", "TEST", 0), ("3.29", "STAT", 0), ("=", "POP_EQ", 1), (".01", "PVAL", 0)])
    text, spans = Augmenter().drop_operators(ex["text"], tags_to_spans(ex["tags"]),
                                             operators="=", collapse=True)
    assert text == "z 3.29, p .01"
    assert [(lab, text[s:e]) for s, e, lab in spans] == [
        ("TEST", "z"), ("STAT", "3.29"), ("PVAL", ".01")]


def test_operator_drop_keeps_an_operator_it_did_not_draw():
    text, spans = Augmenter().drop_operators(TWO["text"], TWO_SPANS,
                                             operators="=", collapse=False)
    assert text == "the slope, t(28)  -2.43, p  .021, and F(1, 73)  3.74, p < .001"
    assert [(lab, text[s:e]) for s, e, lab in spans] == [
        v for v in TWO_VALUES if v[0] != "POP_EQ"]


@pytest.mark.parametrize("seed", range(100))
def test_operator_drop_keeps_every_other_label(seed):
    text, spans = Augmenter(seed=seed).drop_operators(TWO["text"], TWO_SPANS)
    # A draw of ">" alone drops nothing here, and then every label stays.
    dropped = {ch for ch in "=<" if ch not in text}
    assert [(lab, text[s:e]) for s, e, lab in spans] == [
        v for v in TWO_VALUES if not (v[0].startswith("POP_") and v[1] in dropped)]


# -- the assertion fails loudly ---------------------------------------------------------

def test_an_edit_that_cuts_a_label_raises():
    with pytest.raises(AlignmentError):
        Augmenter()._edit(TWO["text"], TWO_SPANS, [(20, 26, "")], "test")


def test_broken_offset_arithmetic_raises(monkeypatch):
    def off_by_one(text, spans, start, end, new):
        return text[:start] + new + text[end:], [(s + 1, e + 1, lab) for s, e, lab in spans]
    monkeypatch.setattr(Augmenter, "_replace", staticmethod(off_by_one))
    with pytest.raises(AlignmentError):
        Augmenter(seed=0).space_digits(TWO["text"], TWO_SPANS, rate=1.0)


# -- the driver -------------------------------------------------------------------------

@pytest.mark.parametrize("flag", NEW_FLAGS)
def test_each_flag_alone_is_reproducible_and_aligned(flag):
    a = Augmenter(seed=3, **{flag: True}).expand(EXAMPLES + [TWO, NEGATIVE], 6, 10)
    b = Augmenter(seed=3, **{flag: True}).expand(EXAMPLES + [TWO, NEGATIVE], 6, 10)
    assert a == b
    assert not check_alignment(a)
    assert digest(a) != BEFORE["default"]


def test_the_originals_are_never_changed():
    before = json.dumps([TWO, NEGATIVE] + EXAMPLES, sort_keys=True)
    Augmenter(seed=1, **{f: True for f in NEW_FLAGS}).expand(
        EXAMPLES + [TWO, NEGATIVE], 8, 5)
    assert json.dumps([TWO, NEGATIVE] + EXAMPLES, sort_keys=True) == before


def test_a_paper_map_switches_off_the_random_operator_damage():
    """After one map, a second random damage would put two fonts in one paper."""
    aug = Augmenter(seed=0, font_map=True, ocr=False, unicode=False)
    mapped = 0
    for _ in range(300):
        made = aug.perturb(TWO)
        text, spans = made["text"], tags_to_spans(made["tags"])
        # "= " before the statistic, perhaps with a line break after it. The
        # map may turn "=" into a space, so only the break is stripped.
        stat_ops = {text[:s].rstrip("\n")[-2] for s, e, lab in spans if lab == "STAT"}
        pop_eq = {text[s:e] for s, e, lab in spans if lab == "POP_EQ"}
        if stat_ops != {"="}:
            mapped += 1
            assert stat_ops == pop_eq and len(pop_eq) == 1
    assert mapped > 50
