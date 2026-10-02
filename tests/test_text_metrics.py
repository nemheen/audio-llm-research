import math

from speechllm.cascade import label_from_freetext, parse_label
from speechllm.data import canonical_reason
from speechllm.evaluate import looks_english
from speechllm.metrics import compression_ratio, is_degenerate, wer, wer_bootstrap_ci
from speechllm.text import normalize


def test_normalize_english():
    assert normalize("It “warms” the throat.") == "it warms the throat"
    assert normalize("Don’t   stop -- now!") == "don't stop now"
    assert normalize(None) == ""


def test_normalize_mongolian_initials():
    assert normalize("З.Энхболд, Улаанбаатар хотын") == "з энхболд улаанбаатар хотын"


def test_wer_is_a_fraction_and_normalised():
    assert wer(["The cat sat."], ["the cat sat"]) == 0.0
    assert math.isclose(wer(["a b c d"], ["a x c d"]), 0.25)
    # insertions can push WER above 1 - this is why v0 reported "1.39"
    assert wer(["a b"], ["a b c d e f"]) == 2.0


def test_wer_ci_brackets_point_estimate():
    refs = ["a b c d"] * 10 + ["e f"] * 10
    hyps = ["a b c d"] * 10 + ["e g"] * 10
    lo, hi = wer_bootstrap_ci(refs, hyps)
    assert lo <= wer(refs, hyps) <= hi


def test_degenerate_detection():
    loop = " ".join(["ffordd o'r"] * 60)
    assert compression_ratio(loop) > 2.4
    assert is_degenerate(loop)
    assert not is_degenerate("The legs are short and black as well.", "The legs are short and black as well.")
    # length rule: more than 3 * n_ref + 5 hypothesis words
    assert not is_degenerate("one two three four five six seven eight nine ten eleven", "one two")
    assert is_degenerate("one two three four five six seven eight nine ten eleven twelve", "one two")


def test_reason_canonicalisation():
    assert canonical_reason("sc-different-language") == "different-language"
    assert canonical_reason("difficult-pronounce") == "difficult-pronunciation"
    assert canonical_reason("tiktok") == "other"


def test_label_parsing():
    assert parse_label('{"label": "offensive-language", "explanation": "x"}') == "offensive-language"
    assert parse_label('Sure! {"label": "nonsense"} it is in a foreign language') == "different-language"
    assert label_from_freetext("Topic: Welsh government policy") == "other"


def test_looks_english():
    assert looks_english("It is widely served and mostly drunk in winter because it “warms” the throat.")
    assert not looks_english("Majani haya ya kijani yenye virutubisho vingi")
    assert not looks_english("Улаанбаатар хотын захирагч")
