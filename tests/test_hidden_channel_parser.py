import importlib.util
from pathlib import Path

SPEC = importlib.util.spec_from_file_location(
    "hc", Path(__file__).resolve().parents[1] / "scripts" / "run_behavioral_ea_hidden_channel.py")
HC = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(HC)

ROW = {"truth": 73, "legacy": 75, "irrelevant": 64}


def test_bare_integer_is_valid():
    assert HC.parse_action("73") == ("valid", 73)
    assert HC.parse_action(" 73\n") == ("valid", 73)


def test_signed_and_prose_single_integer():
    assert HC.parse_action("-12") == ("valid", -12)
    assert HC.parse_action("The answer is 73.") == ("valid", 73)


def test_multiple_integers_invalid():
    assert HC.parse_action("47 + 26 = 73")[0] == "invalid"
    assert HC.parse_action("73 or 75")[0] == "invalid"


def test_comma_grouped_and_spelled_invalid():
    assert HC.parse_action("1,073")[0] == "invalid"
    assert HC.parse_action("seventy-three")[0] == "invalid"
    assert HC.parse_action("")[0] == "invalid"


def test_categories():
    assert HC.categorise(73, ROW) == "truth"
    assert HC.categorise(75, ROW) == "legacy"
    assert HC.categorise(64, ROW) == "irrelevant"
    assert HC.categorise(99, ROW) == "other"
    assert HC.categorise(None, ROW) == "invalid"
