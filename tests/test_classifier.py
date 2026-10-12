import pytest
from llm_animal_experiment.classifier import classify_response

@pytest.mark.parametrize(
    "text,expected",
    [
        ("Yes", "YES"),
        ("yes", "YES"),
        ("YES", "YES"),
        ("Yes.", "YES"),
        ("Yes!", "YES"),
        ("Yes, that's right!", "YES"),
        ("yes, it is a bee.", "YES"),
        ("**Yes**", "YES"),
        ("*Yes*, exactly", "YES"),
        ("\"Yes\"", "YES"),
        ("No", "NO"),
        ("no", "NO"),
        ("NO", "NO"),
        ("No.", "NO"),
        ("No!", "NO"),
        ("No, it is not an elephant.", "NO"),
        ("**No**", "NO"),
        ("*No*, sorry", "NO"),
        ("\"No\"", "NO"),
        ("Nope", "NO"),
        ("Nope, it's not a bee! 🐝❌", "NO"),
        ("🐝 Nope, not a fish either!", "NO"),
        ("🎉 **Yes, it's a bee!** 🐝", "YES"),
        ("🎉 Yes! It's a bee!", "YES"),
        ("Maybe", "INVALID"),
        ("Nopedy", "INVALID"),
        ("🎉🎉", "INVALID"),
        ("I was thinking of an elephant.", "INVALID"),
        ("Bee", "INVALID"),
        ("", "INVALID"),
        (None, "INVALID"),
        ("   ", "INVALID"),
        ("Yesterday was Sunday", "INVALID"),
        ("Nobody knows", "INVALID"),
    ],
)
def test_classify_response(text, expected):
    assert classify_response(text) == expected
