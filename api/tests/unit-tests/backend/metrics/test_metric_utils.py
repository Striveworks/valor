import pytest

from valor_api.backend.metrics.metric_utils import trim_and_load_json


def test_trim_and_load_json():
    input = """this text should be trimmed
{
    "verdicts": [
        {
            "verdict": "yes"
        },
        {
            "verdict": "no",
            "reason": "The statement 'I also think puppies are cute.' is irrelevant to the question about who the cutest cat ever is."
        }
    ]
}"""
    expected = {
        "verdicts": [
            {"verdict": "yes"},
            {
                "verdict": "no",
                "reason": "The statement 'I also think puppies are cute.' is irrelevant to the question about who the cutest cat ever is.",
            },
        ]
    }

    assert trim_and_load_json(input) == expected

    input = """{
    "verdicts": [
        {
            "verdict": "yes"
        }
        {
            "verdict": "no",
            "reason": "The statement 'I also think puppies are cute.' is irrelevant to the question about who the cutest cat ever is."
        }
    ]
}"""

    # Missing a comma
    with pytest.raises(ValueError):
        trim_and_load_json(input)

    input = """
    "sentence": "Hello, world!",
    "value": 3
}"""

    # Missing starting bracket
    with pytest.raises(ValueError):
        trim_and_load_json(input)
