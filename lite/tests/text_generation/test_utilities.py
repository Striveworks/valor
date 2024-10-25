import json

import pytest
from valor_lite.text_generation.exceptions import InvalidLLMResponseError
from valor_lite.text_generation.utilities import trim_and_load_json


def _package_json(data: dict):
    return f"```json \n{json.dumps(data, indent=4)}\n```"


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

    # This function should add an } if none are present.
    input = """{"field": "value" """
    trim_and_load_json(input)

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
    with pytest.raises(InvalidLLMResponseError):
        trim_and_load_json(input)

    input = """
    "sentence": "Hello, world!",
    "value": 3
}"""

    # Missing starting bracket
    with pytest.raises(InvalidLLMResponseError):
        trim_and_load_json(input)
