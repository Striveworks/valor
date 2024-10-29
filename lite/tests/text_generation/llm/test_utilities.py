import json

import pytest
from valor_lite.text_generation.llm.exceptions import InvalidLLMResponseError
from valor_lite.text_generation.llm.utilities import trim_and_load_json


def test_trim_and_load_json():

    # test removal of excess text
    expected = {
        "verdicts": [
            {"verdict": "yes"},
            {
                "verdict": "no",
                "reason": "The statement 'I also think puppies are cute.' is irrelevant to the question about who the cutest cat ever is.",
            },
        ]
    }
    input_string = "this text should be trimmed"
    input_string += json.dumps(expected, indent=4)
    assert trim_and_load_json(input_string) == expected

    # test that a '}' is added it one is not present.
    input = '{"field": "value" '
    assert trim_and_load_json(input) == {"field": "value"}

    # test missing comma edge case
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
    with pytest.raises(InvalidLLMResponseError):
        trim_and_load_json(input)

    # test missing starting bracket
    input = """
    "sentence": "Hello, world!",
    "value": 3
}"""
    with pytest.raises(InvalidLLMResponseError):
        trim_and_load_json(input)
