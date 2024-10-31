import json

import pytest
from valor_lite.text_generation.llm.exceptions import InvalidLLMResponseError
from valor_lite.text_generation.llm.utilities import (
    find_first_signed_integer,
    trim_and_load_json,
)


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
    input_string = (
        "this text should be trimmed"
        + json.dumps(expected, indent=4)
        + "also should be trimmed"
    )
    assert trim_and_load_json(input_string) == expected

    # test missing starting bracket
    input = """
        "sentence": "Hello, world!",
        "value": 3
    }"""
    with pytest.raises(InvalidLLMResponseError) as e:
        trim_and_load_json(input)
    assert "LLM did not include valid brackets in its response" in str(e)

    # test missing closing bracket a '}'
    input = '{"field": "value" '
    with pytest.raises(InvalidLLMResponseError) as e:
        trim_and_load_json(input)
    assert "LLM did not include valid brackets in its response" in str(e)

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
    with pytest.raises(InvalidLLMResponseError) as e:
        trim_and_load_json(input)
    assert "Evaluation LLM responded with invalid JSON." in str(e)

    # test dictionary with non-string keys
    input = "{1: 'a'}"
    with pytest.raises(InvalidLLMResponseError) as e:
        trim_and_load_json(input)
    assert "Evaluation LLM responded with invalid JSON." in str(e)


def test_find_first_signed_integer():

    text = "hello world the number is -101."
    assert find_first_signed_integer(text) == -101

    text = "hidden number 314 in text."
    assert find_first_signed_integer(text) == 314

    text = "only return first number 1, 2, 3"
    assert find_first_signed_integer(text) == 1

    text = "return nothing if no integers"
    assert find_first_signed_integer(text) is None

    text = "floating values are not registered 0.1"
    assert find_first_signed_integer(text) == 0
