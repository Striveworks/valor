import json

import pytest
from valor_lite.text_generation.llm.exceptions import InvalidLLMResponseError
from valor_lite.text_generation.llm.generation import (
    _generate,
    generate_answer_correctness_verdicts,
)
from valor_lite.text_generation.llm.validators import validate_statements


def test__generate(mock_client):

    data = {"claims": ["a", "b", "c"]}

    json_basic = json.dumps(data)
    json_indented = json.dumps(data, indent=4)
    json_with_text = "this should be removed" + json_indented

    # test correct response
    for variation in [
        json_basic,
        json_indented,
        json_with_text,
    ]:
        mock_client.returning = variation
        assert (
            _generate(
                client=mock_client,
                messages=[{}],
                keys={"claims"},
                validator=validate_statements,
            )
            == data
        )

    # test non-dict response
    with pytest.raises(InvalidLLMResponseError):
        mock_client.returning = "some text body with no dict."
        _generate(
            client=mock_client,
            messages=[{}],
            keys={"claims"},
            validator=validate_statements,
        )

    # test missing keyword
    with pytest.raises(InvalidLLMResponseError):
        mock_client.returning = json.dumps({"other": ["a", "b", "c"]})
        _generate(
            client=mock_client,
            messages=[{}],
            keys={"claims"},
            validator=validate_statements,
        )

    # test claims not in list
    with pytest.raises(InvalidLLMResponseError):
        mock_client.returning = json.dumps({"claims": "a"})
        _generate(
            client=mock_client,
            messages=[{}],
            keys={"claims"},
            validator=validate_statements,
        )

    # test claims not string type
    with pytest.raises(InvalidLLMResponseError):
        mock_client.returning = json.dumps({"claims": ["a", 1]})
        _generate(
            client=mock_client,
            messages=[{}],
            keys={"claims"},
            validator=validate_statements,
        )

    # test allowed values
    mock_client.returning = json.dumps({"claims": ["a", "b", "c"]})
    _generate(
        client=mock_client,
        messages=[{}],
        keys={"claims"},
        allowed_values={"a", "b", "c"},
        validator=validate_statements,
    )
    with pytest.raises(InvalidLLMResponseError):
        mock_client.returning = json.dumps({"claims": ["a", "b", "z"]})
        _generate(
            client=mock_client,
            messages=[{}],
            keys={"claims"},
            allowed_values={"a", "b", "c"},
            validator=validate_statements,
        )

    # test enforced length
    mock_client.returning = json.dumps({"claims": ["a", "b", "c"]})
    _generate(
        client=mock_client,
        messages=[{}],
        keys={"claims"},
        enforce_length=3,
        validator=validate_statements,
    )
    with pytest.raises(InvalidLLMResponseError):
        mock_client.returning = json.dumps({"claims": ["a"]})
        _generate(
            client=mock_client,
            messages=[{}],
            keys={"claims"},
            enforce_length=3,
            validator=validate_statements,
        )


def test_generate_answer_correctness_verdicts(mock_client):

    mock_client.returning = json.dumps({"TP": ["a"], "FP": ["b"], "FN": ["c"]})
    with pytest.raises(InvalidLLMResponseError) as e:
        generate_answer_correctness_verdicts(
            client=mock_client,
            system_prompt="",
            query="question?",
            prediction_statements=["x"],
            groundtruth_statements=["m"],
        )
    assert "true positives and false positives" in str(e)

    mock_client.returning = json.dumps(
        {"TP": ["a"], "FP": ["b"], "FN": ["c", "d"]}
    )
    with pytest.raises(InvalidLLMResponseError) as e:
        generate_answer_correctness_verdicts(
            client=mock_client,
            system_prompt="",
            query="question?",
            prediction_statements=["x", "y"],
            groundtruth_statements=["m"],
        )
    assert "false negatives exceeded the number of ground truth" in str(e)
