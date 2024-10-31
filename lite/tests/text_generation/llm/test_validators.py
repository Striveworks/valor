import pytest
from valor_lite.text_generation.llm.exceptions import InvalidLLMResponseError
from valor_lite.text_generation.llm.validators import (
    validate_statements,
    validate_verdicts,
)


def test_validate_statements():
    response = {
        "k1": ["a", "b", "c"],
        "k2": ["a"],
        "k3": "a",
        "k4": {"verdict": "a", "analysis": "b"},
        "k5": {"verdict": "a", "analysis": "b", "other": "c"},
    }

    # test if key is in response
    with pytest.raises(InvalidLLMResponseError) as e:
        validate_statements(
            response=response,
            key="other_key",
        )
    assert "did not include key" in str(e)

    # test proper formatting
    validate_statements(response=response, key="k1")
    validate_statements(response=response, key="k2")
    with pytest.raises(InvalidLLMResponseError):
        validate_statements(response=response, key="k3")
    with pytest.raises(InvalidLLMResponseError):
        validate_statements(response=response, key="k4")
    with pytest.raises(InvalidLLMResponseError):
        validate_statements(response=response, key="k5")

    # test restricted value sets
    validate_statements(
        response=response, key="k1", allowed_values={"a", "b", "c"}
    )
    with pytest.raises(InvalidLLMResponseError):
        validate_statements(response=response, key="k1", allowed_values={"a"})
    validate_statements(
        response=response, key="k2", allowed_values={"a", "b", "c"}
    )
    validate_statements(response=response, key="k2", allowed_values={"a"})

    # test length enforcement
    validate_statements(response=response, key="k1", enforce_length=3)
    with pytest.raises(InvalidLLMResponseError) as e:
        validate_statements(response=response, key="k1", enforce_length=1)
    assert "does not match input size" in str(e)


def test_validate_verdicts():

    response = {
        "k1": [
            {"verdict": "a", "analysis": "abcdef"},
            {"verdict": "b", "analysis": "abcdef"},
            {"verdict": "c", "analysis": "abcdef"},
        ],
        "k2": [
            {"verdict": "a", "analysis": "abcdef"},
            {"verdict": "a", "analysis": "abcdef"},
        ],
        "k3": "a",
        "k4": {"verdict": "a", "analysis": "b"},
        "k5": [{"verdict": "a", "analysis": "b", "other": "c"}],
        "k6": [1, 2, 3],
    }

    # test if key is in response
    with pytest.raises(InvalidLLMResponseError) as e:
        validate_verdicts(
            response=response,
            key="other_key",
        )
    assert "did not include key" in str(e)

    # test proper formatting
    validate_verdicts(response=response, key="k1")
    validate_verdicts(response=response, key="k2")
    with pytest.raises(InvalidLLMResponseError):
        validate_verdicts(response=response, key="k3")
    with pytest.raises(InvalidLLMResponseError):
        validate_verdicts(response=response, key="k4")
    with pytest.raises(InvalidLLMResponseError):
        validate_verdicts(response=response, key="k5")
    with pytest.raises(InvalidLLMResponseError):
        validate_verdicts(response=response, key="k6")

    # test allowed value enforcement
    validate_verdicts(
        response=response, key="k1", allowed_values={"a", "b", "c"}
    )
    with pytest.raises(InvalidLLMResponseError):
        validate_verdicts(response=response, key="k1", allowed_values={"a"})
    validate_verdicts(
        response=response, key="k2", allowed_values={"a", "b", "c"}
    )
    validate_verdicts(response=response, key="k2", allowed_values={"a"})

    # test length enforcement
    validate_verdicts(response=response, key="k1", enforce_length=3)
    with pytest.raises(InvalidLLMResponseError) as e:
        validate_verdicts(response=response, key="k1", enforce_length=1)
    assert "does not match input size" in str(e)
