import pytest
from valor_lite.text_generation import Context, Evaluator, QueryResponse
from valor_lite.text_generation.computation import calculate_hallucination


def test_hallucination_no_context(mock_client):

    evaluator = Evaluator(client=mock_client)
    with pytest.raises(ValueError):
        assert evaluator.compute_hallucination(
            response=QueryResponse(
                query="a",
                response="a",
            )
        )


def test_calculate_hallucination(
    mock_client,
    verdicts_all_yes: str,
    verdicts_all_no: str,
    verdicts_two_yes_one_no: str,
    verdicts_empty: str,
):

    evaluator = Evaluator(client=mock_client)

    mock_client.returning = verdicts_all_yes
    assert (
        calculate_hallucination(
            client=mock_client,
            system_prompt="",
            response="a",
            context=["x", "y", "z"],
        )
        == 1.0
    )
    assert evaluator.compute_hallucination(
        response=QueryResponse(
            query="a",
            response="a",
            context=Context(prediction=["x", "y", "z"]),
        )
    ).to_dict() == {
        "type": "Hallucination",
        "value": 1.0,
        "parameters": {
            "evaluator": "mock",
            "retries": 0,
        },
    }

    mock_client.returning = verdicts_two_yes_one_no
    assert (
        calculate_hallucination(
            client=mock_client,
            system_prompt="",
            response="a",
            context=["x", "y", "z"],
        )
        == 2 / 3
    )
    assert evaluator.compute_hallucination(
        response=QueryResponse(
            query="a",
            response="a",
            context=Context(prediction=["x", "y", "z"]),
        )
    ).to_dict() == {
        "type": "Hallucination",
        "value": 2 / 3,
        "parameters": {
            "evaluator": "mock",
            "retries": 0,
        },
    }

    mock_client.returning = verdicts_all_no
    assert (
        calculate_hallucination(
            client=mock_client,
            system_prompt="",
            response="a",
            context=["x", "y", "z"],
        )
        == 0.0
    )
    assert evaluator.compute_hallucination(
        response=QueryResponse(
            query="a",
            response="a",
            context=Context(prediction=["x", "y", "z"]),
        )
    ).to_dict() == {
        "type": "Hallucination",
        "value": 0.0,
        "parameters": {
            "evaluator": "mock",
            "retries": 0,
        },
    }

    # hallucination score is dependent on context
    with pytest.raises(ValueError):
        mock_client.returning = verdicts_empty
        calculate_hallucination(
            client=mock_client,
            system_prompt="",
            response="a",
            context=[],
        )
    with pytest.raises(ValueError):
        evaluator.compute_hallucination(
            response=QueryResponse(
                query="a",
                response="a",
            )
        )
