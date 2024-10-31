import json

import pytest
from valor_lite.text_generation import Context, Evaluator, QueryResponse
from valor_lite.text_generation.computation import calculate_faithfulness


def test_faithfulness_no_context(mock_client):

    evaluator = Evaluator(client=mock_client)
    with pytest.raises(ValueError):
        assert evaluator.compute_faithfulness(
            response=QueryResponse(
                query="a",
                response="a",
            )
        )


def test_faithfulness_no_claims(mock_client):
    mock_client.returning = json.dumps(
        {
            "claims": [],
        }
    )

    assert (
        calculate_faithfulness(
            client=mock_client,
            system_prompt="",
            response="a",
            context=["x", "y", "z"],
        )
        == 1.0
    )

    evaluator = Evaluator(client=mock_client)
    assert evaluator.compute_faithfulness(
        response=QueryResponse(
            query="a",
            response="a",
            context=Context(prediction=["x", "y", "z"]),
        )
    ).to_dict() == {
        "type": "Faithfulness",
        "value": 1.0,
        "parameters": {
            "evaluator": "mock",
            "retries": 0,
        },
    }


def test_calculate_faithfulness(
    mock_client,
    verdicts_all_yes: str,
    verdicts_all_no: str,
    verdicts_two_yes_one_no: str,
    verdicts_empty: str,
):

    evaluator = Evaluator(client=mock_client)

    mock_client.returning = verdicts_all_yes
    assert (
        calculate_faithfulness(
            client=mock_client,
            system_prompt="",
            response="a",
            context=["x", "y", "z"],
        )
        == 1.0
    )
    assert evaluator.compute_faithfulness(
        response=QueryResponse(
            query="a",
            response="a",
            context=Context(prediction=["x", "y", "z"]),
        )
    ).to_dict() == {
        "type": "Faithfulness",
        "value": 1.0,
        "parameters": {
            "evaluator": "mock",
            "retries": 0,
        },
    }

    mock_client.returning = verdicts_two_yes_one_no
    assert (
        calculate_faithfulness(
            client=mock_client,
            system_prompt="",
            response="a",
            context=["x", "y", "z"],
        )
        == 2 / 3
    )
    assert evaluator.compute_faithfulness(
        response=QueryResponse(
            query="a",
            response="a",
            context=Context(prediction=["x", "y", "z"]),
        )
    ).to_dict() == {
        "type": "Faithfulness",
        "value": 2 / 3,
        "parameters": {
            "evaluator": "mock",
            "retries": 0,
        },
    }

    mock_client.returning = verdicts_all_no
    assert (
        calculate_faithfulness(
            client=mock_client,
            system_prompt="",
            response="a",
            context=["x", "y", "z"],
        )
        == 0.0
    )
    assert evaluator.compute_faithfulness(
        response=QueryResponse(
            query="a",
            response="a",
            context=Context(prediction=["x", "y", "z"]),
        )
    ).to_dict() == {
        "type": "Faithfulness",
        "value": 0.0,
        "parameters": {
            "evaluator": "mock",
            "retries": 0,
        },
    }

    mock_client.returning = verdicts_empty
    assert (
        calculate_faithfulness(
            client=mock_client,
            system_prompt="",
            response="a",
            context=[],
        )
        == 0.0
    )
    with pytest.raises(ValueError):
        evaluator.compute_faithfulness(
            response=QueryResponse(
                query="a",
                response="a",
            )
        )
