import math

import pytest
from valor_lite.text_generation import Context, Evaluator, QueryResponse
from valor_lite.text_generation.computation import calculate_context_precision


def test_context_precision_no_context(mock_client):

    evaluator = Evaluator(client=mock_client)
    with pytest.raises(ValueError):
        assert evaluator.compute_context_precision(
            response=QueryResponse(
                query="a",
                response="a",
            )
        )


def test_calculate_context_precision(
    mock_client,
    verdicts_all_yes: str,
    verdicts_all_no: str,
    verdicts_two_yes_one_no: str,
    verdicts_empty: str,
):

    evaluator = Evaluator(client=mock_client)

    mock_client.returning = verdicts_all_yes
    assert (
        calculate_context_precision(
            client=mock_client,
            system_prompt="",
            query="abcdefg",
            predicted_context=["a", "b", "c"],
            groundtruth_context=["x", "y", "z"],
        )
        == 1.0
    )
    assert evaluator.compute_context_precision(
        response=QueryResponse(
            query="a",
            response="a",
            context=Context(
                groundtruth=["x", "y", "z"],
                prediction=["a", "b", "c"],
            ),
        )
    ).to_dict() == {
        "type": "ContextPrecision",
        "value": 1.0,
        "parameters": {
            "evaluator": "mock",
            "retries": 0,
        },
    }

    mock_client.returning = verdicts_two_yes_one_no
    assert math.isclose(
        calculate_context_precision(
            client=mock_client,
            system_prompt="",
            query="abcdefg",
            predicted_context=["a", "b", "c"],
            groundtruth_context=["x", "y", "z"],
        ),
        5 / 6,
    )
    assert evaluator.compute_context_precision(
        response=QueryResponse(
            query="a",
            response="a",
            context=Context(
                groundtruth=["x", "y", "z"],
                prediction=["a", "b", "c"],
            ),
        )
    ).to_dict() == {
        "type": "ContextPrecision",
        "value": 0.8333333333333333,
        "parameters": {
            "evaluator": "mock",
            "retries": 0,
        },
    }

    mock_client.returning = verdicts_all_no
    assert (
        calculate_context_precision(
            client=mock_client,
            system_prompt="",
            query="abcdefg",
            predicted_context=["a", "b", "c"],
            groundtruth_context=["x", "y", "z"],
        )
        == 0.0
    )
    assert evaluator.compute_context_precision(
        response=QueryResponse(
            query="a",
            response="a",
            context=Context(
                groundtruth=["x", "y", "z"],
                prediction=["a", "b", "c"],
            ),
        )
    ).to_dict() == {
        "type": "ContextPrecision",
        "value": 0.0,
        "parameters": {
            "evaluator": "mock",
            "retries": 0,
        },
    }

    mock_client.returning = verdicts_empty
    assert (
        calculate_context_precision(
            client=mock_client,
            system_prompt="",
            query="abcdefg",
            predicted_context=[],
            groundtruth_context=[],
        )
        == 1.0
    )
    assert evaluator.compute_context_precision(
        response=QueryResponse(
            query="a",
            response="a",
            context=Context(
                groundtruth=[],
                prediction=[],
            ),
        )
    )

    mock_client.returning = verdicts_empty
    assert (
        calculate_context_precision(
            client=mock_client,
            system_prompt="",
            query="abcdefg",
            predicted_context=["a"],
            groundtruth_context=[],
        )
        == 0.0
    )
    assert evaluator.compute_context_precision(
        response=QueryResponse(
            query="a",
            response="a",
            context=Context(
                groundtruth=[],
                prediction=["a"],
            ),
        )
    ).to_dict() == {
        "type": "ContextPrecision",
        "value": 0.0,
        "parameters": {
            "evaluator": "mock",
            "retries": 0,
        },
    }

    mock_client.returning = verdicts_empty
    assert (
        calculate_context_precision(
            client=mock_client,
            system_prompt="",
            query="abcdefg",
            predicted_context=[],
            groundtruth_context=["b"],
        )
        == 0.0
    )
    assert evaluator.compute_context_precision(
        response=QueryResponse(
            query="a",
            response="a",
            context=Context(
                groundtruth=["x"],
                prediction=[],
            ),
        )
    ).to_dict() == {
        "type": "ContextPrecision",
        "value": 0.0,
        "parameters": {
            "evaluator": "mock",
            "retries": 0,
        },
    }
