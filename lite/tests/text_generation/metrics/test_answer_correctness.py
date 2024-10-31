import json

import pytest
from valor_lite.text_generation import Context, Evaluator, QueryResponse
from valor_lite.text_generation.computation import calculate_answer_correctness


def test_answer_correctness_no_context(mock_client):

    evaluator = Evaluator(client=mock_client)
    with pytest.raises(ValueError):
        assert evaluator.compute_answer_correctness(
            response=QueryResponse(
                query="a",
                response="a",
            )
        )


def test_calculate_answer_correctness(mock_client):

    evaluator = Evaluator(client=mock_client)

    mock_client.returning = json.dumps(
        {
            "TP": ["a", "b", "c"],
            "FP": ["d", "e"],
            "FN": ["f"],
            "statements": ["v", "w", "x", "y", "z"],
        }
    )
    assert (
        calculate_answer_correctness(
            client=mock_client,
            system_prompt="",
            query="a",
            response="a",
            groundtruths=["a", "b", "c"],
        )
        == 2 / 3
    )
    assert evaluator.compute_answer_correctness(
        response=QueryResponse(
            query="a",
            response="a",
            context=Context(
                groundtruth=["a", "b", "c"],
            ),
        )
    ).to_dict() == {
        "type": "AnswerCorrectness",
        "value": 2 / 3,
        "parameters": {
            "evaluator": "mock",
            "retries": 0,
        },
    }

    mock_client.returning = json.dumps(
        {"TP": [], "FP": ["a"], "FN": ["b"], "statements": ["c"]}
    )
    assert (
        calculate_answer_correctness(
            client=mock_client,
            system_prompt="",
            query="a",
            response="a",
            groundtruths=["a", "b", "c"],
        )
        == 0
    )
    assert evaluator.compute_answer_correctness(
        response=QueryResponse(
            query="a",
            response="a",
            context=Context(
                groundtruth=["a", "b", "c"],
            ),
        )
    ).to_dict() == {
        "type": "AnswerCorrectness",
        "value": 0,
        "parameters": {
            "evaluator": "mock",
            "retries": 0,
        },
    }
