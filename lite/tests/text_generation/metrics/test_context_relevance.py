import pytest
from valor_lite.text_generation import Context, Evaluator, QueryResponse
from valor_lite.text_generation.computation import calculate_context_relevance


def test_context_relevance_no_context(mock_client):

    evaluator = Evaluator(client=mock_client)
    with pytest.raises(ValueError):
        assert evaluator.compute_context_relevance(
            response=QueryResponse(
                query="a",
                response="a",
            )
        )


def test_calculate_context_relevance(
    mock_client,
    verdicts_all_yes: str,
    verdicts_all_no: str,
    verdicts_two_yes_one_no: str,
    verdicts_empty: str,
):

    evaluator = Evaluator(client=mock_client)

    mock_client.returning = verdicts_all_yes
    assert (
        calculate_context_relevance(
            client=mock_client,
            system_prompt="",
            query="a",
            context=["x", "y", "z"],
        )
        == 1.0
    )
    assert evaluator.compute_context_relevance(
        response=QueryResponse(
            query="a",
            response="a",
            context=Context(
                prediction=["x", "y", "z"],
            ),
        )
    ).to_dict() == {
        "type": "ContextRelevance",
        "value": 1.0,
        "parameters": {
            "evaluator": "mock",
            "retries": 0,
        },
    }

    mock_client.returning = verdicts_two_yes_one_no
    assert (
        calculate_context_relevance(
            client=mock_client,
            system_prompt="",
            query="a",
            context=["x", "y", "z"],
        )
        == 2 / 3
    )
    assert evaluator.compute_context_relevance(
        response=QueryResponse(
            query="a",
            response="a",
            context=Context(
                prediction=["x", "y", "z"],
            ),
        )
    ).to_dict() == {
        "type": "ContextRelevance",
        "value": 2 / 3,
        "parameters": {
            "evaluator": "mock",
            "retries": 0,
        },
    }

    mock_client.returning = verdicts_all_no
    assert (
        calculate_context_relevance(
            client=mock_client,
            system_prompt="",
            query="a",
            context=["x", "y", "z"],
        )
        == 0.0
    )
    assert evaluator.compute_context_relevance(
        response=QueryResponse(
            query="a",
            response="a",
            context=Context(
                prediction=["x", "y", "z"],
            ),
        )
    ).to_dict() == {
        "type": "ContextRelevance",
        "value": 0.0,
        "parameters": {
            "evaluator": "mock",
            "retries": 0,
        },
    }

    mock_client.returning = verdicts_empty
    assert (
        calculate_context_relevance(
            client=mock_client,
            system_prompt="",
            query="a",
            context=[],
        )
        == 0.0
    )
    assert evaluator.compute_context_relevance(
        response=QueryResponse(
            query="a",
            response="a",
            context=Context(
                groundtruth=[],
                prediction=[],
            ),
        )
    ).to_dict() == {
        "type": "ContextRelevance",
        "value": 0.0,
        "parameters": {
            "evaluator": "mock",
            "retries": 0,
        },
    }
