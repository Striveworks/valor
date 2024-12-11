from valor_lite.text_generation import Evaluator, QueryResponse
from valor_lite.text_generation.computation import calculate_answer_relevance


def test_calculate_answer_relevance(
    mock_client,
    verdicts_all_yes: str,
    verdicts_all_no: str,
    verdicts_two_yes_one_no: str,
    verdicts_empty: str,
):

    evaluator = Evaluator(client=mock_client)

    mock_client.returning = verdicts_all_yes
    assert (
        calculate_answer_relevance(
            client=mock_client,
            system_prompt="",
            query="a",
            response="a",
        )
        == 1.0
    )
    assert evaluator.compute_answer_relevance(
        response=QueryResponse(
            query="a",
            response="a",
        )
    ).to_dict() == {
        "type": "AnswerRelevance",
        "value": 1.0,
        "parameters": {
            "evaluator": "mock",
            "retries": 0,
        },
    }

    mock_client.returning = verdicts_two_yes_one_no
    assert (
        calculate_answer_relevance(
            client=mock_client,
            system_prompt="",
            query="a",
            response="a",
        )
        == 2 / 3
    )
    assert evaluator.compute_answer_relevance(
        response=QueryResponse(
            query="a",
            response="a",
        )
    ).to_dict() == {
        "type": "AnswerRelevance",
        "value": 2 / 3,
        "parameters": {
            "evaluator": "mock",
            "retries": 0,
        },
    }

    mock_client.returning = verdicts_all_no
    assert (
        calculate_answer_relevance(
            client=mock_client,
            system_prompt="",
            query="a",
            response="a",
        )
        == 0.0
    )
    assert evaluator.compute_answer_relevance(
        response=QueryResponse(
            query="a",
            response="a",
        )
    ).to_dict() == {
        "type": "AnswerRelevance",
        "value": 0.0,
        "parameters": {
            "evaluator": "mock",
            "retries": 0,
        },
    }

    mock_client.returning = verdicts_empty
    assert (
        calculate_answer_relevance(
            client=mock_client,
            system_prompt="",
            query="a",
            response="a",
        )
        == 0.0
    )
    assert evaluator.compute_answer_relevance(
        response=QueryResponse(
            query="a",
            response="a",
        )
    ).to_dict() == {
        "type": "AnswerRelevance",
        "value": 0.0,
        "parameters": {
            "evaluator": "mock",
            "retries": 0,
        },
    }
