import pytest
from valor_lite.text_generation import Evaluator, QueryResponse
from valor_lite.text_generation.computation import calculate_summary_coherence
from valor_lite.text_generation.llm.exceptions import InvalidLLMResponseError


def test_calculate_summary_coherence(mock_client):

    evaluator = Evaluator(client=mock_client)

    for i in [1, 2, 3, 4, 5]:
        mock_client.returning = str(i)
        assert (
            calculate_summary_coherence(
                client=mock_client,
                system_prompt="",
                text="a",
                summary="b",
            )
            == i
        )
        assert evaluator.compute_summary_coherence(
            response=QueryResponse(
                query="a",
                response="b",
            )
        ).to_dict() == {
            "type": "SummaryCoherence",
            "value": i,
            "parameters": {
                "evaluator": "mock",
                "retries": 0,
            },
        }

    for i in [-1, 0, 6]:
        mock_client.returning = str(i)
        with pytest.raises(InvalidLLMResponseError):
            calculate_summary_coherence(
                client=mock_client,
                system_prompt="",
                text="a",
                summary="b",
            )

        assert evaluator.compute_summary_coherence(
            response=QueryResponse(
                query="a",
                response="b",
            )
        ).to_dict() == {
            "type": "Error",
            "value": {
                "type": "InvalidLLMResponseError",
                "message": f"Summary coherence score was not an integer between 1 and 5: {i}",
            },
            "parameters": {
                "evaluator": "mock",
                "retries": 0,
            },
        }

    for i in ["one", "three"]:
        mock_client.returning = str(i)
        with pytest.raises(InvalidLLMResponseError):
            calculate_summary_coherence(
                client=mock_client,
                system_prompt="",
                text="a",
                summary="b",
            )

        assert evaluator.compute_summary_coherence(
            response=QueryResponse(
                query="a",
                response="b",
            )
        ).to_dict() == {
            "type": "Error",
            "value": {
                "type": "InvalidLLMResponseError",
                "message": f"LLM response was not a valid summary coherence score: {i}",
            },
            "parameters": {
                "evaluator": "mock",
                "retries": 0,
            },
        }
