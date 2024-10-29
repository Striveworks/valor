import pytest
from valor_lite.text_generation.computation import calculate_summary_coherence
from valor_lite.text_generation.llm.exceptions import InvalidLLMResponseError


def test_calculate_summary_coherence(mock_client):

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

    for i in [-1, 0, 6]:
        mock_client.returning = str(i)
        with pytest.raises(InvalidLLMResponseError):
            calculate_summary_coherence(
                client=mock_client,
                system_prompt="",
                text="a",
                summary="b",
            )
