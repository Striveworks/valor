from valor_lite.text_generation.computation import calculate_context_relevance


def test_calculate_context_relevance(
    mock_client,
    verdicts_all_yes: str,
    verdicts_all_no: str,
    verdicts_two_yes_one_no: str,
    verdicts_empty: str,
):

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
