import math

from valor_lite.text_generation.computation import calculate_context_precision


def test_calculate_context_precision(
    mock_client,
    verdicts_all_yes: str,
    verdicts_all_no: str,
    verdicts_two_yes_one_no: str,
    verdicts_empty: str,
):

    mock_client.returning = verdicts_all_yes
    assert (
        calculate_context_precision(
            client=mock_client,
            system_prompt="",
            query="abcdefg",
            retrieved_context=["a", "b", "c"],
            groundtruth_context=["x", "y", "z"],
        )
        == 1.0
    )

    mock_client.returning = verdicts_two_yes_one_no
    assert math.isclose(
        calculate_context_precision(
            client=mock_client,
            system_prompt="",
            query="abcdefg",
            retrieved_context=["a", "b", "c"],
            groundtruth_context=["x", "y", "z"],
        ),
        5 / 6,
    )

    mock_client.returning = verdicts_all_no
    assert (
        calculate_context_precision(
            client=mock_client,
            system_prompt="",
            query="abcdefg",
            retrieved_context=["a", "b", "c"],
            groundtruth_context=["x", "y", "z"],
        )
        == 0.0
    )

    mock_client.returning = verdicts_empty
    assert (
        calculate_context_precision(
            client=mock_client,
            system_prompt="",
            query="abcdefg",
            retrieved_context=[],
            groundtruth_context=[],
        )
        == 1.0
    )

    mock_client.returning = verdicts_empty
    assert (
        calculate_context_precision(
            client=mock_client,
            system_prompt="",
            query="abcdefg",
            retrieved_context=["a"],
            groundtruth_context=[],
        )
        == 0.0
    )

    mock_client.returning = verdicts_empty
    assert (
        calculate_context_precision(
            client=mock_client,
            system_prompt="",
            query="abcdefg",
            retrieved_context=[],
            groundtruth_context=["b"],
        )
        == 0.0
    )
