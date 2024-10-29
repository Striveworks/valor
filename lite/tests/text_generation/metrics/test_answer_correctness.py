import json

from valor_lite.text_generation.computation import calculate_answer_correctness


def test_calculate_answer_correctness(
    mock_client,
):
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
