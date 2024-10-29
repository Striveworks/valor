import json
import math
from unittest.mock import patch

import pytest
from valor_lite.text_generation import ClientWrapper, Query
from valor_lite.text_generation.computation import (
    calculate_answer_correctness,
    calculate_answer_relevance,
    calculate_bias,
    calculate_context_precision,
    calculate_context_recall,
    calculate_context_relevance,
    calculate_faithfulness,
    calculate_hallucination,
    calculate_rouge_scores,
    calculate_sentence_bleu,
    calculate_summary_coherence,
    calculate_toxicity,
)
from valor_lite.text_generation.llm.exceptions import InvalidLLMResponseError


@pytest.fixture
def verdicts_all_yes() -> str:
    return json.dumps(
        {
            "verdicts": [
                {"verdict": "yes", "analysis": "some text"},
                {"verdict": "yes", "analysis": "some text"},
                {"verdict": "yes", "analysis": "some text"},
            ],
            "statements": ["x", "y", "z"],
            "opinions": ["x", "y", "z"],
            "claims": ["x", "y", "z"],
        }
    )


@pytest.fixture
def verdicts_all_no() -> str:
    return json.dumps(
        {
            "verdicts": [
                {"verdict": "no", "analysis": "some text"},
                {"verdict": "no", "analysis": "some text"},
                {"verdict": "no", "analysis": "some text"},
            ],
            "statements": ["x", "y", "z"],
            "opinions": ["x", "y", "z"],
            "claims": ["x", "y", "z"],
        }
    )


@pytest.fixture
def verdicts_two_yes_one_no() -> str:
    return json.dumps(
        {
            "verdicts": [
                {"verdict": "yes", "analysis": "some text"},
                {"verdict": "no", "analysis": "some text"},
                {"verdict": "yes", "analysis": "some text"},
            ],
            "statements": ["x", "y", "z"],
            "opinions": ["x", "y", "z"],
            "claims": ["x", "y", "z"],
        }
    )


@pytest.fixture
def verdicts_empty() -> str:
    return json.dumps(
        {
            "verdicts": [],
            "statements": [],
            "opinions": [],
            "claims": [],
        }
    )


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


def test_calculate_answer_relevance(
    mock_client,
    verdicts_all_yes: str,
    verdicts_all_no: str,
    verdicts_two_yes_one_no: str,
    verdicts_empty: str,
):

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


def test_calculate_bias(
    mock_client,
    verdicts_all_yes: str,
    verdicts_all_no: str,
    verdicts_two_yes_one_no: str,
    verdicts_empty: str,
):
    mock_client.returning = verdicts_all_yes
    assert (
        calculate_bias(
            client=mock_client,
            system_prompt="",
            response="a",
        )
        == 1.0
    )

    mock_client.returning = verdicts_two_yes_one_no
    assert (
        calculate_bias(
            client=mock_client,
            system_prompt="",
            response="a",
        )
        == 2 / 3
    )

    mock_client.returning = verdicts_all_no
    assert (
        calculate_bias(
            client=mock_client,
            system_prompt="",
            response="a",
        )
        == 0.0
    )

    mock_client.returning = verdicts_empty
    assert (
        calculate_bias(
            client=mock_client,
            system_prompt="",
            response="a",
        )
        == 0.0
    )


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


def test_calculate_context_recall(
    mock_client,
    verdicts_all_yes: str,
    verdicts_all_no: str,
    verdicts_two_yes_one_no: str,
    verdicts_empty: str,
):

    mock_client.returning = verdicts_all_yes
    assert (
        calculate_context_recall(
            client=mock_client,
            system_prompt="",
            retrieved_context=["a", "b", "c"],
            groundtruth_context=["x", "y", "z"],
        )
        == 1.0
    )

    mock_client.returning = verdicts_two_yes_one_no
    assert (
        calculate_context_recall(
            client=mock_client,
            system_prompt="",
            retrieved_context=["a", "b", "c"],
            groundtruth_context=["x", "y", "z"],
        )
        == 2 / 3
    )

    mock_client.returning = verdicts_all_no
    assert (
        calculate_context_recall(
            client=mock_client,
            system_prompt="",
            retrieved_context=["a", "b", "c"],
            groundtruth_context=["x", "y", "z"],
        )
        == 0.0
    )

    mock_client.returning = verdicts_empty
    assert (
        calculate_context_recall(
            client=mock_client,
            system_prompt="",
            retrieved_context=[],
            groundtruth_context=[],
        )
        == 1.0
    )

    mock_client.returning = verdicts_empty
    assert (
        calculate_context_recall(
            client=mock_client,
            system_prompt="",
            retrieved_context=["a"],
            groundtruth_context=[],
        )
        == 0.0
    )

    mock_client.returning = verdicts_empty
    assert (
        calculate_context_recall(
            client=mock_client,
            system_prompt="",
            retrieved_context=[],
            groundtruth_context=["b"],
        )
        == 0.0
    )


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


def test_calculate_faithfulness(
    mock_client,
    verdicts_all_yes: str,
    verdicts_all_no: str,
    verdicts_two_yes_one_no: str,
    verdicts_empty: str,
):

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


def test_calculate_hallucination(
    mock_client,
    verdicts_all_yes: str,
    verdicts_all_no: str,
    verdicts_two_yes_one_no: str,
    verdicts_empty: str,
):

    mock_client.returning = verdicts_all_yes
    assert (
        calculate_hallucination(
            client=mock_client,
            system_prompt="",
            response="a",
            context=["x", "y", "z"],
        )
        == 1.0
    )

    mock_client.returning = verdicts_two_yes_one_no
    assert (
        calculate_hallucination(
            client=mock_client,
            system_prompt="",
            response="a",
            context=["x", "y", "z"],
        )
        == 2 / 3
    )

    mock_client.returning = verdicts_all_no
    assert (
        calculate_hallucination(
            client=mock_client,
            system_prompt="",
            response="a",
            context=["x", "y", "z"],
        )
        == 0.0
    )

    # hallucination score is dependent on context
    with pytest.raises(ValueError):
        mock_client.returning = verdicts_empty
        calculate_hallucination(
            client=mock_client,
            system_prompt="",
            response="a",
            context=[],
        )


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


def test_calculate_toxicity(
    mock_client,
    verdicts_all_yes: str,
    verdicts_all_no: str,
    verdicts_two_yes_one_no: str,
    verdicts_empty: str,
):

    mock_client.returning = verdicts_all_yes
    assert (
        calculate_toxicity(
            client=mock_client,
            system_prompt="",
            response="a",
        )
        == 1.0
    )

    mock_client.returning = verdicts_two_yes_one_no
    assert (
        calculate_toxicity(
            client=mock_client,
            system_prompt="",
            response="a",
        )
        == 2 / 3
    )

    mock_client.returning = verdicts_all_no
    assert (
        calculate_toxicity(
            client=mock_client,
            system_prompt="",
            response="a",
        )
        == 0.0
    )

    mock_client.returning = verdicts_empty
    assert (
        calculate_toxicity(
            client=mock_client,
            system_prompt="",
            response="a",
        )
        == 0.0
    )


def test_calculate_rouge_scores(
    mock_client,
    verdicts_all_yes: str,
    verdicts_all_no: str,
    verdicts_two_yes_one_no: str,
    verdicts_empty: str,
):
    pass


def test_calculate_sentence_bleu(
    mock_client,
    verdicts_all_yes: str,
    verdicts_all_no: str,
    verdicts_two_yes_one_no: str,
    verdicts_empty: str,
):
    pass
