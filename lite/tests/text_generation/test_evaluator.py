import json
from unittest.mock import Mock

import pytest
from valor_lite.text_generation import (
    Context,
    Evaluator,
    MistralWrapper,
    OpenAIWrapper,
    Query,
)
from valor_lite.text_generation.exceptions import InvalidLLMResponseError


class BadValueInTestLLMClientsError(Exception):
    """
    Raised when a mock function in test_llm_client.py receives a bad value.
    """

    pass


def _package_json(data: dict):
    return f"```json \n{json.dumps(data, indent=4)}\n```"


@pytest.fixture
def valid_claims() -> str:
    data = {"claims": ["claim 1", "claim 2", "claim 3", "claim 4", "claim 5"]}
    return _package_json(data)


@pytest.fixture
def valid_opinions() -> str:
    data = {"opinions": ["opinion 1", "opinion 2", "opinion 3", "opinion 4"]}
    return _package_json(data)


@pytest.fixture
def valid_statements() -> str:
    data = {
        "statements": [
            "statement 1",
            "statement 2",
            "statement 3",
            "statement 4",
        ]
    }
    return _package_json(data)


@pytest.fixture
def groundtruth_valid_statements() -> str:
    data = {
        "statements": [
            "gt statement 1",
            "gt statement 2",
            "gt statement 3",
            "gt statement 4",
        ]
    }
    return _package_json(data)


@pytest.fixture
def answer_correctness_valid_verdicts() -> str:
    data = {
        "TP": ["statement 1", "statement 2", "statement 4"],
        "FP": ["statement 3"],
        "FN": ["gt statement 1", "gt statement 4"],
    }
    return _package_json(data)


@pytest.fixture
def answer_relevance_valid_verdicts() -> str:
    data = {
        "verdicts": [
            {"verdict": "no"},
            {"verdict": "yes"},
            {"verdict": "idk"},
            {"verdict": "yes"},
        ]
    }
    return _package_json(data)


@pytest.fixture
def bias_valid_verdicts() -> str:
    data = {
        "verdicts": [
            {"verdict": "yes"},
            {"verdict": "no"},
            {"verdict": "yes"},
            {"verdict": "no"},
        ]
    }
    return _package_json(data)


@pytest.fixture
def context_precision_valid_verdicts_1() -> str:
    data = {
        "verdicts": [
            {"verdict": "no"},
            {"verdict": "yes"},
            {"verdict": "no"},
            {"verdict": "no"},
            {"verdict": "yes"},
        ]
    }
    return _package_json(data)


@pytest.fixture
def context_precision_valid_verdicts_2() -> str:
    data = {
        "verdicts": [
            {"verdict": "no"},
            {"verdict": "no"},
            {"verdict": "no"},
            {"verdict": "no"},
            {"verdict": "no"},
        ]
    }
    return _package_json(data)


@pytest.fixture
def context_recall_valid_verdicts() -> str:
    data = {
        "verdicts": [
            {"verdict": "yes"},
            {"verdict": "yes"},
            {"verdict": "no"},
            {"verdict": "yes"},
        ]
    }
    return _package_json(data)


@pytest.fixture
def context_relevance_valid_verdicts() -> str:
    data = {
        "verdicts": [{"verdict": "no"}, {"verdict": "yes"}, {"verdict": "no"}]
    }
    return _package_json(data)


@pytest.fixture
def faithfulness_valid_verdicts() -> str:
    data = {
        "verdicts": [
            {"verdict": "no"},
            {"verdict": "yes"},
            {"verdict": "yes"},
            {"verdict": "yes"},
            {"verdict": "no"},
        ]
    }
    return _package_json(data)


@pytest.fixture
def hallucination_valid_verdicts() -> str:
    data = {
        "verdicts": [{"verdict": "no"}, {"verdict": "yes"}, {"verdict": "yes"}]
    }
    return _package_json(data)


@pytest.fixture
def toxicity_valid_verdicts() -> str:
    data = {
        "verdicts": [
            {"verdict": "yes"},
            {"verdict": "no"},
            {"verdict": "yes"},
            {"verdict": "no"},
        ]
    }
    return _package_json(data)


def _return_valid_answer_correctness_response(*args, **kwargs):
    if "prediction text" in args[1][1]["content"]:
        return valid_statements
    elif "ground truth text" in args[1][1]["content"]:
        return groundtruth_valid_statements
    elif (
        "Return in JSON format with three keys: 'TP', 'FP', and 'FN'"
        in args[1][1]["content"]
    ):
        return answer_correctness_valid_verdicts
    else:
        raise BadValueInTestLLMClientsError


def _return_invalid1_answer_correctness_response(*args, **kwargs):
    return _package_json(
        {"list": ["statement 1", "statement 2", "statement 3", "statement 4"]}
    )


def _return_invalid2_answer_correctness_response(*args, **kwargs):
    if "prediction text" in args[1][1]["content"]:
        return valid_statements
    elif "ground truth text" in args[1][1]["content"]:
        return _package_json(
            {"statements": ["statement 1", 4, "statement 3", "statement 4"]}
        )
    else:
        raise BadValueInTestLLMClientsError


def _return_invalid3_answer_correctness_response(*args, **kwargs):
    if "prediction text" in args[1][1]["content"]:
        return valid_statements
    elif "ground truth text" in args[1][1]["content"]:
        return groundtruth_valid_statements
    elif (
        "Return in JSON format with three keys: 'TP', 'FP', and 'FN'"
        in args[1][1]["content"]
    ):
        return _package_json(
            {
                "TP": ["statement 1", "statement 2", "statement 4"],
                "FP": ["statement 3"],
            }
        )
    else:
        raise BadValueInTestLLMClientsError


def _return_invalid4_answer_correctness_response(*args, **kwargs):
    if "prediction text" in args[1][1]["content"]:
        return valid_statements
    elif "ground truth text" in args[1][1]["content"]:
        return groundtruth_valid_statements
    elif (
        "Return in JSON format with three keys: 'TP', 'FP', and 'FN'"
        in args[1][1]["content"]
    ):
        return _package_json(
            {
                "TP": "statement 1",
                "FP": ["statement 3"],
                "FN": ["gt statement 1", "gt statement 4"],
            }
        )
    else:
        raise BadValueInTestLLMClientsError


def _return_invalid5_answer_correctness_response(*args, **kwargs):
    if "prediction text" in args[1][1]["content"]:
        return valid_statements
    elif "ground truth text" in args[1][1]["content"]:
        return groundtruth_valid_statements
    elif (
        "Return in JSON format with three keys: 'TP', 'FP', and 'FN'"
        in args[1][1]["content"]
    ):
        return _package_json(
            {
                "TP": ["statement 1", "statement 2"],
                "FP": ["statement 3"],
                "FN": ["gt statement 1", "gt statement 4"],
            }
        )
    else:
        raise BadValueInTestLLMClientsError


def _return_invalid6_answer_correctness_response(*args, **kwargs):
    if "prediction text" in args[1][1]["content"]:
        return valid_statements
    elif "ground truth text" in args[1][1]["content"]:
        return groundtruth_valid_statements
    elif (
        "Return in JSON format with three keys: 'TP', 'FP', and 'FN'"
        in args[1][1]["content"]
    ):
        return _package_json(
            {
                "TP": ["statement 1", "statement 2", "statement 4"],
                "FP": ["statement 3"],
                "FN": [
                    "gt statement 1",
                    "gt statement 2",
                    "gt statement 3",
                    "gt statement 4",
                    "too many statements in 'FN'",
                ],
            }
        )
    else:
        raise BadValueInTestLLMClientsError


def _return_valid_answer_relevance_response(*args, **kwargs):
    if "generate a list of STATEMENTS" in args[1][1]["content"]:
        return valid_statements
    elif (
        "generate a list of verdicts that indicate whether each statement is relevant to address the query"
        in args[1][1]["content"]
    ):
        return answer_relevance_valid_verdicts
    else:
        raise BadValueInTestLLMClientsError


def _return_invalid1_answer_relevance_response(*args, **kwargs):
    return _package_json(
        {"list": ["statement 1", "statement 2", "statement 3", "statement 4"]}
    )


def _return_invalid2_answer_relevance_response(*args, **kwargs):
    return _package_json(
        {"statements": ["statement 1", 5, "statement 3", "statement 4"]}
    )


def _return_invalid3_answer_relevance_response(*args, **kwargs):
    if "generate a list of STATEMENTS" in args[1][1]["content"]:
        return valid_statements
    elif (
        "generate a list of verdicts that indicate whether each statement is relevant to address the query"
        in args[1][1]["content"]
    ):
        return _package_json(
            {
                "list": [
                    {
                        "verdict": "no",
                        "reason": "The statement has nothing to do with the query.",
                    },
                    {"verdict": "yes"},
                    {"verdict": "idk"},
                    {"verdict": "yes"},
                ]
            }
        )
    else:
        raise BadValueInTestLLMClientsError


def _return_invalid4_answer_relevance_response(*args, **kwargs):
    if "generate a list of STATEMENTS" in args[1][1]["content"]:
        return valid_statements
    elif (
        "generate a list of verdicts that indicate whether each statement is relevant to address the query"
        in args[1][1]["content"]
    ):
        return _package_json(
            {
                "verdicts": [
                    {
                        "verdict": "no",
                        "reason": "The statement has nothing to do with the query.",
                    },
                    {"verdict": "yes"},
                    {"verdict": "idk"},
                    {"verdict": "unsure"},
                ]
            }
        )
    else:
        raise BadValueInTestLLMClientsError


def _return_valid1_bias_response(*args, **kwargs):
    if "generate a list of OPINIONS" in args[1][1]["content"]:
        return valid_opinions
    elif (
        "generate a list of verdicts to indicate whether EACH opinion is biased"
        in args[1][1]["content"]
    ):
        return bias_valid_verdicts
    else:
        raise BadValueInTestLLMClientsError


def _return_valid2_bias_response(*args, **kwargs):
    return _package_json({"opinions": []})


def _return_invalid1_bias_response(*args, **kwargs):
    return _package_json(
        {
            "verdicts": [
                "opinion 1",
                "verdict 2",
                "these should not be verdicts, these should be opinions",
                "the key above should be 'opinions' not 'verdicts'",
            ]
        }
    )


def _return_invalid2_bias_response(*args, **kwargs):
    return _package_json(
        {
            "opinions": [
                ["a list of opinions"],
                "opinion 2",
                "opinion 3",
                "opinion 4",
            ]
        }
    )


def _return_invalid3_bias_response(*args, **kwargs):
    if "generate a list of OPINIONS" in args[1][1]["content"]:
        return valid_opinions
    elif (
        "generate a list of verdicts to indicate whether EACH opinion is biased"
        in args[1][1]["content"]
    ):
        return _package_json(
            {
                "opinions": [
                    "the key should be 'verdicts' not 'opinions'",
                    "opinion 2",
                    "opinion 3",
                    "opinion 4",
                ]
            }
        )
    else:
        raise BadValueInTestLLMClientsError


def _return_invalid4_bias_response(*args, **kwargs):
    if "generate a list of OPINIONS" in args[1][1]["content"]:
        return valid_opinions
    elif (
        "generate a list of verdicts to indicate whether EACH opinion is biased"
        in args[1][1]["content"]
    ):
        return _package_json(
            {
                "verdicts": [
                    {
                        "verdict": "yes",
                        "reason": "This opinion demonstrates gender bias.",
                    },
                    {"verdict": "idk"},
                    {
                        "verdict": "yes",
                        "reason": "This opinion demonstrates political bias.",
                    },
                    {"verdict": "no"},
                ]
            }
        )
    else:
        raise BadValueInTestLLMClientsError


def _return_valid_context_relevance_response(*args, **kwargs):
    return context_relevance_valid_verdicts


def _return_invalid1_context_relevance_response(*args, **kwargs):
    return _package_json(
        {
            "all_verdicts": [
                {"verdict": "no"},
                {"verdict": "yes"},
                {"verdict": "no"},
            ]
        }
    )


def _return_valid1_context_precision_response(*args, **kwargs):
    return context_precision_valid_verdicts_1


def _return_valid2_context_precision_response(*args, **kwargs):
    return context_precision_valid_verdicts_2


def _return_invalid1_context_precision_response(*args, **kwargs):
    return _package_json(
        {"invalid_key": ["verdict 1", "verdict 2", "verdict 3"]}
    )


def _return_valid_context_recall_response(*args, **kwargs):
    if "generate a list of STATEMENTS" in args[1][1]["content"]:
        return valid_statements
    elif (
        "analyze each ground truth statement and determine if the statement can be attributed to the given context."
        in args[1][1]["content"]
    ):
        return context_recall_valid_verdicts
    else:
        raise BadValueInTestLLMClientsError


def _return_invalid1_context_recall_response(*args, **kwargs):
    return _package_json(
        {
            "invalid_key": [
                "statement 1",
                "statement 2",
                "statement 3",
                "statement 4",
            ]
        }
    )


def _return_invalid2_context_recall_response(*args, **kwargs):
    return _package_json(
        {"statements": [1, "statement 2", "statement 3", "statement 4"]}
    )


def _return_invalid3_context_recall_response(*args, **kwargs):
    if "generate a list of STATEMENTS" in args[1][1]["content"]:
        return valid_statements
    elif (
        "analyze each ground truth statement and determine if the statement can be attributed to the given context."
        in args[1][1]["content"]
    ):
        return _package_json(
            {
                "invalid_key": [
                    "verdict 1",
                    "verdict 2",
                    "verdict 3",
                    "verdict 4",
                ]
            }
        )
    else:
        raise BadValueInTestLLMClientsError


def _return_invalid4_context_recall_response(*args, **kwargs):
    if "generate a list of STATEMENTS" in args[1][1]["content"]:
        return valid_statements
    elif (
        "analyze each ground truth statement and determine if the statement can be attributed to the given context."
        in args[1][1]["content"]
    ):
        return _package_json(
            {"verdicts": ["verdict 1", "verdict 2", "verdict 3"]}
        )
    else:
        raise BadValueInTestLLMClientsError


def _return_valid1_faithfulness_response(*args, **kwargs):
    if (
        "generate a comprehensive list of FACTUAL CLAIMS"
        in args[1][1]["content"]
    ):
        return valid_claims
    elif (
        "generate a list of verdicts to indicate whether EACH claim is implied by the context list"
        in args[1][1]["content"]
    ):
        return faithfulness_valid_verdicts
    else:
        raise BadValueInTestLLMClientsError


def _return_valid2_faithfulness_response(*args, **kwargs):
    return _package_json({"claims": []})


def _return_invalid1_faithfulness_response(*args, **kwargs):
    return _package_json({"invalid_key": ["claim 1", "claim 2"]})


def _return_invalid2_faithfulness_response(*args, **kwargs):
    return _package_json({"claims": [["claim 1", "claim 2"]]})


def _return_invalid3_faithfulness_response(*args, **kwargs):
    if (
        "generate a comprehensive list of FACTUAL CLAIMS"
        in args[1][1]["content"]
    ):
        return valid_claims
    elif (
        "generate a list of verdicts to indicate whether EACH claim is implied by the context list"
        in args[1][1]["content"]
    ):
        return _package_json(
            {
                "bad key": [
                    {"verdict": "no"},
                    {"verdict": "yes"},
                    {"verdict": "yes"},
                    {"verdict": "yes"},
                    {"verdict": "no"},
                ]
            }
        )
    else:
        raise BadValueInTestLLMClientsError


def _return_invalid4_faithfulness_response(*args, **kwargs):
    if (
        "generate a comprehensive list of FACTUAL CLAIMS"
        in args[1][1]["content"]
    ):
        return valid_claims
    elif (
        "generate a list of verdicts to indicate whether EACH claim is implied by the context list"
        in args[1][1]["content"]
    ):
        return _package_json(
            {
                "verdicts": [
                    {"verdict": "no"},
                    {"verdict": "yes"},
                    {"verdict": "yes"},
                    {"verdict": "yes"},
                ]
            }
        )
    else:
        raise BadValueInTestLLMClientsError


def _return_invalid5_faithfulness_response(*args, **kwargs):
    if (
        "generate a comprehensive list of FACTUAL CLAIMS"
        in args[1][1]["content"]
    ):
        return valid_claims
    elif (
        "generate a list of verdicts to indicate whether EACH claim is implied by the context list"
        in args[1][1]["content"]
    ):
        return _package_json(
            {
                "verdicts": [
                    {"verdict": "idk"},
                    {"verdict": "yes"},
                    {"verdict": "yes"},
                    {"verdict": "idk"},
                    {"verdict": "no"},
                ]
            }
        )
    else:
        raise BadValueInTestLLMClientsError


def _return_valid_hallucination_response(*args, **kwargs):
    return hallucination_valid_verdicts


def _return_invalid1_hallucination_response(*args, **kwargs):
    return _package_json(
        {
            "bad key": [
                {"verdict": "yes"},
                {"verdict": "no"},
                {"verdict": "yes"},
            ]
        }
    )


def _return_valid_summary_coherence_response(*args, **kwargs):
    return "5"


def _return_invalid1_summary_coherence_response(*args, **kwargs):
    return "The score is 5."


def _return_invalid2_summary_coherence_response(*args, **kwargs):
    return "0"


def _return_valid1_toxicity_response(*args, **kwargs):
    if "generate a list of OPINIONS" in args[1][1]["content"]:
        return valid_opinions
    elif (
        "generate a list of verdicts to indicate whether EACH opinion is toxic"
        in args[1][1]["content"]
    ):
        return toxicity_valid_verdicts
    else:
        raise BadValueInTestLLMClientsError


def _return_valid2_toxicity_response(*args, **kwargs):
    return _package_json({"opinions": []})


def _return_invalid1_toxicity_response(*args, **kwargs):
    return _package_json(
        {
            "verdicts": [
                "opinion 1",
                "verdict 2",
                "these should not be verdicts, these should be opinions",
                "the key above should be 'opinions' not 'verdicts'",
            ]
        }
    )


def _return_invalid2_toxicity_response(*args, **kwargs):
    return _package_json(
        {"opinions": ["opinion 1", "opinion 2", 0.8, "opinion 4"]}
    )


def _return_invalid3_toxicity_response(*args, **kwargs):
    if "generate a list of OPINIONS" in args[1][1]["content"]:
        return valid_opinions
    elif (
        "generate a list of verdicts to indicate whether EACH opinion is toxic"
        in args[1][1]["content"]
    ):
        return _package_json(
            {"opinions": ["opinion 1", "opinion 2", "opinion 3", "opinion 4"]}
        )
    else:
        raise BadValueInTestLLMClientsError


def _return_invalid4_toxicity_response(*args, **kwargs):
    if "generate a list of OPINIONS" in args[1][1]["content"]:
        return valid_opinions
    elif (
        "generate a list of verdicts to indicate whether EACH opinion is toxic"
        in args[1][1]["content"]
    ):
        return _package_json(
            {
                "verdicts": [
                    {
                        "verdict": "yes",
                        "reason": "This opinion demonstrates gender bias.",
                    },
                    {"verdict": "no"},
                    {
                        "verdict": "yes",
                        "reason": "This opinion demonstrates political bias.",
                    },
                    {"verdict": "idk"},
                ]
            }
        )
    else:
        raise BadValueInTestLLMClientsError


def test_evaluator(
    monkeypatch,
    client,
    valid_claims: str,
    valid_opinions: str,
    valid_statements: str,
    groundtruth_valid_statements: str,
    answer_correctness_valid_verdicts: str,
    answer_relevance_valid_verdicts: str,
    bias_valid_verdicts: str,
    context_precision_valid_verdicts_1: str,
    context_precision_valid_verdicts_2: str,
    context_recall_valid_verdicts: str,
    context_relevance_valid_verdicts: str,
    faithfulness_valid_verdicts: str,
    hallucination_valid_verdicts: str,
    toxicity_valid_verdicts: str,
):
    """
    Check that LLMClient throws NotImplementedErrors for connect and __call__.

    Check the metric computations for LLMClient. The client children inherit all of these metric computations.
    """

    evaluator = Evaluator(client=client)

    # Patch __call__ with a valid response.
    # monkeypatch.setattr(
    #     "valor_lite.text_generation.integrations.MockWrapper.__call__",
    #     _return_valid_answer_correctness_response,
    # )
    assert 0.6666666666666666 == evaluator.compute_answer_correctness(
        Query(
            query="some query",
            response="prediction text",
            context=Context(groundtruth=["ground truth text"], prediction=[]),
        )
    )

    # Need to include ground truth statements.
    monkeypatch.setattr(
        "valor_lite.text_generation.integrations.MockWrapper.__call__",
        _return_valid_answer_correctness_response,
    )
    with pytest.raises(ValueError):
        evaluator.compute_answer_correctness(
            Query(
                query="some query",
                response="prediction text",
            )
        )

    # Needs to have 'statements' key.
    monkeypatch.setattr(
        "valor_lite.text_generation.integrations.MockWrapper.__call__",
        _return_invalid1_answer_correctness_response,
    )
    with pytest.raises(InvalidLLMResponseError):
        evaluator.compute_answer_correctness(
            Query(
                query="some query",
                response="prediction text",
                context=Context(
                    groundtruth=["ground truth text"],
                ),
            )
        )

    # Should fail if ground truth statements are invalid even when prediction statements are valid
    monkeypatch.setattr(
        "valor_lite.text_generation.integrations.MockWrapper.__call__",
        _return_invalid2_answer_correctness_response,
    )
    with pytest.raises(InvalidLLMResponseError):
        evaluator.compute_answer_correctness(
            Query(
                query="some query",
                response="prediction text",
                context=Context(
                    groundtruth=["ground truth text"],
                ),
            )
        )

    # Missing 'FN' in dictionary
    monkeypatch.setattr(
        "valor_lite.text_generation.integrations.MockWrapper.__call__",
        _return_invalid3_answer_correctness_response,
    )
    with pytest.raises(InvalidLLMResponseError):
        evaluator.compute_answer_correctness(
            Query(
                query="some query",
                response="prediction text",
                context=Context(
                    groundtruth=["ground truth text"],
                ),
            )
        )

    # TP has an invalid value.
    monkeypatch.setattr(
        "valor_lite.text_generation.integrations.MockWrapper.__call__",
        _return_invalid4_answer_correctness_response,
    )
    with pytest.raises(InvalidLLMResponseError):
        evaluator.compute_answer_correctness(
            Query(
                query="some query",
                response="prediction text",
                context=Context(
                    groundtruth=["ground truth text"],
                ),
            )
        )

    # Number of TP + FP does not equal the number of prediction statements
    monkeypatch.setattr(
        "valor_lite.text_generation.integrations.MockWrapper.__call__",
        _return_invalid5_answer_correctness_response,
    )
    with pytest.raises(InvalidLLMResponseError):
        evaluator.compute_answer_correctness(
            Query(
                query="some query",
                response="prediction text",
                context=Context(
                    groundtruth=["ground truth text"],
                ),
            )
        )

    # The number of FN is more than the number of ground truth statements
    monkeypatch.setattr(
        "valor_lite.text_generation.integrations.MockWrapper.__call__",
        _return_invalid6_answer_correctness_response,
    )
    with pytest.raises(InvalidLLMResponseError):
        evaluator.compute_answer_correctness(
            Query(
                query="some query",
                response="prediction text",
                context=Context(
                    groundtruth=["ground truth text"],
                ),
            )
        )

    # Patch __call__ with a valid response.
    monkeypatch.setattr(
        "valor_lite.text_generation.integrations.MockWrapper.__call__",
        _return_valid_answer_relevance_response,
    )
    assert 0.5 == evaluator.compute_answer_relevance(
        Query(query="some query", response="some answer")
    )

    # Needs to have 'statements' key.
    monkeypatch.setattr(
        "valor_lite.text_generation.integrations.MockWrapper.__call__",
        _return_invalid1_answer_relevance_response,
    )
    with pytest.raises(InvalidLLMResponseError):
        evaluator.compute_answer_relevance(
            Query(query="some query", response="some answer")
        )

    # Statements must be strings.
    monkeypatch.setattr(
        "valor_lite.text_generation.integrations.MockWrapper.__call__",
        _return_invalid2_answer_relevance_response,
    )
    with pytest.raises(InvalidLLMResponseError):
        evaluator.compute_answer_relevance(
            Query(query="some query", response="some text")
        )

    # Needs to have 'verdicts' key.
    monkeypatch.setattr(
        "valor_lite.text_generation.integrations.MockWrapper.__call__",
        _return_invalid3_answer_relevance_response,
    )
    with pytest.raises(InvalidLLMResponseError):
        evaluator.compute_answer_relevance(
            Query(query="some query", response="some text")
        )

    # Invalid verdict, all verdicts must be yes, no or idk.
    monkeypatch.setattr(
        "valor_lite.text_generation.integrations.MockWrapper.__call__",
        _return_invalid4_answer_relevance_response,
    )
    with pytest.raises(InvalidLLMResponseError):
        evaluator.compute_answer_relevance(
            Query(query="some query", response="some text")
        )

    # Patch __call__ with a valid response.
    monkeypatch.setattr(
        "valor_lite.text_generation.integrations.MockWrapper.__call__",
        _return_valid1_bias_response,
    )
    assert 0.5 == evaluator.compute_bias(
        Query(query="some query", response="some text")
    )

    # No opinions found, so no bias should be reported.
    monkeypatch.setattr(
        "valor_lite.text_generation.integrations.MockWrapper.__call__",
        _return_valid2_bias_response,
    )
    assert 0.0 == evaluator.compute_bias(
        Query(query="some query", response="some text")
    )

    # Key 'verdicts' is returned but the key should be 'opinions'.
    monkeypatch.setattr(
        "valor_lite.text_generation.integrations.MockWrapper.__call__",
        _return_invalid1_bias_response,
    )
    with pytest.raises(InvalidLLMResponseError):
        evaluator.compute_bias(Query(query="some query", response="some text"))

    # Opinions must be strings.
    monkeypatch.setattr(
        "valor_lite.text_generation.integrations.MockWrapper.__call__",
        _return_invalid2_bias_response,
    )
    with pytest.raises(InvalidLLMResponseError):
        evaluator.compute_bias(Query(query="some query", response="some text"))

    # Key 'opinions' is returned but the key should be 'verdicts'.
    monkeypatch.setattr(
        "valor_lite.text_generation.integrations.MockWrapper.__call__",
        _return_invalid3_bias_response,
    )
    with pytest.raises(InvalidLLMResponseError):
        evaluator.compute_bias(Query(query="some query", response="some text"))

    # 'idk' is not a valid bias verdict.
    monkeypatch.setattr(
        "valor_lite.text_generation.integrations.MockWrapper.__call__",
        _return_invalid4_bias_response,
    )
    with pytest.raises(InvalidLLMResponseError):
        evaluator.compute_bias(Query(query="some query", response="some text"))

    # Patch __call__ with a valid response.
    monkeypatch.setattr(
        "valor_lite.text_generation.integrations.MockWrapper.__call__",
        _return_valid1_context_precision_response,
    )
    assert 0.45 == evaluator.compute_context_precision(
        Query(
            query="some query",
            response="",
            context=Context(
                groundtruth=["some ground truth"],
                prediction=[
                    "context 1",
                    "context 2",
                    "context 3",
                    "context 4",
                    "context 5",
                ],
            ),
        )
    )

    # If all verdicts are "no", the returned score should be 0.
    monkeypatch.setattr(
        "valor_lite.text_generation.integrations.MockWrapper.__call__",
        _return_valid2_context_precision_response,
    )
    assert 0.0 == evaluator.compute_context_precision(
        Query(
            query="some query",
            response="",
            context=Context(
                groundtruth=["some ground truth"],
                prediction=[
                    "context 1",
                    "context 2",
                    "context 3",
                    "context 4",
                    "context 5",
                ],
            ),
        )
    )

    # Context precision is meaningless if context_list is empty.
    monkeypatch.setattr(
        "valor_lite.text_generation.integrations.MockWrapper.__call__",
        _return_valid1_context_precision_response,
    )
    with pytest.raises(ValueError):
        evaluator.compute_context_precision(
            Query(
                query="some query",
                response="",
                context=Context(
                    groundtruth=["some ground truth"],
                    prediction=[],
                ),
            )
        )

    # Context precision is meaningless if groundtruth_list is empty.
    monkeypatch.setattr(
        "valor_lite.text_generation.integrations.MockWrapper.__call__",
        _return_valid1_context_precision_response,
    )
    with pytest.raises(ValueError):
        evaluator.compute_context_precision(
            Query(
                query="some query",
                response="",
                context=Context(
                    groundtruth=[],
                    prediction=[
                        "context 1",
                        "context 2",
                        "context 3",
                        "context 4",
                        "context 5",
                    ],
                ),
            )
        )

    # Only 1 context provided but 5 verdicts were returned.
    monkeypatch.setattr(
        "valor_lite.text_generation.integrations.MockWrapper.__call__",
        _return_valid1_context_precision_response,
    )
    with pytest.raises(InvalidLLMResponseError):
        evaluator.compute_context_precision(
            Query(
                query="some query",
                response="",
                context=Context(
                    groundtruth=["some ground truth"],
                    prediction=[
                        "length of context list does not match LLM's response"
                    ],
                ),
            )
        )

    # Key 'invalid_key' is returned but the key should be 'verdicts'.
    monkeypatch.setattr(
        "valor_lite.text_generation.integrations.MockWrapper.__call__",
        _return_invalid1_context_precision_response,
    )
    with pytest.raises(InvalidLLMResponseError):
        evaluator.compute_context_precision(
            Query(
                query="some query",
                response="",
                context=Context(
                    groundtruth=["some ground truth"],
                    prediction=[
                        "context 1",
                        "context 2",
                        "context 3",
                        "context 4",
                        "context 5",
                    ],
                ),
            )
        )

    # Patch __call__ with a valid response.
    monkeypatch.setattr(
        "valor_lite.text_generation.integrations.MockWrapper.__call__",
        _return_valid_context_recall_response,
    )
    assert 0.75 == evaluator.compute_context_recall(
        Query(
            query="some query",
            response="",
            context=Context(
                groundtruth=["some ground truth"],
                prediction=[
                    "context 1",
                    "context 2",
                ],
            ),
        )
    )

    # Context recall is meaningless if context_list is empty.
    monkeypatch.setattr(
        "valor_lite.text_generation.integrations.MockWrapper.__call__",
        _return_valid_context_recall_response,
    )
    with pytest.raises(ValueError):
        evaluator.compute_context_recall(
            Query(
                query="some query",
                response="",
                context=Context(
                    groundtruth=["some ground truth"],
                    prediction=[],
                ),
            )
        )

    # Context recall is meaningless if groundtruth_list is empty.
    monkeypatch.setattr(
        "valor_lite.text_generation.integrations.MockWrapper.__call__",
        _return_valid_context_recall_response,
    )
    with pytest.raises(ValueError):
        evaluator.compute_context_recall(
            Query(
                query="some query",
                response="",
                context=Context(
                    groundtruth=[],
                    prediction=[
                        "context 1",
                        "context 2",
                    ],
                ),
            )
        )

    # Ground truth statements response must have key 'statements'.
    monkeypatch.setattr(
        "valor_lite.text_generation.integrations.MockWrapper.__call__",
        _return_invalid1_context_recall_response,
    )
    with pytest.raises(InvalidLLMResponseError):
        evaluator.compute_context_recall(
            Query(
                query="some query",
                response="",
                context=Context(
                    groundtruth=["some ground truth"],
                    prediction=[
                        "context 1",
                        "context 2",
                    ],
                ),
            )
        )

    # Ground truth statements must be strings.
    monkeypatch.setattr(
        "valor_lite.text_generation.integrations.MockWrapper.__call__",
        _return_invalid2_context_recall_response,
    )
    with pytest.raises(InvalidLLMResponseError):
        evaluator.compute_context_recall(
            Query(
                query="some query",
                response="",
                context=Context(
                    groundtruth=["some ground truth"],
                    prediction=[
                        "context 1",
                        "context 2",
                    ],
                ),
            )
        )

    # Context recall verdicts response must have key 'verdicts'.
    monkeypatch.setattr(
        "valor_lite.text_generation.integrations.MockWrapper.__call__",
        _return_invalid3_context_recall_response,
    )
    with pytest.raises(InvalidLLMResponseError):
        evaluator.compute_context_recall(
            Query(
                query="some query",
                response="",
                context=Context(
                    groundtruth=["some ground truth"],
                    prediction=[
                        "context 1",
                        "context 2",
                    ],
                ),
            )
        )

    # Number of context recall verdicts doesn't match the number of ground truth statements.
    monkeypatch.setattr(
        "valor_lite.text_generation.integrations.MockWrapper.__call__",
        _return_invalid4_context_recall_response,
    )
    with pytest.raises(InvalidLLMResponseError):
        evaluator.compute_context_recall(
            Query(
                query="some query",
                response="",
                context=Context(
                    groundtruth=["some ground truth"],
                    prediction=[
                        "context 1",
                        "context 2",
                    ],
                ),
            )
        )

    # Patch __call__ with a valid response.
    monkeypatch.setattr(
        "valor_lite.text_generation.integrations.MockWrapper.__call__",
        _return_valid_context_relevance_response,
    )
    assert 0.3333333333333333 == evaluator.compute_context_relevance(
        Query(
            query="some query",
            response="",
            context=Context(
                groundtruth=["some ground truth"],
                prediction=["context 1", "context 2", "context 3"],
            ),
        )
    )

    # Context relevance is meaningless if context_list is empty.
    monkeypatch.setattr(
        "valor_lite.text_generation.integrations.MockWrapper.__call__",
        _return_valid_context_relevance_response,
    )
    with pytest.raises(ValueError):
        evaluator.compute_context_relevance(
            Query(
                query="some query",
                response="",
            )
        )

    # Only 1 context provided but 3 verdicts were returned.
    monkeypatch.setattr(
        "valor_lite.text_generation.integrations.MockWrapper.__call__",
        _return_valid_context_relevance_response,
    )
    with pytest.raises(InvalidLLMResponseError):
        evaluator.compute_context_relevance(
            Query(
                query="some query",
                response="",
                context=Context(
                    groundtruth=[],
                    prediction=[
                        "length of context list does not match LLM's response"
                    ],
                ),
            )
        )

    # Key 'all_verdicts' is returned but the key should be 'verdicts'.
    monkeypatch.setattr(
        "valor_lite.text_generation.integrations.MockWrapper.__call__",
        _return_invalid1_context_relevance_response,
    )
    with pytest.raises(InvalidLLMResponseError):
        evaluator.compute_context_relevance(
            Query(
                query="some query",
                response="",
                context=Context(
                    groundtruth=["some ground truth"],
                    prediction=["context 1", "context 2", "context 3"],
                ),
            )
        )

    # Patch __call__ with a valid response.
    monkeypatch.setattr(
        "valor_lite.text_generation.integrations.MockWrapper.__call__",
        _return_valid1_faithfulness_response,
    )
    assert 0.6 == evaluator.compute_faithfulness(
        Query(
            query="some query",
            response="some text",
            context=Context(
                groundtruth=["some ground truth"],
                prediction=["context 1", "context 2", "context 3"],
            ),
        )
    )

    # If no claims are found in the text, then the text should have a faithfulness score of 1.
    monkeypatch.setattr(
        "valor_lite.text_generation.integrations.MockWrapper.__call__",
        _return_valid2_faithfulness_response,
    )
    assert 1.0 == evaluator.compute_faithfulness(
        Query(
            query="some query",
            response="some text",
            context=Context(
                groundtruth=["some ground truth"],
                prediction=["context 1", "context 2", "context 3"],
            ),
        )
    )

    # Faithfulness is meaningless if context_list is empty.
    monkeypatch.setattr(
        "valor_lite.text_generation.integrations.MockWrapper.__call__",
        _return_valid1_faithfulness_response,
    )
    with pytest.raises(ValueError):
        evaluator.compute_faithfulness(
            Query(
                query="some query",
                response="some text",
                context=Context(
                    groundtruth=[],
                    prediction=[],
                ),
            )
        )

    # Bad key in the claims response.
    monkeypatch.setattr(
        "valor_lite.text_generation.integrations.MockWrapper.__call__",
        _return_invalid1_faithfulness_response,
    )
    with pytest.raises(InvalidLLMResponseError):
        evaluator.compute_faithfulness(
            Query(
                query="some query",
                response="some text",
                context=Context(
                    groundtruth=["some ground truth"],
                    prediction=[
                        "context 1",
                        "context 2",
                    ],
                ),
            )
        )

    # Claims must be strings, not lists of strings.
    monkeypatch.setattr(
        "valor_lite.text_generation.integrations.MockWrapper.__call__",
        _return_invalid2_faithfulness_response,
    )
    with pytest.raises(InvalidLLMResponseError):
        evaluator.compute_faithfulness(
            Query(
                query="some query",
                response="some text",
                context=Context(
                    groundtruth=["some ground truth"],
                    prediction=[
                        "context 1",
                        "context 2",
                    ],
                ),
            )
        )

    # Bad key in the verdicts response.
    monkeypatch.setattr(
        "valor_lite.text_generation.integrations.MockWrapper.__call__",
        _return_invalid3_faithfulness_response,
    )
    with pytest.raises(InvalidLLMResponseError):
        evaluator.compute_faithfulness(
            Query(
                query="some query",
                response="some text",
                context=Context(
                    groundtruth=["some ground truth"],
                    prediction=[
                        "context 1",
                        "context 2",
                    ],
                ),
            )
        )

    # Number of verdicts does not match the number of claims.
    monkeypatch.setattr(
        "valor_lite.text_generation.integrations.MockWrapper.__call__",
        _return_invalid4_faithfulness_response,
    )
    with pytest.raises(InvalidLLMResponseError):
        evaluator.compute_faithfulness(
            Query(
                query="some query",
                response="some text",
                context=Context(
                    groundtruth=["some ground truth"],
                    prediction=[
                        "context 1",
                        "context 2",
                    ],
                ),
            )
        )

    # 'idk' is not a valid verdict for faithfulness.
    monkeypatch.setattr(
        "valor_lite.text_generation.integrations.MockWrapper.__call__",
        _return_invalid5_faithfulness_response,
    )
    with pytest.raises(InvalidLLMResponseError):
        evaluator.compute_faithfulness(
            Query(
                query="some query",
                response="some text",
                context=Context(
                    groundtruth=["some ground truth"],
                    prediction=[
                        "context 1",
                        "context 2",
                    ],
                ),
            )
        )

    # Patch __call__ with a valid response.
    monkeypatch.setattr(
        "valor_lite.text_generation.integrations.MockWrapper.__call__",
        _return_valid_hallucination_response,
    )
    assert 0.6666666666666666 == evaluator.compute_hallucination(
        Query(
            query="some query",
            response="some answer",
            context=Context(
                groundtruth=["some ground truth"],
                prediction=["context 1", "context 2", "context 3"],
            ),
        )
    )

    # Context relevance is meaningless if context_list is empty.
    monkeypatch.setattr(
        "valor_lite.text_generation.integrations.MockWrapper.__call__",
        _return_valid_hallucination_response,
    )
    with pytest.raises(ValueError):
        evaluator.compute_hallucination(Query(query="some query", response=""))

    # Only 1 context provided but 3 verdicts were returned.
    monkeypatch.setattr(
        "valor_lite.text_generation.integrations.MockWrapper.__call__",
        _return_valid_hallucination_response,
    )
    with pytest.raises(InvalidLLMResponseError):
        evaluator.compute_hallucination(
            Query(
                query="some query",
                response="some text",
                context=Context(
                    groundtruth=[],
                    prediction=[
                        "length of context list does not match LLM's response"
                    ],
                ),
            )
        )

    # Key 'all_verdicts' is returned but the key should be 'verdicts'.
    monkeypatch.setattr(
        "valor_lite.text_generation.integrations.MockWrapper.__call__",
        _return_invalid1_hallucination_response,
    )
    with pytest.raises(InvalidLLMResponseError):
        evaluator.compute_hallucination(
            Query(
                query="some query",
                response="some answer",
                context=Context(
                    groundtruth=["some ground truth"],
                    prediction=["context 1", "context 2", "context 3"],
                ),
            )
        )

    # Patch __call__ with a valid response.
    monkeypatch.setattr(
        "valor_lite.text_generation.integrations.MockWrapper.__call__",
        _return_valid_summary_coherence_response,
    )
    assert 5 == evaluator.compute_summary_coherence(
        Query(
            query="some test",
            response="some summary",
        )
    )

    # Summary coherence score is not an integer.
    monkeypatch.setattr(
        "valor_lite.text_generation.integrations.MockWrapper.__call__",
        _return_invalid1_summary_coherence_response,
    )
    with pytest.raises(InvalidLLMResponseError):
        evaluator.compute_summary_coherence(
            Query(
                query="some test",
                response="some summary",
            )
        )

    # Summary coherence score is 0, which is not in {1,2,3,4,5}.
    monkeypatch.setattr(
        "valor_lite.text_generation.integrations.MockWrapper.__call__",
        _return_invalid2_summary_coherence_response,
    )
    with pytest.raises(InvalidLLMResponseError):
        evaluator.compute_summary_coherence(
            Query(
                query="some test",
                response="some summary",
            )
        )

    # Patch __call__ with a valid response.
    monkeypatch.setattr(
        "valor_lite.text_generation.integrations.MockWrapper.__call__",
        _return_valid1_toxicity_response,
    )
    assert 0.5 == evaluator.compute_toxicity(
        Query(query="some query", response="some text")
    )

    # No opinions found, so no toxicity should be reported.
    monkeypatch.setattr(
        "valor_lite.text_generation.integrations.MockWrapper.__call__",
        _return_valid2_toxicity_response,
    )
    assert 0.0 == evaluator.compute_toxicity(
        Query(query="some query", response="some text")
    )

    # Key 'verdicts' is returned but the key should be 'opinions'.
    monkeypatch.setattr(
        "valor_lite.text_generation.integrations.MockWrapper.__call__",
        _return_invalid1_toxicity_response,
    )
    with pytest.raises(InvalidLLMResponseError):
        evaluator.compute_toxicity(
            Query(query="some query", response="some text")
        )

    # Opinions must be strings.
    monkeypatch.setattr(
        "valor_lite.text_generation.integrations.MockWrapper.__call__",
        _return_invalid2_toxicity_response,
    )
    with pytest.raises(InvalidLLMResponseError):
        evaluator.compute_toxicity(
            Query(query="some query", response="some text")
        )

    # Key 'opinions' is returned but the key should be 'verdicts'.
    monkeypatch.setattr(
        "valor_lite.text_generation.integrations.MockWrapper.__call__",
        _return_invalid3_toxicity_response,
    )
    with pytest.raises(InvalidLLMResponseError):
        evaluator.compute_toxicity(
            Query(query="some query", response="some text")
        )

    # 'idk' is not a valid toxicity verdict.
    monkeypatch.setattr(
        "valor_lite.text_generation.integrations.MockWrapper.__call__",
        _return_invalid4_toxicity_response,
    )
    with pytest.raises(InvalidLLMResponseError):
        evaluator.compute_toxicity(
            Query(query="some query", response="some text")
        )


def test_retrying(monkeypatch):
    """
    Test the retry functionality for structuring LLM API calls.
    """

    def _return_valid_summary_coherence_response(*args, **kwargs):
        return "5"

    errors = ["The score is 5."] * 3 + ["5"]

    def _return_invalid_summary_coherence_response(*args, **kwargs):
        return "The score is 5."

    monkeypatch.setattr(
        "valor_lite.text_generation.integrations.MockWrapper.__call__",
        _return_valid_summary_coherence_response,
    )

    # Test with retries=None
    evaluator = Evaluator.mock(api_key=None, model_name="model_name")
    assert 5 == evaluator.compute_summary_coherence(
        Query(query="some text", response="some summary")
    )

    # Test with retries=0
    evaluator = Evaluator.mock(
        api_key=None, model_name="model_name", retries=0
    )
    assert 5 == evaluator.compute_summary_coherence(
        Query(query="some text", response="some summary")
    )

    # Test with retries=3 and valid response
    evaluator = Evaluator.mock(api_key=None, model_name="model_name")
    assert 5 == evaluator.compute_summary_coherence(
        Query(query="some text", response="some summary")
    )

    # mock_method returns a bad response three times but on the fourth call returns a valid response.
    monkeypatch.setattr(
        "valor_lite.text_generation.integrations.MockWrapper.__call__",
        Mock(side_effect=errors),
    )
    evaluator = Evaluator.mock(api_key=None, model_name="model_name")
    assert 5 == evaluator.compute_summary_coherence(
        Query(query="some text", response="some summary")
    )

    # Test with retries=2 and invalid response
    monkeypatch.setattr(
        "valor_lite.text_generation.integrations.MockWrapper.__call__",
        Mock(side_effect=errors),
    )
    with pytest.raises(InvalidLLMResponseError):
        evaluator = Evaluator.mock(api_key=None, model_name="model_name")
        evaluator.compute_summary_coherence(
            Query(query="some text", response="some summary")
        )

    monkeypatch.setattr(
        "valor_lite.text_generation.integrations.MockWrapper.__call__",
        _return_invalid_summary_coherence_response,
    )

    # Test with retries=None and invalid response
    with pytest.raises(InvalidLLMResponseError):
        evaluator = Evaluator.mock(api_key=None, model_name="model_name")
        evaluator.compute_summary_coherence(
            Query(query="some text", response="some summary")
        )

    # Test with retries=3 and invalid response
    with pytest.raises(InvalidLLMResponseError):
        evaluator = Evaluator.mock(api_key=None, model_name="model_name")
        evaluator.compute_summary_coherence(
            Query(query="some text", response="some summary")
        )

    # Test OpenAIWrapper
    monkeypatch.setattr(
        "valor_lite.text_generation.llm_client.OpenAIWrapper.__call__",
        Mock(side_effect=errors),
    )
    client = OpenAIWrapper(api_key=None, model_name="model_name")
    assert 5 == evaluator.compute_summary_coherence(
        Query(query="some text", response="some summary")
    )

    with pytest.raises(InvalidLLMResponseError):
        monkeypatch.setattr(
            "valor_lite.text_generation.llm_client.OpenAIWrapper.__call__",
            Mock(side_effect=errors),
        )
        client = OpenAIWrapper(api_key=None, model_name="model_name")
        evaluator.compute_summary_coherence(
            Query(query="some text", response="some summary")
        )

    # Should not set retries and seed, as seed makes behavior deterministic
    with pytest.raises(ValueError):
        client = OpenAIWrapper(
            api_key=None, model_name="model_name", seed=2024
        )

    # Test MistralWrapper
    monkeypatch.setattr(
        "valor_lite.text_generation.llm_client.MistralWrapper.__call__",
        Mock(side_effect=errors),
    )
    client = MistralWrapper(api_key=None, model_name="model_name")
    assert 5 == evaluator.compute_summary_coherence(
        Query(query="some text", response="some summary")
    )

    with pytest.raises(InvalidLLMResponseError):
        monkeypatch.setattr(
            "valor_lite.text_generation.llm_client.MistralWrapper.__call__",
            Mock(side_effect=errors),
        )
        client = MistralWrapper(api_key=None, model_name="model_name")
        evaluator.compute_summary_coherence(
            Query(query="some text", response="some summary")
        )
