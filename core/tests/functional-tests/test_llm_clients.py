import datetime
import os
from unittest.mock import MagicMock, Mock

import pytest

try:
    from mistralai.models import (
        AssistantMessage,
        ChatCompletionChoice,
        ChatCompletionResponse,
        UsageInfo,
    )
    from mistralai.models.sdkerror import SDKError as MistralSDKError

    MISTRALAI_INSTALLED = True
except ImportError:
    MISTRALAI_INSTALLED = False

try:
    from openai import OpenAIError
    from openai.types.chat import ChatCompletionMessage
    from openai.types.chat.chat_completion import ChatCompletion, Choice
    from openai.types.completion_usage import CompletionUsage

    OPENAI_INSTALLED = True
except ImportError:
    OPENAI_INSTALLED = False

from valor_core.exceptions import InvalidLLMResponseError
from valor_core.llm_clients import (
    LLMClient,
    MockLLMClient,
    WrappedMistralAIClient,
    WrappedOpenAIClient,
)

VALID_CLAIMS = """```json
{
    "claims": [
        "claim 1",
        "claim 2",
        "claim 3",
        "claim 4",
        "claim 5"
    ]
}```"""

VALID_OPINIONS = """```json
{
    "opinions": [
        "opinion 1",
        "opinion 2",
        "opinion 3",
        "opinion 4"
    ]
}```"""

VALID_STATEMENTS = """```json
{
    "statements": [
        "statement 1",
        "statement 2",
        "statement 3",
        "statement 4"
    ]
}```"""

GROUNDTRUTH_VALID_STATEMENTS = """```json
{
    "statements": [
        "gt statement 1",
        "gt statement 2",
        "gt statement 3",
        "gt statement 4"
    ]
}```"""

ANSWER_CORRECTNESS_VALID_VERDICTS = """```json
{
    "TP": [
        "statement 1",
        "statement 2",
        "statement 4"
    ],
    "FP": [
        "statement 3"
    ],
    "FN": [
        "gt statement 1",
        "gt statement 4"
    ]
}```"""

ANSWER_RELEVANCE_VALID_VERDICTS = """```json
{
    "verdicts": [
        {"verdict": "no"},
        {"verdict": "yes"},
        {"verdict": "idk"},
        {"verdict": "yes"}
    ]
}```"""

BIAS_VALID_VERDICTS = """```json
{
    "verdicts": [
        {"verdict": "yes"},
        {"verdict": "no"},
        {"verdict": "yes"},
        {"verdict": "no"}
    ]
}```"""

CONTEXT_PRECISION_VALID1_VERDICTS = """```json
{
    "verdicts": [
        {"verdict": "no"},
        {"verdict": "yes"},
        {"verdict": "no"},
        {"verdict": "no"},
        {"verdict": "yes"}
    ]
}```"""

CONTEXT_PRECISION_VALID2_VERDICTS = """```json
{
    "verdicts": [
        {"verdict": "no"},
        {"verdict": "no"},
        {"verdict": "no"},
        {"verdict": "no"},
        {"verdict": "no"}
    ]
}```"""

CONTEXT_RECALL_VALID_VERDICTS = """```json
{
    "verdicts": [
        {"verdict": "yes"},
        {"verdict": "yes"},
        {"verdict": "no"},
        {"verdict": "yes"}
    ]
}```"""

CONTEXT_RELEVANCE_VALID_VERDICTS = """```json
{
    "verdicts": [
        {"verdict": "no"},
        {"verdict": "yes"},
        {"verdict": "no"}
    ]
}```"""

FAITHFULNESS_VALID_VERDICTS = """```json
{
    "verdicts": [
        {"verdict": "no"},
        {"verdict": "yes"},
        {"verdict": "yes"},
        {"verdict": "yes"},
        {"verdict": "no"}
    ]
}```"""

HALLUCINATION_VALID_VERDICTS = """```json
{
    "verdicts": [
        {"verdict": "no"},
        {"verdict": "yes"},
        {"verdict": "yes"}
    ]
}```"""

TOXICITY_VALID_VERDICTS = """```json
{
    "verdicts": [
        {"verdict": "yes"},
        {"verdict": "no"},
        {"verdict": "yes"},
        {"verdict": "no"}
    ]
}```"""


class BadValueInTestLLMClientsError(Exception):
    """
    Raised when a mock function in test_llm_clients.py receives a bad value.
    """

    pass


def test_LLMClient(monkeypatch):
    """
    Check that LLMClient throws NotImplementedErrors for connect and __call__.

    Check the metric computations for LLMClient. The client children inherit all of these metric computations.
    """

    def _return_valid_answer_correctness_response(*args, **kwargs):
        if "prediction text" in args[1][1]["content"]:
            return VALID_STATEMENTS
        elif "ground truth text" in args[1][1]["content"]:
            return GROUNDTRUTH_VALID_STATEMENTS
        elif (
            "Return in JSON format with three keys: 'TP', 'FP', and 'FN'"
            in args[1][1]["content"]
        ):
            return ANSWER_CORRECTNESS_VALID_VERDICTS
        else:
            raise BadValueInTestLLMClientsError

    def _return_invalid1_answer_correctness_response(*args, **kwargs):
        return """```json
{
    "list": [
        "statement 1",
        "statement 2",
        "statement 3",
        "statement 4"
    ]
}```"""

    def _return_invalid2_answer_correctness_response(*args, **kwargs):
        if "prediction text" in args[1][1]["content"]:
            return VALID_STATEMENTS
        elif "ground truth text" in args[1][1]["content"]:
            return """```json
{
    "statements": [
        "statement 1",
        4,
        "statement 3",
        "statement 4"
    ]
}```"""
        else:
            raise BadValueInTestLLMClientsError

    def _return_invalid3_answer_correctness_response(*args, **kwargs):
        if "prediction text" in args[1][1]["content"]:
            return VALID_STATEMENTS
        elif "ground truth text" in args[1][1]["content"]:
            return GROUNDTRUTH_VALID_STATEMENTS
        elif (
            "Return in JSON format with three keys: 'TP', 'FP', and 'FN'"
            in args[1][1]["content"]
        ):
            return """```json
{
    "TP": [
        "statement 1",
        "statement 2",
        "statement 4"
    ],
    "FP": [
        "statement 3"
    ]
}```"""
        else:
            raise BadValueInTestLLMClientsError

    def _return_invalid4_answer_correctness_response(*args, **kwargs):
        if "prediction text" in args[1][1]["content"]:
            return VALID_STATEMENTS
        elif "ground truth text" in args[1][1]["content"]:
            return GROUNDTRUTH_VALID_STATEMENTS
        elif (
            "Return in JSON format with three keys: 'TP', 'FP', and 'FN'"
            in args[1][1]["content"]
        ):
            return """```json
{
    "TP": "statement 1",
    "FP": [
        "statement 3"
    ],
    "FN": [
        "gt statement 1",
        "gt statement 4"
    ]
}```"""
        else:
            raise BadValueInTestLLMClientsError

    def _return_invalid5_answer_correctness_response(*args, **kwargs):
        if "prediction text" in args[1][1]["content"]:
            return VALID_STATEMENTS
        elif "ground truth text" in args[1][1]["content"]:
            return GROUNDTRUTH_VALID_STATEMENTS
        elif (
            "Return in JSON format with three keys: 'TP', 'FP', and 'FN'"
            in args[1][1]["content"]
        ):
            return """```json
{
    "TP": [
        "statement 1",
        "statement 2"
    ],
    "FP": [
        "statement 3"
    ],
    "FN": [
        "gt statement 1",
        "gt statement 4"
    ]
}```"""
        else:
            raise BadValueInTestLLMClientsError

    def _return_invalid6_answer_correctness_response(*args, **kwargs):
        if "prediction text" in args[1][1]["content"]:
            return VALID_STATEMENTS
        elif "ground truth text" in args[1][1]["content"]:
            return GROUNDTRUTH_VALID_STATEMENTS
        elif (
            "Return in JSON format with three keys: 'TP', 'FP', and 'FN'"
            in args[1][1]["content"]
        ):
            return """```json
{
    "TP": [
        "statement 1",
        "statement 2",
        "statement 4"
    ],
    "FP": [
        "statement 3"
    ],
    "FN": [
        "gt statement 1",
        "gt statement 2",
        "gt statement 3",
        "gt statement 4",
        "too many statements in 'FN'"
    ]
}```"""
        else:
            raise BadValueInTestLLMClientsError

    def _return_valid_answer_relevance_response(*args, **kwargs):
        if "generate a list of STATEMENTS" in args[1][1]["content"]:
            return VALID_STATEMENTS
        elif (
            "generate a list of verdicts that indicate whether each statement is relevant to address the query"
            in args[1][1]["content"]
        ):
            return ANSWER_RELEVANCE_VALID_VERDICTS
        else:
            raise BadValueInTestLLMClientsError

    def _return_invalid1_answer_relevance_response(*args, **kwargs):
        return """```json
{
    "list": [
        "statement 1",
        "statement 2",
        "statement 3",
        "statement 4"
    ]
}```"""

    def _return_invalid2_answer_relevance_response(*args, **kwargs):
        return """```json
{
    "statements": [
        "statement 1",
        5,
        "statement 3",
        "statement 4"
    ]
}```"""

    def _return_invalid3_answer_relevance_response(*args, **kwargs):
        if "generate a list of STATEMENTS" in args[1][1]["content"]:
            return VALID_STATEMENTS
        elif (
            "generate a list of verdicts that indicate whether each statement is relevant to address the query"
            in args[1][1]["content"]
        ):
            return """```json
{
    "list": [
        {
            "verdict": "no",
            "reason": "The statement has nothing to do with the query."
        },
        {
            "verdict": "yes"
        },
        {
            "verdict": "idk"
        },
        {
            "verdict": "yes"
        }
    ]
}```"""
        else:
            raise BadValueInTestLLMClientsError

    def _return_invalid4_answer_relevance_response(*args, **kwargs):
        if "generate a list of STATEMENTS" in args[1][1]["content"]:
            return VALID_STATEMENTS
        elif (
            "generate a list of verdicts that indicate whether each statement is relevant to address the query"
            in args[1][1]["content"]
        ):
            return """```json
{
    "verdicts": [
        {
            "verdict": "no",
            "reason": "The statement has nothing to do with the query."
        },
        {
            "verdict": "yes"
        },
        {
            "verdict": "idk"
        },
        {
            "verdict": "unsure"
        }
    ]
}```"""
        else:
            raise BadValueInTestLLMClientsError

    def _return_valid1_bias_response(*args, **kwargs):
        if "generate a list of OPINIONS" in args[1][1]["content"]:
            return VALID_OPINIONS
        elif (
            "generate a list of verdicts to indicate whether EACH opinion is biased"
            in args[1][1]["content"]
        ):
            return BIAS_VALID_VERDICTS
        else:
            raise BadValueInTestLLMClientsError

    def _return_valid2_bias_response(*args, **kwargs):
        return """```json
{
    "opinions": []
}```"""

    def _return_invalid1_bias_response(*args, **kwargs):
        return """```json
{
    "verdicts": [
        "opinion 1",
        "verdict 2",
        "these should not be verdicts, these should be opinions",
        "the key above should be 'opinions' not 'verdicts'"
    ]
}```"""

    def _return_invalid2_bias_response(*args, **kwargs):
        return """```json
{
    "opinions": [
        ["a list of opinions"],
        "opinion 2",
        "opinion 3",
        "opinion 4"
    ]
}```"""

    def _return_invalid3_bias_response(*args, **kwargs):
        if "generate a list of OPINIONS" in args[1][1]["content"]:
            return VALID_OPINIONS
        elif (
            "generate a list of verdicts to indicate whether EACH opinion is biased"
            in args[1][1]["content"]
        ):
            return """```json
{
    "opinions": [
        "the key should be 'verdicts' not 'opinions'",
        "opinion 2",
        "opinion 3",
        "opinion 4"
    ]
}```"""
        else:
            raise BadValueInTestLLMClientsError

    def _return_invalid4_bias_response(*args, **kwargs):
        if "generate a list of OPINIONS" in args[1][1]["content"]:
            return VALID_OPINIONS
        elif (
            "generate a list of verdicts to indicate whether EACH opinion is biased"
            in args[1][1]["content"]
        ):
            return """```json
{
    "verdicts": [
        {
            "verdict": "yes",
            "reason": "This opinion demonstrates gender bias."
        },
        {
            "verdict": "idk"
        },
        {
            "verdict": "yes",
            "reason": "This opinion demonstrates political bias."
        },
        {
            "verdict": "no"
        }
    ]
}```"""
        else:
            raise BadValueInTestLLMClientsError

    def _return_valid_context_relevance_response(*args, **kwargs):
        return CONTEXT_RELEVANCE_VALID_VERDICTS

    def _return_invalid1_context_relevance_response(*args, **kwargs):
        return """```json
{
    "all_verdicts": [
        {"verdict": "no"},
        {"verdict": "yes"},
        {"verdict": "no"}
    ]
}```"""

    def _return_valid1_context_precision_response(*args, **kwargs):
        return CONTEXT_PRECISION_VALID1_VERDICTS

    def _return_valid2_context_precision_response(*args, **kwargs):
        return CONTEXT_PRECISION_VALID2_VERDICTS

    def _return_invalid1_context_precision_response(*args, **kwargs):
        return """```json
{
    "invalid_key": [
        "verdict 1",
        "verdict 2",
        "verdict 3"
    ]
}```"""

    def _return_valid_context_recall_response(*args, **kwargs):
        if "generate a list of STATEMENTS" in args[1][1]["content"]:
            return VALID_STATEMENTS
        elif (
            "analyze each ground truth statement and determine if the statement can be attributed to the given context."
            in args[1][1]["content"]
        ):
            return CONTEXT_RECALL_VALID_VERDICTS
        else:
            raise BadValueInTestLLMClientsError

    def _return_invalid1_context_recall_response(*args, **kwargs):
        return """```json
{
    "invalid_key": [
        "statement 1",
        "statement 2",
        "statement 3",
        "statement 4"
    ]
}```"""

    def _return_invalid2_context_recall_response(*args, **kwargs):
        return """```json
{
    "statements": [
        1,
        "statement 2",
        "statement 3",
        "statement 4"
    ]
}```"""

    def _return_invalid3_context_recall_response(*args, **kwargs):
        if "generate a list of STATEMENTS" in args[1][1]["content"]:
            return VALID_STATEMENTS
        elif (
            "analyze each ground truth statement and determine if the statement can be attributed to the given context."
            in args[1][1]["content"]
        ):
            return """```json
{
    "invalid_key": [
        "verdict 1",
        "verdict 2",
        "verdict 3",
        "verdict 4"
    ]
}```"""
        else:
            raise BadValueInTestLLMClientsError

    def _return_invalid4_context_recall_response(*args, **kwargs):
        if "generate a list of STATEMENTS" in args[1][1]["content"]:
            return VALID_STATEMENTS
        elif (
            "analyze each ground truth statement and determine if the statement can be attributed to the given context."
            in args[1][1]["content"]
        ):
            return """```json
{
    "verdicts": [
        "verdict 1",
        "verdict 2",
        "verdict 3"
    ]
}```"""
        else:
            raise BadValueInTestLLMClientsError

    def _return_valid1_faithfulness_response(*args, **kwargs):
        if (
            "generate a comprehensive list of FACTUAL CLAIMS"
            in args[1][1]["content"]
        ):
            return VALID_CLAIMS
        elif (
            "generate a list of verdicts to indicate whether EACH claim is implied by the context list"
            in args[1][1]["content"]
        ):
            return FAITHFULNESS_VALID_VERDICTS
        else:
            raise BadValueInTestLLMClientsError

    def _return_valid2_faithfulness_response(*args, **kwargs):
        return """```json
{
    "claims": []
}```"""

    def _return_invalid1_faithfulness_response(*args, **kwargs):
        return """```json
{
    "invalid_key": [
        "claim 1",
        "claim 2"
    ]
}```"""

    def _return_invalid2_faithfulness_response(*args, **kwargs):
        return """```json
{
    "claims": [
        [
            "claim 1",
            "claim 2"
        ]
    ]
}```"""

    def _return_invalid3_faithfulness_response(*args, **kwargs):
        if (
            "generate a comprehensive list of FACTUAL CLAIMS"
            in args[1][1]["content"]
        ):
            return VALID_CLAIMS
        elif (
            "generate a list of verdicts to indicate whether EACH claim is implied by the context list"
            in args[1][1]["content"]
        ):
            return """```json
{
    "bad key": [
        {"verdict": "no"},
        {"verdict": "yes"},
        {"verdict": "yes"},
        {"verdict": "yes"},
        {"verdict": "no"},
    ]
}```"""
        else:
            raise BadValueInTestLLMClientsError

    def _return_invalid4_faithfulness_response(*args, **kwargs):
        if (
            "generate a comprehensive list of FACTUAL CLAIMS"
            in args[1][1]["content"]
        ):
            return VALID_CLAIMS
        elif (
            "generate a list of verdicts to indicate whether EACH claim is implied by the context list"
            in args[1][1]["content"]
        ):
            return """```json
{
    "verdicts": [
        {"verdict": "no"},
        {"verdict": "yes"},
        {"verdict": "yes"},
        {"verdict": "yes"}
    ]
}```"""
        else:
            raise BadValueInTestLLMClientsError

    def _return_invalid5_faithfulness_response(*args, **kwargs):
        if (
            "generate a comprehensive list of FACTUAL CLAIMS"
            in args[1][1]["content"]
        ):
            return VALID_CLAIMS
        elif (
            "generate a list of verdicts to indicate whether EACH claim is implied by the context list"
            in args[1][1]["content"]
        ):
            return """```json
{
    "verdicts": [
        {"verdict": "idk"},
        {"verdict": "yes"},
        {"verdict": "yes"},
        {"verdict": "idk"},
        {"verdict": "no"},
    ]
}```"""
        else:
            raise BadValueInTestLLMClientsError

    def _return_valid_hallucination_response(*args, **kwargs):
        return HALLUCINATION_VALID_VERDICTS

    def _return_invalid1_hallucination_response(*args, **kwargs):
        return """```json
{
    "bad key": [
        {"verdict": "yes"},
        {"verdict": "no"},
        {"verdict": "yes"}
    ]
}```"""

    def _return_valid_summary_coherence_response(*args, **kwargs):
        return "5"

    def _return_invalid1_summary_coherence_response(*args, **kwargs):
        return "The score is 5."

    def _return_invalid2_summary_coherence_response(*args, **kwargs):
        return "0"

    def _return_valid1_toxicity_response(*args, **kwargs):
        if "generate a list of OPINIONS" in args[1][1]["content"]:
            return VALID_OPINIONS
        elif (
            "generate a list of verdicts to indicate whether EACH opinion is toxic"
            in args[1][1]["content"]
        ):
            return TOXICITY_VALID_VERDICTS
        else:
            raise BadValueInTestLLMClientsError

    def _return_valid2_toxicity_response(*args, **kwargs):
        return """```json
{
    "opinions": []
}```"""

    def _return_invalid1_toxicity_response(*args, **kwargs):
        return """```json
{
    "verdicts": [
        "opinion 1",
        "verdict 2",
        "these should not be verdicts, these should be opinions",
        "the key above should be 'opinions' not 'verdicts'"
    ]
}```"""

    def _return_invalid2_toxicity_response(*args, **kwargs):
        return """```json
{
    "opinions": [
        "opinion 1",
        "opinion 2",
        0.8,
        "opinion 4"
    ]
}```"""

    def _return_invalid3_toxicity_response(*args, **kwargs):
        if "generate a list of OPINIONS" in args[1][1]["content"]:
            return VALID_OPINIONS
        elif (
            "generate a list of verdicts to indicate whether EACH opinion is toxic"
            in args[1][1]["content"]
        ):
            return """```json
{
    "opinions": [
        "opinion 1",
        "opinion 2",
        "opinion 3",
        "opinion 4"
    ]
}```"""
        else:
            raise BadValueInTestLLMClientsError

    def _return_invalid4_toxicity_response(*args, **kwargs):
        if "generate a list of OPINIONS" in args[1][1]["content"]:
            return VALID_OPINIONS
        elif (
            "generate a list of verdicts to indicate whether EACH opinion is toxic"
            in args[1][1]["content"]
        ):
            return """```json
{
    "verdicts": [
        {
            "verdict": "yes",
            "reason": "This opinion demonstrates gender bias."
        },
        {
            "verdict": "no"
        },
        {
            "verdict": "yes",
            "reason": "This opinion demonstrates political bias."
        },
        {
            "verdict": "idk"
        }
    ]
}```"""
        else:
            raise BadValueInTestLLMClientsError

    client = LLMClient(api_key=None, model_name="model_name")

    # connect(), _process_messages() and __call__() are not implemented for the parent class.
    fake_message = [
        {"role": "system", "content": "You are a helpful assistant."}
    ]
    with pytest.raises(NotImplementedError):
        client.connect()
    with pytest.raises(NotImplementedError):
        client._process_messages(fake_message)
    with pytest.raises(NotImplementedError):
        client(fake_message)

    # Patch __call__ with a valid response.
    monkeypatch.setattr(
        "valor_core.llm_clients.LLMClient.__call__",
        _return_valid_answer_correctness_response,
    )
    assert 0.6666666666666666 == client.answer_correctness(
        "some query", "prediction text", ["ground truth text"]
    )

    # Needs to have 'statements' key.
    monkeypatch.setattr(
        "valor_core.llm_clients.LLMClient.__call__",
        _return_invalid1_answer_correctness_response,
    )
    with pytest.raises(InvalidLLMResponseError):
        client.answer_correctness(
            "some query", "prediction text", ["ground truth text"]
        )

    # Should fail if ground truth statements are invalid even when prediction statements are valid
    monkeypatch.setattr(
        "valor_core.llm_clients.LLMClient.__call__",
        _return_invalid2_answer_correctness_response,
    )
    with pytest.raises(InvalidLLMResponseError):
        client.answer_correctness(
            "some query", "prediction text", ["ground truth text"]
        )

    # Missing 'FN' in dictionary
    monkeypatch.setattr(
        "valor_core.llm_clients.LLMClient.__call__",
        _return_invalid3_answer_correctness_response,
    )
    with pytest.raises(InvalidLLMResponseError):
        client.answer_correctness(
            "some query", "prediction text", ["ground truth text"]
        )

    # TP has an invalid value.
    monkeypatch.setattr(
        "valor_core.llm_clients.LLMClient.__call__",
        _return_invalid4_answer_correctness_response,
    )
    with pytest.raises(InvalidLLMResponseError):
        client.answer_correctness(
            "some query", "prediction text", ["ground truth text"]
        )

    # Number of TP + FP does not equal the number of prediction statements
    monkeypatch.setattr(
        "valor_core.llm_clients.LLMClient.__call__",
        _return_invalid5_answer_correctness_response,
    )
    with pytest.raises(InvalidLLMResponseError):
        client.answer_correctness(
            "some query", "prediction text", ["ground truth text"]
        )

    # The number of FN is more than the number of ground truth statements
    monkeypatch.setattr(
        "valor_core.llm_clients.LLMClient.__call__",
        _return_invalid6_answer_correctness_response,
    )
    with pytest.raises(InvalidLLMResponseError):
        client.answer_correctness(
            "some query", "prediction text", ["ground truth text"]
        )

    # Patch __call__ with a valid response.
    monkeypatch.setattr(
        "valor_core.llm_clients.LLMClient.__call__",
        _return_valid_answer_relevance_response,
    )
    assert 0.5 == client.answer_relevance("some query", "some answer")

    # Needs to have 'statements' key.
    monkeypatch.setattr(
        "valor_core.llm_clients.LLMClient.__call__",
        _return_invalid1_answer_relevance_response,
    )
    with pytest.raises(InvalidLLMResponseError):
        client.answer_relevance("some query", "some text")

    # Statements must be strings.
    monkeypatch.setattr(
        "valor_core.llm_clients.LLMClient.__call__",
        _return_invalid2_answer_relevance_response,
    )
    with pytest.raises(InvalidLLMResponseError):
        client.answer_relevance("some query", "some text")

    # Needs to have 'verdicts' key.
    monkeypatch.setattr(
        "valor_core.llm_clients.LLMClient.__call__",
        _return_invalid3_answer_relevance_response,
    )
    with pytest.raises(InvalidLLMResponseError):
        client.answer_relevance("some query", "some text")

    # Invalid verdict, all verdicts must be yes, no or idk.
    monkeypatch.setattr(
        "valor_core.llm_clients.LLMClient.__call__",
        _return_invalid4_answer_relevance_response,
    )
    with pytest.raises(InvalidLLMResponseError):
        client.answer_relevance("some query", "some text")

    # Patch __call__ with a valid response.
    monkeypatch.setattr(
        "valor_core.llm_clients.LLMClient.__call__",
        _return_valid1_bias_response,
    )
    assert 0.5 == client.bias("some text")

    # No opinions found, so no bias should be reported.
    monkeypatch.setattr(
        "valor_core.llm_clients.LLMClient.__call__",
        _return_valid2_bias_response,
    )
    assert 0.0 == client.bias("some text")

    # Key 'verdicts' is returned but the key should be 'opinions'.
    monkeypatch.setattr(
        "valor_core.llm_clients.LLMClient.__call__",
        _return_invalid1_bias_response,
    )
    with pytest.raises(InvalidLLMResponseError):
        client.bias("some text")

    # Opinions must be strings.
    monkeypatch.setattr(
        "valor_core.llm_clients.LLMClient.__call__",
        _return_invalid2_bias_response,
    )
    with pytest.raises(InvalidLLMResponseError):
        client.bias("some text")

    # Key 'opinions' is returned but the key should be 'verdicts'.
    monkeypatch.setattr(
        "valor_core.llm_clients.LLMClient.__call__",
        _return_invalid3_bias_response,
    )
    with pytest.raises(InvalidLLMResponseError):
        client.bias("some text")

    # 'idk' is not a valid bias verdict.
    monkeypatch.setattr(
        "valor_core.llm_clients.LLMClient.__call__",
        _return_invalid4_bias_response,
    )
    with pytest.raises(InvalidLLMResponseError):
        client.bias("some text")

    # Patch __call__ with a valid response.
    monkeypatch.setattr(
        "valor_core.llm_clients.LLMClient.__call__",
        _return_valid1_context_precision_response,
    )
    assert 0.45 == client.context_precision(
        "some query",
        ["context 1", "context 2", "context 3", "context 4", "context 5"],
        ["some ground truth"],
    )

    # If all verdicts are "no", the returned score should be 0.
    monkeypatch.setattr(
        "valor_core.llm_clients.LLMClient.__call__",
        _return_valid2_context_precision_response,
    )
    assert 0.0 == client.context_precision(
        "some query",
        ["context 1", "context 2", "context 3", "context 4", "context 5"],
        ["some ground truth"],
    )

    # Context precision is meaningless if context_list is empty.
    monkeypatch.setattr(
        "valor_core.llm_clients.LLMClient.__call__",
        _return_valid1_context_precision_response,
    )
    with pytest.raises(ValueError):
        client.context_precision(
            "some query",
            [],
            ["some ground truth"],
        )

    # Only 1 context provided but 5 verdicts were returned.
    monkeypatch.setattr(
        "valor_core.llm_clients.LLMClient.__call__",
        _return_valid1_context_precision_response,
    )
    with pytest.raises(InvalidLLMResponseError):
        client.context_precision(
            "some query",
            ["length of context list does not match LLM's response"],
            ["some ground truth"],
        )

    # Key 'invalid_key' is returned but the key should be 'verdicts'.
    monkeypatch.setattr(
        "valor_core.llm_clients.LLMClient.__call__",
        _return_invalid1_context_precision_response,
    )
    with pytest.raises(InvalidLLMResponseError):
        client.context_precision(
            "some query",
            ["context 1", "context 2", "context 3", "context 4", "context 5"],
            ["some ground truth"],
        )

    # Patch __call__ with a valid response.
    monkeypatch.setattr(
        "valor_core.llm_clients.LLMClient.__call__",
        _return_valid_context_recall_response,
    )
    assert 0.75 == client.context_recall(
        ["context 1", "context 2"],
        ["some ground truth"],
    )

    # Context recall is meaningless if context_list is empty.
    monkeypatch.setattr(
        "valor_core.llm_clients.LLMClient.__call__",
        _return_valid_context_recall_response,
    )
    with pytest.raises(ValueError):
        client.context_recall(
            [],
            ["some ground truth"],
        )

    # Ground truth statements response must have key 'statements'.
    monkeypatch.setattr(
        "valor_core.llm_clients.LLMClient.__call__",
        _return_invalid1_context_recall_response,
    )
    with pytest.raises(InvalidLLMResponseError):
        client.context_recall(
            ["context 1", "context 2"],
            ["some ground truth"],
        )

    # Ground truth statements must be strings.
    monkeypatch.setattr(
        "valor_core.llm_clients.LLMClient.__call__",
        _return_invalid2_context_recall_response,
    )
    with pytest.raises(InvalidLLMResponseError):
        client.context_recall(
            ["context 1", "context 2"],
            ["some ground truth"],
        )

    # Context recall verdicts response must have key 'verdicts'.
    monkeypatch.setattr(
        "valor_core.llm_clients.LLMClient.__call__",
        _return_invalid3_context_recall_response,
    )
    with pytest.raises(InvalidLLMResponseError):
        client.context_recall(
            ["context 1", "context 2"],
            ["some ground truth"],
        )

    # Number of context recall verdicts doesn't match the number of ground truth statements.
    monkeypatch.setattr(
        "valor_core.llm_clients.LLMClient.__call__",
        _return_invalid4_context_recall_response,
    )
    with pytest.raises(InvalidLLMResponseError):
        client.context_recall(
            ["context 1", "context 2"],
            ["some ground truth"],
        )

    # Patch __call__ with a valid response.
    monkeypatch.setattr(
        "valor_core.llm_clients.LLMClient.__call__",
        _return_valid_context_relevance_response,
    )
    assert 0.3333333333333333 == client.context_relevance(
        "some query", ["context 1", "context 2", "context 3"]
    )

    # Context relevance is meaningless if context_list is empty.
    monkeypatch.setattr(
        "valor_core.llm_clients.LLMClient.__call__",
        _return_valid_context_relevance_response,
    )
    with pytest.raises(ValueError):
        client.context_relevance("some query", [])

    # Only 1 context provided but 3 verdicts were returned.
    monkeypatch.setattr(
        "valor_core.llm_clients.LLMClient.__call__",
        _return_valid_context_relevance_response,
    )
    with pytest.raises(InvalidLLMResponseError):
        client.context_relevance(
            "some query",
            ["length of context list does not match LLM's response"],
        )

    # Key 'all_verdicts' is returned but the key should be 'verdicts'.
    monkeypatch.setattr(
        "valor_core.llm_clients.LLMClient.__call__",
        _return_invalid1_context_relevance_response,
    )
    with pytest.raises(InvalidLLMResponseError):
        client.context_relevance(
            "some query", ["context 1", "context 2", "context 3"]
        )

    # Patch __call__ with a valid response.
    monkeypatch.setattr(
        "valor_core.llm_clients.LLMClient.__call__",
        _return_valid1_faithfulness_response,
    )
    assert 0.6 == client.faithfulness("some text", ["context 1", "context 2"])

    # If no claims are found in the text, then the text should have a faithfulness score of 1.
    monkeypatch.setattr(
        "valor_core.llm_clients.LLMClient.__call__",
        _return_valid2_faithfulness_response,
    )
    assert 1.0 == client.faithfulness("some text", ["context 1", "context 2"])

    # Faithfulness is meaningless if context_list is empty.
    monkeypatch.setattr(
        "valor_core.llm_clients.LLMClient.__call__",
        _return_valid1_faithfulness_response,
    )
    with pytest.raises(ValueError):
        client.faithfulness("some text", [])

    # Bad key in the claims response.
    monkeypatch.setattr(
        "valor_core.llm_clients.LLMClient.__call__",
        _return_invalid1_faithfulness_response,
    )
    with pytest.raises(InvalidLLMResponseError):
        client.faithfulness("some text", ["context 1", "context 2"])

    # Claims must be strings, not lists of strings.
    monkeypatch.setattr(
        "valor_core.llm_clients.LLMClient.__call__",
        _return_invalid2_faithfulness_response,
    )
    with pytest.raises(InvalidLLMResponseError):
        client.faithfulness("some text", ["context 1", "context 2"])

    # Bad key in the verdicts response.
    monkeypatch.setattr(
        "valor_core.llm_clients.LLMClient.__call__",
        _return_invalid3_faithfulness_response,
    )
    with pytest.raises(InvalidLLMResponseError):
        client.faithfulness("some text", ["context 1", "context 2"])

    # Number of verdicts does not match the number of claims.
    monkeypatch.setattr(
        "valor_core.llm_clients.LLMClient.__call__",
        _return_invalid4_faithfulness_response,
    )
    with pytest.raises(InvalidLLMResponseError):
        client.faithfulness("some text", ["context 1", "context 2"])

    # 'idk' is not a valid verdict for faithfulness.
    monkeypatch.setattr(
        "valor_core.llm_clients.LLMClient.__call__",
        _return_invalid5_faithfulness_response,
    )
    with pytest.raises(InvalidLLMResponseError):
        client.faithfulness("some text", ["context 1", "context 2"])

    # Patch __call__ with a valid response.
    monkeypatch.setattr(
        "valor_core.llm_clients.LLMClient.__call__",
        _return_valid_hallucination_response,
    )
    assert 0.6666666666666666 == client.hallucination(
        "some answer", ["context 1", "context 2", "context 3"]
    )

    # Context relevance is meaningless if context_list is empty.
    monkeypatch.setattr(
        "valor_core.llm_clients.LLMClient.__call__",
        _return_valid_hallucination_response,
    )
    with pytest.raises(ValueError):
        client.hallucination("some query", [])

    # Only 1 context provided but 3 verdicts were returned.
    monkeypatch.setattr(
        "valor_core.llm_clients.LLMClient.__call__",
        _return_valid_hallucination_response,
    )
    with pytest.raises(InvalidLLMResponseError):
        client.hallucination(
            "some query",
            ["length of context list does not match LLM's response"],
        )

    # Key 'all_verdicts' is returned but the key should be 'verdicts'.
    monkeypatch.setattr(
        "valor_core.llm_clients.LLMClient.__call__",
        _return_invalid1_hallucination_response,
    )
    with pytest.raises(InvalidLLMResponseError):
        client.hallucination(
            "some query", ["context 1", "context 2", "context 3"]
        )

    # Patch __call__ with a valid response.
    monkeypatch.setattr(
        "valor_core.llm_clients.LLMClient.__call__",
        _return_valid_summary_coherence_response,
    )
    assert 5 == client.summary_coherence("some text", "some summary")

    # Summary coherence score is not an integer.
    monkeypatch.setattr(
        "valor_core.llm_clients.LLMClient.__call__",
        _return_invalid1_summary_coherence_response,
    )
    with pytest.raises(InvalidLLMResponseError):
        client.summary_coherence("some text", "some summary")

    # Summary coherence score is 0, which is not in {1,2,3,4,5}.
    monkeypatch.setattr(
        "valor_core.llm_clients.LLMClient.__call__",
        _return_invalid2_summary_coherence_response,
    )
    with pytest.raises(InvalidLLMResponseError):
        client.summary_coherence("some text", "some summary")

    # Patch __call__ with a valid response.
    monkeypatch.setattr(
        "valor_core.llm_clients.LLMClient.__call__",
        _return_valid1_toxicity_response,
    )
    assert 0.5 == client.toxicity("some text")

    # No opinions found, so no toxicity should be reported.
    monkeypatch.setattr(
        "valor_core.llm_clients.LLMClient.__call__",
        _return_valid2_toxicity_response,
    )
    assert 0.0 == client.toxicity("some text")

    # Key 'verdicts' is returned but the key should be 'opinions'.
    monkeypatch.setattr(
        "valor_core.llm_clients.LLMClient.__call__",
        _return_invalid1_toxicity_response,
    )
    with pytest.raises(InvalidLLMResponseError):
        client.toxicity("some text")

    # Opinions must be strings.
    monkeypatch.setattr(
        "valor_core.llm_clients.LLMClient.__call__",
        _return_invalid2_toxicity_response,
    )
    with pytest.raises(InvalidLLMResponseError):
        client.toxicity("some text")

    # Key 'opinions' is returned but the key should be 'verdicts'.
    monkeypatch.setattr(
        "valor_core.llm_clients.LLMClient.__call__",
        _return_invalid3_toxicity_response,
    )
    with pytest.raises(InvalidLLMResponseError):
        client.toxicity("some text")

    # 'idk' is not a valid toxicity verdict.
    monkeypatch.setattr(
        "valor_core.llm_clients.LLMClient.__call__",
        _return_invalid4_toxicity_response,
    )
    with pytest.raises(InvalidLLMResponseError):
        client.toxicity("some text")


def test_LLMClient_retries(monkeypatch):
    """
    Test the retry functionality for structuring LLM API calls.
    """

    def _return_valid_summary_coherence_response(*args, **kwargs):
        return "5"

    errors = ["The score is 5."] * 3 + ["5"]

    def _return_invalid_summary_coherence_response(*args, **kwargs):
        return "The score is 5."

    monkeypatch.setattr(
        "valor_core.llm_clients.LLMClient.__call__",
        _return_valid_summary_coherence_response,
    )

    # Test with retries=None
    client = LLMClient(api_key=None, model_name="model_name", retries=None)
    assert 5 == client.summary_coherence("some text", "some summary")

    # Test with retries=0
    client = LLMClient(api_key=None, model_name="model_name", retries=0)
    assert 5 == client.summary_coherence("some text", "some summary")

    # Test with retries=3 and valid response
    client = LLMClient(api_key=None, model_name="model_name", retries=3)
    assert 5 == client.summary_coherence("some text", "some summary")

    # mock_method returns a bad response three times but on the fourth call returns a valid response.
    monkeypatch.setattr(
        "valor_core.llm_clients.LLMClient.__call__",
        Mock(side_effect=errors),
    )
    client = LLMClient(api_key=None, model_name="model_name", retries=3)
    assert 5 == client.summary_coherence("some text", "some summary")

    # Test with retries=2 and invalid response
    monkeypatch.setattr(
        "valor_core.llm_clients.LLMClient.__call__",
        Mock(side_effect=errors),
    )
    with pytest.raises(InvalidLLMResponseError):
        client = LLMClient(api_key=None, model_name="model_name", retries=2)
        client.summary_coherence("some text", "some summary")

    monkeypatch.setattr(
        "valor_core.llm_clients.LLMClient.__call__",
        _return_invalid_summary_coherence_response,
    )

    # Test with retries=None and invalid response
    with pytest.raises(InvalidLLMResponseError):
        client = LLMClient(api_key=None, model_name="model_name", retries=None)
        client.summary_coherence("some text", "some summary")

    # Test with retries=3 and invalid response
    with pytest.raises(InvalidLLMResponseError):
        client = LLMClient(api_key=None, model_name="model_name", retries=3)
        client.summary_coherence("some text", "some summary")

    # Test WrappedOpenAIClient
    monkeypatch.setattr(
        "valor_core.llm_clients.WrappedOpenAIClient.__call__",
        Mock(side_effect=errors),
    )
    client = WrappedOpenAIClient(
        api_key=None, model_name="model_name", retries=3
    )
    assert 5 == client.summary_coherence("some text", "some summary")

    with pytest.raises(InvalidLLMResponseError):
        monkeypatch.setattr(
            "valor_core.llm_clients.WrappedOpenAIClient.__call__",
            Mock(side_effect=errors),
        )
        client = WrappedOpenAIClient(
            api_key=None, model_name="model_name", retries=2
        )
        client.summary_coherence("some text", "some summary")

    # Should not set retries and seed, as seed makes behavior deterministic
    with pytest.raises(ValueError):
        client = WrappedOpenAIClient(
            api_key=None, model_name="model_name", retries=3, seed=2024
        )

    # Test WrappedMistralAIClient
    monkeypatch.setattr(
        "valor_core.llm_clients.WrappedMistralAIClient.__call__",
        Mock(side_effect=errors),
    )
    client = WrappedMistralAIClient(
        api_key=None, model_name="model_name", retries=3
    )
    assert 5 == client.summary_coherence("some text", "some summary")

    with pytest.raises(InvalidLLMResponseError):
        monkeypatch.setattr(
            "valor_core.llm_clients.WrappedMistralAIClient.__call__",
            Mock(side_effect=errors),
        )
        client = WrappedMistralAIClient(
            api_key=None, model_name="model_name", retries=2
        )
        client.summary_coherence("some text", "some summary")


@pytest.mark.skipif(
    not OPENAI_INSTALLED,
    reason="Openai is not installed.",
)
def test_WrappedOpenAIClient():
    def _create_bad_request(model, messages, seed):
        raise ValueError

    def _create_mock_chat_completion_with_bad_length(
        model, messages, seed
    ) -> ChatCompletion:  # type: ignore - test is not run if openai is not installed
        return ChatCompletion(  # type: ignore - test is not run if openai is not installed
            id="foo",
            model="gpt-3.5-turbo",
            object="chat.completion",
            choices=[
                Choice(  # type: ignore - test is not run if openai is not installed
                    finish_reason="length",
                    index=0,
                    message=ChatCompletionMessage(  # type: ignore - test is not run if openai is not installed
                        content="some response",
                        role="assistant",
                    ),
                )
            ],
            usage=CompletionUsage(  # type: ignore - test is not run if openai is not installed
                completion_tokens=1, prompt_tokens=2, total_tokens=3
            ),
            created=int(datetime.datetime.now().timestamp()),
        )

    def _create_mock_chat_completion_with_content_filter(
        model, messages, seed
    ) -> ChatCompletion:  # type: ignore - test is not run if openai is not installed
        return ChatCompletion(  # type: ignore - test is not run if openai is not installed
            id="foo",
            model="gpt-3.5-turbo",
            object="chat.completion",
            choices=[
                Choice(  # type: ignore - test is not run if openai is not installed
                    finish_reason="content_filter",
                    index=0,
                    message=ChatCompletionMessage(  # type: ignore - test is not run if openai is not installed
                        content="some response",
                        role="assistant",
                    ),
                )
            ],
            usage=CompletionUsage(  # type: ignore - test is not run if openai is not installed
                completion_tokens=1, prompt_tokens=2, total_tokens=3
            ),
            created=int(datetime.datetime.now().timestamp()),
        )

    def _create_mock_chat_completion(model, messages, seed) -> ChatCompletion:  # type: ignore - test is not run if openai is not installed
        return ChatCompletion(  # type: ignore - test is not run if openai is not installed
            id="foo",
            model="gpt-3.5-turbo",
            object="chat.completion",
            choices=[
                Choice(  # type: ignore - test is not run if openai is not installed
                    finish_reason="stop",
                    index=0,
                    message=ChatCompletionMessage(  # type: ignore - test is not run if openai is not installed
                        content="some response",
                        role="assistant",
                    ),
                )
            ],
            usage=CompletionUsage(  # type: ignore - test is not run if openai is not installed
                completion_tokens=1, prompt_tokens=2, total_tokens=3
            ),
            created=int(datetime.datetime.now().timestamp()),
        )

    def _create_mock_chat_completion_none_content(
        model, messages, seed
    ) -> ChatCompletion:  # type: ignore - test is not run if openai is not installed
        return ChatCompletion(  # type: ignore - test is not run if openai is not installed
            id="foo",
            model="gpt-3.5-turbo",
            object="chat.completion",
            choices=[
                Choice(  # type: ignore - test is not run if openai is not installed
                    finish_reason="stop",
                    index=0,
                    message=ChatCompletionMessage(  # type: ignore - test is not run if openai is not installed
                        content=None,
                        role="assistant",
                    ),
                )
            ],
            usage=CompletionUsage(  # type: ignore - test is not run if openai is not installed
                completion_tokens=1, prompt_tokens=2, total_tokens=3
            ),
            created=int(datetime.datetime.now().timestamp()),
        )

    # OpenAI client call should fail as the API key is invalid.
    client = WrappedOpenAIClient(
        api_key="invalid_key", model_name="model_name"
    )
    fake_message = [
        {"role": "system", "content": "You are a helpful assistant."}
    ]
    with pytest.raises(OpenAIError):  # type: ignore - test is not run if openai is not installed
        client.connect()
        client(fake_message)

    # Check that the WrappedOpenAIClient does not alter the messages.
    assert fake_message == client._process_messages(fake_message)

    # The OpenAI Client should be able to connect if the API key is set as the environment variable.
    os.environ["OPENAI_API_KEY"] = "dummy_key"
    client = WrappedOpenAIClient(model_name="model_name")
    client.connect()

    client.client = MagicMock()

    # A bad request should raise a ValueError.
    client.client.chat.completions.create = _create_bad_request
    with pytest.raises(ValueError) as e:
        client(fake_message)

    # The metric computation should fail when the finish reason is bad length.
    client.client.chat.completions.create = (
        _create_mock_chat_completion_with_bad_length
    )
    with pytest.raises(ValueError) as e:
        client(fake_message)
    assert "reached max token limit" in str(e)

    # The metric computation should fail when the finish reason is content filter.
    client.client.chat.completions.create = (
        _create_mock_chat_completion_with_content_filter
    )
    with pytest.raises(ValueError) as e:
        client(fake_message)
    assert "flagged by content filter" in str(e)

    # Should run successfully when the finish reason is stop.
    client.client.chat.completions.create = _create_mock_chat_completion
    assert client(fake_message) == "some response"

    # Should run successfully even when the response content is None.
    client.client.chat.completions.create = (
        _create_mock_chat_completion_none_content
    )
    assert client(fake_message) == ""


@pytest.mark.skipif(
    not MISTRALAI_INSTALLED,
    reason="MistralAI is not installed.",
)
def test_WrappedMistralAIClient():
    def _create_bad_request(model, messages):
        raise ValueError

    def _create_mock_chat_completion_with_bad_length(
        model,
        messages,
    ) -> ChatCompletionResponse:  # type: ignore - test is not run if mistralai is not installed
        return ChatCompletionResponse(  # type: ignore - test is not run if mistralai is not installed
            id="foo",
            model="gpt-3.5-turbo",
            object="chat.completion",
            choices=[
                ChatCompletionChoice(  # type: ignore - test is not run if mistralai is not installed
                    finish_reason="length",
                    index=0,
                    message=AssistantMessage(  # type: ignore - test is not run if mistralai is not installed
                        role="assistant",
                        content="some response",
                        name=None,  # type: ignore - mistralai issue
                        tool_calls=None,
                        tool_call_id=None,  # type: ignore - mistralai issue
                    ),
                )
            ],
            created=int(datetime.datetime.now().timestamp()),
            usage=UsageInfo(  # type: ignore - test is not run if mistralai is not installed
                prompt_tokens=2, total_tokens=4, completion_tokens=199
            ),
        )

    def _create_mock_chat_completion(
        model, messages
    ) -> ChatCompletionResponse:  # type: ignore - test is not run if mistralai is not installed
        return ChatCompletionResponse(  # type: ignore - test is not run if mistralai is not installed
            id="foo",
            model="gpt-3.5-turbo",
            object="chat.completion",
            choices=[
                ChatCompletionChoice(  # type: ignore - test is not run if mistralai is not installed
                    finish_reason="stop",
                    index=0,
                    message=AssistantMessage(  # type: ignore - test is not run if mistralai is not installed
                        role="assistant",
                        content="some response",
                        name=None,  # type: ignore - mistralai issue
                        tool_calls=None,
                        tool_call_id=None,  # type: ignore - mistralai issue
                    ),
                )
            ],
            created=int(datetime.datetime.now().timestamp()),
            usage=UsageInfo(  # type: ignore - test is not run if mistralai is not installed
                prompt_tokens=2, total_tokens=4, completion_tokens=199
            ),
        )

    # Mistral client call should fail as the API key is invalid.
    client = WrappedMistralAIClient(
        api_key="invalid_key", model_name="model_name"
    )
    fake_message = [{"role": "assistant", "content": "content"}]
    with pytest.raises(MistralSDKError):  # type: ignore - test is not run if mistralai is not installed
        client.connect()
        client(fake_message)

    assert fake_message == client._process_messages(fake_message)

    # The Mistral Client should be able to connect if the API key is set as the environment variable.
    os.environ["MISTRAL_API_KEY"] = "dummy_key"
    client = WrappedMistralAIClient(model_name="model_name")
    client.connect()

    client.client = MagicMock()

    # The metric computation should fail if the request fails.
    client.client.chat.complete = _create_bad_request
    with pytest.raises(ValueError) as e:
        client(fake_message)

    # The metric computation should fail when the finish reason is bad length.
    client.client.chat.complete = _create_mock_chat_completion_with_bad_length
    with pytest.raises(ValueError) as e:
        client(fake_message)
    assert "reached max token limit" in str(e)

    # The metric computation should run successfully when the finish reason is stop.
    client.client.chat.complete = _create_mock_chat_completion
    assert client(fake_message) == "some response"


def test_MockLLMClient():
    client = MockLLMClient()

    # The MockLLMClient should not alter the messages.
    messages = [{"role": "system", "content": "You are a helpful assistant."}]
    assert messages == client._process_messages(messages)

    # The MockLLMClient should return nothing by default.
    assert "" == client(messages)


def test_process_message():
    # The messages should pass the validation in _process_messages.
    messages = [
        {
            "role": "system",
            "content": "You are a helpful assistant.",
        },
        {
            "role": "user",
            "content": "What is the weather like today?",
        },
        {
            "role": "assistant",
            "content": "The weather is sunny.",
        },
    ]
    WrappedOpenAIClient()._process_messages(messages=messages)
    WrappedMistralAIClient()._process_messages(messages=messages)
    MockLLMClient()._process_messages(messages=messages)

    # The clients should raise a ValueError because "content" is missing in the second message.
    messages = [
        {
            "role": "system",
            "content": "You are a helpful assistant.",
        },
        {
            "role": "user",
            "value": "What is the weather like today?",
        },
        {
            "role": "assistant",
            "content": "The weather is sunny.",
        },
    ]
    with pytest.raises(ValueError):
        WrappedOpenAIClient()._process_messages(messages=messages)
    with pytest.raises(ValueError):
        WrappedMistralAIClient()._process_messages(messages=messages)
    with pytest.raises(ValueError):
        MockLLMClient()._process_messages(messages=messages)
