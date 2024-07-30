import datetime
import os
from unittest.mock import MagicMock

import pytest
from mistralai.exceptions import MistralException
from mistralai.models.chat_completion import (
    ChatCompletionResponse,
    ChatCompletionResponseChoice,
    ChatMessage,
    FinishReason,
)
from mistralai.models.common import UsageInfo
from openai import OpenAIError
from openai.types.chat import ChatCompletionMessage
from openai.types.chat.chat_completion import ChatCompletion, Choice
from pydantic import ValidationError

from valor_api.backend.core.llm_clients import (
    LLMClient,
    MockLLMClient,
    WrappedMistralAIClient,
    WrappedOpenAIClient,
)
from valor_api.exceptions import (
    BadTestLLMClientsValueError,
    InvalidLLMResponseError,
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

ANSWER_RELEVANCE_VALID_VERDICTS = """```json
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
            "verdict": "yes"
        }
    ]
}```"""

BIAS_VALID_VERDICTS = """```json
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
            "verdict": "no"
        }
    ]
}```"""

CONTEXT_RELEVANCE_VALID_VERDICTS = """```json
{
    "verdicts": [
        {
            "verdict": "no",
            "reason": "This context does not relate to the query."
        },
        {
            "verdict": "yes"
        },
        {
            "verdict": "no",
            "reason": "This context is not useful for answering the query."
        }
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

HALLUCINATION_AGREEMENT_VALID_VERDICTS = """```json
{
    "verdicts": [
        {
            "verdict": "yes"
        },

        {
            "verdict": "no",
            "reason": "The text and context disagree on when Abraham Lincoln was born."
        },
        {
            "verdict": "no",
            "reason": "The text says that Abraham Lincoln lost the election of 1860, but the context says that Abraham Lincoln won the election of 1860."
        }
    ]
}```"""

TOXICITY_VALID_VERDICTS = """```json
{
    "verdicts": [
        {
            "verdict": "yes",
            "reason": "This opinion demonstrates hate."
        },
        {
            "verdict": "no"
        },
        {
            "verdict": "yes",
            "reason": "This opinion demonstrates mockery."
        },
        {
            "verdict": "no"
        }
    ]
}```"""


def test_LLMClient(monkeypatch):
    """
    Check that LLMClient throws NotImplementedErrors for connect and __call__.

    Check the metric computations for LLMClient. The client children inherit all of these metric computations.
    """

    def _return_valid_answer_relevance_response(*args, **kwargs):
        if "generate a list of STATEMENTS" in args[1][1]["content"]:
            return VALID_STATEMENTS
        elif (
            "generate a list of verdicts that indicate whether each statement is relevant to address the input"
            in args[1][1]["content"]
        ):
            return ANSWER_RELEVANCE_VALID_VERDICTS
        else:
            raise BadTestLLMClientsValueError

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
            "generate a list of verdicts that indicate whether each statement is relevant to address the input"
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
            raise BadTestLLMClientsValueError

    def _return_invalid4_answer_relevance_response(*args, **kwargs):
        if "generate a list of STATEMENTS" in args[1][1]["content"]:
            return VALID_STATEMENTS
        elif (
            "generate a list of verdicts that indicate whether each statement is relevant to address the input"
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
            raise BadTestLLMClientsValueError

    def _return_valid1_bias_response(*args, **kwargs):
        if "generate a list of OPINIONS" in args[1][1]["content"]:
            return VALID_OPINIONS
        elif (
            "generate a list of verdicts to indicate whether EACH opinion is biased"
            in args[1][1]["content"]
        ):
            return BIAS_VALID_VERDICTS
        else:
            raise BadTestLLMClientsValueError

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
            raise BadTestLLMClientsValueError

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
            raise BadTestLLMClientsValueError

    def _return_valid_coherence_response(*args, **kwargs):
        return "5"

    def _return_invalid1_coherence_response(*args, **kwargs):
        return "The score is 5."

    def _return_invalid2_coherence_response(*args, **kwargs):
        return "0"

    def _return_valid_context_relevance_response(*args, **kwargs):
        return CONTEXT_RELEVANCE_VALID_VERDICTS

    def _return_invalid1_context_relevance_response(*args, **kwargs):
        return """```json
{
    "all_verdicts": [
        "verdict 1",
        "verdict 2",
        "verdict 3"
    ]
}```"""

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
            raise BadTestLLMClientsValueError

    def _return_valid2_faithfulness_response(*args, **kwargs):
        return """```json
{
    "claims": []
}```"""

    def _return_invalid1_faithfulness_response(*args, **kwargs):
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
    "list": [
        "verdict 1",
        "verdict 2",
        "verdict 3",
        "verdict 4",
        "verdict 5"
    ]
}```"""
        else:
            raise BadTestLLMClientsValueError

    def _return_invalid2_faithfulness_response(*args, **kwargs):
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
    "claims": [
        "verdict 1",
        2,
        "verdict 3",
        "verdict 4",
        "verdict 5"
    ]
}```"""
        else:
            raise BadTestLLMClientsValueError

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
            raise BadTestLLMClientsValueError

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
        {"verdict": "yes"},
    ]
}```"""
        else:
            raise BadTestLLMClientsValueError

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
            raise BadTestLLMClientsValueError

    def _return_valid_hallucination_response(*args, **kwargs):
        return HALLUCINATION_AGREEMENT_VALID_VERDICTS

    def _return_invalid1_hallucination_response(*args, **kwargs):
        return """```json
{
    "bad key": [
        {"verdict": "yes"},
        {"verdict": "no"},
        {"verdict": "yes"}
    ]
}```"""

    def _return_valid1_toxicity_response(*args, **kwargs):
        if "generate a list of OPINIONS" in args[1][1]["content"]:
            return VALID_OPINIONS
        elif (
            "generate a list of verdicts to indicate whether EACH opinion is toxic"
            in args[1][1]["content"]
        ):
            return TOXICITY_VALID_VERDICTS
        else:
            raise BadTestLLMClientsValueError

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
            raise BadTestLLMClientsValueError

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
            raise BadTestLLMClientsValueError

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
        "valor_api.backend.core.llm_clients.LLMClient.__call__",
        _return_valid_answer_relevance_response,
    )
    assert 0.5 == client.answer_relevance("some query", "some answer")

    # Needs to have 'statements' key.
    monkeypatch.setattr(
        "valor_api.backend.core.llm_clients.LLMClient.__call__",
        _return_invalid1_answer_relevance_response,
    )
    with pytest.raises(InvalidLLMResponseError):
        client.answer_relevance("some query", "some text")

    # Statements must be strings.
    monkeypatch.setattr(
        "valor_api.backend.core.llm_clients.LLMClient.__call__",
        _return_invalid2_answer_relevance_response,
    )
    with pytest.raises(InvalidLLMResponseError):
        client.answer_relevance("some query", "some text")

    # Needs to have 'verdicts' key.
    monkeypatch.setattr(
        "valor_api.backend.core.llm_clients.LLMClient.__call__",
        _return_invalid3_answer_relevance_response,
    )
    with pytest.raises(InvalidLLMResponseError):
        client.answer_relevance("some query", "some text")

    # Invalid verdict, all verdicts must be yes, no or idk.
    monkeypatch.setattr(
        "valor_api.backend.core.llm_clients.LLMClient.__call__",
        _return_invalid4_answer_relevance_response,
    )
    with pytest.raises(InvalidLLMResponseError):
        client.answer_relevance("some query", "some text")

    # Patch __call__ with a valid response.
    monkeypatch.setattr(
        "valor_api.backend.core.llm_clients.LLMClient.__call__",
        _return_valid1_bias_response,
    )
    assert 0.5 == client.bias("some text")

    # No opinions found, so no bias should be reported.
    monkeypatch.setattr(
        "valor_api.backend.core.llm_clients.LLMClient.__call__",
        _return_valid2_bias_response,
    )
    assert 0.0 == client.bias("some text")

    # Key 'verdicts' is returned but the key should be 'opinions'.
    monkeypatch.setattr(
        "valor_api.backend.core.llm_clients.LLMClient.__call__",
        _return_invalid1_bias_response,
    )
    with pytest.raises(InvalidLLMResponseError):
        client.bias("some text")

    # Opinions must be strings.
    monkeypatch.setattr(
        "valor_api.backend.core.llm_clients.LLMClient.__call__",
        _return_invalid2_bias_response,
    )
    with pytest.raises(InvalidLLMResponseError):
        client.bias("some text")

    # Key 'opinions' is returned but the key should be 'verdicts'.
    monkeypatch.setattr(
        "valor_api.backend.core.llm_clients.LLMClient.__call__",
        _return_invalid3_bias_response,
    )
    with pytest.raises(InvalidLLMResponseError):
        client.bias("some text")

    # 'idk' is not a valid bias verdict.
    monkeypatch.setattr(
        "valor_api.backend.core.llm_clients.LLMClient.__call__",
        _return_invalid4_bias_response,
    )
    with pytest.raises(InvalidLLMResponseError):
        client.bias("some text")

    # Patch __call__ with a valid response.
    monkeypatch.setattr(
        "valor_api.backend.core.llm_clients.LLMClient.__call__",
        _return_valid_coherence_response,
    )
    assert 5 == client.coherence("some text")

    # Coherence score is not an integer.
    monkeypatch.setattr(
        "valor_api.backend.core.llm_clients.LLMClient.__call__",
        _return_invalid1_coherence_response,
    )
    with pytest.raises(InvalidLLMResponseError):
        client.coherence("some text")

    # Coherence score is 0, which is not in {1,2,3,4,5}.
    monkeypatch.setattr(
        "valor_api.backend.core.llm_clients.LLMClient.__call__",
        _return_invalid2_coherence_response,
    )
    with pytest.raises(InvalidLLMResponseError):
        client.coherence("some text")

    # Patch __call__ with a valid response.
    monkeypatch.setattr(
        "valor_api.backend.core.llm_clients.LLMClient.__call__",
        _return_valid_context_relevance_response,
    )
    assert 0.3333333333333333 == client.context_relevance(
        "some query", ["context 1", "context 2", "context 3"]
    )

    # Context relevance doesn't make sense if no contexts are provided.
    monkeypatch.setattr(
        "valor_api.backend.core.llm_clients.LLMClient.__call__",
        _return_valid_context_relevance_response,
    )
    with pytest.raises(ValueError):
        client.context_relevance("some query", [])

    # Only 1 context provided but 3 verdicts were returned.
    monkeypatch.setattr(
        "valor_api.backend.core.llm_clients.LLMClient.__call__",
        _return_valid_context_relevance_response,
    )
    with pytest.raises(InvalidLLMResponseError):
        client.context_relevance(
            "some query",
            ["length of context list does not match LLM's response"],
        )

    # Key 'all_verdicts' is returned but the key should be 'verdicts'.
    monkeypatch.setattr(
        "valor_api.backend.core.llm_clients.LLMClient.__call__",
        _return_invalid1_context_relevance_response,
    )
    with pytest.raises(InvalidLLMResponseError):
        client.context_relevance(
            "some query", ["context 1", "context 2", "context 3"]
        )

    # Patch __call__ with a valid response.
    monkeypatch.setattr(
        "valor_api.backend.core.llm_clients.LLMClient.__call__",
        _return_valid1_faithfulness_response,
    )
    assert 0.6 == client.faithfulness("some text", ["context 1", "context 2"])

    # If no claims are found in the text, then the text should have a faithfulness score of 1.
    monkeypatch.setattr(
        "valor_api.backend.core.llm_clients.LLMClient.__call__",
        _return_valid2_faithfulness_response,
    )
    assert 1.0 == client.faithfulness("some text", ["context 1", "context 2"])

    # Faithfulness is meaningless if no contexts are provided, so should throw a ValueError.
    monkeypatch.setattr(
        "valor_api.backend.core.llm_clients.LLMClient.__call__",
        _return_valid1_faithfulness_response,
    )
    with pytest.raises(ValueError):
        client.faithfulness("some text", [])

    # Bad key in the response.
    monkeypatch.setattr(
        "valor_api.backend.core.llm_clients.LLMClient.__call__",
        _return_invalid1_faithfulness_response,
    )
    with pytest.raises(InvalidLLMResponseError):
        client.faithfulness("some text", ["context 1", "context 2"])

    # Invalid claim value.
    monkeypatch.setattr(
        "valor_api.backend.core.llm_clients.LLMClient.__call__",
        _return_invalid2_faithfulness_response,
    )
    with pytest.raises(InvalidLLMResponseError):
        client.faithfulness("some text", ["context 1", "context 2"])

    # Bad key in the response.
    monkeypatch.setattr(
        "valor_api.backend.core.llm_clients.LLMClient.__call__",
        _return_invalid3_faithfulness_response,
    )
    with pytest.raises(InvalidLLMResponseError):
        client.faithfulness("some text", ["context 1", "context 2"])

    # Number of verdicts does not match the number of claims.
    monkeypatch.setattr(
        "valor_api.backend.core.llm_clients.LLMClient.__call__",
        _return_invalid4_faithfulness_response,
    )
    with pytest.raises(InvalidLLMResponseError):
        client.faithfulness("some text", ["context 1", "context 2"])

    # 'idk' is not a valid verdict for faithfulness.
    monkeypatch.setattr(
        "valor_api.backend.core.llm_clients.LLMClient.__call__",
        _return_invalid5_faithfulness_response,
    )
    with pytest.raises(InvalidLLMResponseError):
        client.faithfulness("some text", ["context 1", "context 2"])

    # Patch __call__ with a valid response.
    monkeypatch.setattr(
        "valor_api.backend.core.llm_clients.LLMClient.__call__",
        _return_valid_hallucination_response,
    )
    assert 0.6666666666666666 == client.hallucination(
        "some answer", ["context 1", "context 2", "context 3"]
    )

    # Context relevance doesn't make sense if no contexts are provided.
    monkeypatch.setattr(
        "valor_api.backend.core.llm_clients.LLMClient.__call__",
        _return_valid_hallucination_response,
    )
    with pytest.raises(ValueError):
        client.hallucination("some query", [])

    # Only 1 context provided but 3 verdicts were returned.
    monkeypatch.setattr(
        "valor_api.backend.core.llm_clients.LLMClient.__call__",
        _return_valid_hallucination_response,
    )
    with pytest.raises(InvalidLLMResponseError):
        client.hallucination(
            "some query",
            ["length of context list does not match LLM's response"],
        )

    # Key 'all_verdicts' is returned but the key should be 'verdicts'.
    monkeypatch.setattr(
        "valor_api.backend.core.llm_clients.LLMClient.__call__",
        _return_invalid1_hallucination_response,
    )
    with pytest.raises(InvalidLLMResponseError):
        client.hallucination(
            "some query", ["context 1", "context 2", "context 3"]
        )

    # Patch __call__ with a valid response.
    monkeypatch.setattr(
        "valor_api.backend.core.llm_clients.LLMClient.__call__",
        _return_valid1_toxicity_response,
    )
    assert 0.5 == client.toxicity("some text")

    # No opinions found, so no toxicity should be reported.
    monkeypatch.setattr(
        "valor_api.backend.core.llm_clients.LLMClient.__call__",
        _return_valid2_toxicity_response,
    )
    assert 0.0 == client.toxicity("some text")

    # Key 'verdicts' is returned but the key should be 'opinions'.
    monkeypatch.setattr(
        "valor_api.backend.core.llm_clients.LLMClient.__call__",
        _return_invalid1_toxicity_response,
    )
    with pytest.raises(InvalidLLMResponseError):
        client.toxicity("some text")

    # Opinions must be strings.
    monkeypatch.setattr(
        "valor_api.backend.core.llm_clients.LLMClient.__call__",
        _return_invalid2_toxicity_response,
    )
    with pytest.raises(InvalidLLMResponseError):
        client.toxicity("some text")

    # Key 'opinions' is returned but the key should be 'verdicts'.
    monkeypatch.setattr(
        "valor_api.backend.core.llm_clients.LLMClient.__call__",
        _return_invalid3_toxicity_response,
    )
    with pytest.raises(InvalidLLMResponseError):
        client.toxicity("some text")

    # 'idk' is not a valid toxicity verdict.
    monkeypatch.setattr(
        "valor_api.backend.core.llm_clients.LLMClient.__call__",
        _return_invalid4_toxicity_response,
    )
    with pytest.raises(InvalidLLMResponseError):
        client.toxicity("some text")


def test_WrappedOpenAIClient():
    def _create_bad_request(model, messages, seed) -> ChatCompletion:
        raise ValueError

    def _create_mock_chat_completion_with_bad_length(
        model, messages, seed
    ) -> ChatCompletion:
        return ChatCompletion(
            id="foo",
            model="gpt-3.5-turbo",
            object="chat.completion",
            choices=[
                Choice(
                    finish_reason="length",
                    index=0,
                    message=ChatCompletionMessage(
                        content="some response",
                        role="assistant",
                    ),
                )
            ],
            created=int(datetime.datetime.now().timestamp()),
        )

    def _create_mock_chat_completion_with_content_filter(
        model, messages, seed
    ) -> ChatCompletion:
        return ChatCompletion(
            id="foo",
            model="gpt-3.5-turbo",
            object="chat.completion",
            choices=[
                Choice(
                    finish_reason="content_filter",
                    index=0,
                    message=ChatCompletionMessage(
                        content="some response",
                        role="assistant",
                    ),
                )
            ],
            created=int(datetime.datetime.now().timestamp()),
        )

    def _create_mock_chat_completion(model, messages, seed) -> ChatCompletion:
        return ChatCompletion(
            id="foo",
            model="gpt-3.5-turbo",
            object="chat.completion",
            choices=[
                Choice(
                    finish_reason="stop",
                    index=0,
                    message=ChatCompletionMessage(
                        content="some response",
                        role="assistant",
                    ),
                )
            ],
            created=int(datetime.datetime.now().timestamp()),
        )

    def _create_mock_chat_completion_none_content(
        model, messages, seed
    ) -> ChatCompletion:
        return ChatCompletion(
            id="foo",
            model="gpt-3.5-turbo",
            object="chat.completion",
            choices=[
                Choice(
                    finish_reason="stop",
                    index=0,
                    message=ChatCompletionMessage(
                        content=None,
                        role="assistant",
                    ),
                )
            ],
            created=int(datetime.datetime.now().timestamp()),
        )

    # OpenAI client call should fail as the API key is invalid.
    client = WrappedOpenAIClient(
        api_key="invalid_key", model_name="model_name"
    )
    fake_message = [
        {"role": "system", "content": "You are a helpful assistant."}
    ]
    with pytest.raises(OpenAIError):
        client.connect()
        client(fake_message)

    # Check that the WrappedOpenAIClient does not alter the messages.
    assert fake_message == client._process_messages(fake_message)

    # OpenAI only allows the roles of system, user and assistant.
    invalid_message = [{"role": "invalid", "content": "Some content."}]
    with pytest.raises(ValueError):
        client._process_messages(invalid_message)

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


def test_WrappedMistralAIClient():
    def _create_bad_request(model, messages) -> ChatCompletion:
        raise ValueError

    def _create_mock_chat_completion_with_bad_length(
        model,
        messages,
    ) -> ChatCompletionResponse:
        return ChatCompletionResponse(
            id="foo",
            model="gpt-3.5-turbo",
            object="chat.completion",
            choices=[
                ChatCompletionResponseChoice(
                    finish_reason=FinishReason("length"),
                    index=0,
                    message=ChatMessage(
                        role="role",
                        content="some content",
                        name=None,
                        tool_calls=None,
                        tool_call_id=None,
                    ),
                )
            ],
            created=int(datetime.datetime.now().timestamp()),
            usage=UsageInfo(
                prompt_tokens=2, total_tokens=4, completion_tokens=199
            ),
        )

    def _create_mock_chat_completion(
        model, messages
    ) -> ChatCompletionResponse:
        return ChatCompletionResponse(
            id="foo",
            model="gpt-3.5-turbo",
            object="chat.completion",
            choices=[
                ChatCompletionResponseChoice(
                    finish_reason=FinishReason("stop"),
                    index=0,
                    message=ChatMessage(
                        role="role",
                        content="some response",
                        name=None,
                        tool_calls=None,
                        tool_call_id=None,
                    ),
                )
            ],
            created=int(datetime.datetime.now().timestamp()),
            usage=UsageInfo(
                prompt_tokens=2, total_tokens=4, completion_tokens=199
            ),
        )

    # Mistral client call should fail as the API key is invalid.
    client = WrappedMistralAIClient(
        api_key="invalid_key", model_name="model_name"
    )
    fake_message = [{"role": "role", "content": "content"}]
    with pytest.raises(MistralException):
        client.connect()
        client(fake_message)

    assert [
        ChatMessage(
            role="role",
            content="content",
            name=None,
            tool_calls=None,
            tool_call_id=None,
        )
    ] == client._process_messages(fake_message)

    # The Mistral Client should be able to connect if the API key is set as the environment variable.
    os.environ["MISTRAL_API_KEY"] = "dummy_key"
    client = WrappedMistralAIClient(model_name="model_name")
    client.connect()

    client.client = MagicMock()

    # The metric computation should fail if the request fails.
    client.client.chat = _create_bad_request
    with pytest.raises(ValueError) as e:
        client(fake_message)

    # The metric computation should fail when the finish reason is bad length.
    client.client.chat = _create_mock_chat_completion_with_bad_length
    with pytest.raises(ValueError) as e:
        client(fake_message)
    assert "reached max token limit" in str(e)

    # The metric computation should run successfully when the finish reason is stop.
    client.client.chat = _create_mock_chat_completion
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

    # The clients should raise a ValidationError because "content" is missing in the second message.
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
    with pytest.raises(ValidationError):
        WrappedOpenAIClient()._process_messages(messages=messages)
    with pytest.raises(ValidationError):
        WrappedMistralAIClient()._process_messages(messages=messages)
    with pytest.raises(ValidationError):
        MockLLMClient()._process_messages(messages=messages)
