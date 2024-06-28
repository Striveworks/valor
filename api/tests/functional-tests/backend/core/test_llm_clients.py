import datetime
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

from valor_api.backend.core.llm_clients import (
    InvalidLLMResponseError,
    LLMClient,
    WrappedMistralAIClient,
    WrappedOpenAIClient,
)

ANSWER_RELEVANCE_VALID_STATEMENTS = """```json
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


def test_LLMClient(monkeypatch):
    """Check that this parent class mostly throws NotImplementedErrors, since its methods are intended to be overridden by its children."""

    def _return_valid_answer_relevance_response(*args, **kwargs):
        if "generate a list of statements" in args[1][1]["content"]:
            return ANSWER_RELEVANCE_VALID_STATEMENTS
        elif (
            "determine whether each statement is relevant to address the input"
            in args[1][1]["content"]
        ):
            return ANSWER_RELEVANCE_VALID_VERDICTS
        else:
            raise ValueError

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
        if "generate a list of statements" in args[1][1]["content"]:
            return ANSWER_RELEVANCE_VALID_STATEMENTS
        elif (
            "determine whether each statement is relevant to address the input"
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
            raise ValueError

    def _return_invalid4_answer_relevance_response(*args, **kwargs):
        if "generate a list of statements" in args[1][1]["content"]:
            return ANSWER_RELEVANCE_VALID_STATEMENTS
        elif (
            "determine whether each statement is relevant to address the input"
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
            raise ValueError

    def _return_valid_coherence_response(*args, **kwargs):
        return "5"

    def _return_invalid_response(*args, **kwargs):
        return "some bad response"

    client = LLMClient(api_key=None, model_name="model_name")

    fake_message = [{"key": "value"}]
    with pytest.raises(NotImplementedError):
        client.connect()

    fake_message = client.process_messages(fake_message)
    with pytest.raises(NotImplementedError):
        client(fake_message)

    # patch __call__ with a valid response
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

    # Invalid verdict.
    monkeypatch.setattr(
        "valor_api.backend.core.llm_clients.LLMClient.__call__",
        _return_invalid4_answer_relevance_response,
    )
    with pytest.raises(InvalidLLMResponseError):
        client.answer_relevance("some query", "some text")

    # patch __call__ with a valid response
    monkeypatch.setattr(
        "valor_api.backend.core.llm_clients.LLMClient.__call__",
        _return_valid_coherence_response,
    )
    assert 5 == client.coherence("some text")

    # patch __call__ with an invalid response
    monkeypatch.setattr(
        "valor_api.backend.core.llm_clients.LLMClient.__call__",
        _return_invalid_response,
    )
    with pytest.raises(InvalidLLMResponseError):
        client.coherence("some text")


def test_WrappedOpenAIClient():
    def create_bad_request(model, messages, seed) -> ChatCompletion:
        raise ValueError

    def create_mock_chat_completion_with_bad_length(
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

    def create_mock_chat_completion_with_content_filter(
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

    def create_mock_chat_completion(model, messages, seed) -> ChatCompletion:
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

    client = WrappedOpenAIClient(
        api_key="invalid_key", model_name="model_name"
    )

    fake_message = [{"key": "value"}]

    with pytest.raises(OpenAIError):
        client.connect()
        client(fake_message)

    assert fake_message == client.process_messages(fake_message)

    client.client = MagicMock()

    # test bad request
    client.client.chat.completions.create = create_bad_request
    with pytest.raises(ValueError) as e:
        client(fake_message)

    # test good request with bad finish reasons
    client.client.chat.completions.create = (
        create_mock_chat_completion_with_bad_length
    )
    with pytest.raises(ValueError) as e:
        client(fake_message)
    assert "reached max token limit" in str(e)

    client.client.chat.completions.create = (
        create_mock_chat_completion_with_content_filter
    )
    with pytest.raises(ValueError) as e:
        client(fake_message)
    assert "flagged by content filter" in str(e)

    # test good request
    client.client.chat.completions.create = create_mock_chat_completion
    assert client(fake_message) == "some response"


def test_WrappedMistralAIClient():
    def create_bad_request(model, messages) -> ChatCompletion:
        raise ValueError

    def create_mock_chat_completion_with_bad_length(
        model, messages
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

    def create_mock_chat_completion(model, messages) -> ChatCompletionResponse:
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
    ] == client.process_messages(fake_message)

    client.client = MagicMock()

    # test bad request
    client.client.chat = create_bad_request
    with pytest.raises(ValueError) as e:
        client(fake_message)

    # test good request with bad finish reasons
    client.client.chat = create_mock_chat_completion_with_bad_length
    with pytest.raises(ValueError) as e:
        client(fake_message)
    assert "reached max token limit" in str(e)

    # test good request
    client.client.chat = create_mock_chat_completion
    assert client(fake_message) == "some response"
