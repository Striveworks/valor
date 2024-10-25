import datetime
import json
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

from valor_lite.text_generation.exceptions import InvalidLLMResponseError
from valor_lite.text_generation.integrations import (
    MistralWrapper,
    MockWrapper,
    OpenAIWrapper,
    _validate_messages,
)


def test__validate_messages():
    # Valid messages.
    _validate_messages(
        messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "Hello!"},
        ],
    )

    # messages must be a list
    with pytest.raises(TypeError):
        _validate_messages(
            messages={"role": "system", "content": "You are a helpful assistant."}  # type: ignore - testing
        )

    # messages must be a list of dictionaries
    with pytest.raises(TypeError):
        _validate_messages(
            messages=["You are a helpful assistant."]  # type: ignore - testing
        )

    # Each message must have 'role' and 'content' keys
    with pytest.raises(ValueError):
        _validate_messages(
            messages=[
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user"},
            ]
        )

    # Role values must be strings
    with pytest.raises(TypeError):
        _validate_messages(
            messages=[
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": 1, "content": "Hello!"},  # type: ignore - testing
            ]
        )

    # Content values must be a string
    with pytest.raises(TypeError):
        _validate_messages(
            messages=[
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": 1},  # type: ignore - testing
            ]
        )


@pytest.mark.skipif(
    not OPENAI_INSTALLED,
    reason="Openai is not installed.",
)
def test_openai_client():
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
    client = OpenAIWrapper(api_key="invalid_key", model_name="model_name")
    fake_message = [
        {"role": "system", "content": "You are a helpful assistant."}
    ]
    with pytest.raises(OpenAIError):  # type: ignore - test is not run if openai is not installed
        client.connect()
        client(fake_message)

    # Check that the OpenAIWrapper does not alter the messages.
    assert fake_message == client._process_messages(fake_message)

    # The OpenAI Client should be able to connect if the API key is set as the environment variable.
    os.environ["OPENAI_API_KEY"] = "dummy_key"
    client = OpenAIWrapper(model_name="model_name")
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
def test_mistral_client():
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

    def _create_mock_chat_completion_choices_is_None(
        model, messages
    ) -> ChatCompletionResponse:  # type: ignore - test is not run if mistralai is not installed
        return ChatCompletionResponse(  # type: ignore - test is not run if mistralai is not installed
            id="foo",
            model="gpt-3.5-turbo",
            object="chat.completion",
            choices=None,
            created=int(datetime.datetime.now().timestamp()),
            usage=UsageInfo(  # type: ignore - test is not run if mistralai is not installed
                prompt_tokens=2, total_tokens=4, completion_tokens=199
            ),
        )

    def _create_mock_chat_completion_bad_message_content(
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
    client = MistralWrapper(api_key="invalid_key", model_name="model_name")
    fake_message = [{"role": "assistant", "content": "content"}]
    with pytest.raises(MistralSDKError):  # type: ignore - test is not run if mistralai is not installed
        client.connect()
        client(fake_message)

    assert fake_message == client._process_messages(fake_message)

    # The Mistral Client should be able to connect if the API key is set as the environment variable.
    os.environ["MISTRAL_API_KEY"] = "dummy_key"
    client = MistralWrapper(model_name="model_name")
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

    # The metric computation should run successfully even when choices is None.
    client.client.chat.complete = _create_mock_chat_completion_choices_is_None
    assert client(fake_message) == ""

    # The metric computation should fail when the message doesn't contain any content.
    client.client.chat.complete = (
        _create_mock_chat_completion_bad_message_content
    )
    with pytest.raises(TypeError) as e:
        client(fake_message)
        assert "Mistral AI response was not a string." in str(e)


def test_mock_client():
    client = MockWrapper()

    # The MockWrapper should not alter the messages.
    messages = [{"role": "system", "content": "You are a helpful assistant."}]
    assert messages == client._process_messages(messages)

    # The MockWrapper should return nothing by default.
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
    OpenAIWrapper(model_name="gpt-3.5-turbo")._process_messages(
        messages=messages
    )
    MistralWrapper(model_name="mistral-small-latest")._process_messages(
        messages=messages
    )
    MockWrapper()._process_messages(messages=messages)

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
        OpenAIWrapper(model_name="gpt-3.5-turbo")._process_messages(
            messages=messages
        )
    with pytest.raises(ValueError):
        MistralWrapper(model_name="mistral-small-latest")._process_messages(
            messages=messages
        )
    with pytest.raises(ValueError):
        MockWrapper()._process_messages(messages=messages)
