import pytest
from pydantic import ValidationError

from valor_api.backend.core.llm_clients import (
    LLMClient,
    MockLLMClient,
    WrappedMistralAIClient,
    WrappedOpenAIClient,
)


def test_process_message():

    # The messages should process successfully.
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
    LLMClient().process_messages(messages=messages)
    WrappedOpenAIClient().process_messages(messages=messages)
    WrappedMistralAIClient().process_messages(messages=messages)
    MockLLMClient().process_messages(messages=messages)

    # Missing "content" in the second message, so it should raise a ValidationError.
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
        LLMClient().process_messages(messages=messages)
    with pytest.raises(ValidationError):
        WrappedOpenAIClient().process_messages(messages=messages)
    with pytest.raises(ValidationError):
        WrappedMistralAIClient().process_messages(messages=messages)
    with pytest.raises(ValidationError):
        MockLLMClient().process_messages(messages=messages)
