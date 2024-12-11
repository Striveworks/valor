import os
from typing import Protocol


def _validate_messages(messages: list[dict[str, str]]):
    """
    Validate that the input is a list of dictionaries with "role" and "content" keys.

    Parameters
    ----------
    messages: list[dict[str, str]]
        The messages formatted according to the OpenAI standard. Each message in messages is a dictionary with "role" and "content" keys.
    """
    if not isinstance(messages, list):
        raise TypeError(
            f"messages must be a list, got {type(messages)} instead."
        )

    if not all(isinstance(message, dict) for message in messages):
        raise TypeError("messages must be a list of dictionaries.")

    if not all(
        "role" in message and "content" in message for message in messages
    ):
        raise ValueError(
            'messages must be a list of dictionaries with "role" and "content" keys.'
        )

    if not all(isinstance(message["role"], str) for message in messages):
        raise TypeError("All roles in messages must be strings.")

    if not all(isinstance(message["content"], str) for message in messages):
        raise TypeError("All content in messages must be strings.")


class ClientWrapper(Protocol):
    def __call__(
        self,
        messages: list[dict[str, str]],
    ) -> str:
        ...


class OpenAIWrapper:
    """
    Wrapper for calls to OpenAI's API.

    Attributes
    ----------
    model_name : str
        The model to use. Defaults to "gpt-3.5-turbo".
    api_key : str, optional
        The OpenAI API key to use. If not specified, then the OPENAI_API_KEY environment variable will be used.
    seed : int, optional
        An optional seed can be provided to GPT to get deterministic results.
    total_prompt_tokens : int
        A total count of tokens used for prompt inputs.
    total_completion_tokens : int
        A total count of tokens used to generate responses.
    """

    def __init__(
        self,
        model_name: str,
        api_key: str | None = None,
        seed: int | None = None,
    ):
        """
        Wrapper for calls to OpenAI's API.

        Parameters
        ----------
        model_name : str
            The model to use (e.g. "gpt-3.5-turbo").
        api_key : str, optional
            The OpenAI API key to use. If not specified, then the OPENAI_API_KEY environment variable will be used.
        seed : int, optional
            An optional seed can be provided to GPT to get deterministic results.
        """

        from openai import OpenAI

        if api_key is None:
            self.client = OpenAI()
        else:
            self.client = OpenAI(api_key=api_key)

        self.model_name = model_name
        self.seed = seed

        # logs
        self.total_prompt_tokens = 0
        self.total_completion_tokens = 0

    def __call__(
        self,
        messages: list[dict[str, str]],
    ) -> str:
        """
        Call to the API.

        Parameters
        ----------
        messages: list[dict[str, str]]
            The messages formatted according to the OpenAI standard. Each message in messages is a dictionary with "role" and "content" keys.

        Returns
        -------
        str
            The response from the API.
        """
        _validate_messages(messages=messages)

        openai_response = self.client.chat.completions.create(
            model=self.model_name,
            messages=messages,  # type: ignore - mistralai issue
            seed=self.seed,
        )

        response = openai_response.choices[0].message.content
        if openai_response.usage is not None:
            self.total_prompt_tokens += openai_response.usage.prompt_tokens
            self.total_completion_tokens += (
                openai_response.usage.completion_tokens
            )
        finish_reason = openai_response.choices[
            0
        ].finish_reason  # Enum: "stop" "length" "content_filter" "tool_calls" "function_call"

        if finish_reason == "length":
            raise ValueError(
                "OpenAI response reached max token limit. Resulting evaluation is likely invalid or of low quality."
            )
        elif finish_reason == "content_filter":
            raise ValueError(
                "OpenAI response was flagged by content filter. Resulting evaluation is likely invalid or of low quality."
            )

        if response is None:
            response = ""
        return response


class MistralWrapper:
    """
    Wrapper for calls to Mistral's API.

    Attributes
    ----------
    api_key : str, optional
        The Mistral API key to use. If not specified, then the MISTRAL_API_KEY environment variable will be used.
    model_name : str
        The model to use. Defaults to "mistral-small-latest".
    """

    def __init__(
        self,
        model_name: str,
        api_key: str | None = None,
    ):
        """
        Creates an instance of the Mistral interface.

        Parameters
        ----------
        model_name : str
            The model to use (e.g. "mistral-small-latest").
        api_key : str, optional
            The Mistral API key to use. If not specified, then the MISTRAL_API_KEY environment variable will be used.
        """

        from mistralai import Mistral

        if api_key is None:
            self.client = Mistral(api_key=os.getenv("MISTRAL_API_KEY"))
        else:
            self.client = Mistral(api_key=api_key)

        self.model_name = model_name

    def __call__(
        self,
        messages: list[dict[str, str]],
    ) -> str:
        """
        Call to the API.

        Parameters
        ----------
        messages: list[dict[str, str]]
            The messages formatted according to the OpenAI standard. Each message in messages is a dictionary with "role" and "content" keys.

        Returns
        -------
        str
            The response from the API.
        """
        _validate_messages(messages)

        mistral_response = self.client.chat.complete(
            model=self.model_name,
            messages=messages,  # type: ignore - mistral complaining about native types
        )
        if (
            mistral_response is None
            or mistral_response.choices is None
            or mistral_response.choices[0].message is None
            or mistral_response.choices[0].message.content is None
        ):
            return ""

        response = mistral_response.choices[0].message.content

        finish_reason = mistral_response.choices[
            0
        ].finish_reason  # Enum: "stop" "length" "model_length" "error" "tool_calls"

        if finish_reason == "length":
            raise ValueError(
                "Mistral response reached max token limit. Resulting evaluation is likely invalid or of low quality."
            )

        if not isinstance(response, str):
            raise TypeError("Mistral AI response was not a string.")

        return response
