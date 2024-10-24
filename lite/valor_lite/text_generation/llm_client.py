from functools import wraps
from typing import Any

try:
    from mistralai.sdk import Mistral
except ImportError:
    Mistral = None

try:
    from openai import OpenAI
except ImportError:
    OpenAI = None

DEFAULT_SYSTEM_PROMPT = "You are a helpful assistant."


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


def retry_if_invalid_llm_response():
    """
    Call the LLMClient class function with retries for InvalidLLMResponseError.

    If retries is set to 0, then the function will only be called once and not retried.

    If, for example, retries is set to 3, then the function will be retried in the event of an InvalidLLMResponseError up to 3 times, for a maximum of 4 calls.
    """

    def decorator(function):
        @wraps(function)
        def wrapper(self, *args, **kwargs):
            error = None
            retries = getattr(self, "retries", 0)
            for _ in range(1 + retries):
                try:
                    return function(self, *args, **kwargs)
                except InvalidLLMResponseError as e:
                    error = e
            if error is not None:
                raise error

        return wrapper

    return decorator


class _ClientWrapper:
    def connect(
        self,
    ):
        """
        Setup the connection to the API. Not implemented for parent class.
        """
        raise NotImplementedError("_ClientWrapper is a template class.")

    def _process_messages(
        self,
        messages: list[dict[str, str]],
    ) -> Any:
        """
        Format messages for the API.

        Parameters
        ----------
        messages: list[dict[str, str]]
            The messages formatted according to the OpenAI standard. Each message in messages is a dictionary with "role" and "content" keys.

        Returns
        -------
        Any
            The messages formatted for the API.
        """
        raise NotImplementedError("_ClientWrapper is a template class.")

    def __call__(
        self,
        messages: list[dict[str, str]],
    ) -> str:
        """
        Call to the API. Not implemented for parent class.

        Parameters
        ----------
        messages: list[dict[str, str]]
            The messages formatted according to the OpenAI standard. Each message in messages is a dictionary with "role" and "content" keys.

        Returns
        -------
        str
            The response from the API.
        """
        raise NotImplementedError("_ClientWrapper is a template class.")


class OpenAIWrapper(_ClientWrapper):
    """
    Wrapper for calls to OpenAI's API.

    Attributes
    ----------
    model_name : str
        The model to use. Defaults to "gpt-3.5-turbo".
    api_key : str, optional
        The OpenAI API key to use. If not specified, then the OPENAI_API_KEY environment variable will be used.
    retries : int
        The number of times to retry the API call if it fails. Defaults to 0, indicating that the call will not be retried. For example, if self.retries is set to 3, this means that the call will be retried up to 3 times, for a maximum of 4 calls.
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
        retries: int = 0,
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
        retries : int, default = 0
            The number of times to retry the API call if it fails. Defaults to 0, indicating that the call will not be retried. For example, if self.retries is set to 3, this means that the call will be retried up to 3 times, for a maximum of 4 calls.
        seed : int, optional
            An optional seed can be provided to GPT to get deterministic results.
        """
        if seed is not None:
            if retries != 0:
                raise ValueError(
                    "Seed is provided, but retries is not 0. Retries should be 0 when seed is provided."
                )

        self.api_key = api_key
        self.model_name = model_name
        self.retries = retries
        self.seed = seed

        # logs
        self.total_prompt_tokens = 0
        self.total_completion_tokens = 0

    def connect(
        self,
    ):
        """
        Setup the connection to the API.
        """
        if OpenAI is None:
            raise ImportError(
                "OpenAI must be installed to use the OpenAIWrapper."
            )

        if self.api_key is None:
            self.client = OpenAI()
        else:
            self.client = OpenAI(api_key=self.api_key)

    def _process_messages(
        self,
        messages: list[dict[str, str]],
    ) -> list[dict[str, str]]:
        """
        Format messages for the API.

        Parameters
        ----------
        messages: list[dict[str, str]]
            The messages formatted according to the OpenAI standard. Each message in messages is a dictionary with "role" and "content" keys.

        Returns
        -------
        list[dict[str, str]]
            The messages are left in the OpenAI standard.
        """
        # Validate that the input is a list of dictionaries with "role" and "content" keys.
        _validate_messages(messages=messages)  # type: ignore

        return messages

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
        processed_messages = self._process_messages(messages)
        openai_response = self.client.chat.completions.create(
            model=self.model_name,
            messages=processed_messages,  # type: ignore - mistralai issue
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


class MistralWrapper(_ClientWrapper):
    """
    Wrapper for calls to Mistral's API.

    Attributes
    ----------
    api_key : str, optional
        The Mistral API key to use. If not specified, then the MISTRAL_API_KEY environment variable will be used.
    model_name : str
        The model to use. Defaults to "mistral-small-latest".
    retries : int
        The number of times to retry the API call if it fails. Defaults to 0, indicating that the call will not be retried. For example, if self.retries is set to 3, this means that the call will be retried up to 3 times, for a maximum of 4 calls.
    """

    def __init__(
        self,
        model_name: str,
        api_key: str | None = None,
        retries: int = 0,
    ):
        """
        Creates an instance of the Mistral interface.

        Parameters
        ----------
        model_name : str
            The model to use (e.g. "mistral-small-latest").
        api_key : str, optional
            The Mistral API key to use. If not specified, then the MISTRAL_API_KEY environment variable will be used.
        retries : int, default=0
            The number of times to retry the API call if it fails. Defaults to 0, indicating that the call will not be retried. For example, if self.retries is set to 3, this means that the call will be retried up to 3 times, for a maximum of 4 calls.
        """
        self.api_key = api_key
        self.model_name = model_name
        self.retries = retries

    def connect(
        self,
    ):
        """
        Setup the connection to the API.
        """
        if Mistral is None:
            raise ImportError(
                "Mistral must be installed to use the MistralWrapper."
            )
        if self.api_key is None:
            self.client = Mistral()
        else:
            self.client = Mistral(api_key=self.api_key)

    def _process_messages(
        self,
        messages: list[dict[str, str]],
    ) -> Any:
        """
        Format messages for Mistral's API.

        Parameters
        ----------
        messages: list[dict[str, str]]
            The messages formatted according to the OpenAI standard. Each message in messages is a dictionary with "role" and "content" keys.

        Returns
        -------
        Any
            The messages formatted for Mistral's API. With mistralai>=1.0.0, the messages can be left in the OpenAI standard.
        """
        # Validate that the input is a list of dictionaries with "role" and "content" keys.
        _validate_messages(messages=messages)  # type: ignore

        return messages

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
        processed_messages = self._process_messages(messages)
        mistral_response = self.client.chat.complete(
            model=self.model_name,
            messages=processed_messages,
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


class MockWrapper(_ClientWrapper):
    """
    A mocked LLM client for testing purposes.
    """

    def __init__(
        self,
        **kwargs,
    ):
        """
        Neither the api_key nor the model_name are required for the mock client.
        """
        super().__init__(**kwargs)

    def connect(
        self,
    ):
        """
        No connection is required for the mock client.
        """
        pass

    def _process_messages(
        self,
        messages: list[dict[str, str]],
    ) -> list[dict[str, str]]:
        """
        Format messages for the API.

        Parameters
        ----------
        messages: list[dict[str, str]]
            The messages formatted according to the OpenAI standard. Each message in messages is a dictionary with "role" and "content" keys.

        Returns
        -------
        list[dict[str, str]]
            The messages are left in the OpenAI format.
        """
        # Validate that the input is a list of dictionaries with "role" and "content" keys.
        _validate_messages(messages=messages)  # type: ignore

        return messages

    def __call__(
        self,
        messages: list[dict[str, str]],
    ) -> str:
        """
        Call to the API. Returns "" by default, or metric specific mock responses.

        Parameters
        ----------
        messages: list[dict[str, str]]
            The messages formatted according to the OpenAI standard. Each message in messages is a dictionary with "role" and "content" keys.

        Returns
        -------
        str
            The response from the API.
        """
        response = None

        processed_messages = self._process_messages(messages)
        if len(processed_messages) >= 2:
            # Generate claims
            if (
                "generate a comprehensive list of FACTUAL CLAIMS"
                in processed_messages[1]["content"]
            ):
                response = """```json
    {
        "claims": [
            "The capital of the UK is London.",
            "The capital of South Korea is Seoul.",
            "The capital of Argentina is Canada."
        ]
    }```"""

            # Generate opinions
            elif (
                "generate a list of OPINIONS"
                in processed_messages[1]["content"]
            ):
                response = """```json
    {
        "opinions": [
            "I like the color green.",
            "People from Canada are nicer than people from other countries."
        ]
    }```"""

            # Generate statements
            elif (
                "generate a list of STATEMENTS"
                in processed_messages[1]["content"]
            ):
                response = """```json
    {
        "statements": [
            "The capital of the UK is London.",
            "London is the largest city in the UK by population and GDP."
        ]
    }```"""

            # Answer correctness verdicts
            elif (
                "Return in JSON format with three keys: 'TP', 'FP', and 'FN'"
                in processed_messages[1]["content"]
            ):
                response = """```json
{
    "TP": [
        "London is the largest city in the UK by GDP"
    ],
    "FP": [
        "London is the largest city in the UK by population"
    ],
    "FN": [
        "In 2021, financial services made up more than 20% of London's output"
    ]
}```"""

            # Answer relevance verdicts
            elif (
                "generate a list of verdicts that indicate whether each statement is relevant to address the query"
                in processed_messages[1]["content"]
            ):
                response = """```json
    {
        "verdicts": [
            {"verdict": "yes"},
            {"verdict": "no"}
        ]
    }```"""

            # Bias verdicts
            elif (
                "generate a list of verdicts to indicate whether EACH opinion is biased"
                in processed_messages[1]["content"]
            ):
                response = """```json
    {
        "verdicts": [
            {"verdict": "no"},
            {"verdict": "yes"}
        ]
    }```"""

            # Summary coherence score
            elif (
                "Your task is to rate the summary based on its coherence"
                in processed_messages[1]["content"]
            ):
                response = "4"

            # Context precision verdicts
            elif (
                "generate a list of verdicts to determine whether each context in the context list is useful for producing the ground truth answer to the query"
                in processed_messages[1]["content"]
            ):
                response = """```json
    {
        "verdicts": [
            {"verdict": "yes"},
            {"verdict": "no"},
            {"verdict": "no"},
            {"verdict": "yes"}
        ]
    }```"""

            # Context recall verdicts
            elif (
                "analyze each ground truth statement and determine if the statement can be attributed to the given context"
                in processed_messages[1]["content"]
            ):
                response = """```json
    {
        "verdicts": [
            {"verdict": "yes"},
            {"verdict": "yes"}
        ]
    }```"""

            # Context relevance verdicts
            elif (
                "generate a list of verdicts to indicate whether each context is relevant to the provided query"
                in processed_messages[1]["content"]
            ):
                response = """```json
    {
        "verdicts": [
            {"verdict": "yes"},
            {"verdict": "yes"},
            {"verdict": "no"},
            {"verdict": "yes"}
        ]
    }```"""

            # Faithfulness verdicts
            elif (
                "generate a list of verdicts to indicate whether EACH claim is implied by the context list"
                in processed_messages[1]["content"]
            ):
                response = """```json
    {
        "verdicts": [
            {"verdict": "yes"},
            {"verdict": "no"},
            {"verdict": "no"}
        ]
    }```"""

            # Hallucination agreement verdicts
            elif (
                "generate a list of verdicts to indicate whether the given text contradicts EACH context"
                in processed_messages[1]["content"]
            ):
                response = """```json
    {
        "verdicts": [
            {"verdict": "no"},
            {"verdict": "no"},
            {"verdict": "yes"},
            {"verdict": "no"}
        ]
    }```"""

            # Toxicity verdicts
            elif (
                "generate a list of verdicts to indicate whether EACH opinion is toxic"
                in processed_messages[1]["content"]
            ):
                response = """```json
    {
        "verdicts": [
            {"verdict": "no"},
            {"verdict": "no"}
        ]
    }```"""

        if response is None:
            response = ""
        return response
