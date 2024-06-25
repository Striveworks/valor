from typing import Any

import openai
from mistralai.client import MistralClient
from mistralai.models.chat_completion import ChatMessage


def _get_instruction(metric: str):
    """Fetch the instruction associated with a particular metric type."""
    instruction_dict = {
        "coherence": """You are a helpful assistant. You will grade the user's text. Your task is to rate the text based on its coherence. Please make sure you read and understand these instructions carefully. Please keep this document open while reviewing, and refer to it as needed.
Evaluation Criteria:
Coherence (1-5) - the collective quality of all sentences. We align this dimension with the DUC quality question of structure and coherence whereby ”the summary should be well-structured and well-organized. The summary should not just be a heap of related information, but should build from sentence to sentence to a coherent body of information about a topic.”
Evaluation Steps:
1. Read the text carefully and identify the main topic and key points.
2. Assign a score for coherence on a scale of 1 to 5, where 1 is the lowest and 5 is the highest based on the Evaluation Criteria. Respond with just the number 1 to 5."""
    }

    return instruction_dict[metric]


class LLMClient:
    """
    Parent class for all LLM clients.

    Attributes
    ----------
    api_key : str, optional
        The API key to use.
    model_name : str
        The model to use.
    """

    api_key: str | None = None
    model_name: str

    def __init__(
        self,
        api_key: str | None = None,
        model_name: str | None = None,
    ):
        """
        Set the API key and model name (if provided).
        """
        self.api_key = api_key
        if model_name is not None:
            self.model_name = model_name

    def connect(
        self,
    ):
        """
        Setup the connection to the API. Not implemented for parent class.
        """
        raise NotImplementedError

    def process_messages(
        self,
        messages: list[dict[str, str]],
    ) -> Any:
        """
        Process messages from the API. By default, we don't do any processing and just return the inputted messages.
        """
        return messages

    def __call__(
        self,
        messages: list[dict[str, str]],
    ) -> Any:
        """
        Call to the API. Not implemented for parent class.
        """
        raise NotImplementedError

    def coherence(
        self,
        text: str,
    ) -> int:
        """
        Computes coherence, the collective quality of all sentences, for a single piece of text.

        Parameters
        ----------
        text: str
            The text to be evaluated.

        Returns
        -------
        int
            The coherence score will be evaluated as an integer, with 1 indicating the lowest coherence and 5 the highest coherence.
        """
        messages = [
            {"role": "system", "content": _get_instruction("coherence")},
            {"role": "user", "content": text},
        ]

        response = self(messages)

        try:
            # Valid responses: "5", "\n5", "5\n", "5.", " 5", "5 {explanation}", etc.
            ret = int(response.strip()[0])
            assert ret in [1, 2, 3, 4, 5]
        except Exception:
            raise ValueError(
                f"LLM response was not a valid coherence score: {response}"
            )

        return ret


class WrappedOpenAIClient(LLMClient):
    """
    Wrapper for calls to OpenAI's API.

    Attributes
    ----------
    api_key : str, optional
        The OpenAI API key to use. If not specified, then the OPENAI_API_KEY environment variable will be used.
    seed : int, optional
        An optional seed can be provided to GPT to get deterministic results.
    model_name : str
        The model to use. Defaults to "gpt-3.5-turbo".
    """

    api_key: str | None = None
    seed: int | None = None
    model_name: str = "gpt-3.5-turbo"  # gpt-3.5-turbo gpt-4-turbo gpt-4o

    def __init__(
        self,
        api_key: str | None = None,
        seed: int | None = None,
        model_name: str | None = None,
    ):
        """
        Set the API key, seed and model name (if provided).
        """
        self.api_key = api_key
        if seed is not None:
            self.seed = seed
        if model_name is not None:
            self.model_name = model_name

    def connect(
        self,
    ):
        """
        Setup the connection to the API.
        """
        if self.api_key is None:
            self.client = openai.OpenAI()
        else:
            self.client = openai.OpenAI(api_key=self.api_key)

    def __call__(
        self,
        messages: list[dict[str, str]],
    ) -> Any:
        """
        Call to the API.
        """
        processed_messages = self.process_messages(messages)
        try:
            openai_response = self.client.chat.completions.create(
                model=self.model_name,
                messages=processed_messages,
                seed=self.seed,
            )
        except Exception as e:
            raise ValueError(f"OpenAI API request failed with error: {e}")

        finish_reason = openai_response.choices[0].finish_reason
        response = openai_response.choices[0].message.content

        if finish_reason == "length":
            raise ValueError(
                "OpenAI response reached max token limit. Resulting evaluation is likely invalid or of low quality."
            )
        elif finish_reason == "content_filter":
            raise ValueError(
                "OpenAI response was flagged by content filter. Resulting evaluation is likely invalid or of low quality."
            )

        return response


class WrappedMistralAIClient(LLMClient):
    """
    Wrapper for calls to Mistral's API.

    Attributes
    ----------
    api_key : str, optional
        The Mistral API key to use. If not specified, then the MISTRAL_API_KEY environment variable will be used.
    model_name : str
        The model to use. Defaults to "mistral-small-latest".
    """

    # url: str
    api_key: str | None = None
    model_name: str = (
        "mistral-small-latest"  # mistral-small-latest mistral-large-latest
    )

    def __init__(
        self,
        api_key: str | None = None,
        model_name: str | None = None,
    ):
        """
        Set the API key and model name (if provided).
        """
        self.api_key = api_key
        if model_name is not None:
            self.model_name = model_name

    def connect(
        self,
    ):
        """
        Setup the connection to the API.
        """
        if self.api_key is None:
            self.client = MistralClient()
        else:
            self.client = MistralClient(api_key=self.api_key)

    def process_messages(
        self,
        messages: list[dict[str, str]],
    ) -> Any:
        """
        All messages should be formatted according to the standard set by OpenAI, and should be modified
        as needed for other models. This function takes in messages in the OpenAI standard format and converts
        them to the format required by the model.
        """
        ret = []
        for i in range(len(messages)):
            ret.append(
                ChatMessage(
                    role=messages[i]["role"],
                    content=messages[i]["content"],
                )
            )
        return ret

    def __call__(
        self,
        messages: list[dict[str, str]],
    ) -> Any:
        """
        Call to the API.
        """
        processed_messages = self.process_messages(messages)
        try:
            mistral_response = self.client.chat(
                model=self.model_name,
                messages=processed_messages,
            )
        except Exception as e:
            raise ValueError(f"Mistral API request failed with error: {e}")

        response = mistral_response.choices[0].message.content
        finish_reason = mistral_response.choices[0].finish_reason

        if finish_reason == "length":
            raise ValueError(
                "MistralAI response reached max token limit. Resulting evaluation is likely invalid or of low quality."
            )

        return response
