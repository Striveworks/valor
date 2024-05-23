from typing import Any

import openai
from mistralai.client import MistralClient
from mistralai.models.chat_completion import ChatMessage

COHERENCE_INSTRUCTION = """You are a helpful assistant. You will grade the user's text. Your task is to rate the text based on its coherence. Please make sure you read and understand these instructions carefully. Please keep this document open while reviewing, and refer to it as needed.
Evaluation Criteria:
Coherence (1-5) - the collective quality of all sentences. We align this dimension with the DUC quality question of structure and coherence whereby ”the summary should be well-structured and well-organized. The summary should not just be a heap of related information, but should build from sentence to sentence to a coherent body of information about a topic.”
Evaluation Steps:
1. Read the text carefully and identify the main topic and key points.
2. Assign a score for coherence on a scale of 1 to 5, where 1 is the lowest and 5 is the highest based on the Evaluation Criteria. Respond with just the number 1 to 5."""


class LLMClient:
    """
    Wrapper for calls to an LLM API.

    Parameters
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
    ):
        """
        TODO should we use an __attrs_post_init__ instead?
        """
        self.api_key = api_key

    def connect(
        self,
    ):
        """
        TODO This is separated for now because I want to mock connecting to the LLM API.
        """
        raise NotImplementedError

    def process_messages(
        self,
        messages: list[dict[str, str]],
    ) -> Any:
        """
        All messages should be formatted according to the standard set by OpenAI, and should be modified
        as needed for other models. This function takes in messages in the OpenAI standard format and converts
        them to the format required by the model.
        """
        raise NotImplementedError

    def __call__(
        self,
        messages: list[dict[str, str]],
    ) -> Any:
        """
        Call to the API.
        """
        raise NotImplementedError

    def coherence(
        self,
        text: str,
    ) -> int:
        """
        Computes coherence for a single piece of text.

        Parameters
        ----------
        text : TODO

        Returns
        -------
        int
        """
        messages = [
            {"role": "system", "content": COHERENCE_INSTRUCTION},
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


class OpenAIValorClient(LLMClient):
    """
    Wrapper for calls to OpenAI's API.

    Parameters
    ----------
    api_key : str, optional
        The OpenAI API key to use. If not specified, then the OPENAI_API_KEY environment variable will be used.
    seed : int, optional
        An optional seed can be provided to GPT to get deterministic results.
    model_name : str
        The model to use. Defaults to "gpt-3.5-turbo".
    """

    # url: str
    api_key: str | None = None
    seed: int | None = None
    model_name: str = "gpt-3.5-turbo"  # gpt-3.5-turbo gpt-4-turbo

    def __init__(
        self,
        api_key: str | None = None,
        seed: int | None = None,
    ):
        """
        TODO should we use an __attrs_post_init__ instead?
        """
        self.api_key = api_key
        self.seed = seed

    def connect(
        self,
    ):
        """
        TODO This is separated for now because I want to mock connecting to the OpenAI API.
        """
        if self.api_key is None:
            self.client = openai.OpenAI()
        else:
            self.client = openai.OpenAI(api_key=self.api_key)

    def process_messages(
        self,
        messages: list[dict[str, str]],
    ) -> Any:
        """
        All messages should be formatted according to the standard set by OpenAI, and should be modified
        as needed for other models. This function takes in messages in the OpenAI standard format and converts
        them to the format required by the model.
        """
        return messages

    def __call__(
        self,
        messages: list[dict[str, str]],
    ) -> Any:
        """
        Call to the API.

        TODO possibly change this to a call with the API. This would remove the openai python dependence.
        """
        processed_messages = self.process_messages(messages)
        try:
            openai_response = self.client.chat.completions.create(
                model=self.model_name,
                messages=processed_messages,
                seed=self.seed,
            )
        # TODO should we both catching this if we aren't going to do anything?
        except openai.BadRequestError as e:
            raise ValueError(f"OpenAI request failed with error: {e}")

        # token_usage = openai_response.usage  # TODO Could report token usage to user. Could use token length to determine if input is too larger, although this would require us to know the model's context window size.
        finish_reason = openai_response.choices[0].finish_reason
        response = openai_response.choices[0].message.content

        # TODO Only keep these if we can test them.
        if finish_reason == "length":
            raise ValueError(
                "OpenAI response reached max token limit. Resulting evaluation is likely invalid or of low quality."
            )
        elif finish_reason == "content_filter":
            raise ValueError(
                "OpenAI response was flagged by content filter. Resulting evaluation is likely invalid or of low quality."
            )

        return response


class MistralValorClient(LLMClient):
    """
    Wrapper for calls to Mistral's API.

    Parameters
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
    ):
        """
        TODO should we use an __attrs_post_init__ instead?
        """
        self.api_key = api_key

    def connect(
        self,
    ):
        """
        TODO This is separated for now because I want to mock connecting to the Mistral API.
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
                    role=messages[i]["role"], content=messages[i]["content"]
                )
            )
        return ret

    def __call__(
        self,
        messages: list[dict[str, str]],
    ) -> Any:
        """
        Call to the API.

        TODO possibly change this to a call with the API. This would remove the openai python dependence.
        """
        processed_messages = self.process_messages(messages)
        mistral_response = self.client.chat(
            model=self.model_name,
            messages=processed_messages,
        )
        # TODO Are there any errors we should catch in a try except block?

        # token_usage = mistral_response.usage  # TODO Could report token usage to user. Could use token length to determine if input is too larger, although this would require us to know the model's context window size.
        # finish_reason = mistral_response.choices[0].finish_reason
        response = mistral_response.choices[0].message.content

        # TODO Possibly add errors depending on the finish reason?

        return response
