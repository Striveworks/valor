from typing import Any

import openai

from valor_api import schemas
from valor_api.backend.models import Label

COHERENCE_INSTRUCTION = """You are a helpful assistant. You will grade the user's text. Your task is to rate the text based on its coherence. Please make sure you read and understand these instructions carefully. Please keep this document open while reviewing, and refer to it as needed.
Evaluation Criteria:
Coherence (1-5) - the collective quality of all sentences. We align this dimension with the DUC quality question of structure and coherence whereby ”the summary should be well-structured and well-organized. The summary should not just be a heap of related information, but should build from sentence to sentence to a coherent body of information about a topic.”
Evaluation Steps:
1. Read the text carefully and identify the main topic and key points.
2. Assign a score for coherence on a scale of 1 to 5, where 1 is the lowest and 5 is the highest based on the Evaluation Criteria. Respond with just the number 1 to 5."""


class OpenAIClient:
    """
    Wrapper for calls to OpenAI

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
    api_key: str | None = None  # If api_key is not specified, then OPENAI_API_KEY environment variable will be used.
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

    def __call__(
        self,
        messages: list[dict],
    ) -> Any:
        """
        TODO possibly change this to a call with the API. This would remove the openai python dependence.
        """
        try:
            openai_response = self.client.chat.completions.create(
                model=self.model_name, messages=messages, seed=self.seed
            )
        # TODO should we both catching this if we aren't going to do anything?
        except openai.BadRequestError as e:
            raise ValueError(f"OpenAI request failed with error: {e}")

        # token_usage = openai_response.usage  # TODO Could report token usage to user. Could use token length to determine if input is too larger, although this would require us to know the model's context window size.
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

    def coherence(
        self,
        text: str,
        label: Label,
    ) -> schemas.CoherenceMetric | None:
        """
        Computes coherence for a single piece of text.

        Parameters
        ----------
        text : TODO
        label_key : TODO

        Returns
        -------
        schemas.CoherenceMetric | None
        """
        messages = [
            {"role": "system", "content": COHERENCE_INSTRUCTION},
            {"role": "user", "content": text},
        ]

        response = self(messages)
        # TODO Should we take just the first word of the response, in case GPT outputs a number followed by an explanation?

        try:
            response = int(response)
            assert response in [1, 2, 3, 4, 5]
        except Exception:
            raise ValueError(
                f"OpenAI response was not a valid coherence score: {response}"
            )

        return schemas.CoherenceMetric(
            label=label,
            value=response,
        )
