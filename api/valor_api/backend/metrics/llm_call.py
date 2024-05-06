from typing import Any

from openai import OpenAI

from valor_api import schemas

COHERENCE_INSTRUCTION = """You are a helpful assistant. You will grade the user's text. Your task is to rate the text based on its coherence. Please make sure you read and understand these instructions carefully. Please keep this document open while reviewing, and refer to it as needed.
Evaluation Criteria:
Coherence (1-5) - the collective quality of all sentences. We align this dimension with the DUC quality question of structure and coherence whereby ”the summary should be well-structured and well-organized. The summary should not just be a heap of related information, but should build from sentence to sentence to a coherent body of information about a topic.”
Evaluation Steps:
1. Read the text carefully and identify the main topic and key points.
2. Assign a score for coherence on a scale of 1 to 5, where 1 is the lowest and 5 is the highest based on the Evaluation Criteria. Respond with just the number 1 to 5."""


class OpenAIClient:
    """
    Wrapper for calls to OpenAI
    """

    # url: str
    api_key: str | None = None
    model_name: str = "gpt-3.5-turbo"  # gpt-3.5-turbo gpt-4-turbo

    def __init__(
        self,
        api_key: str | None = None,
    ):
        """
        TODO should we use an __attrs_post_init__ instead?
        """
        self.api_key = api_key
        self.client = OpenAI(api_key=self.api_key)

    def __call__(
        self,
        messages: list[dict],
    ) -> Any:
        """
        TODO possibly change this to a call with the API. This would remove the openai python dependence.
        """
        response = self.client.chat.completions.create(
            model=self.model_name, messages=messages
        )
        response = response.choices[0].message.content
        return response

    def coherence(
        self,
        text: str,
        label_key: str,
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

        try:
            response = int(response)
        except ValueError:
            return None

        if response not in [1, 2, 3, 4, 5]:
            return None

        return schemas.CoherenceMetric(
            label_key=label_key,
            value=response,
        )
