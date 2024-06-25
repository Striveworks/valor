from typing import Any

import openai
from mistralai.client import MistralClient
from mistralai.models.chat_completion import ChatMessage

DEFAULT_SYSTEM_PROMPT = "You are a helpful assistant."

def get_coherence_instruction(text: str):
    """
    This instruction was adapted from appendix A of Deepeval's paper G-EVAL: NLG Evaluation using GPT-4 with Better Human Alignment (https://arxiv.org/pdf/2303.16634).
    The main adaptation is a generalization of the metric to more task types. The example prompt in Deepeval was specific to summarization, but the below prompt could apply to any text generation task.
    Crucially, no context is used. Instead, the coherence of the source text is evaluated entirely based on the source text. This generalizes the prompt and also prevents the evaluation from being influenced by the quality of sentences in the context.
    Parameters
    ----------
    text: str
        The text to be evaluated.
    Returns
    -------
    str
        The instruction for the llm.
    """
    return f"""
    You are a helpful assistant. You will grade the source text. Your task is to rate the source text based on its coherence. Please make sure you read and understand these instructions carefully. Please keep this document open while reviewing, and refer to it as needed.

    Evaluation Criteria:
    Coherence (1-5) - the collective quality of all sentences. We align this dimension with the DUC quality question of structure and coherence whereby ”the summary should be well-structured and well-organized. The summary should not just be a heap of related information, but should build from sentence to sentence to a coherent body of information about a topic.”
    
    Evaluation Steps:
    1. Read the source text carefully and identify the main topic and key points.
    2. Check if the source text presents the information in a clear and logical order. Examine the collective quality of the sentences.
    3. Assign a score for coherence on a scale of 1 to 5, where 1 is the lowest and 5 is the highest based on the Evaluation Criteria. Respond with just the number 1 to 5.
    
    Source Text:
    {text}
    """
  

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
        Format messages for the API.

        Parameters
        ----------
        messages: list[dict[str, str]]
            The messages formatted according to the OpenAI standard. Each message in messages is a dictionary with "role" and "content" keys.

        Returns
        -------
        Any
            The messages formatted for the API. By default, the messages are left in the OpenAI format.
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
            {"role": "system", "content": DEFAULT_SYSTEM_PROMPT},
            {"role": "user", "content": get_coherence_instruction(text)},
        ]

        response = self(messages)

        try:
            # Valid responses: "5", "\n5", "5\n", "5.", " 5", "5 {explanation}", etc.
            ret = int(response.strip()[0])
            assert ret in {1, 2, 3, 4, 5}
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
    model_name: str = "gpt-3.5-turbo"

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

        openai_response = self.client.chat.completions.create(
            model=self.model_name,
            messages=processed_messages,
            seed=self.seed,
        )

        finish_reason = openai_response.choices[
            0
        ].finish_reason  # Enum: "stop" "length" "content_filter" "tool_calls" "function_call"

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

    api_key: str | None = None
    model_name: str = "mistral-small-latest"

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
        Format messages for Mistral's API.

        Parameters
        ----------
        messages: list[dict[str, str]]
            The messages formatted according to the OpenAI standard. Each message in messages is a dictionary with "role" and "content" keys.

        Returns
        -------
        Any
            The messages formatted for Mistral's API. Each message is converted to a mistralai.models.chat_completion.ChatMessage object.
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
        mistral_response = self.client.chat(
            model=self.model_name,
            messages=processed_messages,
        )

        finish_reason = mistral_response.choices[
            0
        ].finish_reason  # Enum: "stop" "length" "model_length" "error" "tool_calls"
        response = mistral_response.choices[0].message.content
        finish_reason = mistral_response.choices[0].finish_reason

        if finish_reason == "length":
            raise ValueError(
                "MistralAI response reached max token limit. Resulting evaluation is likely invalid or of low quality."
            )

        if finish_reason == "length":
            raise ValueError(
                "Mistral response reached max token limit. Resulting evaluation is likely invalid or of low quality."
            )

        return response
