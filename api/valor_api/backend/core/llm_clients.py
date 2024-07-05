from typing import Any

import openai
from mistralai.client import MistralClient
from mistralai.models.chat_completion import ChatMessage
from pydantic import BaseModel

from valor_api.backend.metrics.metric_utils import trim_and_load_json
from valor_api.exceptions import InvalidLLMResponseError

DEFAULT_SYSTEM_PROMPT = "You are a helpful assistant."


class Message(BaseModel):
    role: str
    content: str


class Messages(BaseModel):
    messages: list[Message]


def _generate_statements_instruction(text: str) -> str:
    """
    Instruction template was copied from DeepEval's codebase https://github.com/confident-ai/deepeval/blob/main/deepeval/metrics/answer_relevancy/template.py.

    Parameters
    ----------
    text: str
        The text to extract statements from.

    Returns
    -------
    str
        The instruction for the llm.
    """
    return f"""Given the text, breakdown and generate a list of statements presented. Ambiguous statements and single words can also be considered as statements.

Example:
Example text: Shoes. The shoes can be refunded at no extra cost. Thanks for asking the question!

{{
    "statements": ["Shoes.", "Shoes can be refunded at no extra cost", "Thanks for asking the question!"]
}}
===== END OF EXAMPLE ======

**
IMPORTANT: Please make sure to only return in JSON format, with the "statements" key mapping to a list of strings. No words or explanation is needed.
**

Text:
{text}

JSON:
"""


def _generate_anwswer_relevance_verdicts_instruction(
    query: str, statements: str | list[str]
) -> str:
    """
    Instruction template was copied from DeepEval's codebase https://github.com/confident-ai/deepeval/blob/main/deepeval/metrics/answer_relevancy/template.py.

    Parameters
    ----------
    query: str
        The query to evaluate the statements against.
    statements: str
        The statements to evaluate the validity of.

    Returns
    -------
    str
        The instruction for the llm.
    """
    return f"""For the provided list of statements, determine whether each statement is relevant to address the input.
Please generate a list of JSON with two keys: `verdict` and `reason`.
The 'verdict' key should STRICTLY be either a 'yes', 'idk' or 'no'. Answer 'yes' if the statement is relevant to addressing the original input, 'no' if the statement is irrelevant, and 'idk' if it is ambiguous (eg., not directly relevant but could be used as a supporting point to address the input).
The 'reason' is the reason for the verdict.
Provide a 'reason' ONLY if the answer is 'no'.
The provided statements are statements made in the actual output.

**
IMPORTANT: Please make sure to only return in JSON format, with the 'verdicts' key mapping to a list of JSON objects.
Example input: What should I do if there is an earthquake?
Example statements: ["Shoes.", "Thanks for asking the question!", "Is there anything else I can help you with?", "Duck and hide"]
Example JSON:
{{
    "verdicts": [
        {{
            "verdict": "no",
            "reason": "The 'Shoes.' statement made in the actual output is completely irrelevant to the input, which asks about what to do in the event of an earthquake."
        }},
        {{
            "verdict": "idk"
        }},
        {{
            "verdict": "idk"
        }},
        {{
            "verdict": "yes"
        }}
    ]
}}

Since you are going to generate a verdict for each statement, the number of 'verdicts' SHOULD BE STRICTLY EQUAL to the number of `statements`.
**

Input:
{query}

Statements:
{statements}

JSON:
"""


def _get_coherence_instruction(text: str) -> str:
    """
    This instruction was adapted from appendix A of DeepEval's paper G-EVAL: NLG Evaluation using GPT-4 with Better Human Alignment (https://arxiv.org/pdf/2303.16634).
    The main adaptation is a generalization of the metric to more task types. The example prompt in DeepEval was specific to summarization, but the below prompt could apply to any text generation task.
    Crucially, unlike DeepEval, no context is used. Instead, the coherence of the text is evaluated entirely based on the text. This generalizes the prompt and also prevents the evaluation from being influenced by the quality of sentences in the context.
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
    You are a helpful assistant. You will grade the text. Your task is to rate the text based on its coherence. Please make sure you read and understand these instructions carefully. Please keep this document open while reviewing, and refer to it as needed.

    Evaluation Criteria:
    Coherence (1-5) - the collective quality of all sentences. We align this dimension with the DUC quality question of structure and coherence whereby ”the summary should be well-structured and well-organized. The summary should not just be a heap of related information, but should build from sentence to sentence to a coherent body of information about a topic.”

    Evaluation Steps:
    1. Read the text carefully and identify the main topic and key points.
    2. Check if the text presents the information in a clear and logical order. Examine the collective quality of all sentences.
    3. Assign a score for coherence on a scale of 1 to 5, where 1 is the lowest and 5 is the highest based on the Evaluation Criteria. Respond with just the number 1 to 5.

    Text to Evaluate:
    {text}

    Coherence Score (1-5):
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
        # Validate that the input is a list of dictionaries with "role" and "content" keys.
        _ = Messages(messages=messages)  # type: ignore

        return messages

    def __call__(
        self,
        messages: list[dict[str, str]],
    ) -> Any:
        """
        Call to the API. Not implemented for parent class.

        Parameters
        ----------
        messages: list[dict[str, str]]
            The messages formatted according to the OpenAI standard. Each message in messages is a dictionary with "role" and "content" keys.

        Returns
        -------
        Any
            The response from the API.
        """
        raise NotImplementedError

    def _generate_statements(
        self,
        text: str,
    ) -> list[str]:
        """
        Generate a list of statements from a piece of text, using a call to the LLM API.

        Parameters
        ----------
        text: str
            The text to extract statements from.

        Returns
        -------
        list[str]
            The list of statements extracted from the text.
        """
        messages = [
            {"role": "system", "content": DEFAULT_SYSTEM_PROMPT},
            {
                "role": "user",
                "content": _generate_statements_instruction(text),
            },
        ]

        response = self(messages)
        response = trim_and_load_json(response)
        if type(response) != dict or "statements" not in response:
            raise InvalidLLMResponseError(
                f"LLM response was not a dictionary or 'statements' was not in response: {response}"
            )
        statements = response["statements"]
        if type(statements) != list or not all(
            type(statement) == str for statement in statements
        ):
            raise InvalidLLMResponseError(
                f"LLM response was not a valid list of statements (list[str]): {response}"
            )
        return statements

    def _generate_anwswer_relevance_verdicts(
        self,
        query: str,
        statements: list[str],
    ) -> list[dict[str, str]]:
        """
        Generates a list of answer relevance verdicts for a list of statements, using a call to the LLM API.

        Parameters
        ----------
        query: str
            The query to evaluate the statements against.
        statements: list[str]
            The statements to evaluate the validity of.

        Returns
        -------
        list[dict[str,str]]
            The list of verdicts for each statement. Each verdict is a dictionary with the "verdict" and optionally a "reason".
        """
        messages = [
            {"role": "system", "content": DEFAULT_SYSTEM_PROMPT},
            {
                "role": "user",
                "content": _generate_anwswer_relevance_verdicts_instruction(
                    query, statements
                ),
            },
        ]

        response = self(messages)
        response = trim_and_load_json(response)
        if type(response) != dict or "verdicts" not in response:
            raise InvalidLLMResponseError(
                f"LLM response was not a valid list of verdicts: {response}"
            )

        verdicts = response["verdicts"]
        if (
            type(verdicts) != list
            or len(verdicts) != len(statements)
            or not all(
                verdict["verdict"] in ["yes", "no", "idk"]
                for verdict in verdicts
            )
        ):
            raise InvalidLLMResponseError(
                f"LLM response was not a valid list of verdicts: {response}"
            )

        return verdicts

    def answer_relevance(
        self,
        query: str,
        text: str,
    ) -> float:
        """
        Compute answer relevance, the proportion of statements that are relevant to the query, for a single piece of text.

        Parameters
        ----------
        query: str
            The query to evaluate the statements against.
        text: str
            The text to extract statements from.

        Returns
        -------
        float
            The answer relevance score will be evaluated as a float between 0 and 1, with 1 indicating that all statements are relevant to the query.
        """
        statements = self._generate_statements(text)
        verdicts = self._generate_anwswer_relevance_verdicts(query, statements)
        return sum(
            1 for verdict in verdicts if verdict["verdict"] == "yes"
        ) / len(verdicts)

    def coherence(
        self,
        text: str,
    ) -> int:
        """
        Compute coherence, the collective quality of all sentences, for a single piece of text.

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
            {"role": "user", "content": _get_coherence_instruction(text)},
        ]

        response = self(messages)

        try:
            # Valid responses: "5", "\n5", "5\n", "5.", " 5", "5 {explanation}", etc.
            ret = int(response.strip()[0])
        except Exception:
            raise InvalidLLMResponseError(
                f"LLM response was not a valid coherence score: {response}"
            )

        if ret not in {1, 2, 3, 4, 5}:
            raise InvalidLLMResponseError(
                f"Coherence score was not an integer between 1 and 5: {ret}"
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

        Parameters
        ----------
        messages: list[dict[str, str]]
            The messages formatted according to the OpenAI standard. Each message in messages is a dictionary with "role" and "content" keys.

        Returns
        -------
        Any
            The response from the API.
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
        # Validate that the input is a list of dictionaries with "role" and "content" keys.
        _ = Messages(messages=messages)  # type: ignore

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

        Parameters
        ----------
        messages: list[dict[str, str]]
            The messages formatted according to the OpenAI standard. Each message in messages is a dictionary with "role" and "content" keys.

        Returns
        -------
        Any
            The response from the API.
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
                "Mistral response reached max token limit. Resulting evaluation is likely invalid or of low quality."
            )

        return response


class MockLLMClient(LLMClient):
    """
    A mocked LLM client for testing purposes.

    Attributes
    ----------
    api_key : str, optional
        The API key to use.
    model_name : str
        The model to use. A model_name is not required for testing purposes.
    """

    def __init__(
        self,
        **kwargs,
    ):
        """
        Neither the api_key nor model_name are required for the mock client.
        """
        pass

    def connect(
        self,
    ):
        """
        No connection is required for the mock client.
        """
        pass

    def __call__(
        self,
        messages: list[dict[str, str]],
    ) -> Any:
        """
        Call to the API. Returns "" by default, or metric specific mock responses.

        Parameters
        ----------
        messages: list[dict[str, str]]
            The messages formatted according to the OpenAI standard. Each message in messages is a dictionary with "role" and "content" keys.

        Returns
        -------
        Any
            The response from the API.
        """
        processed_messages = self.process_messages(messages)
        if len(processed_messages) >= 2:
            if (
                "generate a list of statements"
                in processed_messages[1]["content"]
            ):
                return """```json
    {
        "statements": [
            "The capital of the UK is London.",
            "London is the largest city in the UK by population and GDP."
        ]
    }```"""

            elif (
                "determine whether each statement is relevant to address the input"
                in processed_messages[1]["content"]
            ):
                return """```json
    {
        "verdicts": [
            {
                "verdict": "yes"
            },
            {
                "verdict": "no",
                "reason": "The detail in this statement is not necessary for answering the question."
            }
        ]
    }```"""
            elif (
                "Coherence (1-5) - the collective quality of all sentences."
                in processed_messages[1]["content"]
            ):
                return "4"

        return ""
