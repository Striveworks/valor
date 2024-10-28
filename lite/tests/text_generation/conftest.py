import json

import pytest
from valor_lite.text_generation.integrations import _validate_messages


def _package_json(data: dict):
    return f"```json \n{json.dumps(data, indent=4)}\n```"


class MockWrapper:
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
        pass

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
                response = _package_json(
                    {
                        "claims": [
                            "The capital of the UK is London.",
                            "The capital of South Korea is Seoul.",
                            "The capital of Argentina is Canada.",
                        ]
                    }
                )

            # Generate opinions
            elif (
                "generate a list of OPINIONS"
                in processed_messages[1]["content"]
            ):
                response = _package_json(
                    {
                        "opinions": [
                            "I like the color green.",
                            "People from Canada are nicer than people from other countries.",
                        ]
                    }
                )

            # Generate statements
            elif (
                "generate a list of STATEMENTS"
                in processed_messages[1]["content"]
            ):
                response = _package_json(
                    {
                        "statements": [
                            "The capital of the UK is London.",
                            "London is the largest city in the UK by population and GDP.",
                        ]
                    }
                )

            # Answer correctness verdicts
            elif (
                "Return in JSON format with three keys: 'TP', 'FP', and 'FN'"
                in processed_messages[1]["content"]
            ):
                response = _package_json(
                    {
                        "TP": ["London is the largest city in the UK by GDP"],
                        "FP": [
                            "London is the largest city in the UK by population"
                        ],
                        "FN": [
                            "In 2021, financial services made up more than 20% of London's output"
                        ],
                    }
                )

            # Answer relevance verdicts
            elif (
                "generate a list of verdicts that indicate whether each statement is relevant to address the query"
                in processed_messages[1]["content"]
            ):
                response = _package_json(
                    {"verdicts": [{"verdict": "yes"}, {"verdict": "no"}]}
                )

            # Bias verdicts
            elif (
                "generate a list of verdicts to indicate whether EACH opinion is biased"
                in processed_messages[1]["content"]
            ):
                response = _package_json(
                    {"verdicts": [{"verdict": "no"}, {"verdict": "yes"}]}
                )

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
                response = _package_json(
                    {
                        "verdicts": [
                            {"verdict": "yes"},
                            {"verdict": "no"},
                            {"verdict": "no"},
                            {"verdict": "yes"},
                        ]
                    }
                )

            # Context recall verdicts
            elif (
                "analyze each ground truth statement and determine if the statement can be attributed to the given context"
                in processed_messages[1]["content"]
            ):
                response = _package_json(
                    {"verdicts": [{"verdict": "yes"}, {"verdict": "yes"}]}
                )

            # Context relevance verdicts
            elif (
                "generate a list of verdicts to indicate whether each context is relevant to the provided query"
                in processed_messages[1]["content"]
            ):
                response = _package_json(
                    {
                        "verdicts": [
                            {"verdict": "yes"},
                            {"verdict": "yes"},
                            {"verdict": "no"},
                            {"verdict": "yes"},
                        ]
                    }
                )

            # Faithfulness verdicts
            elif (
                "generate a list of verdicts to indicate whether EACH claim is implied by the context list"
                in processed_messages[1]["content"]
            ):
                response = _package_json(
                    {
                        "verdicts": [
                            {"verdict": "yes"},
                            {"verdict": "no"},
                            {"verdict": "no"},
                        ]
                    }
                )

            # Hallucination agreement verdicts
            elif (
                "generate a list of verdicts to indicate whether the given text contradicts EACH context"
                in processed_messages[1]["content"]
            ):
                response = _package_json(
                    {
                        "verdicts": [
                            {"verdict": "no"},
                            {"verdict": "no"},
                            {"verdict": "yes"},
                            {"verdict": "no"},
                        ]
                    }
                )

            # Toxicity verdicts
            elif (
                "generate a list of verdicts to indicate whether EACH opinion is toxic"
                in processed_messages[1]["content"]
            ):
                response = _package_json(
                    {"verdicts": [{"verdict": "no"}, {"verdict": "no"}]}
                )

        if response is None:
            response = ""
        return response


@pytest.fixture
def client() -> MockWrapper:
    return MockWrapper()
