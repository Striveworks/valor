import json
import re
from typing import Any

from valor_lite.text_generation.llm.exceptions import InvalidLLMResponseError


def trim_and_load_json(text: str) -> dict[str, Any]:
    """
    Trims and loads input_string as a dictionary. Adapted from DeepEval https://github.com/confident-ai/deepeval/blob/dc117a5ea2160dbb61909c537908a41f7da4dfe7/deepeval/metrics/utils.py#L50

    Parameters
    ----------
    input_string : str
        The input string to trim and load as a json.

    Returns
    -------
    dict
        A dictionary.
    """

    pattern = r"\{[\s\S]*\}"
    match = re.search(pattern, text)
    if not match:
        raise InvalidLLMResponseError(
            f"LLM did not include valid brackets in its response: {text}"
        )
    extracted_text = match.group()

    try:
        return json.loads(extracted_text)
    except json.JSONDecodeError as e:
        raise InvalidLLMResponseError(
            f"Evaluation LLM responded with invalid JSON. JSONDecodeError: {str(e)}"
        )


def find_first_signed_integer(text: str) -> int | None:
    match = re.search(r"-?\d+", text)
    if not match:
        return None
    return int(match.group())
