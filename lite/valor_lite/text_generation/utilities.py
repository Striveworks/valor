import json
from typing import Any, Sequence

from valor_lite.text_generation.exceptions import InvalidLLMResponseError
from valor_lite.text_generation.metric import MetricType, ROUGEType


def trim_and_load_json(input_string: str) -> Any:
    """
    Trims and loads input_string as a json. Adapted from DeepEval https://github.com/confident-ai/deepeval/blob/dc117a5ea2160dbb61909c537908a41f7da4dfe7/deepeval/metrics/utils.py#L50

    Parameters
    ----------
    input_string : str
        The input string to trim and load as a json.

    Returns
    -------
    Any
        The json object.
    """
    start = input_string.find("{")
    end = input_string.rfind("}") + 1

    if end == 0 and start != -1:
        input_string = input_string + "}"
        end = len(input_string)

    jsonStr = input_string[start:end] if start != -1 and end != 0 else ""

    try:
        return json.loads(jsonStr)
    except json.JSONDecodeError as e:
        raise InvalidLLMResponseError(
            "Evaluation LLM outputted an invalid JSON. Please use a better evaluation model. JSONDecodeError: "
            + str(e)
        )


def validate_text_gen_metrics_to_return(
    metrics_to_return: Sequence[str | MetricType],
) -> None:
    """
    Validate that the provided metrics are appropriate for text generation.

    Parameters
    ----------
    metrics_to_return : Sequence[str | MetricType]
        A list of metrics to check.

    Raises
    ------
    ValueError
        If any of the provided metrics are not supported for text generation.
    """
    if not set(metrics_to_return).issubset(MetricType.text_generation()):
        raise ValueError(
            f"The following metrics are not supported for text generation: '{set(metrics_to_return) - MetricType.text_generation()}'"
        )


def validate_metric_parameters(
    metrics_to_return: list[MetricType],
    metric_params: dict[str, dict],
):
    # check that the keys of metric parameters are all in metrics_to_return
    if not set(metric_params.keys()).issubset(
        [metric.value for metric in metrics_to_return]
    ):
        raise ValueError(
            "The keys of metric_params must be a subset of the metrics_to_return."
        )

    if MetricType.BLEU in metric_params:
        bleu_params = metric_params[MetricType.BLEU.value]

        if not isinstance(bleu_params, dict):
            raise TypeError(
                f"Expected BLEU parameters to be of type 'dict', got {type(bleu_params).__name__}"
            )

        if "weights" in bleu_params:
            bleu_weights = bleu_params["weights"]
            if not isinstance(bleu_weights, list):
                raise TypeError(
                    f"Expected BLEU weights to be of type 'list', got {type(bleu_weights).__name__}"
                )

            if not all(
                isinstance(weight, (int, float)) and 0 <= weight
                for weight in bleu_weights
            ):
                raise ValueError(
                    "BLEU metric weights must be a list of non-negative integers or floats."
                )

            if sum(bleu_weights) != 1:
                raise ValueError("BLEU metric weights must sum to 1.")

    if MetricType.ROUGE in metric_params:
        rouge_params = metric_params[MetricType.ROUGE.value]
        if not isinstance(rouge_params, dict):
            raise TypeError(
                f"Expected ROUGE parameters to be of type 'dict', got {type(rouge_params).__name__}"
            )

        if "rouge_types" in rouge_params:
            rouge_types = rouge_params["rouge_types"]
            if not isinstance(rouge_types, (list, set)):
                raise TypeError(
                    f"Expected rouge_types to be of type 'list' or 'set', got {type(rouge_types).__name__}"
                )

            if not set(rouge_types).issubset(set(ROUGEType)):
                raise ValueError(
                    f"Some ROUGE types are not supported: {set(rouge_types) - set(ROUGEType)}"
                )

        if "use_stemmer" in rouge_params:
            use_stemmer = rouge_params["use_stemmer"]
            if not isinstance(use_stemmer, bool):
                raise TypeError(
                    f"Expected use_stemmer to be of type 'bool', got {type(use_stemmer).__name__}"
                )
