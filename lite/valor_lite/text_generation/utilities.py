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


def validate_metrics_to_return(
    metrics_to_return: Sequence[str | MetricType],
):
    """
    Validate that the provided metrics are appropriate for text generation.

    Parameters
    ----------
    metrics_to_return : Sequence[str | MetricType]
        A list of metrics to check.
    """
    if not isinstance(metrics_to_return, (list, Sequence)):
        raise TypeError(
            f"Expected 'metrics_to_return' to be of type 'list' or 'Sequence', got {type(metrics_to_return).__name__}"
        )

    if not all(
        isinstance(metric, (str, MetricType)) for metric in metrics_to_return
    ):
        raise TypeError(
            "All items in 'metrics_to_return' must be of type 'str' or 'MetricType'"
        )

    if not set(metrics_to_return).issubset(MetricType.text_generation()):
        raise ValueError(
            f"The following metrics are not supported for text generation: '{set(metrics_to_return) - MetricType.text_generation()}'"
        )

    if len(metrics_to_return) != len(set(metrics_to_return)):
        raise ValueError(
            "There are duplicate metrics in 'metrics_to_return'. Please remove the duplicates."
        )


def validate_llm_api_params(
    llm_api_params: dict[str, str | int | dict] | None = None,
):
    """
    Validate the LLM API parameters.

    Parameters
    ----------
    llm_api_params : dict[str, str | int | dict], optional
        A dictionary of parameters for the LLM API. Only required by LLM-guided text generation metrics.
    """
    if llm_api_params is None:
        return

    if not isinstance(llm_api_params, dict):
        raise TypeError(
            f"Expected 'llm_api_params' to be of type 'dict' or 'None', got {type(llm_api_params).__name__}"
        )

    if not all(isinstance(key, str) for key in llm_api_params.keys()):
        raise TypeError("All keys in 'llm_api_params' must be of type 'str'")

    if not all(
        isinstance(value, (str, int, dict))
        for value in llm_api_params.values()
    ):
        raise TypeError(
            "All values in 'llm_api_params' must be of type 'str', 'int' or 'dict'"
        )

    if not ("client" in llm_api_params or "api_url" in llm_api_params):
        raise ValueError("Need to specify the client or api_url.")

    if "client" in llm_api_params and "api_url" in llm_api_params:
        raise ValueError("Cannot specify both client and api_url.")


def validate_metric_params(
    metrics_to_return: Sequence[str | MetricType],
    metric_params: dict[str, dict] | None = None,
):
    """
    Validate the metric parameters for text generation metrics.

    Parameters
    ----------
    metrics_to_return : Sequence[str | MetricType]
        A list of metrics to calculate during the evaluation.
    metric_params : dict, optional
        A dictionary of optional parameters to pass in to specific metrics.
    """
    if metric_params is None:
        return

    if not isinstance(metric_params, dict):
        raise TypeError(
            f"Expected 'metric_params' to be of type 'dict' or 'None', got {type(metric_params).__name__}"
        )
    if not all(isinstance(key, str) for key in metric_params.keys()):
        raise TypeError("All keys in 'metric_params' must be of type 'str'")

    if not all(isinstance(value, dict) for value in metric_params.values()):
        raise TypeError("All values in 'metric_params' must be of type 'dict'")

    if not set(metric_params.keys()).issubset(
        [
            metric.value if isinstance(metric, MetricType) else metric
            for metric in metrics_to_return
        ]
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
