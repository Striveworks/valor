from valor_lite.text_generation.metric import MetricType


def validate_text_gen_metrics_to_return(
    metrics_to_return: list[MetricType],
) -> None:
    """
    Validate that the provided metrics are appropriate for text generation.

    Parameters
    ----------
    metrics_to_return : List[MetricType]
        A list of metrics that need to be validated against the task type.

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
        if "weights" in bleu_params:
            bleu_weights = bleu_params["weights"]
            if not all(
                isinstance(weight, (int, float)) and 0 <= weight
                for weight in bleu_weights
            ):
                raise ValueError(
                    "BLEU metric weights must be a list of non-negative integers or floats."
                )
            if sum(bleu_weights) != 1:
                raise ValueError("BLEU metric weights must sum to 1.")
