import json
from dataclasses import dataclass

from valor_lite.text_generation.metric import MetricType


@dataclass
class EvaluationParameters:
    """
    Defines optional parameters for evaluation methods.

    Attributes
    ----------
    metrics: list[str], optional
        The list of metrics to compute, store, and return to the user.
    llm_api_params: dict[str, str | int | dict], optional
        A dictionary of parameters for the LLM API. Only required by some text generation metrics.
    metric_params: dict[str, dict], optional
        A dictionary of optional parameters to pass in to specific metrics.
    """

    metrics_to_return: list[MetricType] | None = None
    llm_api_params: dict[str, str | int | dict] | None = None
    metric_params: dict[str, dict] | None = None

    def __post_init__(self):
        """Validate instantiated class."""

        if not isinstance(self.metrics_to_return, (list, type(None))):
            raise TypeError(
                f"Expected 'metrics_to_return' to be of type 'list' or 'None', got {type(self.metrics_to_return).__name__}"
            )
        if self.metrics_to_return is not None and not all(
            isinstance(metric, MetricType) for metric in self.metrics_to_return
        ):
            raise TypeError(
                "All items in 'metrics_to_return' must be of type 'MetricType'"
            )

        if self.llm_api_params is not None:
            if not isinstance(self.llm_api_params, dict):
                raise TypeError(
                    f"Expected 'llm_api_params' to be of type 'dict' or 'None', got {type(self.llm_api_params).__name__}"
                )
            if not all(
                isinstance(key, str) for key in self.llm_api_params.keys()
            ):
                raise TypeError(
                    "All keys in 'llm_api_params' must be of type 'str'"
                )

            if not all(
                isinstance(value, (str, int, dict))
                for value in self.llm_api_params.values()
            ):
                raise TypeError(
                    "All values in 'llm_api_params' must be of type 'str', 'int' or 'dict'"
                )

        if self.metric_params is not None:
            if not isinstance(self.metric_params, dict):
                raise TypeError(
                    f"Expected 'metric_params' to be of type 'dict' or 'None', got {type(self.llm_api_params).__name__}"
                )
            if not all(
                isinstance(key, str) for key in self.metric_params.keys()
            ):
                raise TypeError(
                    "All keys in 'metric_params' must be of type 'str'"
                )

            if not all(
                isinstance(value, dict)
                for value in self.metric_params.values()
            ):
                raise TypeError(
                    "All values in 'metric_params' must be of type 'dict'"
                )


@dataclass
class Evaluation:
    parameters: EvaluationParameters
    metrics: list[dict]
    meta: dict | None = None

    def __str__(self) -> str:
        """Dumps the object into a JSON formatted string."""
        return json.dumps(self.__dict__, indent=4)

    def __post_init__(self):
        """Validate instantiated class."""

        if not isinstance(self.parameters, EvaluationParameters):
            raise TypeError(
                f"Expected 'parameters' to be of type 'EvaluationParameters', got {type(self.parameters).__name__}"
            )

        if not isinstance(self.metrics, list):
            raise TypeError(
                f"Expected 'metrics' to be of type 'list', got {type(self.metrics).__name__}"
            )
        if not all(isinstance(metric, dict) for metric in self.metrics):
            raise TypeError("All items in 'metrics' must be of type 'dict'")

        if not isinstance(self.meta, (dict, type(None))):
            raise TypeError(
                f"Expected 'meta' to be of type 'dict' or 'None', got {type(self.meta).__name__}"
            )

    def to_dict(self) -> dict:
        """
        Defines how a `valor.Evaluation` object is serialized into a dictionary.

        Returns
        ----------
        dict
            A dictionary describing an evaluation.
        """
        return {
            "parameters": self.parameters.__dict__,
            "metrics": self.metrics,
            "meta": self.meta,
        }
