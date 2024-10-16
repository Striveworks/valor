import json
from dataclasses import dataclass

from valor_lite.text_generation import enums


@dataclass
class Datum:
    """
    A class used to store information about a datum for either a 'GroundTruth' or a 'Prediction'.

    Attributes
    ----------
    uid : str
        The UID of the datum.
    text : str, optional
        If the datum is a piece of text, then this field should contain the text.
    metadata : dict[str, Any]
        A dictionary of metadata that describes the datum.

    Examples
    --------
    >>> Datum(uid="uid1")
    >>> Datum(uid="uid1", metadata={})
    >>> Datum(uid="uid1", metadata={"foo": "bar", "pi": 3.14})
    >>> Datum(uid="uid2", text="What is the capital of Kenya?")
    """

    uid: str | None = None
    text: str | None = None
    metadata: dict | None = None

    def __post_init__(
        self,
    ):
        """Validate instantiated class."""

        if not isinstance(self.uid, (str, type(None))):
            raise TypeError(
                f"Expected 'uid' to be of type 'str' or 'None', got {type(self.uid).__name__}"
            )
        if not isinstance(self.text, (str, type(None))):
            raise TypeError(
                f"Expected 'text' to be of type 'str' or 'None', got {type(self.text).__name__}"
            )
        if not isinstance(self.metadata, (dict, type(None))):
            raise TypeError(
                f"Expected 'metadata' to be of type 'dict' or 'None', got {type(self.metadata).__name__}"
            )


@dataclass
class Annotation:
    """
    A class used to annotate `GroundTruths` and `Predictions`.

    Attributes
    ----------
    metadata: dict[str, Any]
        A dictionary of metadata that describes the `Annotation`.
    text: str, optional
        A piece of text to assign to the 'Annotation'.
    context_list: list[str], optional
        A list of contexts to assign to the 'Annotation'.

    Examples
    --------
    Text Generation Annotation with text and context_list. Not all text generation tasks require both text and context.
    >>> annotation = Annotation(
    ...     text="Abraham Lincoln was the 16th President of the United States.",
    ...     context_list=["Lincoln was elected the 16th president of the United States in 1860.", "Abraham Lincoln was born on February 12, 1809, in a one-room log cabin on the Sinking Spring Farm in Hardin County, Kentucky."],
    ... )
    """

    metadata: dict | None = None
    text: str | None = None
    context_list: list[str] | None = None

    def __post_init__(self):
        """Validate instantiated class."""

        if not isinstance(self.metadata, (dict, type(None))):
            raise TypeError(
                f"Expected 'metadata' to be of type 'dict' or 'None', got {type(self.metadata).__name__}"
            )

        if not isinstance(self.text, (str, type(None))):
            raise TypeError(
                f"Expected 'text' to be of type 'str' or 'None', got {type(self.text).__name__}"
            )

        if self.context_list is not None:
            if not isinstance(self.context_list, list):
                raise TypeError(
                    f"Expected 'context_list' to be of type 'list' or 'None', got {type(self.context_list).__name__}"
                )

            if not all(
                isinstance(context, str) for context in self.context_list
            ):
                raise TypeError(
                    "All items in 'context_list' must be of type 'str'"
                )


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

    metrics_to_return: list[enums.MetricType] | None = None
    llm_api_params: dict[str, str | int | dict] | None = None
    metric_params: dict[str, dict] | None = None

    def __post_init__(self):
        """Validate instantiated class."""

        if not isinstance(self.metrics_to_return, (list, type(None))):
            raise TypeError(
                f"Expected 'metrics_to_return' to be of type 'list' or 'None', got {type(self.metrics_to_return).__name__}"
            )
        if self.metrics_to_return is not None and not all(
            isinstance(metric, enums.MetricType)
            for metric in self.metrics_to_return
        ):
            raise TypeError(
                "All items in 'metrics_to_return' must be of type 'enums.MetricType'"
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


@dataclass
class GroundTruth:
    """
    An object describing a ground truth (e.g., a human generated answer).

    Attributes
    ----------
    datum : Datum
        The datum associated with the groundtruth.
    annotations : list[Annotation]
        The list of annotations associated with the groundtruth.

    Examples
    --------
    >>> GroundTruth(
    ...     datum=Datum(uid="uid1"),
    ...     annotations=[
    ...         Annotation(
    ...             text="The answer is 6*7=42.",
    ...         )
    ...     ]
    ... )
    """

    datum: Datum
    annotations: list[Annotation]

    def __post_init__(
        self,
    ):
        """Validate instantiated class."""

        if not isinstance(self.datum, Datum):
            raise TypeError(
                f"Expected 'datum' to be of type 'Datum', got {type(self.datum).__name__}"
            )

        if not isinstance(self.annotations, list):
            raise TypeError(
                f"Expected 'annotations' to be of type 'list', got {type(self.annotations).__name__}"
            )
        if not all(
            isinstance(annotation, Annotation)
            for annotation in self.annotations
        ):
            raise TypeError(
                "All items in 'annotations' must be of type 'Annotation'"
            )


@dataclass
class Prediction:
    """
    An object describing a prediction (e.g., an LLM generated answer).

    Attributes
    ----------
    datum : Datum
        The datum associated with the prediction.
    annotations : list[Annotation]
        The list of annotations associated with the prediction.

    Examples
    --------
    >>> Prediction(
    ...     datum=Datum(uid="uid1"),
    ...     annotations=[
    ...         Annotation(
    ...             text="The answer is forty two.",
    ...         )
    ...     ]
    ... )
    """

    datum: Datum
    annotations: list[Annotation]

    def __post_init__(self):
        """Validate instantiated class."""

        if not isinstance(self.datum, Datum):
            raise TypeError(
                f"Expected 'datum' to be of type 'Datum', got {type(self.datum).__name__}"
            )

        if not isinstance(self.annotations, list):
            raise TypeError(
                f"Expected 'annotations' to be of type 'list', got {type(self.annotations).__name__}"
            )
        if not all(
            isinstance(annotation, Annotation)
            for annotation in self.annotations
        ):
            raise TypeError(
                "All items in 'annotations' must be of type 'Annotation'"
            )
