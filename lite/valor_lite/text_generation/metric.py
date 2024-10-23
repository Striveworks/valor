from dataclasses import dataclass
from enum import Enum

from valor_lite.schemas import BaseMetric


class MetricType(str, Enum):
    AnswerCorrectness = "AnswerCorrectness"
    AnswerRelevance = "AnswerRelevance"
    Bias = "Bias"
    BLEU = "BLEU"
    ContextPrecision = "ContextPrecision"
    ContextRecall = "ContextRecall"
    ContextRelevance = "ContextRelevance"
    Faithfulness = "Faithfulness"
    Hallucination = "Hallucination"
    ROUGE = "ROUGE"
    SummaryCoherence = "SummaryCoherence"
    Toxicity = "Toxicity"

    @classmethod
    def text_generation(cls) -> set["MetricType"]:
        """
        All text generation metrics.
        """
        return {
            cls.AnswerCorrectness,
            cls.AnswerRelevance,
            cls.Bias,
            cls.BLEU,
            cls.ContextPrecision,
            cls.ContextRecall,
            cls.ContextRelevance,
            cls.Faithfulness,
            cls.Hallucination,
            cls.ROUGE,
            cls.SummaryCoherence,
            cls.Toxicity,
        }

    @classmethod
    def llm_guided(cls) -> set["MetricType"]:
        """
        All text generation metrics that use an LLM for part of the metric computation.
        """
        return {
            cls.AnswerCorrectness,
            cls.AnswerRelevance,
            cls.Bias,
            cls.ContextPrecision,
            cls.ContextRecall,
            cls.ContextRelevance,
            cls.Faithfulness,
            cls.Hallucination,
            cls.SummaryCoherence,
            cls.Toxicity,
        }

    @classmethod
    def text_comparison(cls) -> set["MetricType"]:
        """
        All text generation metrics that use ground truth text as part of the metric computation.

        These metrics require ground truths to be used. They are not usable in certain settings, like streaming, where ground truths are not available.
        """
        return {
            cls.AnswerCorrectness,
            cls.BLEU,
            cls.ContextPrecision,
            cls.ContextRecall,
            cls.ROUGE,
        }


class ROUGEType(str, Enum):
    ROUGE1 = "rouge1"
    ROUGE2 = "rouge2"
    ROUGEL = "rougeL"
    ROUGELSUM = "rougeLsum"


class MetricStatus(str, Enum):
    SUCCESS = "success"
    FAILED = "error"


def _validate_text_generation_value(
    value: int | float | dict[str, float] | None,
    status: MetricStatus,
):
    if not isinstance(status, MetricStatus):
        raise TypeError(
            f"Expected status to be of type MetricStatus but received {type(status)}"
        )
    if status == MetricStatus.SUCCESS:
        if not isinstance(value, (int, float)):
            raise TypeError(
                f"Expected value to be int or float, got {type(value).__name__}"
            )
        if not 0 <= value <= 1:
            raise ValueError(
                f"Expected value to be between 0 and 1, got {value}"
            )


class Metric(BaseMetric):
    """
    Text Generation Metric.

    Attributes
    ----------
    type : str
        The metric type.
    value : int | float | dict
        The metric value.
    parameters : dict[str, Any]
        A dictionary containing metric parameters.
    """

    def __post_init__(self):
        """Validate instantiated class."""
        if not isinstance(self.status, str):
            raise TypeError(
                f"Expected status to be a string, got {type(self.status).__name__}"
            )
        if self.status not in {"success", "error"}:
            raise ValueError(
                f"Expected status to be either 'success' or 'error', got {self.status}"
            )
        if self.status == "error":
            if self.value is not None:
                raise ValueError(
                    f"Expected value to be None when status is 'error', got {self.value}"
                )
        if not isinstance(self.parameters, dict):
            raise TypeError(
                f"Expected parameters to be a dict, got {type(self.parameters).__name__}"
            )

    @classmethod
    def AnswerCorrectness(
        cls,
        value: int | float | None,
        status: MetricStatus,
        parameters: dict,
    ):
        """
        Defines an answer correctness metric.

        Parameters
        ----------
        value : int | float | None
            The answer correctness score between 0 and 1, with higher values indicating that the answer is more correct. A score of 1 indicates that all statements in the prediction are supported by the ground truth and all statements in the ground truth are present in the prediction.
        status : str
            The status of the metric. The status should be "success" if the metric was calculated successfully and "error" if there was an error in calculating the metric.
        parameters : dict
            Any parameters associated with the metric, as well as any datum or prediction parameters that are relevant to the metric.
        """
        _validate_text_generation_value(value, status)
        return cls(
            type=MetricType.AnswerCorrectness,
            value={
                "value": value,
                "status": status.value,
            },
            parameters={
                **parameters,
            },
        )

    @classmethod
    def AnswerRelevance(
        cls,
        value: int | float | None,
        status: MetricStatus,
        parameters: dict,
    ):
        """
        Defines an answer relevance metric.

        Parameters
        ----------
        value : int | float | None
            The number of statements in the answer that are relevant to the query divided by the total number of statements in the answer.
        status : str
            The status of the metric. The status should be "success" if the metric was calculated successfully and "error" if there was an error in calculating the metric.
        parameters : dict
            Any parameters associated with the metric, as well as any datum or prediction parameters that are relevant to the metric.
        """
        _validate_text_generation_value(value, status)
        return cls(
            type=MetricType.AnswerRelevance,
            value={
                "value": value,
                "status": status.value,
            },
            parameters={
                **parameters,
            },
        )

    @classmethod
    def BLEU(
        cls,
        value: int | float | None,
        status: MetricStatus,
        weights: list[float],
    ):
        """
        Defines a BLEU metric.

        Parameters
        ----------
        value : int | float | None
            The BLEU score for an individual datapoint.
        status : str
            The status of the metric. The status should be "success" if the metric was calculated successfully and "error" if there was an error in calculating the metric.
        parameters : dict
            Any parameters associated with the metric, as well as any datum or prediction parameters that are relevant to the metric.
        """
        _validate_text_generation_value(value, status)
        return cls(
            type=MetricType.BLEU,
            value={
                "value": value,
                "status": status.value,
            },
            parameters={
                "weights": weights,
            },
        )

    @classmethod
    def Bias(
        cls,
        value: int | float | None,
        status: MetricStatus,
        parameters: dict,
    ):
        """
        Defines a bias metric.

        Parameters
        ----------
        value : int | float | None
            The bias score for a datum. This is a float between 0 and 1, with 1 indicating that all opinions in the datum text are biased and 0 indicating that there is no bias.
        status : str
            The status of the metric. The status should be "success" if the metric was calculated successfully and "error" if there was an error in calculating the metric.
        parameters : dict
            Any parameters associated with the metric, as well as any datum or prediction parameters that are relevant to the metric.
        """
        _validate_text_generation_value(value, status)
        return cls(
            type=MetricType.Bias,
            value={
                "value": value,
                "status": status.value,
            },
            parameters={
                **parameters,
            },
        )

    @classmethod
    def ContextPrecision(
        cls,
        value: int | float | None,
        status: MetricStatus,
        parameters: dict,
    ):
        """
        Defines a context precision metric.

        Parameters
        ----------
        value : int | float | None
            The context precision score for a datum. This is a float between 0 and 1, with 0 indicating that none of the contexts are useful to arrive at the ground truth answer to the query and 1 indicating that all contexts are useful to arrive at the ground truth answer to the query. The score is more heavily influenced by earlier contexts in the list of contexts than later contexts.
        status : str
            The status of the metric. The status should be "success" if the metric was calculated successfully and "error" if there was an error in calculating the metric.
        parameters : dict
            Any parameters associated with the metric, as well as any datum or prediction parameters that are relevant to the metric.
        """
        _validate_text_generation_value(value, status)
        return cls(
            type=MetricType.ContextPrecision,
            value={
                "value": value,
                "status": status.value,
            },
            parameters={
                **parameters,
            },
        )

    @classmethod
    def ContextRecall(
        cls,
        value: int | float | None,
        status: MetricStatus,
        parameters: dict,
    ):
        """
        Defines a context recall metric.

        Parameters
        ----------
        value : int | float | None
            The context recall score for a datum. This is a float between 0 and 1, with 1 indicating that all ground truth statements are attributable to the context list.
        status : str
            The status of the metric. The status should be "success" if the metric was calculated successfully and "error" if there was an error in calculating the metric.
        parameters : dict
            Any parameters associated with the metric, as well as any datum or prediction parameters that are relevant to the metric.
        """
        _validate_text_generation_value(value, status)
        return cls(
            type=MetricType.ContextRecall,
            value={
                "value": value,
                "status": status.value,
            },
            parameters={
                **parameters,
            },
        )

    @classmethod
    def ContextRelevance(
        cls,
        value: int | float | None,
        status: MetricStatus,
        parameters: dict,
    ):
        """
        Defines a context relevance metric.

        Parameters
        ----------
        value : int | float | None
            The context relevance score for a datum. This is a float between 0 and 1, with 0 indicating that none of the contexts are relevant and 1 indicating that all of the contexts are relevant.
        status : str
            The status of the metric. The status should be "success" if the metric was calculated successfully and "error" if there was an error in calculating the metric.
        parameters : dict
            Any parameters associated with the metric, as well as any datum or prediction parameters that are relevant to the metric.
        """
        _validate_text_generation_value(value, status)
        return cls(
            type=MetricType.ContextRelevance,
            value={
                "value": value,
                "status": status.value,
            },
            parameters={
                **parameters,
            },
        )

    @classmethod
    def Faithfulness(
        cls,
        value: int | float | None,
        status: MetricStatus,
        parameters: dict,
    ):
        """
        Defines a faithfulness metric.

        Parameters
        ----------
        value : int | float | None
            The faithfulness score for a datum. This is a float between 0 and 1, with 1 indicating that all claims in the text are implied by the contexts.
        status : str
            The status of the metric. The status should be "success" if the metric was calculated successfully and "error" if there was an error in calculating the metric.
        parameters : dict
            Any parameters associated with the metric, as well as any datum or prediction parameters that are relevant to the metric.
        """
        _validate_text_generation_value(value, status)
        return cls(
            type=MetricType.Faithfulness,
            value={
                "value": value,
                "status": status.value,
            },
            parameters={
                **parameters,
            },
        )

    @classmethod
    def Hallucination(
        cls,
        value: int | float | None,
        status: MetricStatus,
        parameters: dict,
    ):
        """
        Defines a hallucination metric.

        Parameters
        ----------
        value : int | float | None
            The hallucination score for a datum. This is a float between 0 and 1, with 1 indicating that all contexts are contradicted by the text.
        status : str
            The status of the metric. The status should be "success" if the metric was calculated successfully and "error" if there was an error in calculating the metric.
        parameters : dict
            Any parameters associated with the metric, as well as any datum or prediction parameters that are relevant to the metric.
        """
        _validate_text_generation_value(value, status)
        return cls(
            type=MetricType.Hallucination,
            value={
                "value": value,
                "status": status.value,
            },
            parameters={
                **parameters,
            },
        )

    @classmethod
    def ROUGE(
        cls,
        value: int | float | None,
        status: MetricStatus,
        types: list[ROUGEType],
        use_stemmer: bool,
    ):
        """
        Defines a ROUGE metric.

        Parameters
        ----------
        value : int | float | None
            A JSON containing individual ROUGE scores calculated in different ways. `rouge1` is unigram-based scoring, `rouge2` is bigram-based scoring, `rougeL` is scoring based on sentences (i.e., splitting on "." and ignoring "\n"), and `rougeLsum` is scoring based on splitting the text using "\n".
        status : str
            The status of the metric. The status should be "success" if the metric was calculated successfully and "error" if there was an error in calculating the metric.
        parameters : dict
            Any parameters associated with the metric, as well as any datum or prediction parameters that are relevant to the metric.
        """
        if status == "success":
            if not isinstance(value, dict):
                raise TypeError(
                    f"Expected value to be a dict[str, float], got {type(value).__name__}"
                )
            if not all(isinstance(k, str) for k in value.keys()):
                raise TypeError(
                    f"Expected keys in value to be strings, got {type(next(iter(value.keys()))).__name__}"
                )
            if not all(isinstance(v, (int, float)) for v in value.values()):
                raise TypeError(
                    f"Expected the values in self.value to be int or float, got {type(next(iter(value.values()))).__name__}"
                )
            if not all(0 <= v <= 1 for v in value.values()):
                raise ValueError(
                    f"Expected values in value to be between 0 and 1, got {value}"
                )
        return cls(
            type=MetricType.ROUGE,
            value={
                "value": value,
                "status": status.value,
            },
            parameters={
                "types": [t.value for t in types],
                "use_stemmer": use_stemmer,
            },
        )

    @classmethod
    def SummaryCoherence(
        cls,
        value: int | None,
        status: MetricStatus,
        parameters: dict,
    ):
        """
        Defines a summary coherence metric.

        Parameters
        ----------
        value : int | float | None
            The summary coherence score for a datum. This is an integer with 1 being the lowest summary coherence and 5 the highest summary coherence.
        status : str
            The status of the metric. The status should be "success" if the metric was calculated successfully and "error" if there was an error in calculating the metric.
        parameters : dict
            Any parameters associated with the metric, as well as any datum or prediction parameters that are relevant to the metric.
        """
        if status == "success":
            if not isinstance(value, int):
                raise TypeError(
                    f"Expected value to be int, got {type(value).__name__}"
                )
            if value not in [1, 2, 3, 4, 5]:
                raise ValueError(
                    f"Expected value to be between 1 and 5, got {value}"
                )
        return cls(
            type=MetricType.SummaryCoherence,
            value={
                "value": value,
                "status": status.value,
            },
            parameters={
                **parameters,
            },
        )

    @classmethod
    def Toxicity(
        cls,
        value: int | float | None,
        status: MetricStatus,
        parameters: dict,
    ):
        """
        Defines a toxicity metric.

        Parameters
        ----------
        value : int | float | None
            The toxicity score for a datum. This is a value between 0 and 1, with 1 indicating that all opinions in the datum text are toxic and 0 indicating that there is no toxicity.
        status : str
            The status of the metric. The status should be "success" if the metric was calculated successfully and "error" if there was an error in calculating the metric.
        parameters : dict
            Any parameters associated with the metric, as well as any datum or prediction parameters that are relevant to the metric.
        """
        _validate_text_generation_value(value, status)
        return cls(
            type=MetricType.SummaryCoherence,
            value={
                "value": value,
                "status": status.value,
            },
            parameters={
                **parameters,
            },
        )
