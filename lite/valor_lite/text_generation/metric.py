from dataclasses import dataclass
from enum import Enum


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
        MetricTypes for text-generation tasks.
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


class ROUGEType(str, Enum):
    ROUGE1 = "rouge1"
    ROUGE2 = "rouge2"
    ROUGEL = "rougeL"
    ROUGELSUM = "rougeLsum"


@dataclass
class _TextGenerationMetricBase:
    """
    Defines a base class for text generation metrics.

    Attributes
    ----------
    status : str
        The status of the metric. The status should be "success" if the metric was calculated successfully and "error" if there was an error in calculating the metric.
    value : int | float | dict[str, float] | None
        The metric value. Different metrics have different value types. The value should only be None when the status is "error".
    parameters : dict
        Any parameters associated with the metric, as well as any datum or prediction parameters that are relevant to the metric.
    """

    status: str
    value: int | float | dict[str, float] | None
    parameters: dict
    __type__ = "BaseClass"

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

    def to_dict(self):
        """Converts a metric object into a dictionary."""
        return {
            "type": self.__type__,
            "status": self.status,
            "value": self.value,
            "parameters": self.parameters,
        }


class AnswerCorrectnessMetric(_TextGenerationMetricBase):
    """
    Defines an answer correctness metric.

    Attributes
    ----------
    status : str
        The status of the metric. The status should be "success" if the metric was calculated successfully and "error" if there was an error in calculating the metric.
    value : int | float | None
        The answer correctness score between 0 and 1, with higher values indicating that the answer is more correct. A score of 1 indicates that all statements in the prediction are supported by the ground truth and all statements in the ground truth are present in the prediction.
    parameters : dict
        Any parameters associated with the metric, as well as any datum or prediction parameters that are relevant to the metric.
    """

    __type__ = "AnswerCorrectness"

    def __post_init__(self):
        """Validate instantiated class."""
        super().__post_init__()
        if self.status == "success":
            if not isinstance(self.value, (int, float)):
                raise TypeError(
                    f"Expected value to be int or float, got {type(self.value).__name__}"
                )
            if not 0 <= self.value <= 1:
                raise ValueError(
                    f"Expected value to be between 0 and 1, got {self.value}"
                )


class AnswerRelevanceMetric(_TextGenerationMetricBase):
    """
    Defines an answer relevance metric.

    Attributes
    ----------
    status : str
        The status of the metric. The status should be "success" if the metric was calculated successfully and "error" if there was an error in calculating the metric.
    value : int | float | None
        The number of statements in the answer that are relevant to the query divided by the total number of statements in the answer.
    parameters : dict
        Any parameters associated with the metric, as well as any datum or prediction parameters that are relevant to the metric.
    """

    __type__ = "AnswerRelevance"

    def __post_init__(self):
        """Validate instantiated class."""
        super().__post_init__()
        if self.status == "success":
            if not isinstance(self.value, (int, float)):
                raise TypeError(
                    f"Expected value to be int or float, got {type(self.value).__name__}"
                )
            if not 0 <= self.value <= 1:
                raise ValueError(
                    f"Expected value to be between 0 and 1, got {self.value}"
                )


class BLEUMetric(_TextGenerationMetricBase):
    """
    Defines a BLEU metric.

    Attributes
    ----------
    status : str
        The status of the metric. The status should be "success" if the metric was calculated successfully and "error" if there was an error in calculating the metric.
    value : int | float | None
        The BLEU score for an individual datapoint.
    parameters : dict
        Any parameters associated with the metric, as well as any datum or prediction parameters that are relevant to the metric.
    """

    __type__ = "BLEU"

    def __post_init__(self):
        """Validate instantiated class."""
        super().__post_init__()
        if self.status == "success":
            if not isinstance(self.value, (int, float)):
                raise TypeError(
                    f"Expected value to be int or float, got {type(self.value).__name__}"
                )
            if not 0 <= self.value <= 1:
                raise ValueError(
                    f"Expected value to be between 0 and 1, got {self.value}"
                )


class BiasMetric(_TextGenerationMetricBase):
    """
    Defines a bias metric.

    Attributes
    ----------
    status : str
        The status of the metric. The status should be "success" if the metric was calculated successfully and "error" if there was an error in calculating the metric.
    value : int | float | None
        The bias score for a datum. This is a float between 0 and 1, with 1 indicating that all opinions in the datum text are biased and 0 indicating that there is no bias.
    parameters : dict
        Any parameters associated with the metric, as well as any datum or prediction parameters that are relevant to the metric.
    """

    __type__ = "Bias"

    def __post_init__(self):
        """Validate instantiated class."""
        super().__post_init__()
        if self.status == "success":
            if not isinstance(self.value, (int, float)):
                raise TypeError(
                    f"Expected value to be int or float, got {type(self.value).__name__}"
                )
            if not 0 <= self.value <= 1:
                raise ValueError(
                    f"Expected value to be between 0 and 1, got {self.value}"
                )


class ContextPrecisionMetric(_TextGenerationMetricBase):
    """
    Defines a context precision metric.

    Attributes
    ----------
    status : str
        The status of the metric. The status should be "success" if the metric was calculated successfully and "error" if there was an error in calculating the metric.
    value : int | float | None
        The context precision score for a datum. This is a float between 0 and 1, with 0 indicating that none of the contexts are useful to arrive at the ground truth answer to the query and 1 indicating that all contexts are useful to arrive at the ground truth answer to the query. The score is more heavily influenced by earlier contexts in the list of contexts than later contexts.
    parameters : dict
        Any parameters associated with the metric, as well as any datum or prediction parameters that are relevant to the metric.
    """

    __type__ = "ContextPrecision"

    def __post_init__(self):
        """Validate instantiated class."""
        super().__post_init__()
        if self.status == "success":
            if not isinstance(self.value, (int, float)):
                raise TypeError(
                    f"Expected value to be int or float, got {type(self.value).__name__}"
                )
            if not 0 <= self.value <= 1:
                raise ValueError(
                    f"Expected value to be between 0 and 1, got {self.value}"
                )


class ContextRecallMetric(_TextGenerationMetricBase):
    """
    Defines a context recall metric.

    Attributes
    ----------
    status : str
        The status of the metric. The status should be "success" if the metric was calculated successfully and "error" if there was an error in calculating the metric.
    value : int | float | None
        The context recall score for a datum. This is a float between 0 and 1, with 1 indicating that all ground truth statements are attributable to the context list.
    parameters : dict
        Any parameters associated with the metric, as well as any datum or prediction parameters that are relevant to the metric.
    """

    __type__ = "ContextRecall"

    def __post_init__(self):
        """Validate instantiated class."""
        super().__post_init__()
        if self.status == "success":
            if not isinstance(self.value, (int, float)):
                raise TypeError(
                    f"Expected value to be int or float, got {type(self.value).__name__}"
                )
            if not 0 <= self.value <= 1:
                raise ValueError(
                    f"Expected value to be between 0 and 1, got {self.value}"
                )


class ContextRelevanceMetric(_TextGenerationMetricBase):
    """
    Defines a context relevance metric.

    Attributes
    ----------
    status : str
        The status of the metric. The status should be "success" if the metric was calculated successfully and "error" if there was an error in calculating the metric.
    value : int | float | None
        The context relevance score for a datum. This is a float between 0 and 1, with 0 indicating that none of the contexts are relevant and 1 indicating that all of the contexts are relevant.
    parameters : dict
        Any parameters associated with the metric, as well as any datum or prediction parameters that are relevant to the metric.
    """

    __type__ = "ContextRelevance"

    def __post_init__(self):
        """Validate instantiated class."""
        super().__post_init__()
        if self.status == "success":
            if not isinstance(self.value, (int, float)):
                raise TypeError(
                    f"Expected value to be int or float, got {type(self.value).__name__}"
                )
            if not 0 <= self.value <= 1:
                raise ValueError(
                    f"Expected value to be between 0 and 1, got {self.value}"
                )


class FaithfulnessMetric(_TextGenerationMetricBase):
    """
    Defines a faithfulness metric.

    Attributes
    ----------
    status : str
        The status of the metric. The status should be "success" if the metric was calculated successfully and "error" if there was an error in calculating the metric.
    value : int | float | None
        The faithfulness score for a datum. This is a float between 0 and 1, with 1 indicating that all claims in the text are implied by the contexts.
    parameters : dict
        Any parameters associated with the metric, as well as any datum or prediction parameters that are relevant to the metric.
    """

    __type__ = "Faithfulness"

    def __post_init__(self):
        """Validate instantiated class."""
        super().__post_init__()
        if self.status == "success":
            if not isinstance(self.value, (int, float)):
                raise TypeError(
                    f"Expected value to be int or float, got {type(self.value).__name__}"
                )
            if not 0 <= self.value <= 1:
                raise ValueError(
                    f"Expected value to be between 0 and 1, got {self.value}"
                )


class HallucinationMetric(_TextGenerationMetricBase):
    """
    Defines a hallucination metric.

    Attributes
    ----------
    status : str
        The status of the metric. The status should be "success" if the metric was calculated successfully and "error" if there was an error in calculating the metric.
    value : int | float | None
        The hallucination score for a datum. This is a float between 0 and 1, with 1 indicating that all contexts are contradicted by the text.
    parameters : dict
        Any parameters associated with the metric, as well as any datum or prediction parameters that are relevant to the metric.
    """

    __type__ = "Hallucination"

    def __post_init__(self):
        """Validate instantiated class."""
        super().__post_init__()
        if self.status == "success":
            if not isinstance(self.value, (int, float)):
                raise TypeError(
                    f"Expected value to be int or float, got {type(self.value).__name__}"
                )
            if not 0 <= self.value <= 1:
                raise ValueError(
                    f"Expected value to be between 0 and 1, got {self.value}"
                )


class ROUGEMetric(_TextGenerationMetricBase):
    """
    Defines a ROUGE metric.

    Attributes
    ----------
    status : str
        The status of the metric. The status should be "success" if the metric was calculated successfully and "error" if there was an error in calculating the metric.
    value : dict[str, int | float] | None
        A JSON containing individual ROUGE scores calculated in different ways. `rouge1` is unigram-based scoring, `rouge2` is bigram-based scoring, `rougeL` is scoring based on sentences (i.e., splitting on "." and ignoring "\n"), and `rougeLsum` is scoring based on splitting the text using "\n".
    parameters : dict
        Any parameters associated with the metric, as well as any datum or prediction parameters that are relevant to the metric.
    """

    __type__ = "ROUGE"

    def __post_init__(self):
        """Validate instantiated class."""
        super().__post_init__()
        if self.status == "success":
            if not isinstance(self.value, dict):
                raise TypeError(
                    f"Expected value to be a dict[str, float], got {type(self.value).__name__}"
                )
            if not all(isinstance(k, str) for k in self.value.keys()):
                raise TypeError(
                    f"Expected keys in value to be strings, got {type(next(iter(self.value.keys()))).__name__}"
                )
            if not all(
                isinstance(v, (int, float)) for v in self.value.values()
            ):
                raise TypeError(
                    f"Expected the values in self.value to be int or float, got {type(next(iter(self.value.values()))).__name__}"
                )
            if not all(0 <= v <= 1 for v in self.value.values()):
                raise ValueError(
                    f"Expected values in value to be between 0 and 1, got {self.value}"
                )


class SummaryCoherenceMetric(_TextGenerationMetricBase):
    """
    Defines a summary coherence metric.

    Attributes
    ----------
    status : str
        The status of the metric. The status should be "success" if the metric was calculated successfully and "error" if there was an error in calculating the metric.
    value : int | None
        The summary coherence score for a datum. This is an integer with 1 being the lowest summary coherence and 5 the highest summary coherence.
    parameters : dict
        Any parameters associated with the metric, as well as any datum or prediction parameters that are relevant to the metric.
    """

    __type__ = "SummaryCoherence"

    def __post_init__(self):
        """Validate instantiated class."""
        super().__post_init__()
        if self.status == "success":
            if not isinstance(self.value, int):
                raise TypeError(
                    f"Expected value to be int, got {type(self.value).__name__}"
                )
            if self.value not in [1, 2, 3, 4, 5]:
                raise ValueError(
                    f"Expected value to be between 1 and 5, got {self.value}"
                )


class ToxicityMetric(_TextGenerationMetricBase):
    """
    Defines a toxicity metric.

    Attributes
    ----------
    status : str
        The status of the metric. The status should be "success" if the metric was calculated successfully and "error" if there was an error in calculating the metric.
    value : int | float | None
        The toxicity score for a datum. This is a value between 0 and 1, with 1 indicating that all opinions in the datum text are toxic and 0 indicating that there is no toxicity.
    parameters : dict
        Any parameters associated with the metric, as well as any datum or prediction parameters that are relevant to the metric.
    """

    __type__ = "Toxicity"

    def __post_init__(self):
        """Validate instantiated class."""
        super().__post_init__()
        if self.status == "success":
            if not isinstance(self.value, (int, float)):
                raise TypeError(
                    f"Expected value to be int or float, got {type(self.value).__name__}"
                )
            if not 0 <= self.value <= 1:
                raise ValueError(
                    f"Expected value to be between 0 and 1, got {self.value}"
                )
