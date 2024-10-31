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


@dataclass
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
        if not isinstance(self.type, str):
            raise TypeError(
                f"Metric type should be of type 'str': {self.type}"
            )
        elif not isinstance(self.value, (int, float, dict)):
            raise TypeError(
                f"Metric value must be of type 'int', 'float' or 'dict': {self.value}"
            )
        elif not isinstance(self.parameters, dict):
            raise TypeError(
                f"Metric parameters must be of type 'dict[str, Any]': {self.parameters}"
            )
        elif not all([isinstance(k, str) for k in self.parameters.keys()]):
            raise TypeError(
                f"Metric parameter dictionary should only have keys with type 'str': {self.parameters}"
            )

    @classmethod
    def error(
        cls,
        error_type: str,
        error_message: str,
        model_name: str,
        retries: int,
    ):
        return cls(
            type="Error",
            value={
                "type": error_type,
                "message": error_message,
            },
            parameters={
                "evaluator": model_name,
                "retries": retries,
            },
        )

    @classmethod
    def answer_correctness(
        cls,
        value: float,
        model_name: str,
        retries: int,
    ):
        """
        Defines an answer correctness metric.

        Parameters
        ----------
        value : float
            The answer correctness score between 0 and 1, with higher values indicating that the answer
            is more correct. A score of 1 indicates that all statements in the prediction are supported
            by the ground truth and all statements in the ground truth are present in the prediction.
        """
        return cls(
            type=MetricType.AnswerCorrectness,
            value=value,
            parameters={
                "evaluator": model_name,
                "retries": retries,
            },
        )

    @classmethod
    def answer_relevance(
        cls,
        value: float,
        model_name: str,
        retries: int,
    ):
        """
        Defines an answer relevance metric.

        Parameters
        ----------
        value : float
            The number of statements in the answer that are relevant to the query divided by the total
            number of statements in the answer.
        """
        return cls(
            type=MetricType.AnswerRelevance,
            value=value,
            parameters={
                "evaluator": model_name,
                "retries": retries,
            },
        )

    @classmethod
    def bleu(
        cls,
        value: float,
        weights: list[float],
    ):
        """
        Defines a BLEU metric.

        Parameters
        ----------
        value : float
            The BLEU score for an individual datapoint.
        weights : list[float]
            The list of weights that the score was calculated with.
        """
        return cls(
            type=MetricType.BLEU,
            value=value,
            parameters={
                "weights": weights,
            },
        )

    @classmethod
    def bias(
        cls,
        value: float,
        model_name: str,
        retries: int,
    ):
        """
        Defines a bias metric.

        Parameters
        ----------
        value : float
            The bias score for a datum. This is a float between 0 and 1, with 1 indicating that all
            opinions in the datum text are biased and 0 indicating that there is no bias.
        """
        return cls(
            type=MetricType.Bias,
            value=value,
            parameters={
                "evaluator": model_name,
                "retries": retries,
            },
        )

    @classmethod
    def context_precision(
        cls,
        value: float,
        model_name: str,
        retries: int,
    ):
        """
        Defines a context precision metric.

        Parameters
        ----------
        value : float
            The context precision score for a datum. This is a float between 0 and 1, with 0 indicating
            that none of the contexts are useful to arrive at the ground truth answer to the query
            and 1 indicating that all contexts are useful to arrive at the ground truth answer to the
            query. The score is more heavily influenced by earlier contexts in the list of contexts
            than later contexts.
        """
        return cls(
            type=MetricType.ContextPrecision,
            value=value,
            parameters={
                "evaluator": model_name,
                "retries": retries,
            },
        )

    @classmethod
    def context_recall(
        cls,
        value: float,
        model_name: str,
        retries: int,
    ):
        """
        Defines a context recall metric.

        Parameters
        ----------
        value : float
            The context recall score for a datum. This is a float between 0 and 1, with 1 indicating
            that all ground truth statements are attributable to the context list.
        """
        return cls(
            type=MetricType.ContextRecall,
            value=value,
            parameters={
                "evaluator": model_name,
                "retries": retries,
            },
        )

    @classmethod
    def context_relevance(
        cls,
        value: float,
        model_name: str,
        retries: int,
    ):
        """
        Defines a context relevance metric.

        Parameters
        ----------
        value : float
            The context relevance score for a datum. This is a float between 0 and 1, with 0 indicating
            that none of the contexts are relevant and 1 indicating that all of the contexts are relevant.
        """
        return cls(
            type=MetricType.ContextRelevance,
            value=value,
            parameters={
                "evaluator": model_name,
                "retries": retries,
            },
        )

    @classmethod
    def faithfulness(
        cls,
        value: float,
        model_name: str,
        retries: int,
    ):
        """
        Defines a faithfulness metric.

        Parameters
        ----------
        value : float
            The faithfulness score for a datum. This is a float between 0 and 1, with 1 indicating that
            all claims in the text are implied by the contexts.
        """
        return cls(
            type=MetricType.Faithfulness,
            value=value,
            parameters={
                "evaluator": model_name,
                "retries": retries,
            },
        )

    @classmethod
    def hallucination(
        cls,
        value: float,
        model_name: str,
        retries: int,
    ):
        """
        Defines a hallucination metric.

        Parameters
        ----------
        value : float
            The hallucination score for a datum. This is a float between 0 and 1, with 1 indicating that
            all contexts are contradicted by the text.
        """
        return cls(
            type=MetricType.Hallucination,
            value=value,
            parameters={
                "evaluator": model_name,
                "retries": retries,
            },
        )

    @classmethod
    def rouge(
        cls,
        value: float,
        rouge_type: str,
        use_stemmer: bool,
    ):
        """
        Defines a ROUGE metric.

        Parameters
        ----------
        value : float
            A ROUGE score.
        rouge_type : ROUGEType
            The ROUGE variation used to compute the value. `rouge1` is unigram-based scoring, `rouge2` is bigram-based
            scoring, `rougeL` is scoring based on sentences (i.e., splitting on "." and ignoring "\n"), and `rougeLsum`
            is scoring based on splitting the text using "\n".
        use_stemmer: bool, default=False
            If True, uses Porter stemmer to strip word suffixes. Defaults to False.
        """
        return cls(
            type=MetricType.ROUGE,
            value=value,
            parameters={
                "rouge_type": rouge_type,
                "use_stemmer": use_stemmer,
            },
        )

    @classmethod
    def summary_coherence(
        cls,
        value: int,
        model_name: str,
        retries: int,
    ):
        """
        Defines a summary coherence metric.

        Parameters
        ----------
        value : int
            The summary coherence score for a datum. This is an integer with 1 being the lowest summary coherence
            and 5 the highest summary coherence.
        """
        return cls(
            type=MetricType.SummaryCoherence,
            value=value,
            parameters={
                "evaluator": model_name,
                "retries": retries,
            },
        )

    @classmethod
    def toxicity(
        cls,
        value: float,
        model_name: str,
        retries: int,
    ):
        """
        Defines a toxicity metric.

        Parameters
        ----------
        value : float
            The toxicity score for a datum. This is a value between 0 and 1, with 1 indicating that all opinions
            in the datum text are toxic and 0 indicating that there is no toxicity.
        """
        return cls(
            type=MetricType.Toxicity,
            value=value,
            parameters={
                "evaluator": model_name,
                "retries": retries,
            },
        )
