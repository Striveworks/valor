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

    @classmethod
    def error(
        cls,
        error_type: str,
        error_message: str,
    ):
        return cls(
            type="Error",
            value={
                "type": error_type,
                "message": error_message,
            },
            parameters={},
        )

    @classmethod
    def answer_correctness(
        cls,
        value: float,
    ):
        """
        Defines an answer correctness metric.

        Parameters
        ----------
        value : float
            The answer correctness score between 0 and 1, with higher values indicating that the answer is more correct. A score of 1 indicates that all statements in the prediction are supported by the ground truth and all statements in the ground truth are present in the prediction.
        """
        if (value < 0) or (value > 1):
            raise ValueError("Value should be a float between 0 and 1.")
        return cls(
            type=MetricType.AnswerCorrectness,
            value=value,
            parameters={},
        )

    @classmethod
    def answer_relevance(
        cls,
        value: float,
    ):
        """
        Defines an answer relevance metric.

        Parameters
        ----------
        value : float
            The number of statements in the answer that are relevant to the query divided by the total number of statements in the answer.
        """
        if (value < 0) or (value > 1):
            raise ValueError("Value should be a float between 0 and 1.")
        return cls(
            type=MetricType.AnswerRelevance,
            value=value,
            parameters={},
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
        if (value < 0) or (value > 1):
            raise ValueError("Value should be a float between 0 and 1.")
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
    ):
        """
        Defines a bias metric.

        Parameters
        ----------
        value : float
            The bias score for a datum. This is a float between 0 and 1, with 1 indicating that all opinions in the datum text are biased and 0 indicating that there is no bias.
        """
        if (value < 0) or (value > 1):
            raise ValueError("Value should be a float between 0 and 1.")
        return cls(
            type=MetricType.Bias,
            value=value,
            parameters={},
        )

    @classmethod
    def context_precision(
        cls,
        value: float,
    ):
        """
        Defines a context precision metric.

        Parameters
        ----------
        value : float
            The context precision score for a datum. This is a float between 0 and 1, with 0 indicating that none of the contexts are useful to arrive at the ground truth answer to the query and 1 indicating that all contexts are useful to arrive at the ground truth answer to the query. The score is more heavily influenced by earlier contexts in the list of contexts than later contexts.
        """
        if (value < 0) or (value > 1):
            raise ValueError("Value should be a float between 0 and 1.")
        return cls(
            type=MetricType.ContextPrecision,
            value=value,
            parameters={},
        )

    @classmethod
    def context_recall(
        cls,
        value: float,
    ):
        """
        Defines a context recall metric.

        Parameters
        ----------
        value : float
            The context recall score for a datum. This is a float between 0 and 1, with 1 indicating that all ground truth statements are attributable to the context list.
        """
        if (value < 0) or (value > 1):
            raise ValueError("Value should be a float between 0 and 1.")
        return cls(
            type=MetricType.ContextRecall,
            value=value,
            parameters={},
        )

    @classmethod
    def context_relevance(
        cls,
        value: float,
    ):
        """
        Defines a context relevance metric.

        Parameters
        ----------
        value : float
            The context relevance score for a datum. This is a float between 0 and 1, with 0 indicating that none of the contexts are relevant and 1 indicating that all of the contexts are relevant.
        """
        if (value < 0) or (value > 1):
            raise ValueError("Value should be a float between 0 and 1.")
        return cls(
            type=MetricType.ContextRelevance,
            value=value,
            parameters={},
        )

    @classmethod
    def faithfulness(
        cls,
        value: float,
    ):
        """
        Defines a faithfulness metric.

        Parameters
        ----------
        value : float
            The faithfulness score for a datum. This is a float between 0 and 1, with 1 indicating that all claims in the text are implied by the contexts.
        """
        if (value < 0) or (value > 1):
            raise ValueError("Value should be a float between 0 and 1.")
        return cls(
            type=MetricType.Faithfulness,
            value=value,
            parameters={},
        )

    @classmethod
    def hallucination(cls, value: float):
        """
        Defines a hallucination metric.

        Parameters
        ----------
        value : float
            The hallucination score for a datum. This is a float between 0 and 1, with 1 indicating that all contexts are contradicted by the text.
        """
        if (value < 0) or (value > 1):
            raise ValueError("Value should be a float between 0 and 1.")
        return cls(
            type=MetricType.Hallucination,
            value=value,
            parameters={},
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
            The ROUGE variation used to compute the value. `rouge1` is unigram-based scoring, `rouge2` is bigram-based scoring, `rougeL` is scoring based on sentences (i.e., splitting on "." and ignoring "\n"), and `rougeLsum` is scoring based on splitting the text using "\n".
        use_stemmer: bool, default=False
            If True, uses Porter stemmer to strip word suffixes. Defaults to False.
        """
        if (value < 0) or (value > 1):
            raise ValueError("Value should be a float between 0 and 1.")
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
    ):
        """
        Defines a summary coherence metric.

        Parameters
        ----------
        value : int
            The summary coherence score for a datum. This is an integer with 1 being the lowest summary coherence and 5 the highest summary coherence.
        """
        if value not in [1, 2, 3, 4, 5]:
            raise ValueError(
                f"Expected value to be between 1 and 5, got {value}"
            )
        return cls(
            type=MetricType.SummaryCoherence,
            value=value,
            parameters={},
        )

    @classmethod
    def toxicity(
        cls,
        value: float,
    ):
        """
        Defines a toxicity metric.

        Parameters
        ----------
        value : float
            The toxicity score for a datum. This is a value between 0 and 1, with 1 indicating that all opinions in the datum text are toxic and 0 indicating that there is no toxicity.
        """
        if (value < 0) or (value > 1):
            raise ValueError("Value should be a float between 0 and 1.")
        return cls(
            type=MetricType.Toxicity,
            value=value,
            parameters={},
        )
