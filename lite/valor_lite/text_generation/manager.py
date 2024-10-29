from functools import wraps

from valor_lite.text_generation.annotation import Query
from valor_lite.text_generation.computation import (
    calculate_answer_correctness,
    calculate_answer_relevance,
    calculate_bias,
    calculate_context_precision,
    calculate_context_recall,
    calculate_context_relevance,
    calculate_faithfulness,
    calculate_hallucination,
    calculate_rouge_scores,
    calculate_sentence_bleu,
    calculate_summary_coherence,
    calculate_toxicity,
)
from valor_lite.text_generation.llm.exceptions import InvalidLLMResponseError
from valor_lite.text_generation.llm.integrations import (
    ClientWrapper,
    MistralWrapper,
    OpenAIWrapper,
)
from valor_lite.text_generation.metric import Metric, MetricType


def llm_guided(fn):
    """
    Call the LLMClient class function with retries for InvalidLLMResponseError.

    If retries is set to 0, then the function will only be called once and not retried.

    If, for example, retries is set to 3, then the function will be retried in the
    event of an InvalidLLMResponseError up to 3 times, for a maximum of 4 calls.
    """

    @wraps(fn)
    def wrapper(self, *args, **kwargs):
        if self.client is None:
            raise ValueError(
                f"{fn.__name__} requires the definition of an LLM client."
            )
        error = None
        retries = getattr(self, "retries", 0)
        for _ in range(1 + retries):
            try:
                return fn(self, *args, **kwargs)
            except InvalidLLMResponseError as e:
                error = e
        if error is not None:
            return Metric.error(
                error_type=type(error).__name__,
                error_message=str(error),
            )

    return wrapper


class Evaluator:
    """
    Parent class for all LLM clients.

    Attributes
    ----------
    client : ClientWrapper, optional
        An optional client to compute llm-guided metrics.
    retries : int
        The number of times to retry the API call if it fails. Defaults to 0, indicating
        that the call will not be retried.
    """

    def __init__(
        self,
        client: ClientWrapper,
        retries: int = 0,
    ):
        """
        Creates an instance of a generic LLM client.

        Parameters
        ----------
        client : ClientWrapper, optional
            Any LLM client that conforms to _ClientWrapper. Required for LLM-guided metrics.
        retries : int, default=0
            The number of times to retry the API call if it fails. Defaults to 0, indicating
            that the call will not be retried.
        """
        self.client = client
        self.retries = retries
        self.default_system_prompt = "You are a helpful assistant."

    @classmethod
    def openai(
        cls,
        model_name: str = "gpt-3.5-turbo",
        api_key: str | None = None,
        retries: int = 0,
        seed: int | None = None,
    ):
        """
        Create an evaluator using OpenAI's client.

        Parameters
        ----------
        model_name : str, default="gpt-3.5-turbo"
            The model to use. Defaults to "gpt-3.5-turbo".
        api_key : str, optional
            The OpenAI API key to use. If not specified, then the OPENAI_API_KEY environment
            variable will be used.
        retries : int, default=0
            The number of times to retry the API call if it fails. Defaults to 0, indicating
            that the call will not be retried. For example, if self.retries is set to 3,
            this means that the call will be retried up to 3 times, for a maximum of 4 calls.
        seed : int, optional
            An optional seed can be provided to GPT to get deterministic results.
        """
        if seed is not None:
            if retries != 0:
                raise ValueError(
                    "Seed is provided, but retries is not 0. Retries should be 0 when seed is provided."
                )
        client = OpenAIWrapper(
            api_key=api_key,
            model_name=model_name,
            seed=seed,
        )
        return cls(client=client, retries=retries)

    @classmethod
    def mistral(
        cls,
        model_name: str = "mistral-small-latest",
        api_key: str | None = None,
        retries: int = 0,
    ):
        """
        Create an evaluator using the Mistral API.

        Parameters
        ----------
        model_name : str, default="mistral-small-latest"
            The model to use. Defaults to "mistral-small-latest".
        api_key : str, optional
            The Mistral API key to use. If not specified, then the MISTRAL_API_KEY environment
            variable will be used.
        retries : int, default=0
            The number of times to retry the API call if it fails. Defaults to 0, indicating
            that the call will not be retried. For example, if self.retries is set to 3,
            this means that the call will be retried up to 3 times, for a maximum of 4 calls.
        """
        client = MistralWrapper(
            api_key=api_key,
            model_name=model_name,
        )
        return cls(client=client, retries=retries)

    @llm_guided
    def compute_answer_correctness(
        self,
        query: Query,
    ) -> Metric:
        """
        Compute answer correctness. Answer correctness is computed as an f1 score obtained
        by comparing prediction statements to ground truth statements.

        If there are multiple ground truths, then the f1 score is computed for each ground
        truth and the maximum score is returned.

        This metric was adapted from RAGAS. We follow a similar prompting strategy and
        computation, however we do not do a weighted sum with an answer similarity score
        using embeddings.

        Parameters
        ----------
        query: Query
            A user query with ground truth and generated response.

        Returns
        -------
        Metric
            The answer correctness score between 0 and 1. Higher values indicate that the
            answer is more correct. A score of 1 indicates that all statements in the
            prediction are supported by the ground truth and all statements in the ground
            truth are present in the prediction.
        """
        if not query.context or len(query.context.groundtruth) == 0:
            raise ValueError(
                "Answer correctness requires the definition of ground truth context."
            )

        result = calculate_answer_correctness(
            client=self.client,
            system_prompt=self.default_system_prompt,
            query=query.query,
            response=query.response,
            groundtruths=query.context.groundtruth,
        )
        return Metric.answer_correctness(value=result)

    @llm_guided
    def compute_answer_relevance(self, query: Query) -> Metric:
        """
        Compute answer relevance, the proportion of statements that are relevant to the
        query, for a single piece of text.

        Parameters
        ----------
        query: Query
            A user query with ground truth and generated response.

        Returns
        -------
        Metric
            The answer relevance score between 0 and 1. A score of 1 indicates that all
            statements are relevant to the query.
        """
        result = calculate_answer_relevance(
            client=self.client,
            system_prompt=self.default_system_prompt,
            query=query.query,
            response=query.response,
        )
        return Metric.answer_relevance(value=result)

    @llm_guided
    def compute_bias(
        self,
        query: Query,
    ) -> Metric:
        """
        Compute bias, the portion of opinions that are biased.

        Parameters
        ----------
        query: Query
            A user query with ground truth and generated response.

        Returns
        -------
        float
            The bias score between 0 and 1. A score of 1 indicates that all opinions in
            the text are biased.
        """
        result = calculate_bias(
            client=self.client,
            system_prompt=self.default_system_prompt,
            response=query.response,
        )
        return Metric.bias(value=result)

    @llm_guided
    def compute_context_precision(
        self,
        query: Query,
    ) -> Metric:
        """
        Compute context precision, a score for evaluating the retrieval
        mechanism of a RAG model.

        First, an LLM is prompted to determine if each context in the context
        list is useful for producing the ground truth answer to the query.

        If there are multiple ground truths, then the verdict is "yes" for a
        context if that context is useful for producing any of the ground truth
        answers, and "no" otherwise.

        Then, using these verdicts, the context precision score is computed as
        a weighted sum of the precision at k for each k from 1 to the length
        of the context list.

        Note that the earlier a piece of context appears in the context list,
        the more important it is in the computation of this score. For example,
        the first context in the context list will be included in every precision
        at k computation, so will have a large influence on the final score,
        whereas the last context will only be used for the last precision at
        k computation, so will have a small influence on the final score.

        Parameters
        ----------
        query: Query
            A user query with ground truth and generated response.

        Returns
        -------
        Metric
            The context precision score between 0 and 1. A higher score indicates
            better context precision.
        """
        if not query.context or len(query.context.prediction) == 0:
            raise ValueError(
                "Context precision requires context in the prediction response."
            )
        if len(query.context.groundtruth) == 0:
            raise ValueError(
                "Context precision requires ground truth contexts."
            )

        result = calculate_context_precision(
            client=self.client,
            system_prompt=self.default_system_prompt,
            query=query.query,
            retrieved_context=query.context.prediction,
            groundtruth_context=query.context.groundtruth,
        )
        return Metric.context_precision(value=result)

    @llm_guided
    def compute_context_recall(
        self,
        query: Query,
    ) -> Metric:
        """
        Compute context recall, a score for evaluating the retrieval mechanism of a RAG model.

        The context recall score is the proportion of statements in the ground truth
        that are attributable to the context list.

        If multiple ground truths are provided, then the context recall score is
        computed for each ground truth and the maximum score is returned.

        Parameters
        ----------
        query: Query
            A user query with ground truth and generated response.

        Returns
        -------
        Metric
            The context recall score between 0 and 1. A score of 1 indicates that
            all ground truth statements are attributable to the contexts in the context list.
        """
        if not query.context or len(query.context.prediction) == 0:
            raise ValueError(
                "Context recall requires context in the prediction response."
            )
        if len(query.context.groundtruth) == 0:
            raise ValueError("Context recall requires ground truth contexts.")

        result = calculate_context_recall(
            client=self.client,
            system_prompt=self.default_system_prompt,
            retrieved_context=query.context.prediction,
            groundtruth_context=query.context.groundtruth,
        )
        return Metric.context_recall(value=result)

    @llm_guided
    def compute_context_relevance(
        self,
        query: Query,
    ) -> Metric:
        """
        Compute context relevance, the proportion of contexts in the context list
        that are relevant to the query.

        Parameters
        ----------
        query: Query
            A user query with ground truth and generated response.

        Returns
        -------
        Metric
            The context relevance score between 0 and 1. A score of 0 indicates
            that none of the contexts are relevant and a score of 1 indicates
            that all of the contexts are relevant.
        """
        if not query.context or len(query.context.prediction) == 0:
            raise ValueError(
                "Context relevance requires context in the prediction response."
            )

        result = calculate_context_relevance(
            client=self.client,
            system_prompt=self.default_system_prompt,
            query=query.query,
            context=query.context.prediction,
        )
        return Metric.context_relevance(value=result)

    @llm_guided
    def compute_faithfulness(
        self,
        query: Query,
    ) -> Metric:
        """
        Compute the faithfulness score. The faithfulness score is the proportion
        of claims in the text that are implied by the list of contexts. Claims
        that contradict the list of contexts and claims that are unrelated to
        the list of contexts both count against the score.

        Parameters
        ----------
        query: Query
            A user query with ground truth and generated response.

        Returns
        -------
        Metric
            The faithfulness score between 0 and 1. A score of 1 indicates that
            all claims in the text are implied by the list of contexts.
        """
        if not query.context:
            raise ValueError(
                "Faithfulness requires context in the prediction response."
            )

        result = calculate_faithfulness(
            client=self.client,
            system_prompt=self.default_system_prompt,
            response=query.response,
            context=query.context.prediction,
        )
        return Metric.faithfulness(value=result)

    @llm_guided
    def compute_hallucination(
        self,
        query: Query,
    ) -> Metric:
        """
        Compute the hallucination score, the proportion of contexts in the context
        list that are contradicted by the text.

        Parameters
        ----------
        query: Query
            A user query with ground truth and generated response.

        Returns
        -------
        Metric
            The hallucination score between 0 and 1. A score of 1 indicates that
            all contexts are contradicted by the text.
        """
        if not query.context or len(query.context.prediction) == 0:
            raise ValueError(
                "Hallucination requires context in the prediction response."
            )

        result = calculate_hallucination(
            client=self.client,
            system_prompt=self.default_system_prompt,
            response=query.response,
            context=query.context.prediction,
        )
        return Metric.hallucination(value=result)

    @llm_guided
    def compute_summary_coherence(
        self,
        query: Query,
    ) -> Metric:
        """
        Compute summary coherence, the collective quality of a summary.

        Parameters
        ----------
        query: Query
            A user query with ground truth and generated response.

        Returns
        -------
        Metric
            The summary coherence score between 1 and 5. A score of 1 indicates
            the lowest summary coherence and a score of 5 indicates the highest
            summary coherence.
        """
        result = calculate_summary_coherence(
            client=self.client,
            system_prompt=self.default_system_prompt,
            text=query.query,
            summary=query.response,
        )
        return Metric.summary_coherence(value=result)

    @llm_guided
    def compute_toxicity(
        self,
        query: Query,
    ) -> Metric:
        """
        Compute toxicity, the portion of opinions that are toxic.

        Parameters
        ----------
        query: Query
            A user query with ground truth and generated response.

        Returns
        -------
        Metric
            The toxicity score will be evaluated as a float between 0 and 1, with
            1 indicating that all opinions in the text are toxic.
        """
        result = calculate_toxicity(
            client=self.client,
            system_prompt=self.default_system_prompt,
            response=query.response,
        )
        return Metric.toxicity(value=result)

    @staticmethod
    def compute_rouge(
        query: Query,
        rouge_types: list[str] = [
            "rouge1",
            "rouge2",
            "rougeL",
            "rougeLsum",
        ],
        use_stemmer: bool = False,
    ) -> list[Metric]:
        """
        Calculate ROUGE scores for a prediction given some set of references.

        Parameters
        ----------
        query: Query
            A user query with ground truth and generated response.
        rouge_types : list[str], optional
            A list of rouge types to calculate.
            Defaults to ['rouge1', 'rouge2', 'rougeL', 'rougeLsum'].
        use_stemmer: bool, default=False
            If True, uses Porter stemmer to strip word suffixes. Defaults to False.

        Returns
        -------
        list[Metric]
        """
        if not query.context:
            raise ValueError("Context is required for rouge metric.")
        results = calculate_rouge_scores(
            prediction=query.response,
            references=query.context.groundtruth,
            rouge_types=rouge_types,
            use_stemmer=use_stemmer,
        )
        return [
            Metric.rouge(
                value=result,
                rouge_type=rouge_type,
                use_stemmer=use_stemmer,
            )
            for rouge_type, result in results.items()
        ]

    @staticmethod
    def compute_sentence_bleu(
        query: Query,
        weights: list[float] = [0.25, 0.25, 0.25, 0.25],
    ) -> Metric:
        """
        Calculate sentence BLEU scores for a set of prediction - ground truth pairs.

        Parameters
        ----------
        query: Query
            A user query with ground truth and generated response.
        weights: list[float], default=[0.25, 0.25, 0.25, 0.25]
            The default BLEU calculates a score for up to 4-grams using uniform
            weights (this is called BLEU-4). To evaluate your translations with
            higher/lower order ngrams, use customized weights. Example: when accounting
            for up to 5-grams with uniform weights (this is called BLEU-5) use [1/5]*5
        """
        if not query.context:
            raise ValueError("Context is required for sentence BLEU metric.")
        result = calculate_sentence_bleu(
            prediction=query.response,
            references=query.context.groundtruth,
            weights=weights,
        )
        return Metric.bleu(
            value=result,
            weights=weights,
        )

    def compute_all(
        self,
        query: Query,
        bleu_weights: list[float] = [0.25, 0.25, 0.25, 0.25],
        rouge_types: list[str] = [
            "rouge1",
            "rouge2",
            "rougeL",
            "rougeLsum",
        ],
        rouge_use_stemmer: bool = False,
    ) -> dict[MetricType, list[Metric]]:
        """
        Computes all available metrics.

        Parameters
        ----------
        query: Query
            A user query with ground truth and generated response.
        bleu_weights: list[float], default=[0.25, 0.25, 0.25, 0.25]
            The default BLEU calculates a score for up to 4-grams using uniform
            weights (this is called BLEU-4). To evaluate your translations with
            higher/lower order ngrams, use customized weights. Example: when accounting
            for up to 5-grams with uniform weights (this is called BLEU-5) use [1/5]*5
        rouge_types : list[str], optional
            A list of rouge types to calculate.
            Defaults to ['rouge1', 'rouge2', 'rougeL', 'rougeLsum'].
        rouge_use_stemmer: bool, default=False
            If True, uses Porter stemmer to strip word suffixes. Defaults to False.
        """
        results = dict()
        results[MetricType.AnswerCorrectness] = [
            self.compute_answer_correctness(query)
        ]
        results[MetricType.AnswerRelevance] = [
            self.compute_answer_relevance(query)
        ]
        results[MetricType.Bias] = [self.compute_bias(query)]
        results[MetricType.ContextPrecision] = [
            self.compute_context_precision(query)
        ]
        results[MetricType.ContextRecall] = [
            self.compute_context_recall(query)
        ]
        results[MetricType.ContextRelevance] = [
            self.compute_context_relevance(query)
        ]
        results[MetricType.Faithfulness] = [self.compute_faithfulness(query)]
        results[MetricType.Hallucination] = [self.compute_hallucination(query)]
        results[MetricType.SummaryCoherence] = [
            self.compute_summary_coherence(query)
        ]
        results[MetricType.Toxicity] = [self.compute_toxicity(query)]
        results[MetricType.ROUGE] = self.compute_rouge(
            query, rouge_types=rouge_types, use_stemmer=rouge_use_stemmer
        )
        results[MetricType.BLEU] = [
            self.compute_sentence_bleu(query, weights=bleu_weights)
        ]
        return results
