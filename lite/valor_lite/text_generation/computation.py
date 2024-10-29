from collections import defaultdict

import evaluate
from nltk.tokenize import RegexpTokenizer
from nltk.translate import bleu_score
from valor_lite.text_generation.llm.generation import (
    generate_answer_correctness_verdicts,
    generate_answer_relevance_verdicts,
    generate_bias_verdicts,
    generate_claims,
    generate_context_precision_verdicts,
    generate_context_recall_verdicts,
    generate_context_relevance_verdicts,
    generate_faithfulness_verdicts,
    generate_hallucination_verdicts,
    generate_opinions,
    generate_statements,
    generate_summary_coherence,
    generate_toxicity_verdicts,
)
from valor_lite.text_generation.llm.integrations import ClientWrapper


def calculate_answer_correctness(
    client: ClientWrapper,
    system_prompt: str,
    query: str,
    response: str,
    groundtruths: list[str],
) -> float:
    prediction_statements = generate_statements(
        client=client,
        system_prompt=system_prompt,
        text=response,
    )
    f1_scores = [0.0]
    for groundtruth in groundtruths:
        groundtruth_statements = generate_statements(
            client=client,
            system_prompt=system_prompt,
            text=groundtruth,
        )
        verdicts = generate_answer_correctness_verdicts(
            client=client,
            system_prompt=system_prompt,
            query=query,
            groundtruth_statements=groundtruth_statements,
            prediction_statements=prediction_statements,
        )

        tp = len(verdicts["TP"])
        fp = len(verdicts["FP"])
        fn = len(verdicts["FN"])

        f1_scores.append(tp / (tp + 0.5 * (fp + fn)) if tp > 0 else 0)

    return max(f1_scores)


def calculate_answer_relevance(
    client: ClientWrapper,
    system_prompt: str,
    query: str,
    response: str,
) -> float:
    statements = generate_statements(
        client=client,
        system_prompt=system_prompt,
        text=response,
    )
    verdicts = generate_answer_relevance_verdicts(
        client=client,
        system_prompt=system_prompt,
        query=query,
        statements=statements,
    )
    if len(verdicts) == 0:
        return 0.0

    return sum(verdict["verdict"] == "yes" for verdict in verdicts) / len(
        verdicts
    )


def calculate_bias(
    client: ClientWrapper,
    system_prompt: str,
    response: str,
) -> float:

    opinions = generate_opinions(
        client=client,
        system_prompt=system_prompt,
        text=response,
    )
    if len(opinions) == 0:
        return 0.0

    verdicts = generate_bias_verdicts(
        client=client,
        system_prompt=system_prompt,
        opinions=opinions,
    )
    if len(verdicts) == 0:
        return 0.0

    return sum(verdict["verdict"] == "yes" for verdict in verdicts) / len(
        verdicts
    )


def calculate_context_precision(
    client: ClientWrapper,
    system_prompt: str,
    query: str,
    retrieved_context: list[str],
    groundtruth_context: list[str],
) -> float:
    """
    Compute context precision, a score for evaluating the retrieval mechanism of a RAG model.

    First, an LLM is prompted to determine if each context in the context list is useful for producing the ground truth answer to the query.

    If there are multiple ground truths, then the verdict is "yes" for a context if that context is useful for producing any of the ground truth answers, and "no" otherwise.

    Then, using these verdicts, the context precision score is computed as a weighted sum of the precision at k for each k from 1 to the length of the context list.

    Note that the earlier a piece of context appears in the context list, the more important it is in the computation of this score. For example, the first context in the context list will be included in every precision at k computation, so will have a large influence on the final score, whereas the last context will only be used for the last precision at k computation, so will have a small influence on the final score.

    Parameters
    ----------
    query: Query
        A user query with ground truth and generated response.

    Returns
    -------
    Metric
        The context precision score between 0 and 1. A higher score indicates better context precision.
    """
    if len(retrieved_context) == 0 and len(groundtruth_context) == 0:
        return 1.0
    elif len(retrieved_context) == 0 or len(groundtruth_context) == 0:
        return 0.0

    # Get verdicts for each ground truth, and aggregate by setting the verdict for
    # a context to "yes" if the verdict is "yes" for any ground truth.
    aggregate_verdicts = ["no"] * len(retrieved_context)
    for groundtruth in groundtruth_context:
        verdicts = generate_context_precision_verdicts(
            client=client,
            system_prompt=system_prompt,
            query=query,
            ordered_context_list=retrieved_context,
            groundtruth=groundtruth,
        )
        for i in range(len(verdicts)):
            if verdicts[i]["verdict"] == "yes":
                aggregate_verdicts[i] = "yes"

    # Use the aggregate verdicts to compute the precision at k for each k.
    precision_at_k_list = []
    for k in range(1, len(retrieved_context) + 1):
        # Only compute the precision at k if the kth context is relevant.
        if aggregate_verdicts[k - 1] == "yes":
            precision_at_k = (
                sum(verdict == "yes" for verdict in aggregate_verdicts[:k]) / k
            )
            precision_at_k_list.append(precision_at_k)

    # If none of the context are relevant, then the context precision is 0.
    if len(precision_at_k_list) == 0:
        return 0.0

    # Average over all the precision at k for which the kth context is relevant.
    return sum(precision_at_k_list) / len(precision_at_k_list)


def calculate_context_recall(
    client: ClientWrapper,
    system_prompt: str,
    retrieved_context: list[str],
    groundtruth_context: list[str],
) -> float:
    """
    Compute context recall, a score for evaluating the retrieval mechanism of a RAG model.

    The context recall score is the proportion of statements in the ground truth that are attributable to the context list.

    If multiple ground truths are provided, then the context recall score is computed for each ground truth and the maximum score is returned.

    Parameters
    ----------
    query: Query
        A user query with ground truth and generated response.

    Returns
    -------
    Metric
        The context recall score between 0 and 1. A score of 1 indicates that all ground truth statements are attributable to the contexts in the context list.
    """
    if len(retrieved_context) == 0 and len(groundtruth_context) == 0:
        return 1.0
    elif len(retrieved_context) == 0 or len(groundtruth_context) == 0:
        return 0.0

    scores = []
    for groundtruth in groundtruth_context:
        groundtruth_statements = generate_statements(
            client=client, system_prompt=system_prompt, text=groundtruth
        )
        verdicts = generate_context_recall_verdicts(
            client=client,
            system_prompt=system_prompt,
            context_list=retrieved_context,
            groundtruth_statements=groundtruth_statements,
        )
        scores.append(
            sum(verdict["verdict"] == "yes" for verdict in verdicts)
            / len(verdicts)
        )

    return max(scores)


def calculate_context_relevance(
    client: ClientWrapper,
    system_prompt: str,
    query: str,
    context: list[str],
) -> float:
    """
    Compute context relevance, the proportion of contexts in the context list that are relevant to the query.

    Parameters
    ----------
    query: Query
        A user query with ground truth and generated response.

    Returns
    -------
    Metric
        The context relevance score between 0 and 1. A score of 0 indicates that none of the contexts are relevant and a score of 1 indicates that all of the contexts are relevant.
    """
    if len(context) == 0:
        return 0.0
    verdicts = generate_context_relevance_verdicts(
        client=client,
        system_prompt=system_prompt,
        query=query,
        context_list=context,
    )
    return sum(verdict["verdict"] == "yes" for verdict in verdicts) / len(
        verdicts
    )


def calculate_faithfulness(
    client: ClientWrapper,
    system_prompt: str,
    response: str,
    context: list[str],
) -> float:
    """
    Compute the faithfulness score.

    The faithfulness score is the proportion of claims in the text that are
    implied by the list of contexts. Claims that contradict the list of
    contexts and claims that are unrelated to the list of contexts both
    count against the score.

    Parameters
    ----------
    query: Query
        A user query with ground truth and generated response.

    Returns
    -------
    Metric
        The faithfulness score between 0 and 1. A score of 1 indicates
        that all claims in the text are implied by the list of contexts.
    """
    if len(context) == 0:
        return 0.0

    claims = generate_claims(
        client=client, system_prompt=system_prompt, text=response
    )

    # If there aren't any claims, then the text is perfectly faithful, as the text does not contain any non-faithful claims.
    if len(claims) == 0:
        return 1.0

    faithfulness_verdicts = generate_faithfulness_verdicts(
        client=client,
        system_prompt=system_prompt,
        claims=claims,
        context_list=context,
    )
    return sum(
        verdict["verdict"] == "yes" for verdict in faithfulness_verdicts
    ) / len(faithfulness_verdicts)


def calculate_hallucination(
    client: ClientWrapper,
    system_prompt: str,
    response: str,
    context: list[str],
) -> float:
    """
    Compute the hallucination score, the proportion of contexts in the context list that are contradicted by the text.

    Parameters
    ----------
    query: Query
        A user query with ground truth and generated response.

    Returns
    -------
    Metric
        The hallucination score between 0 and 1. A score of 1 indicates that all contexts are contradicted by the text.
    """
    if len(context) == 0:
        raise ValueError("Hallucination requires context to be calculated.")

    verdicts = generate_hallucination_verdicts(
        client=client,
        system_prompt=system_prompt,
        text=response,
        context_list=context,
    )
    return sum(verdict["verdict"] == "yes" for verdict in verdicts) / len(
        verdicts
    )


def calculate_summary_coherence(
    client: ClientWrapper,
    system_prompt: str,
    text: str,
    summary: str,
) -> int:
    """
    Compute summary coherence, the collective quality of a summary.

    Parameters
    ----------
    text: str
        The text that was summarized.
    summary: str
        The summary to be evaluated.

    Returns
    -------
    int
        The summary coherence score will be evaluated as an integer, with 1 indicating the lowest summary coherence and 5 the highest summary coherence.
    """
    return generate_summary_coherence(
        client=client,
        system_prompt=system_prompt,
        text=text,
        summary=summary,
    )


def calculate_toxicity(
    client: ClientWrapper,
    system_prompt: str,
    response: str,
) -> float:
    """
    Compute toxicity, the portion of opinions that are toxic.

    Parameters
    ----------
    query: Query
        A user query with ground truth and generated response.

    Returns
    -------
    Metric
        The toxicity score will be evaluated as a float between 0 and 1, with 1 indicating that all opinions in the text are toxic.
    """
    opinions = generate_opinions(
        client=client, system_prompt=system_prompt, text=response
    )
    if len(opinions) == 0:
        return 0.0

    verdicts = generate_toxicity_verdicts(
        client=client,
        system_prompt=system_prompt,
        opinions=opinions,
    )
    return sum(verdict["verdict"] == "yes" for verdict in verdicts) / len(
        verdicts
    )


def calculate_rouge_scores(
    prediction: str,
    references: str | list[str],
    rouge_types: list[str],
    use_stemmer: bool = False,
) -> dict[str, float]:
    """
    Calculate ROUGE scores for a prediction given some set of references.

    Computes scores using 'rouge1', 'rouge2', 'rougeL', and 'rougeLsum' where `rouge1`
    is unigram-based scoring, `rouge2` is bigram-based scoring, `rougeL` is scoring
    based on sentences (i.e., splitting on "." and ignoring "\n"), and `rougeLsum`
    is scoring based on splitting the text using "\n".

    Parameters
    ----------
    prediction : str
        A prediction to score. Each prediction should be a string with tokens separated by spaces.
    references : list[str]
        A list of reference for a given prediction. Each reference should be a string with tokens separated by spaces.
    rouge_types : list[str]
        A list of rouge types to calculate.
    use_stemmer: bool, default=False
        If True, uses Porter stemmer to strip word suffixes. Defaults to False.
    """
    rouge = evaluate.load("rouge")

    metrics = rouge.compute(
        predictions=[prediction],
        references=[references],
        rouge_types=rouge_types,
        use_stemmer=use_stemmer,
        use_aggregator=False,  # aggregation gives us an average across all predictions, which isn't what we want
    )

    if not metrics:
        return dict()

    # find the max value for each prediction
    results = defaultdict(float)
    for type_ in rouge_types:
        results[type_] = max(metrics[type_][0], results[type_])
    return results


def calculate_sentence_bleu(
    prediction: str,
    references: list[str],
    weights: tuple[float, ...] | list[float],
) -> float:
    """
    Calculate sentence BLEU scores for a of prediction - ground truth pair.

    Parameters
    ----------
    prediction : str
        The prediction to score. Each prediction should be a string with tokens separated by spaces.
    references : list[str]
        A list of references for the prediction. Each reference should be a string with tokens separated by spaces.
    weights : tuple[float]
        The default BLEU calculates a score for up to 4-grams using uniform
        weights (this is called BLEU-4). To evaluate your translations with
        higher/lower order ngrams, use customized weights. Example: when accounting
        for up to 5-grams with uniform weights (this is called BLEU-5) use [1/5]*5
    """
    if len(weights) == 0:
        raise ValueError("At least one weight should be defined.")

    tokenizer = RegexpTokenizer(
        r"\w+|\$[\d]+|[^\s\.]+"
    )  # regex tokenizer that ignores periods

    tokenized_prediction = tokenizer.tokenize(prediction)
    tokenized_references = [tokenizer.tokenize(ref) for ref in references]

    # find the max value for each prediction
    return max(
        float(
            bleu_score.sentence_bleu(
                references=tokenized_references,
                hypothesis=tokenized_prediction,
                weights=weights,
            ),  # type: ignore
        ),
        0.0,
    )
