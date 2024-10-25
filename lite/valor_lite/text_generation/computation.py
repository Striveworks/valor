from collections import defaultdict
from enum import Enum

import evaluate
from nltk.tokenize import RegexpTokenizer
from nltk.translate import bleu_score


def calculate_rouge_scores(
    prediction: str,
    references: list[str],
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
        results[type_] = max(metrics[type_], results[type_])
    return results


def calculate_sentence_bleu(
    prediction: str,
    references: list[str],
    weights: list[float],
) -> float:
    """
    Calculate sentence BLEU scores for a of prediction - ground truth pair.

    Parameters
    ----------
    prediction : str
        The prediction to score. Each prediction should be a string with tokens separated by spaces.
    references : list[str]
        A list of references for the prediction. Each reference should be a string with tokens separated by spaces.
    weights : list[float]
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
