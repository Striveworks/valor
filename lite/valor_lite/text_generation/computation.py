from collections import defaultdict
from typing import Sequence

import evaluate
from nltk.tokenize import RegexpTokenizer
from nltk.translate import bleu_score
from valor_lite.text_generation.metric import ROUGEType


def calculate_rouge_scores(
    predictions: list[str],
    references: list[list[str]],
    use_stemmer: bool = False,
) -> list[dict[str, dict[str, float]]]:
    """
    Calculate ROUGE scores for a prediction (or list of predictions) given some set of references.

    Parameters
    ----------
    prediction: list[str]
        A list of predictions to score. Each prediction should be a string with tokens separated by spaces.
    references: list[list[str]]
        A list of reference for a given prediction. Each reference should be a string with tokens separated by spaces.
    rouge_types: list[ROUGEType]
        A list of rouge types to calculate. Defaults to ['rouge1', 'rouge2', 'rougeL', 'rougeLsum'], where `rouge1` is unigram-based scoring, `rouge2` is bigram-based scoring, `rougeL` is scoring based on sentences (i.e., splitting on "." and ignoring "\n"), and `rougeLsum` is scoring based on splitting the text using "\n".
    use_stemmer: bool
        If True, uses Porter stemmer to strip word suffixes. Defaults to False.
    """
    rouge = evaluate.load("rouge")
    rouge_types = [
        ROUGEType.ROUGE1,
        ROUGEType.ROUGE2,
        ROUGEType.ROUGEL,
        ROUGEType.ROUGELSUM,
    ]

    metrics = rouge.compute(
        predictions=predictions,
        references=references,
        rouge_types=rouge_types,
        use_stemmer=use_stemmer,
        use_aggregator=False,  # aggregation gives us an average across all predictions, which isn't what we want
    )

    if not metrics:
        raise ValueError("No ROUGE metrics were returned.")

    # find the max value for each prediction
    output = defaultdict(lambda: defaultdict(float))
    for i, prediction in enumerate(predictions):
        for type_ in rouge_types:
            output[prediction][type_] = max(
                metrics[type_][i], output[prediction][type_]
            )

    return [
        {"prediction": prediction, "value": dict(value)}
        for prediction, value in output.items()
    ]


def calculate_sentence_bleu(
    predictions: list[str],
    references: list[list[str]],
    weights: list[float],
) -> list[dict[str, float]]:
    """
    Calculate sentence BLEU scores for a set of prediction - ground truth pairs.

    Parameters
    ----------
    predictions: list[str]
        The predictions to score. Each prediction should be a string with tokens separated by spaces.
    references: list[list[str]
        A list of reference for each prediction or a list of several references per prediction. Each reference should be a string with tokens separated by spaces.
    weights: list[float]
        The default BLEU calculates a score for up to 4-grams using uniform
        weights (this is called BLEU-4). To evaluate your translations with
        higher/lower order ngrams, use customized weights. Example: when accounting
        for up to 5-grams with uniform weights (this is called BLEU-5) use [1/5]*5
    """
    if len(weights) == 0:
        raise ValueError("At least one weight should be defined.")

    output = defaultdict(float)
    tokenizer = RegexpTokenizer(
        r"\w+|\$[\d]+|[^\s\.]+"
    )  # regex tokenizer that ignores periods

    for pred, refs in zip(predictions, references):

        tokenized_prediction = tokenizer.tokenize(pred)
        tokenized_references = [tokenizer.tokenize(ref) for ref in refs]

        # find the max value for each prediction
        output[pred] = max(
            float(
                bleu_score.sentence_bleu(
                    references=tokenized_references,
                    hypothesis=tokenized_prediction,
                    weights=weights,
                ),  # type: ignore
            ),
            output[pred],
        )

    return [
        {"prediction": key, "value": value} for key, value in output.items()
    ]
