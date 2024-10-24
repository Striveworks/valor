from collections import defaultdict
from enum import Enum

import evaluate
from nltk.tokenize import RegexpTokenizer
from nltk.translate import bleu_score
from valor_lite.text_generation.metric import ROUGEType


class ROUGEType(str, Enum):
    ROUGE1 = "rouge1"
    ROUGE2 = "rouge2"
    ROUGEL = "rougeL"
    ROUGELSUM = "rougeLsum"


def calculate_rouge_scores(
    prediction: str,
    references: list[str],
    rouge_types: list[ROUGEType] | None = None,
    use_stemmer: bool = False,
) -> dict[str, float]:
    """
    Calculate ROUGE scores for a prediction (or list of predictions) given some set of references.

    Parameters
    ----------
    prediction: str
        A prediction to score. Each prediction should be a string with tokens separated by spaces.
    references: list[list[str]]
        A list of reference for a given prediction. Each reference should be a string with tokens separated by spaces.
    rouge_types: list[ROUGEType], default=all
        A list of rouge types to calculate. Defaults to ['rouge1', 'rouge2', 'rougeL', 'rougeLsum'], where `rouge1` is unigram-based scoring, `rouge2` is bigram-based scoring, `rougeL` is scoring based on sentences (i.e., splitting on "." and ignoring "\n"), and `rougeLsum` is scoring based on splitting the text using "\n".
    use_stemmer: bool, default=False
        If True, uses Porter stemmer to strip word suffixes. Defaults to False.
    """
    if rouge_types is None or len(rouge_types) == 0:
        return {}

    rouge = evaluate.load("rouge")
    rouge_types = [
        ROUGEType.ROUGE1,
        ROUGEType.ROUGE2,
        ROUGEType.ROUGEL,
        ROUGEType.ROUGELSUM,
    ]

    metrics = rouge.compute(
        predictions=[prediction],
        references=[references],
        rouge_types=rouge_types,
        use_stemmer=use_stemmer,
        use_aggregator=False,  # aggregation gives us an average across all predictions, which isn't what we want
    )

    if not metrics:
        raise ValueError("No ROUGE metrics were returned.")

    # find the max value for each prediction
    output = defaultdict(float)
    for type_ in rouge_types:
        output[type_] = max(metrics[type_], output[type_])
    return output


def calculate_sentence_bleu(
    prediction: str,
    references: list[str],
    weights: list[float],
) -> dict[str, float]:
    """
    Calculate sentence BLEU scores for a set of prediction - ground truth pairs.

    Parameters
    ----------
    prediction: str
        The predictions to score. Each prediction should be a string with tokens separated by spaces.
    references: list[str]
        A list of reference for each prediction or a list of several references per prediction. Each reference should be a string with tokens separated by spaces.
    weights: list[float]
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
