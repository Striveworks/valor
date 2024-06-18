# %%
# required dependencies: absl-py, nltk, rouge_score, evaluate
import evaluate
import pytest
from nltk.translate import bleu_score

# %%

predictions = ["hello there", "general kenobi"]
references = [["hello", "there"], ["general kenobi", "general yoda"]]


def _calculate_rouge_scores(
    predictions: list[str],
    references: list[list[str]],
    rouge_types: list[str] = ["rouge1", "rouge2", "rougeL", "rougeLsum"],
    use_stemmer: bool = False,
) -> list[dict]:
    """
    Calculate ROUGE scores for a set of prediction-groundtruth pairs.

    Parameters
    ----------
    predictions: list[str]
        A list of predictions to score. Each prediction should be a string with tokens separated by spaces.
    references: list[str] | list[list[str]
        A list of reference for each prediction or a list of several references per prediction. Each reference should be a string with tokens separated by spaces.
    rouge_types: list[str]
        A list of rouge types to calculate. Defaults to ['rouge1', 'rouge2', 'rougeL', 'rougeLsum'].
    use_stemmer: bool
        If True, uses Porter stemmer to strip word suffixes. Defaults to False.
    """
    rouge = evaluate.load("rouge")

    metrics = rouge.compute(
        predictions=predictions,
        references=references,
        rouge_types=rouge_types,
        use_stemmer=use_stemmer,
        use_aggregator=False,  # TODO do we want to use non-aggregate metrics in some way?
    )

    assert metrics is not None  # handle type error

    return [
        {
            "prediction": predictions[i],
            "references": references[i],
            "value": {key: value[i] for key, value in metrics.items()},
        }
        for i in range(len(predictions))
    ]


_calculate_rouge_scores(predictions=predictions, references=references)

# %%


def _calculate_sentence_bleu(
    prediction: str,
    references: list[str],
    weights: list[float] = [0.25, 0.25, 0.25, 0.25],
) -> dict:
    """
    Calculate sentence BLEU scores for a set of prediction-groundtruth pairs.

    Parameters
    ----------
    prediction: list[str]
        The prediction to score. Each prediction should be a string with tokens separated by spaces.
    references: list[str] | list[list[str]
        A list of reference for each prediction or a list of several references per prediction. Each reference should be a string with tokens separated by spaces.
    weights: list[float]
        The default BLEU calculates a score for up to 4-grams using uniform
        weights (this is called BLEU-4). To evaluate your translations with
        higher/lower order ngrams, use customized weights. Example: when accounting
        for up to 5-grams with uniform weights (this is called BLEU-5) use [1/5]*5
    """
    if (
        not predictions
        or not references
        or not weights
        or isinstance(references, str)
        or len(weights) == 0
    ):
        raise ValueError(
            "Received incorrect inputs. predictions should be a string, references a list of strings, and weights a list/tuple of floats"
        )
    split_prediction = prediction.split()
    split_references = [reference.split() for reference in references]
    return {
        "prediction": split_prediction,
        "references": split_references,
        "value": bleu_score.sentence_bleu(
            references=references,
            hypothesis=prediction,
            weights=weights,
        ),
    }


examples = [
    {
        "prediction": "Mary loves Joe",
        "references": ["Mary loves Joe"],
        "weights": (1,),
        "expected_value": 1.0,
    },  # perfect match
    {
        "prediction": "Mary loves Joe",
        "references": ["Mary loves Joe"],
        "weights": [
            1,
        ],
        "expected_value": 1.0,
    },  # perfect match, weights are a list
    {
        "prediction": "MARY LOVES JOE",
        "references": ["Mary loves Joe"],
        "weights": (1,),
        "expected_value": 0.29,
    },  # perfect match, case sensitive
    {
        "prediction": "Mary loves Joe",
        "references": ["MARY LOVES JOE"],
        "weights": (1,),
        "expected_value": 0.29,
    },  # perfect match, case sensitive
    {
        "prediction": "Mary loves Joe",
        "references": ["MARY LOVES JOE"],
        "weights": (0, 1),
        "expected_value": 0.08,
    },  # perfect match, case sensitive, BLEU-2
    {
        "prediction": "Mary loves Joe",
        "references": ["Mary loves Joe"],
        "weights": (0, 1),
        "expected_value": 1.0,
    },  # BLEU-2
    {
        "prediction": "Mary loves Joe",
        "references": ["Mary loves Joe"],
        "weights": [0.25] * 4,
        "expected_value": 1.0,
    },  # BLEU-4
    {
        "prediction": "Mary loves Joe",
        "references": ["Mary loves Jane"],
        "weights": (1,),
        "expected_value": 0.86,
    },  # off by one
    {
        "prediction": "Mary loves Joe",
        "references": ["Mary loves Jane"],
        "weights": (0, 1),
        "expected_value": 0.79,
    },  # off by one BLEU-2
    {
        "prediction": "Mary loves Joe",
        "references": ["Mary loves Jane"],
        "weights": (0, 0, 1),
        "expected_value": 0.78,
    },  # off by one BLEU-3
    {
        "prediction": "Mary loves Joe",
        "references": ["Mary loves Jane"],
        "weights": (0, 0, 0, 1),
        "expected_value": 0.76,
    },  # off by one BLEU-4
    {
        "prediction": "mary loves joe",
        "references": ["MARY LOVES JOE"],
        "weights": (1,),
        "expected_value": 0.14,
    },  # different cases
    {
        "prediction": "mary loves joe",
        "references": ["MARY LOVES JOE"],
        "weights": [0, 1],
        "expected_value": 0,
    },  # different cases BLEU-10
    {
        "prediction": "mary loves joe",
        "references": ["MARY LOVES JOE"],
        "weights": [0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
        "expected_value": 0,
    },  # different cases BLEU-10
]

expected_errors = [
    {
        "prediction": "Mary loves Joe",
        "references": "Mary loves Joe",
        "weights": (1,),
        "error": ValueError,
    },  # references isn't a list
    {
        "prediction": None,
        "references": "Mary loves Joe",
        "weights": (1,),
        "error": ValueError,
    },  # prediction shouldn't be None
    {
        "prediction": "Mary loves Joe",
        "references": None,
        "weights": (1,),
        "error": ValueError,
    },  # references shouldn't be None
    {
        "prediction": "Mary loves Joe",
        "references": ["Mary loves Joe"],
        "weights": None,
        "error": ValueError,
    },  # weights shouldn't be None
]

for example in examples:
    output = _calculate_sentence_bleu(
        prediction=example["prediction"],
        references=example["references"],
        weights=example["weights"],
    )
    assert round(output["value"], 2) == example["expected_value"]

for example in expected_errors:
    print(example)
    with pytest.raises(example["error"]):
        _calculate_sentence_bleu(
            prediction=example["prediction"],
            references=example["references"],
            weights=example["weights"],
        )

# %%

# func = bleu_score.sentence_bleu
func = bleu_score.sentence_bleu
ans1 = func(
    ["The candidate has no alignment to any of the references".split()],
    "John loves Mary".split(),
    (1,),
)

ans2 = func(
    ["John loves Mary".split()],
    "John loves Mary".split(),
    (1,),
)

ans3 = func(
    ["John loves Mary".split()],
    "John loves Paige".split(),
    (1,),
)

ans4 = func(
    ["John loves Mary".split()],
    "John loves Mary".split(),
    (0, 1),
)

ans5 = func(
    ["John loves Mary".split()],
    "John loves Mary".split(),
)

print(ans1, ans2, ans3, ans4, ans5)


# %%
