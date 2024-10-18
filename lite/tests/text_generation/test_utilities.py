import pytest
from valor_lite.text_generation.exceptions import InvalidLLMResponseError
from valor_lite.text_generation.metric import MetricType
from valor_lite.text_generation.utilities import (
    trim_and_load_json,
    validate_metric_parameters,
    validate_text_gen_metrics_to_return,
)


def test_trim_and_load_json():
    input = """this text should be trimmed
{
    "verdicts": [
        {
            "verdict": "yes"
        },
        {
            "verdict": "no",
            "reason": "The statement 'I also think puppies are cute.' is irrelevant to the question about who the cutest cat ever is."
        }
    ]
}"""
    expected = {
        "verdicts": [
            {"verdict": "yes"},
            {
                "verdict": "no",
                "reason": "The statement 'I also think puppies are cute.' is irrelevant to the question about who the cutest cat ever is.",
            },
        ]
    }

    assert trim_and_load_json(input) == expected

    # This function should add an } if none are present.
    input = """{"field": "value" """
    trim_and_load_json(input)

    input = """{
    "verdicts": [
        {
            "verdict": "yes"
        }
        {
            "verdict": "no",
            "reason": "The statement 'I also think puppies are cute.' is irrelevant to the question about who the cutest cat ever is."
        }
    ]
}"""

    # Missing a comma
    with pytest.raises(InvalidLLMResponseError):
        trim_and_load_json(input)

    input = """
    "sentence": "Hello, world!",
    "value": 3
}"""

    # Missing starting bracket
    with pytest.raises(InvalidLLMResponseError):
        trim_and_load_json(input)


def test_validate_text_gen_metrics_to_return():
    metrics_to_return = [
        MetricType.AnswerCorrectness,
        MetricType.AnswerRelevance,
        MetricType.Bias,
        MetricType.BLEU,
        MetricType.ContextPrecision,
        MetricType.ContextRecall,
        MetricType.ContextRelevance,
        MetricType.Faithfulness,
        MetricType.Hallucination,
        MetricType.ROUGE,
        MetricType.SummaryCoherence,
        MetricType.Toxicity,
    ]
    validate_text_gen_metrics_to_return(metrics_to_return=metrics_to_return)

    metrics_to_return = [
        "AnswerCorrectness",
        "AnswerRelevance",
        "Bias",
        "BLEU",
        "ContextPrecision",
        "ContextRecall",
        "ContextRelevance",
        "Faithfulness",
        "Hallucination",
        "ROUGE",
        "SummaryCoherence",
        "Toxicity",
    ]
    validate_text_gen_metrics_to_return(metrics_to_return=metrics_to_return)

    metrics_to_return = ["Accuracy", "F1"]
    with pytest.raises(ValueError):
        validate_text_gen_metrics_to_return(
            metrics_to_return=metrics_to_return
        )


def test_validate_metric_parameters():
    validate_metric_parameters(
        metrics_to_return=[MetricType.BLEU, MetricType.ROUGE],
        metric_params={
            "BLEU": {
                "weights": [0.25, 0.25, 0.25, 0.25],
            },
            "ROUGE": {
                "rouge_types": ["rouge1", "rouge2", "rougeL"],
                "use_stemmer": True,
            },
        },
    )

    # A metric in metric_params is not in metrics_to_return
    with pytest.raises(ValueError):
        validate_metric_parameters(
            metrics_to_return=[MetricType.ROUGE],
            metric_params={
                "BLEU": {
                    "weights": [0.25, 0.25, 0.25, 0.25],
                },
            },
        )

    # BLEU metric parameters should be a dictionary.
    with pytest.raises(TypeError):
        validate_metric_parameters(
            metrics_to_return=[MetricType.BLEU],
            metric_params={
                "BLEU": [0.25, 0.25, 0.25, 0.25],  # type: ignore - testing
            },
        )

    # BLEU weights should be a list
    with pytest.raises(TypeError):
        validate_metric_parameters(
            metrics_to_return=[MetricType.BLEU],
            metric_params={
                "BLEU": {
                    "weights": 1.0,  # type: ignore - testing
                },
            },
        )

    # BLEU weights should all be non-negative
    with pytest.raises(ValueError):
        validate_metric_parameters(
            metrics_to_return=[MetricType.BLEU],
            metric_params={
                "BLEU": {
                    "weights": [0.5, 0.25, -0.25, 0.5],
                },
            },
        )

    # BLEU weights should sum to 1
    with pytest.raises(ValueError):
        validate_metric_parameters(
            metrics_to_return=[MetricType.BLEU],
            metric_params={
                "BLEU": {
                    "weights": [0.5, 0.4, 0.3, 0.2],
                },
            },
        )

    # ROUGE metric parameters should be a dictionary.
    with pytest.raises(TypeError):
        validate_metric_parameters(
            metrics_to_return=[MetricType.ROUGE],
            metric_params={
                "ROUGE": ["use_stemmer"],  # type: ignore - testing
            },
        )

    # rouge_types should be a list or set
    with pytest.raises(TypeError):
        validate_metric_parameters(
            metrics_to_return=[MetricType.ROUGE],
            metric_params={
                "ROUGE": {
                    "rouge_types": "rouge1",  # type: ignore - testing
                },
            },
        )

    # rouge_types should be a subset of ROUGEType
    with pytest.raises(ValueError):
        validate_metric_parameters(
            metrics_to_return=[MetricType.ROUGE],
            metric_params={
                "ROUGE": {
                    "rouge_types": ["rouge1", "rouge2", "rouge3"],
                },
            },
        )

    # use_stemmer should be a boolean
    with pytest.raises(TypeError):
        validate_metric_parameters(
            metrics_to_return=[MetricType.ROUGE],
            metric_params={
                "ROUGE": {
                    "use_stemmer": "True",  # type: ignore - testing
                },
            },
        )
