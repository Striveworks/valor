import pytest
from valor_lite.text_generation.exceptions import InvalidLLMResponseError
from valor_lite.text_generation.metric import MetricType
from valor_lite.text_generation.utilities import (
    trim_and_load_json,
    validate_llm_api_params,
    validate_metric_params,
    validate_metrics_to_return,
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


def test_validate_metrics_to_return():
    # Can pass in metrics as MetricType or as strings
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
    validate_metrics_to_return(metrics_to_return=metrics_to_return)

    # Can specify any collection of metrics
    metrics_to_return = [
        MetricType.Hallucination,
    ]
    validate_metrics_to_return(metrics_to_return=metrics_to_return)

    # Can specify metrics as strings
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
    validate_metrics_to_return(metrics_to_return=metrics_to_return)

    # metrics_to_return should be a list
    metrics_to_return = True
    with pytest.raises(TypeError):
        validate_metrics_to_return(metrics_to_return=metrics_to_return)  # type: ignore - testing

    # metrics_to_return should be a list of strings or MetricType
    metrics_to_return = [True]
    with pytest.raises(TypeError):
        validate_metrics_to_return(metrics_to_return=metrics_to_return)  # type: ignore - testing

    # Accuracy and F1 are classification metrics, not text generation metrics
    metrics_to_return = ["Accuracy", "F1"]
    with pytest.raises(ValueError):
        validate_metrics_to_return(metrics_to_return=metrics_to_return)

    # Should not have duplicate metrics
    metrics_to_return = ["Hallucination", "Hallucination"]
    with pytest.raises(ValueError):
        validate_metrics_to_return(metrics_to_return=metrics_to_return)


def test_validate_llm_api_params():
    # Valid call with openai
    validate_llm_api_params(
        llm_api_params={
            "client": "openai",
            "data": {
                "seed": 2024,
                "model": "gpt-4o",
                "retries": 0,
            },
        },
    )

    # None is valid as some text generation metrics don't require llm-guidance
    validate_llm_api_params(
        llm_api_params=None,
    )

    # llm_api_params should be a dictionary
    with pytest.raises(TypeError):
        validate_llm_api_params(
            llm_api_params="openai",  # type: ignore - testing
        )

    # llm api keys should be strings
    with pytest.raises(TypeError):
        validate_llm_api_params(
            llm_api_params={
                1: {  # type: ignore - testing
                    "weights": [0.25, 0.25, 0.25, 0.25],
                },
            },
        )

    # llm api values should be strings, integers or dictionaries
    with pytest.raises(TypeError):
        validate_llm_api_params(
            llm_api_params={
                "client": 1.0,  # type: ignore - testing
            },
        )

    # Need to specify the client or api_url (api_url has not been implemented)
    with pytest.raises(ValueError):
        validate_llm_api_params(
            llm_api_params={
                "data": {
                    "seed": 2024,
                    "model": "gpt-4o",
                },
            },
        )

    # Cannot specify both a client and api_url.
    with pytest.raises(ValueError):
        validate_llm_api_params(
            llm_api_params={
                "client": "openai",
                "api_url": "openai.com",
                "data": {
                    "seed": 2024,
                    "model": "gpt-4o",
                },
            },
        )


def test_validate_metric_params():
    # None is valid if using default metric parameters
    validate_metric_params(
        metrics_to_return=["BLEU"],
        metric_params=None,
    )

    # Valid metric parameters for BLEU and ROUGE
    validate_metric_params(
        metrics_to_return=["BLEU", "ROUGE"],
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

    # metric_params should be a dictionary
    with pytest.raises(TypeError):
        validate_metric_params(
            metrics_to_return=["BLEU"],
            metric_params=[0.25, 0.25, 0.25, 0.25],  # type: ignore - testing
        )

    # metric_params keys should be strings
    with pytest.raises(TypeError):
        validate_metric_params(
            metrics_to_return=[MetricType.BLEU],
            metric_params={
                1: {  # type: ignore - testing
                    "weights": [0.25, 0.25, 0.25, 0.25],
                },
            },
        )

    # metric_params values should be a dictionary
    with pytest.raises(TypeError):
        validate_metric_params(
            metrics_to_return=["BLEU"],
            metric_params={
                "BLEU": [0.25, 0.25, 0.25, 0.25],  # type: ignore - testing
            },
        )

    # A metric in metric_params is not in metrics_to_return
    with pytest.raises(ValueError):
        validate_metric_params(
            metrics_to_return=["ROUGE"],
            metric_params={
                "BLEU": {
                    "weights": [0.25, 0.25, 0.25, 0.25],
                },
            },
        )

    # BLEU weights should be a list
    with pytest.raises(TypeError):
        validate_metric_params(
            metrics_to_return=["BLEU"],
            metric_params={
                "BLEU": {
                    "weights": 1.0,  # type: ignore - testing
                },
            },
        )

    # BLEU weights should all be non-negative
    with pytest.raises(ValueError):
        validate_metric_params(
            metrics_to_return=["BLEU"],
            metric_params={
                "BLEU": {
                    "weights": [0.5, 0.25, -0.25, 0.5],
                },
            },
        )

    # BLEU weights should sum to 1
    with pytest.raises(ValueError):
        validate_metric_params(
            metrics_to_return=["BLEU"],
            metric_params={
                "BLEU": {
                    "weights": [0.5, 0.4, 0.3, 0.2],
                },
            },
        )

    # rouge_types should be a list or set
    with pytest.raises(TypeError):
        validate_metric_params(
            metrics_to_return=["ROUGE"],
            metric_params={
                "ROUGE": {
                    "rouge_types": "rouge1",  # type: ignore - testing
                },
            },
        )

    # rouge_types should be a subset of ROUGEType
    with pytest.raises(ValueError):
        validate_metric_params(
            metrics_to_return=["ROUGE"],
            metric_params={
                "ROUGE": {
                    "rouge_types": ["rouge1", "rouge2", "rouge3"],
                },
            },
        )

    # use_stemmer should be a boolean
    with pytest.raises(TypeError):
        validate_metric_params(
            metrics_to_return=["ROUGE"],
            metric_params={
                "ROUGE": {
                    "use_stemmer": "True",  # type: ignore - testing
                },
            },
        )
