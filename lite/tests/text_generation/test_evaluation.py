import pytest
from valor_lite.text_generation import evaluation, metric


def test_EvaluationParameters():
    evaluation.EvaluationParameters()

    # Typical evaluation parameters for a text generation task
    evaluation.EvaluationParameters(
        metrics_to_return=[
            metric.MetricType.AnswerCorrectness,
            metric.MetricType.BLEU,
            metric.MetricType.ContextPrecision,
            metric.MetricType.ContextRecall,
        ],
        llm_api_params={
            "client": "openai",
            "api_key": "test_key",
            "data": {
                "seed": 2024,
                "model": "gpt-4o",
            },
        },
        metric_params={
            "BLEU": {
                "weights": [0.5, 0.3, 0.1, 0.1],
            },
        },
    )

    with pytest.raises(TypeError):
        evaluation.EvaluationParameters(metrics_to_return={"bad": "inputs"})  # type: ignore
