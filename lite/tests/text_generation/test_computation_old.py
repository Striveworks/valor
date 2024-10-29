from typing import Callable
from unittest.mock import patch

import pytest
from valor_lite.text_generation import Context, Query
from valor_lite.text_generation.computation import (
    _calculate_rouge_scores,
    _calculate_sentence_bleu,
    _setup_llm_client,
    evaluate_text_generation,
)
from valor_lite.text_generation.exceptions import InvalidLLMResponseError


def mocked_connection(self):
    pass



def mocked_compute_answer_correctness(
    rag_queries: list[str],
    rag_predictions: list[str],
    rag_references: list[str],
) -> Callable:
    def compute_answer_correctness(
        self,
        query: str,
        prediction: str,
        groundtruth_list: list[str],
    ):
        ret_dict = {
            (
                rag_queries[0],
                rag_predictions[0],
                tuple([rag_references[0], "some other text", "some final text"]),
            ): 0.8,
            (
                rag_queries[1],
                rag_predictions[1],
                tuple([rag_references[1], "some other text", "some final text"]),
            ): 1.0,
            (
                rag_queries[2],
                rag_predictions[2],
                tuple([rag_references[2], "some other text", "some final text"]),
            ): 0.0,
        }
        if (query, prediction, tuple(groundtruth_list)) in ret_dict:
            return ret_dict[(query, prediction, tuple(groundtruth_list))]
        return 0.0
    return compute_answer_correctness


def mocked_compute_answer_relevance(
    rag_queries: list[str],
    rag_predictions: list[str],
) -> Callable:
    def compute_answer_relevance(
        self,
        query: str,
        text: str,
    ):
        ret_dict = {
            (rag_queries[0], rag_predictions[0]): 0.6666666666666666,
            (rag_queries[1], rag_predictions[1]): 0.2,
            (rag_queries[2], rag_predictions[2]): 0.2,
        }
        return ret_dict[(query, text)]
    return compute_answer_relevance


@pytest.fixture
def mocked_compute_bias(
    rag_predictions: list[str],
    content_gen_predictions: list[str],
):
    def compute_bias(
        self,
        text: str,
    ):
        ret_dict = {
            rag_predictions[0]: 0.0,
            rag_predictions[1]: 0.0,
            rag_predictions[2]: 0.0,
            content_gen_predictions[0]: 0.2,
            content_gen_predictions[1]: 0.0,
            content_gen_predictions[2]: 0.0,
        }
        return ret_dict[text]
    return compute_bias


def mocked_compute_context_precision(
    rag_queries: list[str],
    rag_context: list[list[str]],
    rag_references: list[str],
):
    def compute_context_precision(
        self,
        query: str,
        ordered_context_list: list[str],
        groundtruth_list: list[str],
    ):
        ret_dict = {
            (
                rag_queries[0],
                tuple(rag_context[0]),
                tuple([rag_references[0], "some other text", "some final text"]),
            ): 1.0,
            (
                rag_queries[1],
                tuple(rag_context[1]),
                tuple([rag_references[1], "some other text", "some final text"]),
            ): 1.0,
            (
                rag_queries[2],
                tuple(rag_context[2]),
                tuple([rag_references[2], "some other text", "some final text"]),
            ): 1.0,
        }
        if (
            query,
            tuple(ordered_context_list),
            tuple(groundtruth_list),
        ) in ret_dict:
            return ret_dict[
                (query, tuple(ordered_context_list), tuple(groundtruth_list))
            ]
        return 0.0
    return compute_context_precision


def mocked_context_recall(
    rag_context: list[list[str]],
    rag_references: list[str],
) -> Callable:
    def compute_context_recall(
        self,
        context_list: list[str],
        groundtruth_list: list[str],
    ):
        ret_dict = {
            (
                tuple(rag_context[0]),
                tuple([rag_references[0], "some other text", "some final text"]),
            ): 0.8,
            (
                tuple(rag_context[1]),
                tuple([rag_references[1], "some other text", "some final text"]),
            ): 0.5,
            (
                tuple(rag_context[2]),
                tuple([rag_references[2], "some other text", "some final text"]),
            ): 0.2,
        }
        if (tuple(context_list), tuple(groundtruth_list)) in ret_dict:
            return ret_dict[(tuple(context_list), tuple(groundtruth_list))]
        return 0.0
    return compute_context_recall


def mocked_context_relevance(
    rag_queries: list[str],
    rag_context: list[list[str]],
) -> Callable:
    def compute_context_relevance(
        self,
        query: str,
        context_list: list[str],
    ):
        ret_dict = {
            (rag_queries[0], tuple(rag_context[0])): 0.75,
            (rag_queries[1], tuple(rag_context[1])): 1.0,
            (rag_queries[2], tuple(rag_context[2])): 0.25,
        }
        return ret_dict[(query, tuple(context_list))]
    return compute_context_relevance


def mocked_faithfulness(
    rag_predictions: list[str],
    rag_context: list[list[str]],
):
    def compute_faithfulness(
        self,
        text: str,
        context_list: list[str],
    ):
        ret_dict = {
            (rag_predictions[0], tuple(rag_context[0])): 0.4,
            (rag_predictions[1], tuple(rag_context[1])): 0.55,
            (rag_predictions[2], tuple(rag_context[2])): 0.6666666666666666,
        }
        return ret_dict[(text, tuple(context_list))]
    return compute_faithfulness


def mocked_hallucination(
    self,
    text: str,
    context_list: list[str],
):
    def compute_hallucination
    ret_dict = {
        (rag_predictions[0], tuple(rag_context[0])): 0.0,
        (rag_predictions[1], tuple(rag_context[1])): 0.0,
        (rag_predictions[2], tuple(rag_context[2])): 0.25,
    }
    return ret_dict[(text, tuple(context_list))]


def mocked_summary_coherence(
    self,
    text: str,
    summary: str,
):
    ret_dict = {
        (SUMMARIZATION_TEXTS[0], SUMMARIZATION_PREDICTIONS[0]): 4,
        (SUMMARIZATION_TEXTS[1], SUMMARIZATION_PREDICTIONS[1]): 5,
    }
    return ret_dict[(text, summary)]


def mocked_toxicity(
    self,
    text: str,
):
    ret_dict = {
        rag_predictions[0]: 0.0,
        rag_predictions[1]: 0.0,
        rag_predictions[2]: 0.0,
        content_gen_predictions[0]: 0.4,
        content_gen_predictions[1]: 0.0,
        content_gen_predictions[2]: 0.0,
    }
    return ret_dict[text]


def mocked_invalid_llm_response(*args, **kwargs):
    raise InvalidLLMResponseError


def mocked_compute_rouge_none(*args, **kwargs):
    """
    Dummy docstring
    """
    return None





@patch(
    "valor_lite.text_generation.llm_client.WrappedOpenAIClient.connect",
    mocked_connection,
)
@patch(
    "valor_lite.text_generation.llm_client.WrappedMistralAIClient.connect",
    mocked_connection,
)
@patch(
    "valor_lite.text_generation.llm_client.LLMClient.answer_correctness",
    mocked_answer_correctness,
)
@patch(
    "valor_lite.text_generation.llm_client.LLMClient.answer_relevance",
    mocked_answer_relevance,
)
@patch(
    "valor_lite.text_generation.llm_client.LLMClient.bias",
    mocked_bias,
)
@patch(
    "valor_lite.text_generation.llm_client.LLMClient.context_precision",
    mocked_context_precision,
)
@patch(
    "valor_lite.text_generation.llm_client.LLMClient.context_recall",
    mocked_context_recall,
)
@patch(
    "valor_lite.text_generation.llm_client.LLMClient.context_relevance",
    mocked_context_relevance,
)
@patch(
    "valor_lite.text_generation.llm_client.LLMClient.faithfulness",
    mocked_faithfulness,
)
@patch(
    "valor_lite.text_generation.llm_client.LLMClient.hallucination",
    mocked_hallucination,
)
@patch(
    "valor_lite.text_generation.llm_client.LLMClient.toxicity",
    mocked_toxicity,
)
def test_evaluate_text_generation_rag(
    rag_gts: list[annotation.GroundTruth],
    rag_preds: list[annotation.Prediction],
):
    """
    Tests the evaluate_text_generation function for RAG.
    """
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
        "Toxicity",
    ]

    metrics = evaluate_text_generation(
        predictions=rag_preds,
        groundtruths=rag_gts,
        metrics_to_return=metrics_to_return,
        llm_api_params={
            "client": "openai",
            "data": {
                "seed": 2024,
                "model": "gpt-4o",
            },
        },
        metric_params={
            "ROUGE": {
                "use_stemmer": False,
            },
        },
    )

    expected_values = {
        "uid0": {
            "AnswerCorrectness": 0.8,
            "AnswerRelevance": 0.6666666666666666,
            "Bias": 0.0,
            "BLEU": 0.3502270395690205,
            "ContextPrecision": 1.0,
            "ContextRecall": 0.8,
            "ContextRelevance": 0.75,
            "Faithfulness": 0.4,
            "Hallucination": 0.0,
            "ROUGE": {
                "rouge1": 0.5925925925925926,
                "rouge2": 0.5569620253164557,
                "rougeL": 0.5925925925925926,
                "rougeLsum": 0.5925925925925926,
            },
            "Toxicity": 0.0,
        },
        "uid1": {
            "AnswerCorrectness": 1.0,
            "AnswerRelevance": 0.2,
            "Bias": 0.0,
            "BLEU": 1.0,
            "ContextPrecision": 1.0,
            "ContextRecall": 0.5,
            "ContextRelevance": 1.0,
            "Faithfulness": 0.55,
            "Hallucination": 0.0,
            "ROUGE": {
                "rouge1": 1.0,
                "rouge2": 1.0,
                "rougeL": 1.0,
                "rougeLsum": 1.0,
            },
            "Toxicity": 0.0,
        },
        "uid2": {
            "AnswerCorrectness": 0.0,
            "AnswerRelevance": 0.2,
            "Bias": 0.0,
            "BLEU": 0.05434912989707719,
            "ContextPrecision": 1.0,
            "ContextRecall": 0.2,
            "ContextRelevance": 0.25,
            "Faithfulness": 0.6666666666666666,
            "Hallucination": 0.25,
            "ROUGE": {
                "rouge1": 0.18666666666666668,
                "rouge2": 0.0821917808219178,
                "rougeL": 0.18666666666666668,
                "rougeLsum": 0.18666666666666668,
            },
            "Toxicity": 0.0,
        },
    }

    assert metrics
    assert len(metrics) == len(metrics_to_return) * len(expected_values)
    for metric in metrics:
        assert isinstance(metric["parameters"], dict)
        assert isinstance(metric["parameters"]["datum_uid"], str)
        assert (
            expected_values[metric["parameters"]["datum_uid"]].get(
                metric["type"]
            )
            == metric["value"]
        )

    # Test that mistral is accepted as a valid client.
    _ = evaluate_text_generation(
        predictions=rag_preds,
        groundtruths=rag_gts,
        metrics_to_return=["AnswerRelevance", "BLEU"],
        llm_api_params={
            "client": "mistral",
            "data": {
                "model": "mistral-small-latest",
            },
        },
        metric_params={
            "BLEU": {
                "weights": [0.5, 0.25, 0.25, 0],
            },
        },
    )

    # Test that manually specifying the api key works.
    _ = evaluate_text_generation(
        predictions=rag_preds,
        groundtruths=rag_gts,
        metrics_to_return=["ContextRelevance"],
        llm_api_params={
            "client": "openai",
            "api_key": "test_key",
            "data": {
                "seed": 2024,
                "model": "gpt-4o",
            },
        },
    )

    # Test the mock client.
    _ = evaluate_text_generation(
        predictions=rag_preds,
        groundtruths=rag_gts,
        metrics_to_return=metrics_to_return,
        llm_api_params={
            "client": "mock",
            "data": {
                "model": "some model",
            },
        },
        metric_params={
            "BLEU": {
                "weights": [0.5, 0.25, 0.25, 0],
            },
            "ROUGE": {
                "rouge_types": [
                    ROUGEType.ROUGE1,
                    ROUGEType.ROUGE2,
                    ROUGEType.ROUGEL,
                ],
                "use_stemmer": True,
            },
        },
    )

    # If an llm-guided metric is requested, then llm_api_params must be specified.
    with pytest.raises(ValueError):
        _ = evaluate_text_generation(
            predictions=rag_preds,
            groundtruths=rag_gts,
            metrics_to_return=metrics_to_return,
        )


@patch(
    "valor_lite.text_generation.llm_client.WrappedOpenAIClient.connect",
    mocked_connection,
)
@patch(
    "valor_lite.text_generation.llm_client.LLMClient.bias",
    mocked_bias,
)
@patch(
    "valor_lite.text_generation.llm_client.LLMClient.toxicity",
    mocked_toxicity,
)
def test_evaluate_text_generation_content_gen(
    content_gen_gts: list[annotation.GroundTruth],
    content_gen_preds: list[annotation.Prediction],
):
    """
    Tests the evaluate_text_generation function for content generation.
    """
    metrics_to_return = [
        "Bias",
        "Toxicity",
    ]

    # default request
    metrics = evaluate_text_generation(
        predictions=content_gen_preds,
        groundtruths=content_gen_gts,
        metrics_to_return=metrics_to_return,
        llm_api_params={
            "client": "openai",
            "data": {
                "seed": 2024,
                "model": "gpt-4o",
            },
        },
    )

    expected_values = {
        "uid0": {
            "Bias": 0.2,
            "Toxicity": 0.4,
        },
        "uid1": {
            "Bias": 0.0,
            "Toxicity": 0.0,
        },
        "uid2": {
            "Bias": 0.0,
            "Toxicity": 0.0,
        },
    }

    assert metrics
    assert len(metrics) == len(metrics_to_return) * len(expected_values)
    for metric in metrics:
        assert isinstance(metric["parameters"], dict)
        assert (
            expected_values[metric["parameters"]["datum_uid"]].get(
                metric["type"]
            )
            == metric["value"]
        )


@patch(
    "valor_lite.text_generation.llm_client.WrappedOpenAIClient.connect",
    mocked_connection,
)
@patch(
    "valor_lite.text_generation.llm_client.LLMClient.summary_coherence",
    mocked_summary_coherence,
)
def test_evaluate_text_generation_summarization(
    summarization_gts: list[annotation.GroundTruth],
    summarization_preds: list[annotation.Prediction],
):
    """
    Tests the evaluate_text_generation function for summarization.
    """
    metrics_to_return = [
        "SummaryCoherence",
    ]

    # default request
    metrics = evaluate_text_generation(
        predictions=summarization_preds,
        groundtruths=summarization_gts,
        metrics_to_return=metrics_to_return,
        llm_api_params={
            "client": "openai",
            "data": {
                "seed": 2024,
                "model": "gpt-4o",
            },
        },
    )

    expected_values = {
        "uid0": {
            "SummaryCoherence": 4,
        },
        "uid1": {
            "SummaryCoherence": 5,
        },
    }

    assert metrics
    assert len(metrics) == len(metrics_to_return) * len(expected_values)
    for metric in metrics:
        assert isinstance(metric["parameters"], dict)
        assert (
            expected_values[metric["parameters"]["datum_uid"]].get(
                metric["type"]
            )
            == metric["value"]
        )


@patch(
    "valor_lite.text_generation.llm_client.WrappedOpenAIClient.connect",
    mocked_connection,
)
@patch(
    "valor_lite.text_generation.llm_client.LLMClient.answer_correctness",
    mocked_invalid_llm_response,
)
@patch(
    "valor_lite.text_generation.llm_client.LLMClient.answer_relevance",
    mocked_invalid_llm_response,
)
@patch(
    "valor_lite.text_generation.llm_client.LLMClient.bias",
    mocked_invalid_llm_response,
)
@patch(
    "valor_lite.text_generation.llm_client.LLMClient.context_precision",
    mocked_invalid_llm_response,
)
@patch(
    "valor_lite.text_generation.llm_client.LLMClient.context_recall",
    mocked_invalid_llm_response,
)
@patch(
    "valor_lite.text_generation.llm_client.LLMClient.context_relevance",
    mocked_invalid_llm_response,
)
@patch(
    "valor_lite.text_generation.llm_client.LLMClient.faithfulness",
    mocked_invalid_llm_response,
)
@patch(
    "valor_lite.text_generation.llm_client.LLMClient.hallucination",
    mocked_invalid_llm_response,
)
@patch(
    "valor_lite.text_generation.llm_client.LLMClient.summary_coherence",
    mocked_invalid_llm_response,
)
@patch(
    "valor_lite.text_generation.llm_client.LLMClient.toxicity",
    mocked_invalid_llm_response,
)
def test_evaluate_text_generation_invalid_llm_response(
    rag_gts: list[annotation.GroundTruth],
    rag_preds: list[annotation.Prediction],
    summarization_gts: list[annotation.GroundTruth],
    summarization_preds: list[annotation.Prediction],
):
    llm_api_params = {
        "client": "openai",
        "data": {
            "seed": 2024,
            "model": "gpt-4o",
        },
    }

    metrics = evaluate_text_generation(
        predictions=rag_preds,
        groundtruths=rag_gts,
        metrics_to_return=[
            "AnswerCorrectness",
            "AnswerRelevance",
            "Bias",
            "ContextPrecision",
            "ContextRecall",
            "ContextRelevance",
            "Faithfulness",
            "Hallucination",
            "Toxicity",
        ],
        llm_api_params=llm_api_params,
    )
    for metric in metrics:
        assert (
            metric["status"] == "error"
        ), f"Unexpected status: {metric['status']}, for metric: {metric}"

    metrics = evaluate_text_generation(
        predictions=summarization_preds,
        groundtruths=summarization_gts,
        metrics_to_return=[
            "SummaryCoherence",
        ],
        llm_api_params=llm_api_params,
    )
    for metric in metrics:
        assert (
            metric["status"] == "error"
        ), f"Unexpected status: {metric['status']}, for metric: {metric}"


def test__calculate_rouge_scores():
    examples = [
        {
            "prediction": "Mary loves Joe",
            "references": [
                "Mary loves Joe",
            ],
            "rouge1": 1.0,
            "rouge2": 1.0,
            "rougeL": 1.0,
            "rougeLsum": 1.0,
        },  # perfect match
        {
            "prediction": "MARY LOVES JOE",
            "references": ["Mary loves Joe"],
            "rouge1": 1.0,
            "rouge2": 1.0,
            "rougeL": 1.0,
            "rougeLsum": 1.0,
        },  # perfect match, case sensitive
        {
            "prediction": "Mary loves Joe",
            "references": ["MARY LOVES JOE"],
            "rouge1": 1.0,
            "rouge2": 1.0,
            "rougeL": 1.0,
            "rougeLsum": 1.0,
        },  # perfect match, case sensitive
        {
            "prediction": "Mary loves Joe",
            "references": ["Mary loves Jane"],
            "rouge1": 0.67,
            "rouge2": 0.5,
            "rougeL": 0.67,
            "rougeLsum": 0.67,
        },  # off by one
        {
            "prediction": "flipping the roaring white dolphin",
            "references": ["flip the roaring white dolphin"],
            "rouge1": 0.8,
            "rouge2": 0.75,
            "rougeL": 0.8,
            "rougeLsum": 0.8,
            "use_stemmer": False,
        },  # incorrect match without stemming
        {
            "prediction": "flipping the roaring white dolphin",
            "references": ["flip the roaring white dolphin"],
            "rouge1": 1,
            "rouge2": 1,
            "rougeL": 1,
            "rougeLsum": 1,
            "use_stemmer": True,
        },  # correct match with stemming
        {
            "prediction": "flipping the roaring white dolphin",
            "references": [
                "some random sentence",
                "some other sentence",
                "some final reference",
                "flip the roaring white dolphin",
            ],
            "rouge1": 1,
            "rouge2": 1,
            "rougeL": 1,
            "rougeLsum": 1,
            "use_stemmer": True,
        },  # test multiple references
    ]

    multiple_prediction_examples = [
        {
            "prediction": ["Mary loves Joe", "Mary loves Jack"],
            "references": [
                ["Mary loves June", "some other sentence"],
                ["some other sentence", "the big fox hunts rabbits"],
            ],
            "expected_value": [
                {
                    "prediction": "Mary loves Joe",
                    "value": {
                        "rouge1": 0.6666666666666666,
                        "rouge2": 0.5,
                        "rougeL": 0.6666666666666666,
                        "rougeLsum": 0.6666666666666666,
                    },
                },
                {
                    "prediction": "Mary loves Jack",
                    "value": {
                        "rouge1": 0.0,
                        "rouge2": 0.0,
                        "rougeL": 0.0,
                        "rougeLsum": 0.0,
                    },
                },
            ],
        },  # off by one
        {
            "prediction": [
                "flipping the roaring white dolphin",
                "Mary loves Joe",
            ],
            "references": [
                [
                    "some random sentence",
                    "some other sentence",
                    "some final reference",
                    "flip the roaring white dolphin",
                ],
                ["beep bop", "Mary loves June"],
            ],
            "expected_value": [
                {
                    "prediction": "flipping the roaring white dolphin",
                    "value": {
                        "rouge1": 1.0,
                        "rouge2": 1.0,
                        "rougeL": 1.0,
                        "rougeLsum": 1.0,
                    },
                },
                {
                    "prediction": "Mary loves Joe",
                    "value": {
                        "rouge1": 0.6666666666666666,
                        "rouge2": 0.5,
                        "rougeL": 0.6666666666666666,
                        "rougeLsum": 0.6666666666666666,
                    },
                },
            ],
            "use_stemmer": True,
        },  # test multiple references and multiple predictions
    ]

    expected_errors = [
        {
            "prediction": ["Mary loves Joe", "Mary loves Jack"],
            "references": [["Mary loves June"]],
            "error": ValueError,
            "weights": (1,),
        },  # mismatched predictions and references
        {
            "prediction": ["Mary loves Joe", "Mary loves Jack"],
            "references": ["Mary loves June"],
            "error": ValueError,
        },  # incorrect use of multiple predictions
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
            "prediction": 123,
            "references": None,
            "weights": (1,),
            "error": ValueError,
        },  # prediction must be str or list
    ]

    # test single prediction examples
    for example in examples:
        output = _calculate_rouge_scores(
            predictions=example["prediction"],
            references=example["references"],
            use_stemmer=example.get("use_stemmer", False),
        )[0]
        assert all(
            round(output["value"][key], 2) == example[key]
            for key in ["rouge1", "rouge2", "rougeL", "rougeLsum"]
        ), f"Error for example {example} with output {output}."

    # test multiple prediction examples
    for example in multiple_prediction_examples:
        metrics = _calculate_rouge_scores(
            predictions=example["prediction"],
            references=example["references"],
            use_stemmer=example.get("use_stemmer", False),
        )
        assert metrics == example["expected_value"]

    for example in expected_errors:
        with pytest.raises(example["error"]):
            _calculate_rouge_scores(
                predictions=example["prediction"],
                references=example["references"],
            )


@patch(
    "evaluate.EvaluationModule.compute",
    mocked_compute_rouge_none,
)
def test__calculate_rouge_scores_with_none():
    prediction = "Mary loves Joe"
    references = ["Mary loves Joe"]

    with pytest.raises(ValueError):
        _calculate_rouge_scores(
            predictions=prediction,
            references=references,
        )


def test__calculate_bleu_scores():
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
            "expected_value": 0,
        },  # perfect match, case sensitive
        {
            "prediction": "Mary loves Joe",
            "references": ["MARY LOVES JOE"],
            "weights": (1,),
            "expected_value": 0,
        },  # perfect match, case sensitive
        {
            "prediction": "Mary loves Joe",
            "references": ["MARY LOVES JOE"],
            "weights": (0, 1),
            "expected_value": 0,
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
            "expected_value": 0,
        },  # BLEU-4
        {
            "prediction": "Mary loves Joe",
            "references": ["Mary loves Jane"],
            "weights": (1,),
            "expected_value": 0.67,
        },  # off by one
        {
            "prediction": "Mary loves Joe",
            "references": ["Mary loves Jane"],
            "weights": (0, 1),
            "expected_value": 0.5,
        },  # off by one BLEU-2
        {
            "prediction": "Mary loves Joe",
            "references": ["Mary loves Jane"],
            "weights": (0, 0, 1),
            "expected_value": 0,
        },  # off by one BLEU-3
        {
            "prediction": "Mary loves Joe",
            "references": ["Mary loves Jane"],
            "weights": (0, 0, 0, 1),
            "expected_value": 0,
        },  # off by one BLEU-4
        {
            "prediction": "mary loves joe",
            "references": ["MARY LOVES JOE"],
            "weights": (1,),
            "expected_value": 0,
        },  # different cases
        {
            "prediction": "mary loves joe",
            "references": ["MARY LOVES JOE"],
            "weights": [0, 1],
            "expected_value": 0,
        },  # different cases BLEU-2
        {
            "prediction": "mary loves joe",
            "references": ["MARY LOVES JOE"],
            "weights": [0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
            "expected_value": 0,
        },  # different cases BLEU-10
        {
            "prediction": "flip the roaring white dolphin",
            "references": [
                "some random sentence",
                "some other sentence",
                "some final reference",
                "flip the roaring white dolphin",
            ],
            "weights": [0, 1],
            "expected_value": 1,
        },  # test multiple references
        {
            "prediction": ["Mary loves Joe"],
            "references": [["Mary loves Joe"]],
            "weights": (1,),
            "expected_value": 1.0,
        },  # test multiple predictions, each with a list of references
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
        {
            "prediction": 0.3,
            "references": ["Mary loves Joe"],
            "weights": (1,),
            "error": ValueError,
        },  # prediction should be a string or list of strings
    ]

    for example in examples:
        output = _calculate_sentence_bleu(
            predictions=example["prediction"],
            references=example["references"],
            weights=example["weights"],
        )
        assert (
            round(output[0]["value"], 2) == example["expected_value"]
        ), f"Error for example {example} with output {output}."

    for example in expected_errors:
        with pytest.raises(example["error"]):
            _calculate_sentence_bleu(
                predictions=example["prediction"],
                references=example["references"],
                weights=example["weights"],
            )
