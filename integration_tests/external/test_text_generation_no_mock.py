""" These integration tests should be run with a back end at http://localhost:8000
that is no auth
"""

from valor import Client, Dataset, GroundTruth, Model, Prediction
from valor.enums import EvaluationStatus, MetricType

LLM_API_PARAMS = {
    "openai": {
        "client": "openai",
        "data": {
            "model": "gpt-4o",
            "seed": 2024,
        },
    },
    "mistral": {
        "client": "mistral",
        "data": {
            "model": "mistral-large-latest",
        },
    },
}


def _get_metrics(
    dataset_name: str,
    model_name: str,
    gt_questions: list[GroundTruth],
    pred_answers: list[Prediction],
    metrics_to_return: list[MetricType],
    llm_client: str,
    timeout: int = 60,
):
    dataset = Dataset.create(dataset_name)
    model = Model.create(model_name)

    for gt in gt_questions:
        dataset.add_groundtruth(gt)

    dataset.finalize()

    for pred in pred_answers:
        model.add_prediction(dataset, pred)

    model.finalize_inferences(dataset)

    eval_job = model.evaluate_text_generation(
        datasets=dataset,
        metrics_to_return=metrics_to_return,
        llm_api_params=LLM_API_PARAMS[llm_client],
    )
    assert eval_job.id
    try:
        eval_status = eval_job.wait_for_completion(timeout=timeout)
        if eval_status != EvaluationStatus.DONE:
            raise Exception(
                f"Evaluation was not successful for {llm_client} and {metrics_to_return} with status {eval_status}."
            )
    except TimeoutError as e:
        raise Exception(
            f"Evaluation timed out for {llm_client} and {metrics_to_return}.\nTimeoutError: {e}"
        )

    # Check that the right number of metrics are returned.
    assert len(eval_job.metrics) == (
        len(pred_answers) * len(metrics_to_return)
    )

    return eval_job.metrics


def test_answer_correctness_with_openai(
    client: Client,
    answer_correctness_gt_questions: list[GroundTruth],
    answer_correctness_pred_answers: list[Prediction],
    dataset_name: str,
    model_name: str,
):
    metrics = _get_metrics(
        dataset_name=dataset_name,
        model_name=model_name,
        gt_questions=answer_correctness_gt_questions,
        pred_answers=answer_correctness_pred_answers,
        metrics_to_return=[MetricType.AnswerCorrectness],
        llm_client="openai",
    )

    expected_metrics = {
        "uid0": {
            "AnswerCorrectness": 0.5,
        },
        "uid1": {
            "AnswerCorrectness": 1.0,
        },
    }

    # Check that the returned metrics have the right format.
    for m in metrics:
        uid = m["parameters"]["datum_uid"]
        metric_name = m["type"]
        assert (
            expected_metrics[uid][metric_name] == m["value"]
        ), f"Failed for {uid} and {metric_name}"


def test_answer_relevance_with_openai(
    client: Client,
    answer_relevance_gt_questions: list[GroundTruth],
    answer_relevance_pred_answers: list[Prediction],
    dataset_name: str,
    model_name: str,
):
    metrics = _get_metrics(
        dataset_name=dataset_name,
        model_name=model_name,
        gt_questions=answer_relevance_gt_questions,
        pred_answers=answer_relevance_pred_answers,
        metrics_to_return=[MetricType.AnswerRelevance],
        llm_client="openai",
    )

    expected_metrics = {
        "uid0": {
            "AnswerRelevance": 1.0,
        },
        "uid1": {
            "AnswerRelevance": 0.0,
        },
    }

    # Check that the returned metrics match the expected values.
    for m in metrics:
        uid = m["parameters"]["datum_uid"]
        metric_name = m["type"]
        assert (
            expected_metrics[uid][metric_name] == m["value"]
        ), f"Failed for {uid} and {metric_name}"


def test_bias_with_openai(
    client: Client,
    bias_gt_questions: list[GroundTruth],
    bias_pred_answers: list[Prediction],
    dataset_name: str,
    model_name: str,
):
    metrics = _get_metrics(
        dataset_name=dataset_name,
        model_name=model_name,
        gt_questions=bias_gt_questions,
        pred_answers=bias_pred_answers,
        metrics_to_return=[MetricType.Bias],
        llm_client="openai",
    )

    expected_metrics = {
        "uid0": {
            "Bias": 0.3333333333333333,
        },
        "uid1": {
            "Bias": 0.0,
        },
    }

    # Check that the returned metrics have the right format.
    for m in metrics:
        uid = m["parameters"]["datum_uid"]
        metric_name = m["type"]
        assert (
            expected_metrics[uid][metric_name] == m["value"]
        ), f"Failed for {uid} and {metric_name}"


def test_context_relevance_with_openai(
    client: Client,
    context_relevance_gt_questions: list[GroundTruth],
    context_relevance_pred_answers: list[Prediction],
    dataset_name: str,
    model_name: str,
):
    metrics = _get_metrics(
        dataset_name=dataset_name,
        model_name=model_name,
        gt_questions=context_relevance_gt_questions,
        pred_answers=context_relevance_pred_answers,
        metrics_to_return=[MetricType.ContextRelevance],
        llm_client="openai",
    )

    expected_metrics = {
        "uid0": {
            "ContextRelevance": 0.25,
        },
        "uid1": {
            "ContextRelevance": 0.75,
        },
    }

    # Check that the returned metrics have the right format.
    for m in metrics:
        uid = m["parameters"]["datum_uid"]
        metric_name = m["type"]
        assert (
            expected_metrics[uid][metric_name] == m["value"]
        ), f"Failed for {uid} and {metric_name}"


def test_context_precision_with_openai(
    client: Client,
    context_precision_gt_questions: list[GroundTruth],
    context_precision_pred_answers: list[Prediction],
    dataset_name: str,
    model_name: str,
):
    metrics = _get_metrics(
        dataset_name=dataset_name,
        model_name=model_name,
        gt_questions=context_precision_gt_questions,
        pred_answers=context_precision_pred_answers,
        metrics_to_return=[MetricType.ContextPrecision],
        llm_client="openai",
    )

    expected_metrics = {
        "uid0": {
            "ContextPrecision": 0.5,
        },
        "uid1": {
            "ContextPrecision": 0.8333333333333333,
        },
    }

    # Check that the returned metrics have the right format.
    for m in metrics:
        uid = m["parameters"]["datum_uid"]
        metric_name = m["type"]
        assert (
            expected_metrics[uid][metric_name] == m["value"]
        ), f"Failed for {uid} and {metric_name}"


def test_context_recall_with_openai(
    client: Client,
    context_recall_gt_questions: list[GroundTruth],
    context_recall_pred_answers: list[Prediction],
    dataset_name: str,
    model_name: str,
):
    metrics = _get_metrics(
        dataset_name=dataset_name,
        model_name=model_name,
        gt_questions=context_recall_gt_questions,
        pred_answers=context_recall_pred_answers,
        metrics_to_return=[MetricType.ContextRecall],
        llm_client="openai",
    )

    expected_metrics = {
        "uid0": {
            "ContextRecall": 0.5,
        },
        "uid1": {
            "ContextRecall": 1.0,
        },
    }

    # Check that the returned metrics have the right format.
    for m in metrics:
        uid = m["parameters"]["datum_uid"]
        metric_name = m["type"]
        assert (
            expected_metrics[uid][metric_name] == m["value"]
        ), f"Failed for {uid} and {metric_name}"


def test_faithfulness_with_openai(
    client: Client,
    faithfulness_gt_questions: list[GroundTruth],
    faithfulness_pred_answers: list[Prediction],
    dataset_name: str,
    model_name: str,
):
    metrics = _get_metrics(
        dataset_name=dataset_name,
        model_name=model_name,
        gt_questions=faithfulness_gt_questions,
        pred_answers=faithfulness_pred_answers,
        metrics_to_return=[MetricType.Faithfulness],
        llm_client="openai",
    )

    expected_metrics = {
        "uid0": {
            "Faithfulness": 0.5,
        },
        "uid1": {
            "Faithfulness": 0.6666666666666666,
        },
    }

    # Check that the returned metrics have the right format.
    for m in metrics:
        uid = m["parameters"]["datum_uid"]
        metric_name = m["type"]
        assert (
            expected_metrics[uid][metric_name] == m["value"]
        ), f"Failed for {uid} and {metric_name}"


def test_hallucination_with_openai(
    client: Client,
    hallucination_gt_questions: list[GroundTruth],
    hallucination_pred_answers: list[Prediction],
    dataset_name: str,
    model_name: str,
):
    metrics = _get_metrics(
        dataset_name=dataset_name,
        model_name=model_name,
        gt_questions=hallucination_gt_questions,
        pred_answers=hallucination_pred_answers,
        metrics_to_return=[MetricType.Hallucination],
        llm_client="openai",
    )

    expected_metrics = {
        "uid0": {
            "Hallucination": 0.3333333333333333,
        },
        "uid1": {
            "Hallucination": 0.25,
        },
    }

    # Check that the returned metrics have the right format.
    for m in metrics:
        uid = m["parameters"]["datum_uid"]
        metric_name = m["type"]
        assert (
            expected_metrics[uid][metric_name] == m["value"]
        ), f"Failed for {uid} and {metric_name}"


def test_summary_coherence_with_openai(
    client: Client,
    summary_coherence_gt_questions: list[GroundTruth],
    summary_coherence_pred_answers: list[Prediction],
    dataset_name: str,
    model_name: str,
):
    metrics = _get_metrics(
        dataset_name=dataset_name,
        model_name=model_name,
        gt_questions=summary_coherence_gt_questions,
        pred_answers=summary_coherence_pred_answers,
        metrics_to_return=[MetricType.SummaryCoherence],
        llm_client="openai",
    )

    # Check that the returned metrics have the right format.
    assert len(metrics) == 1
    assert metrics[0]["parameters"]["datum_uid"] == "uid0"
    assert metrics[0]["type"] == "SummaryCoherence"

    # Check that the summary coherence was rated >= 3.
    assert metrics[0]["value"] in {3, 4, 5}


def test_toxicity_with_openai(
    client: Client,
    toxicity_gt_questions: list[GroundTruth],
    toxicity_pred_answers: list[Prediction],
    dataset_name: str,
    model_name: str,
):
    metrics = _get_metrics(
        dataset_name=dataset_name,
        model_name=model_name,
        gt_questions=toxicity_gt_questions,
        pred_answers=toxicity_pred_answers,
        metrics_to_return=[MetricType.Toxicity],
        llm_client="openai",
    )

    expected_metrics = {
        "uid0": {
            "Toxicity": 0.0,
        },
        "uid1": {
            "Toxicity": 1.0,
        },
        "uid2": {
            "Toxicity": 0.0,
        },
    }

    # Check that the returned metrics have the right format.
    for m in metrics:
        uid = m["parameters"]["datum_uid"]
        metric_name = m["type"]
        assert (
            expected_metrics[uid][metric_name] == m["value"]
        ), f"Failed for {uid} and {metric_name}"


def test_answer_correctness_with_mistral(
    client: Client,
    answer_correctness_gt_questions: list[GroundTruth],
    answer_correctness_pred_answers: list[Prediction],
    dataset_name: str,
    model_name: str,
):
    metrics = _get_metrics(
        dataset_name=dataset_name,
        model_name=model_name,
        gt_questions=answer_correctness_gt_questions,
        pred_answers=answer_correctness_pred_answers,
        metrics_to_return=[MetricType.AnswerCorrectness],
        llm_client="mistral",
    )

    expected_metrics = {
        "uid0": {
            "AnswerCorrectness": 0.5,
        },
        "uid1": {
            "AnswerCorrectness": 1.0,
        },
    }

    # Check that the returned metrics have the right format.
    for m in metrics:
        uid = m["parameters"]["datum_uid"]
        metric_name = m["type"]
        assert (
            expected_metrics[uid][metric_name] == m["value"]
        ), f"Failed for {uid} and {metric_name}"


def test_answer_relevance_with_mistral(
    client: Client,
    answer_relevance_gt_questions: list[GroundTruth],
    answer_relevance_pred_answers: list[Prediction],
    dataset_name: str,
    model_name: str,
):
    metrics = _get_metrics(
        dataset_name=dataset_name,
        model_name=model_name,
        gt_questions=answer_relevance_gt_questions,
        pred_answers=answer_relevance_pred_answers,
        metrics_to_return=[MetricType.AnswerRelevance],
        llm_client="mistral",
    )

    expected_metrics = {
        "uid0": {
            "AnswerRelevance": 1.0,
        },
        "uid1": {
            "AnswerRelevance": 0.0,
        },
    }

    # Check that the returned metrics have the right format.
    for m in metrics:
        uid = m["parameters"]["datum_uid"]
        metric_name = m["type"]
        assert (
            expected_metrics[uid][metric_name] == m["value"]
        ), f"Failed for {uid} and {metric_name}"


def test_bias_with_mistral(
    client: Client,
    bias_gt_questions: list[GroundTruth],
    bias_pred_answers: list[Prediction],
    dataset_name: str,
    model_name: str,
):
    metrics = _get_metrics(
        dataset_name=dataset_name,
        model_name=model_name,
        gt_questions=bias_gt_questions,
        pred_answers=bias_pred_answers,
        metrics_to_return=[MetricType.Bias],
        llm_client="mistral",
    )

    expected_metrics = {
        "uid0": {
            "Bias": 0.3333333333333333,
        },
        "uid1": {
            "Bias": 0.0,
        },
    }

    # Check that the returned metrics have the right format.
    for m in metrics:
        uid = m["parameters"]["datum_uid"]
        metric_name = m["type"]
        assert (
            expected_metrics[uid][metric_name] == m["value"]
        ), f"Failed for {uid} and {metric_name}"


def test_context_precision_with_mistral(
    client: Client,
    context_precision_gt_questions: list[GroundTruth],
    context_precision_pred_answers: list[Prediction],
    dataset_name: str,
    model_name: str,
):
    metrics = _get_metrics(
        dataset_name=dataset_name,
        model_name=model_name,
        gt_questions=context_precision_gt_questions,
        pred_answers=context_precision_pred_answers,
        metrics_to_return=[MetricType.ContextPrecision],
        llm_client="mistral",
    )

    expected_metrics = {
        "uid0": {
            "ContextPrecision": 0.5,
        },
        "uid1": {
            "ContextPrecision": 0.8333333333333333,
        },
    }

    # Check that the returned metrics have the right format.
    for m in metrics:
        uid = m["parameters"]["datum_uid"]
        metric_name = m["type"]
        assert (
            expected_metrics[uid][metric_name] == m["value"]
        ), f"Failed for {uid} and {metric_name}"


def test_context_recall_with_mistral(
    client: Client,
    context_recall_gt_questions: list[GroundTruth],
    context_recall_pred_answers: list[Prediction],
    dataset_name: str,
    model_name: str,
):
    metrics = _get_metrics(
        dataset_name=dataset_name,
        model_name=model_name,
        gt_questions=context_recall_gt_questions,
        pred_answers=context_recall_pred_answers,
        metrics_to_return=[MetricType.ContextRecall],
        llm_client="mistral",
    )

    expected_metrics = {
        "uid0": {
            "ContextRecall": 0.5,
        },
        "uid1": {
            "ContextRecall": 1.0,
        },
    }

    # Check that the returned metrics have the right format.
    for m in metrics:
        uid = m["parameters"]["datum_uid"]
        metric_name = m["type"]
        assert (
            expected_metrics[uid][metric_name] == m["value"]
        ), f"Failed for {uid} and {metric_name}"


def test_context_relevance_with_mistral(
    client: Client,
    context_relevance_gt_questions: list[GroundTruth],
    context_relevance_pred_answers: list[Prediction],
    dataset_name: str,
    model_name: str,
):
    metrics = _get_metrics(
        dataset_name=dataset_name,
        model_name=model_name,
        gt_questions=context_relevance_gt_questions,
        pred_answers=context_relevance_pred_answers,
        metrics_to_return=[MetricType.ContextRelevance],
        llm_client="mistral",
    )

    expected_metrics = {
        "uid0": {
            "ContextRelevance": 0.25,
        },
        "uid1": {
            "ContextRelevance": 0.75,
        },
    }

    # Check that the returned metrics have the right format.
    for m in metrics:
        uid = m["parameters"]["datum_uid"]
        metric_name = m["type"]
        assert (
            expected_metrics[uid][metric_name] == m["value"]
        ), f"Failed for {uid} and {metric_name}"


def test_faithfulness_with_mistral(
    client: Client,
    faithfulness_gt_questions: list[GroundTruth],
    faithfulness_pred_answers: list[Prediction],
    dataset_name: str,
    model_name: str,
):
    metrics = _get_metrics(
        dataset_name=dataset_name,
        model_name=model_name,
        gt_questions=faithfulness_gt_questions,
        pred_answers=faithfulness_pred_answers,
        metrics_to_return=[MetricType.Faithfulness],
        llm_client="mistral",
    )

    expected_metrics = {
        "uid0": {
            "Faithfulness": 0.5,
        },
        "uid1": {
            "Faithfulness": 0.6666666666666666,
        },
    }

    # Check that the returned metrics have the right format.
    for m in metrics:
        uid = m["parameters"]["datum_uid"]
        metric_name = m["type"]
        assert (
            expected_metrics[uid][metric_name] == m["value"]
        ), f"Failed for {uid} and {metric_name}"


def test_hallucination_with_mistral(
    client: Client,
    hallucination_gt_questions: list[GroundTruth],
    hallucination_pred_answers: list[Prediction],
    dataset_name: str,
    model_name: str,
):
    metrics = _get_metrics(
        dataset_name=dataset_name,
        model_name=model_name,
        gt_questions=hallucination_gt_questions,
        pred_answers=hallucination_pred_answers,
        metrics_to_return=[MetricType.Hallucination],
        llm_client="mistral",
    )

    expected_metrics = {
        "uid0": {
            "Hallucination": 0.3333333333333333,
        },
        "uid1": {
            "Hallucination": 0.25,
        },
    }

    # Check that the returned metrics have the right format.
    for m in metrics:
        uid = m["parameters"]["datum_uid"]
        metric_name = m["type"]
        assert (
            expected_metrics[uid][metric_name] == m["value"]
        ), f"Failed for {uid} and {metric_name}"


def test_summary_coherence_with_mistral(
    client: Client,
    summary_coherence_gt_questions: list[GroundTruth],
    summary_coherence_pred_answers: list[Prediction],
    dataset_name: str,
    model_name: str,
):
    metrics = _get_metrics(
        dataset_name=dataset_name,
        model_name=model_name,
        gt_questions=summary_coherence_gt_questions,
        pred_answers=summary_coherence_pred_answers,
        metrics_to_return=[MetricType.SummaryCoherence],
        llm_client="mistral",
    )

    # Check that the returned metrics have the right format.
    assert len(metrics) == 1
    assert metrics[0]["parameters"]["datum_uid"] == "uid0"
    assert metrics[0]["type"] == "SummaryCoherence"

    # Check that the summary coherence was rated >= 3.
    assert metrics[0]["value"] in {3, 4, 5}


def test_toxicity_with_mistral(
    client: Client,
    toxicity_gt_questions: list[GroundTruth],
    toxicity_pred_answers: list[Prediction],
    dataset_name: str,
    model_name: str,
):
    metrics = _get_metrics(
        dataset_name=dataset_name,
        model_name=model_name,
        gt_questions=toxicity_gt_questions,
        pred_answers=toxicity_pred_answers,
        metrics_to_return=[MetricType.Toxicity],
        llm_client="mistral",
    )

    expected_metrics = {
        "uid0": {
            "Toxicity": 0.0,
        },
        "uid1": {
            "Toxicity": 1.0,
        },
        "uid2": {
            "Toxicity": 0.0,
        },
    }

    # Check that the returned metrics have the right format.
    for m in metrics:
        uid = m["parameters"]["datum_uid"]
        metric_name = m["type"]
        assert (
            expected_metrics[uid][metric_name] == m["value"]
        ), f"Failed for {uid} and {metric_name}"
