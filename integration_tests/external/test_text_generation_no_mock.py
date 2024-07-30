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
    assert eval_job.wait_for_completion(timeout=90) == EvaluationStatus.DONE

    # Check that the right number of metrics are returned.
    assert len(eval_job.metrics) == (
        len(pred_answers) * len(metrics_to_return)
    )

    return eval_job.metrics


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


def test_llm_evaluation_coherence_with_openai(
    client: Client,
    coherence_gt_questions: list[GroundTruth],
    coherence_pred_answers: list[Prediction],
    dataset_name: str,
    model_name: str,
):
    metrics = _get_metrics(
        dataset_name=dataset_name,
        model_name=model_name,
        gt_questions=coherence_gt_questions,
        pred_answers=coherence_pred_answers,
        metrics_to_return=[MetricType.Coherence],
        llm_client="openai",
    )

    expected_metrics = {
        "uid0": {
            "Coherence": 1,
        },
        "uid2": {
            "Coherence": 5,
        },
    }

    # Check that the returned metrics have the right format.
    for m in metrics:
        uid = m["parameters"]["datum_uid"]
        metric_name = m["type"]
        assert (
            expected_metrics[uid][metric_name] == m["value"]
        ), f"Failed for {uid} and {metric_name}"


def test_llm_evaluation_answer_relevance_with_mistral(
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


def test_llm_evaluation_coherence_with_mistral(
    client: Client,
    coherence_gt_questions: list[GroundTruth],
    coherence_pred_answers: list[Prediction],
    dataset_name: str,
    model_name: str,
):
    metrics = _get_metrics(
        dataset_name=dataset_name,
        model_name=model_name,
        gt_questions=coherence_gt_questions,
        pred_answers=coherence_pred_answers,
        metrics_to_return=[MetricType.Coherence],
        llm_client="mistral",
    )

    expected_metrics = {
        "uid0": {
            "Coherence": 1,
        },
        "uid2": {
            "Coherence": 5,
        },
    }

    # Check that the returned metrics have the right format.
    for m in metrics:
        uid = m["parameters"]["datum_uid"]
        metric_name = m["type"]
        assert (
            expected_metrics[uid][metric_name] == m["value"]
        ), f"Failed for {uid} and {metric_name}"
