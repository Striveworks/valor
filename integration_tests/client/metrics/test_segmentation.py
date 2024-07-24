""" These integration tests should be run with a back end at http://localhost:8000
that is no auth
"""

import random

from valor import (
    Client,
    Dataset,
    Datum,
    Filter,
    GroundTruth,
    Label,
    Model,
    Prediction,
)
from valor.enums import EvaluationStatus, MetricType


def test_evaluate_segmentation(
    client: Client,
    dataset_name: str,
    model_name: str,
    gt_semantic_segs1: list[GroundTruth],
    gt_semantic_segs2: list[GroundTruth],
    pred_semantic_segs: list[Prediction],
):
    dataset = Dataset.create(dataset_name)
    model = Model.create(model_name)

    for gt in gt_semantic_segs1 + gt_semantic_segs2:
        dataset.add_groundtruth(gt)
    dataset.finalize()

    for pred in pred_semantic_segs:
        model.add_prediction(dataset, pred)
    model.finalize_inferences(dataset)

    eval_job = model.evaluate_segmentation(dataset)
    assert eval_job.missing_pred_labels == [
        {"key": "k3", "value": "v3", "score": None}
    ]
    assert eval_job.ignored_pred_labels == [
        {"key": "k1", "value": "v1", "score": None}
    ]
    assert eval_job.wait_for_completion(timeout=30) == EvaluationStatus.DONE

    metrics = eval_job.metrics

    assert len(metrics) == 4
    assert set(
        [
            (m["label"]["key"], m["label"]["value"])
            for m in metrics
            if "label" in m
        ]
    ) == {("k2", "v2"), ("k3", "v3")}
    assert set([m["type"] for m in metrics]) == {"IOU", "mIOU"}

    # check metadata
    assert eval_job.meta["datums"] == 2
    assert eval_job.meta["labels"] == 3
    assert eval_job.meta["duration"] <= 5  # usually ~.25

    # check that metrics arg works correctly
    selected_metrics = random.sample(
        [MetricType.IOU, MetricType.mIOU],
        1,
    )
    eval_job_random_metrics = model.evaluate_segmentation(
        dataset, metrics_to_return=selected_metrics
    )
    assert (
        eval_job_random_metrics.wait_for_completion(timeout=30)
        == EvaluationStatus.DONE
    )
    assert set(
        [metric["type"] for metric in eval_job_random_metrics.metrics]
    ) == set(selected_metrics)

    # check that passing None to metrics returns the assumed list of default metrics
    default_metrics = ["IOU", "mIOU"]

    eval_job = model.evaluate_segmentation(dataset, metrics_to_return=None)
    assert eval_job.wait_for_completion(timeout=30) == EvaluationStatus.DONE
    assert set([metric["type"] for metric in eval_job.metrics]) == set(
        default_metrics
    )


def test_evaluate_segmentation_with_filter(
    client: Client,
    dataset_name: str,
    model_name: str,
    gt_semantic_segs1: list[GroundTruth],
    gt_semantic_segs2: list[GroundTruth],
    pred_semantic_segs: list[Prediction],
):
    dataset = Dataset.create(dataset_name)
    model = Model.create(model_name)

    for gt in gt_semantic_segs1:
        gt.datum.metadata["color"] = "red"
        dataset.add_groundtruth(gt)
    for gt in gt_semantic_segs2:
        gt.datum.metadata["color"] = "blue"
        dataset.add_groundtruth(gt)

    dataset.finalize()

    for pred in pred_semantic_segs:
        model.add_prediction(dataset, pred)

    model.finalize_inferences(dataset)

    color = Datum.metadata["color"]
    eval_job = model.evaluate_segmentation(
        dataset,
        filters=Filter(datums=(color == "red")),
    )
    assert eval_job.wait_for_completion(timeout=30) == EvaluationStatus.DONE

    assert eval_job.missing_pred_labels == []
    assert eval_job.ignored_pred_labels == []

    metrics = eval_job.metrics

    assert len(metrics) == 2
    assert set(
        [
            (m["label"]["key"], m["label"]["value"])
            for m in metrics
            if "label" in m
        ]
    ) == {("k2", "v2")}
    assert set([m["type"] for m in metrics]) == {"IOU", "mIOU"}


def test_evaluate_segmentation_with_label_maps(
    client: Client,
    dataset_name: str,
    model_name: str,
    gt_semantic_segs1: list[GroundTruth],
    gt_semantic_segs2: list[GroundTruth],
    pred_semantic_segs: list[Prediction],
):
    dataset = Dataset.create(dataset_name)
    model = Model.create(model_name)

    for gt in gt_semantic_segs1:
        gt.datum.metadata["color"] = "red"
        dataset.add_groundtruth(gt)
    for gt in gt_semantic_segs2:
        gt.datum.metadata["color"] = "blue"
        dataset.add_groundtruth(gt)

    dataset.finalize()

    for pred in pred_semantic_segs:
        model.add_prediction(dataset, pred)

    model.finalize_inferences(dataset)

    # check the baseline case

    eval_job = model.evaluate_segmentation(dataset)
    assert eval_job.missing_pred_labels == [
        {"key": "k3", "value": "v3", "score": None}
    ]
    assert eval_job.ignored_pred_labels == [
        {"key": "k1", "value": "v1", "score": None}
    ]
    assert eval_job.wait_for_completion(timeout=30) == EvaluationStatus.DONE

    metrics = eval_job.metrics

    assert len(metrics) == 4
    assert set(
        [
            (m["label"]["key"], m["label"]["value"])
            for m in metrics
            if "label" in m
        ]
    ) == {("k2", "v2"), ("k3", "v3")}
    assert set([m["type"] for m in metrics]) == {"IOU", "mIOU"}

    # now do the same thing, but with a label map
    eval_job = model.evaluate_segmentation(
        dataset,
        label_map={
            Label(key=f"k{i}", value=f"v{i}"): Label(key="foo", value="bar")
            for i in range(1, 4)
        },
    )

    # no labels are missing, since the missing labels have been mapped to a grouper label
    assert eval_job.missing_pred_labels == []
    assert eval_job.ignored_pred_labels == []
    assert eval_job.wait_for_completion(timeout=30) == EvaluationStatus.DONE

    metrics = eval_job.metrics

    # there's now only two metrics, since all three (k, v) combinations have been mapped to (foo, bar)
    assert len(metrics) == 2
    assert set(
        [
            (m["label"]["key"], m["label"]["value"])
            for m in metrics
            if "label" in m
        ]
    ) == {("foo", "bar"), ("foo", "bar")}
    assert set([m["type"] for m in metrics]) == {"IOU", "mIOU"}

    # check metadata
    assert eval_job.meta["datums"] == 2
    assert eval_job.meta["labels"] == 3
    assert eval_job.meta["annotations"] == 4
    assert eval_job.meta["duration"] <= 5  # usually .35

    # test only passing in one metric or the other
    eval_job = model.evaluate_segmentation(
        dataset,
        metrics_to_return=[MetricType.IOU],
        label_map={
            Label(key=f"k{i}", value=f"v{i}"): Label(key="foo", value="bar")
            for i in range(1, 4)
        },
    )

    assert eval_job.wait_for_completion(timeout=30) == EvaluationStatus.DONE
    assert set([m["type"] for m in eval_job.metrics]) == set(["IOU"])

    eval_job = model.evaluate_segmentation(
        dataset,
        metrics_to_return=[MetricType.mIOU],
        label_map={
            Label(key=f"k{i}", value=f"v{i}"): Label(key="foo", value="bar")
            for i in range(1, 4)
        },
    )

    assert eval_job.wait_for_completion(timeout=30) == EvaluationStatus.DONE
    assert set([m["type"] for m in eval_job.metrics]) == set([MetricType.mIOU])


def test_evaluate_segmentation_model_with_no_predictions(
    client: Client,
    gt_semantic_segs1: list[GroundTruth],
    gt_semantic_segs2: list[GroundTruth],
    dataset_name: str,
    model_name: str,
):
    dataset = Dataset.create(dataset_name)
    for gt in gt_semantic_segs1 + gt_semantic_segs2:
        dataset.add_groundtruth(gt)
    dataset.finalize()

    model = Model.create(model_name)
    for gt in gt_semantic_segs1 + gt_semantic_segs2:
        pd = Prediction(datum=gt.datum, annotations=[])
        model.add_prediction(dataset, pd)
    model.finalize_inferences(dataset)

    expected_metrics = [
        {"type": "IOU", "value": 0.0, "label": {"key": "k2", "value": "v2"}},
        {"type": "IOU", "value": 0.0, "label": {"key": "k3", "value": "v3"}},
        {"type": "mIOU", "parameters": {"label_key": "k2"}, "value": 0.0},
        {"type": "mIOU", "parameters": {"label_key": "k3"}, "value": 0.0},
    ]

    evaluation = model.evaluate_segmentation(dataset)
    assert evaluation.wait_for_completion(timeout=30) == EvaluationStatus.DONE

    computed_metrics = evaluation.metrics

    assert all([metric["value"] == 0 for metric in computed_metrics])
    assert all([metric in computed_metrics for metric in expected_metrics])
    assert all([metric in expected_metrics for metric in computed_metrics])
