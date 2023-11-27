""" These integration tests should be run with a backend at http://localhost:8000
that is no auth
"""
from velour import Dataset, Datum, GroundTruth, Model, Prediction
from velour.client import Client


def test_evaluate_segmentation(
    client: Client,
    dataset_name: str,
    model_name: str,
    gt_semantic_segs1: list[GroundTruth],
    gt_semantic_segs2: list[GroundTruth],
    pred_semantic_segs: list[Prediction],
):
    dataset = Dataset.create(client, dataset_name)
    model = Model.create(client, model_name)

    for gt in gt_semantic_segs1 + gt_semantic_segs2:
        dataset.add_groundtruth(gt)

    for pred in pred_semantic_segs:
        model.add_prediction(pred)

    dataset.finalize()
    model.finalize_inferences(dataset)

    eval_job = model.evaluate_segmentation(dataset, timeout=30)
    assert eval_job.missing_pred_labels == [
        {"key": "k3", "value": "v3", "score": None}
    ]
    assert eval_job.ignored_pred_labels == [
        {"key": "k1", "value": "v1", "score": None}
    ]

    metrics = eval_job.metrics["metrics"]

    assert len(metrics) == 3
    assert set(
        [
            (m["label"]["key"], m["label"]["value"])
            for m in metrics
            if "label" in m
        ]
    ) == {("k2", "v2"), ("k3", "v3")}
    assert set([m["type"] for m in metrics]) == {"IOU", "mIOU"}


def test_evaluate_segmentation_with_filter(
    client: Client,
    dataset_name: str,
    model_name: str,
    gt_semantic_segs1: list[GroundTruth],
    gt_semantic_segs2: list[GroundTruth],
    pred_semantic_segs: list[Prediction],
):
    dataset = Dataset.create(client, dataset_name)
    model = Model.create(client, model_name)

    for gt in gt_semantic_segs1:
        gt.datum.metadata["color"] = "red"
        dataset.add_groundtruth(gt)
    for gt in gt_semantic_segs2:
        gt.datum.metadata["color"] = "blue"
        dataset.add_groundtruth(gt)

    for pred in pred_semantic_segs:
        model.add_prediction(pred)

    dataset.finalize()
    model.finalize_inferences(dataset)

    eval_job = model.evaluate_segmentation(
        dataset,
        filters=[
            Datum.metadata["color"] == "red",
        ],
        timeout=30,
    )
    assert eval_job.missing_pred_labels == [
        {"key": "k3", "value": "v3", "score": None}
    ]
    assert eval_job.ignored_pred_labels == [
        {"key": "k1", "value": "v1", "score": None}
    ]

    metrics = eval_job.metrics["metrics"]

    assert len(metrics) == 2
    assert set(
        [
            (m["label"]["key"], m["label"]["value"])
            for m in metrics
            if "label" in m
        ]
    ) == {("k2", "v2")}
    assert set([m["type"] for m in metrics]) == {"IOU", "mIOU"}
