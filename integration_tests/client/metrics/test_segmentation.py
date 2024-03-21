""" These integration tests should be run with a back end at http://localhost:8000
that is no auth
"""

from client.valor import (
    Client,
    Dataset,
    Datum,
    GroundTruth,
    Label,
    Model,
    Prediction,
)
from client.valor.enums import EvaluationStatus


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
    assert color is str
    eval_job = model.evaluate_segmentation(
        dataset,
        filter_by=[
            color == "red",
        ],
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

    assert len(metrics) == 3
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
