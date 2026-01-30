from copy import deepcopy
from pathlib import Path
from uuid import uuid4

import pyarrow.compute as pc
import pytest

from valor_lite.object_detection import (
    BoundingBox,
    Detection,
    Loader,
    MetricType,
)


@pytest.fixture
def one_detection(
    rect1: tuple[float, float, float, float],
    rect3: tuple[float, float, float, float],
) -> list[Detection]:
    """Combines the labels from basic_detections_first_class and basic_detections_second_class."""
    return [
        Detection(
            uid="uid1",
            groundtruths=[
                BoundingBox(
                    uid=str(uuid4()),
                    xmin=rect1[0],
                    xmax=rect1[1],
                    ymin=rect1[2],
                    ymax=rect1[3],
                    labels=["v1"],
                    metadata={
                        "gt_rect": "rect1",
                    },
                ),
                BoundingBox(
                    uid=str(uuid4()),
                    xmin=rect3[0],
                    xmax=rect3[1],
                    ymin=rect3[2],
                    ymax=rect3[3],
                    labels=["v2"],
                    metadata={
                        "gt_rect": "rect3",
                    },
                ),
            ],
            predictions=[
                BoundingBox(
                    uid=str(uuid4()),
                    xmin=rect1[0],
                    xmax=rect1[1],
                    ymin=rect1[2],
                    ymax=rect1[3],
                    labels=["v1"],
                    scores=[0.3],
                    metadata={
                        "pd_rect": "rect1",
                    },
                ),
            ],
        ),
    ]


@pytest.fixture
def two_detections(
    one_detection: list[Detection],
    rect2: tuple[float, float, float, float],
) -> list[Detection]:
    return one_detection + [
        Detection(
            uid="uid2",
            groundtruths=[
                BoundingBox(
                    uid=str(uuid4()),
                    xmin=rect2[0],
                    xmax=rect2[1],
                    ymin=rect2[2],
                    ymax=rect2[3],
                    labels=["v1"],
                    metadata={
                        "gt_rect": "rect2",
                    },
                ),
            ],
            predictions=[
                BoundingBox(
                    uid=str(uuid4()),
                    xmin=rect2[0],
                    xmax=rect2[1],
                    ymin=rect2[2],
                    ymax=rect2[3],
                    labels=["v2"],
                    scores=[0.98],
                    metadata={
                        "pd_rect": "rect2",
                    },
                ),
            ],
        ),
    ]


@pytest.fixture
def four_detections(two_detections: list[Detection]) -> list[Detection]:
    det1 = two_detections[0]
    det2 = two_detections[1]
    det3 = deepcopy(two_detections[0])
    det4 = deepcopy(two_detections[1])

    det3.uid = "uid3"
    det4.uid = "uid4"

    for det in [det1, det2, det3, det4]:
        for gidx, gt in enumerate(det.groundtruths):
            gt.uid = f"{det.uid}_gt_{gidx}"
        for pidx, pd in enumerate(det.predictions):
            pd.uid = f"{det.uid}_pd_{pidx}"

    return [det1, det2, det3, det4]


def _generate_random_detections(
    n_detections: int, n_boxes: int, labels: str
) -> list[Detection]:
    from random import choice, uniform

    def bbox(is_prediction):
        xmin, ymin = uniform(0, 10), uniform(0, 10)
        xmax, ymax = uniform(xmin, 15), uniform(ymin, 15)
        kw = {"scores": [uniform(0, 1)]} if is_prediction else {}
        return BoundingBox(
            str(uuid4()),
            xmin,
            xmax,
            ymin,
            ymax,
            [choice(labels)],
            metadata=None,
            **kw,
        )

    return [
        Detection(
            uid=f"uid{i}",
            groundtruths=[bbox(is_prediction=False) for _ in range(n_boxes)],
            predictions=[bbox(is_prediction=True) for _ in range(n_boxes)],
        )
        for i in range(n_detections)
    ]


def test_filtering_one_detection(
    loader: Loader,
    tmp_path: Path,
    one_detection: list[Detection],
):
    """
    Basic object detection test that combines the labels of basic_detections_first_class and basic_detections_second_class.

    groundtruths
        datum uid1
            box 1 - label v1 - tp
            box 3 - label v2 - fn unmatched ground truths

    predictions
        datum uid1
            box 1 - label v1 - score 0.3 - tp
    """
    loader.add_bounding_boxes(one_detection)
    evaluator = loader.finalize()

    # test evaluation
    filtered_evaluator = evaluator.filter(
        datums=pc.field("datum_uid") == "uid1",
        path=tmp_path / "filtered",
    )
    metrics = filtered_evaluator.compute_precision_recall(
        iou_thresholds=[0.5], score_thresholds=[0.5]
    )

    actual_metrics = [m.to_dict() for m in metrics[MetricType.AP]]
    expected_metrics = [
        {
            "type": "AP",
            "value": 1.0,
            "parameters": {
                "iou_threshold": 0.5,
                "label": "v1",
            },
        },
        {
            "type": "AP",
            "value": 0.0,
            "parameters": {
                "iou_threshold": 0.5,
                "label": "v2",
            },
        },
    ]
    for m in actual_metrics:
        assert m in expected_metrics
    for m in expected_metrics:
        assert m in actual_metrics


def test_filtering_two_detections(
    loader: Loader,
    tmp_path: Path,
    two_detections: list[Detection],
):
    """
    Basic object detection test that combines the labels of basic_detections_first_class and basic_detections_second_class.

    groundtruths
        datum uid1
            box 1 - label v1 - tp
            box 3 - label v2 - fn unmatched ground truths
        datum uid2
            box 2 - label v1 - fn misclassification

    predictions
        datum uid1
            box 1 - label v1 - score 0.3 - tp
        datum uid2
            box 2 - label v2 - score 0.98 - fp misclassification
    """
    loader.add_bounding_boxes(two_detections)
    evaluator = loader.finalize()

    # test evaluation
    filtered_evaluator = evaluator.filter(
        datums=pc.field("datum_uid") == "uid1",
        path=tmp_path / "filtered",
    )
    metrics = filtered_evaluator.compute_precision_recall(
        iou_thresholds=[0.5], score_thresholds=[0.5]
    )

    actual_metrics = [m.to_dict() for m in metrics[MetricType.AP]]
    expected_metrics = [
        {
            "type": "AP",
            "value": 1.0,
            "parameters": {"iou_threshold": 0.5, "label": "v1"},
        },
        {
            "type": "AP",
            "value": 0.0,
            "parameters": {"iou_threshold": 0.5, "label": "v2"},
        },
    ]
    for m in actual_metrics:
        assert m in expected_metrics
    for m in expected_metrics:
        assert m in actual_metrics


def test_filtering_four_detections(
    loader: Loader,
    tmp_path: Path,
    four_detections: list[Detection],
):
    """
    Basic object detection test that combines the labels of basic_detections_first_class and basic_detections_second_class.

    groundtruths
        datum uid1
            box 1 - label v1 - tp
            box 3 - label v2 - fn unmatched ground truths
        datum uid2
            box 2 - label v1 - fn misclassification
        datum uid3
            box 1 - label v1 - tp
            box 3 - label v2 - fn unmatched ground truths
        datum uid4
            box 2 - label v1 - fn misclassification

    predictions
        datum uid1
            box 1 - label v1 - score 0.3 - tp
        datum uid2
            box 2 - label v2 - score 0.98 - fp misclassification
        datum uid3
            box 1 - label v1 - score 0.3 - tp
        datum uid4
            box 2 - label v2 - score 0.98 - fp misclassification
    """
    loader.add_bounding_boxes(four_detections)
    evaluator = loader.finalize()

    # test evaluation
    filtered_evaluator = evaluator.filter(
        datums=pc.field("datum_uid") == "uid1",
        path=tmp_path / "filtered",
    )
    metrics = filtered_evaluator.compute_precision_recall(
        iou_thresholds=[0.5], score_thresholds=[0.5]
    )

    actual_metrics = [m.to_dict() for m in metrics[MetricType.AP]]
    expected_metrics = [
        {
            "type": "AP",
            "value": 1.0,
            "parameters": {"iou_threshold": 0.5, "label": "v1"},
        },
        {
            "type": "AP",
            "value": 0.0,
            "parameters": {"iou_threshold": 0.5, "label": "v2"},
        },
    ]
    for m in actual_metrics:
        assert m in expected_metrics
    for m in expected_metrics:
        assert m in actual_metrics


def test_filtering_all_detections(
    loader: Loader,
    tmp_path: Path,
    four_detections: list[Detection],
):
    """
    Basic object detection test that combines the labels of basic_detections_first_class and basic_detections_second_class.

    groundtruths
        datum uid1
            box 1 - label v1 - tp
            box 3 - label v2 - fn unmatched ground truths
        datum uid2
            box 2 - label v1 - fn misclassification
        datum uid3
            box 1 - label v1 - tp
            box 3 - label v2 - fn unmatched ground truths
        datum uid4
            box 2 - label v1 - fn misclassification

    predictions
        datum uid1
            box 1 - label v1 - score 0.3 - tp
        datum uid2
            box 2 - label v2 - score 0.98 - fp misclassification
        datum uid3
            box 1 - label v1 - score 0.3 - tp
        datum uid4
            box 2 - label v2 - score 0.98 - fp misclassification
    """
    loader.add_bounding_boxes(four_detections)
    evaluator = loader.finalize()

    # test evaluation
    filtered_evaluator = evaluator.filter(
        predictions=pc.field("pd_uid") == "uid1_pd_0",
        path=tmp_path / "filtered",
    )
    metrics = filtered_evaluator.compute_precision_recall(
        iou_thresholds=[0.5], score_thresholds=[0.5]
    )

    evaluator.compute_confusion_matrix(
        iou_thresholds=[0.5],
        score_thresholds=[0.5],
    )
    actual_metrics = [m.to_dict() for m in metrics[MetricType.AP]]
    assert len(actual_metrics) == 2
    expected_metrics = [
        {
            "type": "AP",
            "parameters": {"iou_threshold": 0.5, "label": "v1"},
            "value": 0.25742574257425743,
        },
        {
            "type": "AP",
            "parameters": {"iou_threshold": 0.5, "label": "v2"},
            "value": 0.0,
        },
    ]
    for m in actual_metrics:
        assert m in expected_metrics
    for m in expected_metrics:
        assert m in actual_metrics


def test_filtering_random_detections(
    loader: Loader,
    tmp_path: Path,
):
    loader.add_bounding_boxes(_generate_random_detections(13, 4, "abc"))
    evaluator = loader.finalize()

    filtered_evaluator = evaluator.filter(
        predictions=pc.field("pd_uid") == "uid1_pd_0",
        path=tmp_path / "filtered",
    )
    filtered_evaluator.compute_precision_recall(
        iou_thresholds=[0.5], score_thresholds=[0.5]
    )


def test_filtering_four_detections_by_indices(
    loader: Loader,
    tmp_path: Path,
    four_detections: list[Detection],
):
    """
    Basic object detection test that combines the labels of basic_detections_first_class and basic_detections_second_class.

    groundtruths
        datum uid1
            box 1 - label v1 - tp
            box 3 - label v2 - fn unmatched ground truths
        datum uid2
            box 2 - label v1 - fn misclassification
        datum uid3
            box 1 - label v1 - tp
            box 3 - label v2 - fn unmatched ground truths
        datum uid4
            box 2 - label v1 - fn misclassification

    predictions
        datum uid1
            box 1 - label v1 - score 0.3 - tp
        datum uid2
            box 2 - label v2 - score 0.98 - fp misclassification
        datum uid3
            box 1 - label v1 - score 0.3 - tp
        datum uid4
            box 2 - label v2 - score 0.98 - fp misclassification
    """
    loader.add_bounding_boxes(four_detections)
    evaluator = loader.finalize()

    # test evaluation
    filtered_evaluator = evaluator.filter(
        datums=pc.field("datum_id") == 0,
        path=tmp_path / "filtered",
    )
    metrics = filtered_evaluator.compute_precision_recall(
        iou_thresholds=[0.5], score_thresholds=[0.5]
    )

    actual_metrics = [m.to_dict() for m in metrics[MetricType.AP]]
    expected_metrics = [
        {
            "type": "AP",
            "value": 1.0,
            "parameters": {"iou_threshold": 0.5, "label": "v1"},
        },
        {
            "type": "AP",
            "value": 0.0,
            "parameters": {"iou_threshold": 0.5, "label": "v2"},
        },
    ]
    for m in actual_metrics:
        assert m in expected_metrics
    for m in expected_metrics:
        assert m in actual_metrics


def test_filtering_four_detections_by_annotation_metadata(
    loader: Loader,
    tmp_path: Path,
    four_detections: list[Detection],
):
    """
    Basic object detection test that combines the labels of basic_detections_first_class and basic_detections_second_class.

    groundtruths
        datum uid1
            box 1 - label v1 - tp
            box 3 - label v2 - fn unmatched ground truths
        datum uid2
            box 2 - label v1 - fn misclassification
        datum uid3
            box 1 - label v1 - tp
            box 3 - label v2 - fn unmatched ground truths
        datum uid4
            box 2 - label v1 - fn misclassification

    predictions
        datum uid1
            box 1 - label v1 - score 0.3 - tp
        datum uid2
            box 2 - label v2 - score 0.98 - fp misclassification
        datum uid3
            box 1 - label v1 - score 0.3 - tp
        datum uid4
            box 2 - label v2 - score 0.98 - fp misclassification
    """
    loader.add_bounding_boxes(four_detections)
    evaluator = loader.finalize()

    # remove all FN groundtruths
    filtered_evaluator = evaluator.filter(
        groundtruths=pc.field("gt_rect") == "rect1",
        path=tmp_path / "filtered1",
    )
    metrics = filtered_evaluator.compute_precision_recall(
        iou_thresholds=[0.5], score_thresholds=[0.1]
    )
    actual_metrics = [m.to_dict() for m in metrics[MetricType.Counts]]
    expected_metrics = [
        {
            "type": "Counts",
            "value": {"tp": 2, "fp": 0, "fn": 0},
            "parameters": {
                "iou_threshold": 0.5,
                "score_threshold": 0.1,
                "label": "v1",
            },
        },
        {
            "type": "Counts",
            "value": {"tp": 0, "fp": 2, "fn": 0},
            "parameters": {
                "iou_threshold": 0.5,
                "score_threshold": 0.1,
                "label": "v2",
            },
        },
    ]
    for m in actual_metrics:
        assert m in expected_metrics
    for m in expected_metrics:
        assert m in actual_metrics

    # remove TP ground truths
    filtered_evaluator = evaluator.filter(
        groundtruths=pc.field("gt_rect") != "rect1",
        path=tmp_path / "filtered2",
    )
    metrics = filtered_evaluator.compute_precision_recall(
        iou_thresholds=[0.5], score_thresholds=[0.1]
    )
    actual_metrics = [m.to_dict() for m in metrics[MetricType.Counts]]
    expected_metrics = [
        {
            "type": "Counts",
            "value": {"tp": 0, "fp": 2, "fn": 2},
            "parameters": {
                "iou_threshold": 0.5,
                "score_threshold": 0.1,
                "label": "v1",
            },
        },
        {
            "type": "Counts",
            "value": {"tp": 0, "fp": 2, "fn": 2},
            "parameters": {
                "iou_threshold": 0.5,
                "score_threshold": 0.1,
                "label": "v2",
            },
        },
    ]
    for m in actual_metrics:
        assert m in expected_metrics
    for m in expected_metrics:
        assert m in actual_metrics


def test_filtering_four_detections_inline(
    loader: Loader,
    four_detections: list[Detection],
):
    """
    Basic object detection test that combines the labels of basic_detections_first_class and basic_detections_second_class.

    groundtruths
        datum uid1
            box 1 - label v1 - tp
            box 3 - label v2 - fn unmatched ground truths
        datum uid2
            box 2 - label v1 - fn misclassification
        datum uid3
            box 1 - label v1 - tp
            box 3 - label v2 - fn unmatched ground truths
        datum uid4
            box 2 - label v1 - fn misclassification

    predictions
        datum uid1
            box 1 - label v1 - score 0.3 - tp
        datum uid2
            box 2 - label v2 - score 0.98 - fp misclassification
        datum uid3
            box 1 - label v1 - score 0.3 - tp
        datum uid4
            box 2 - label v2 - score 0.98 - fp misclassification
    """
    loader.add_bounding_boxes(four_detections)
    evaluator = loader.finalize()

    # test evaluation
    metrics = evaluator.compute_precision_recall(
        iou_thresholds=[0.5],
        score_thresholds=[0.5],
        datums=pc.field("datum_uid") == "uid1",
    )

    actual_metrics = [m.to_dict() for m in metrics[MetricType.AP]]
    expected_metrics = [
        {
            "type": "AP",
            "value": 1.0,
            "parameters": {"iou_threshold": 0.5, "label": "v1"},
        },
        {
            "type": "AP",
            "value": 0.0,
            "parameters": {"iou_threshold": 0.5, "label": "v2"},
        },
    ]
    for m in actual_metrics:
        assert m in expected_metrics
    for m in expected_metrics:
        assert m in actual_metrics


def test_get_info(
    loader: Loader,
    four_detections: list[Detection],
):
    """
    Basic object detection test that combines the labels of basic_detections_first_class and basic_detections_second_class.

    groundtruths
        datum uid1
            box 1 - label v1 - tp
            box 3 - label v2 - fn unmatched ground truths
        datum uid2
            box 2 - label v1 - fn misclassification
        datum uid3
            box 1 - label v1 - tp
            box 3 - label v2 - fn unmatched ground truths
        datum uid4
            box 2 - label v1 - fn misclassification

    predictions
        datum uid1
            box 1 - label v1 - score 0.3 - tp
        datum uid2
            box 2 - label v2 - score 0.98 - fp misclassification
        datum uid3
            box 1 - label v1 - score 0.3 - tp
        datum uid4
            box 2 - label v2 - score 0.98 - fp misclassification
    """
    loader.add_bounding_boxes(four_detections)
    evaluator = loader.finalize()

    # remove all FN groundtruths
    info = evaluator.get_info(
        groundtruths=pc.field("gt_rect") == "rect1",
    )
    assert info.number_of_datums == 4
    assert info.number_of_labels == 2
    assert info.number_of_groundtruth_annotations == 2
    assert info.number_of_prediction_annotations == 4

    # remove TP ground truths
    info = evaluator.get_info(
        groundtruths=pc.field("gt_rect") != "rect1",
    )
    assert info.number_of_datums == 4
    assert info.number_of_labels == 2
    assert info.number_of_groundtruth_annotations == 4
    assert info.number_of_prediction_annotations == 4

    # remove all predictions
    info = evaluator.get_info(
        predictions=pc.field("pd_rect") == "rect3",
    )
    assert info.number_of_datums == 4
    assert info.number_of_labels == 2
    assert info.number_of_groundtruth_annotations == 6
    assert info.number_of_prediction_annotations == 0

    # remove TP predictions
    info = evaluator.get_info(
        predictions=pc.field("pd_rect") != "rect1",
    )
    assert info.number_of_datums == 4
    assert info.number_of_labels == 2
    assert info.number_of_groundtruth_annotations == 6
    assert info.number_of_prediction_annotations == 2


def test_filtering_labels(
    loader: Loader, torchmetrics_detections: list[Detection], tmp_path: Path
):
    loader.add_bounding_boxes(torchmetrics_detections)
    evaluator = loader.finalize()

    scores = [0.1]
    ious = [0.1]

    assert evaluator._index_to_label == {
        0: "4",
        1: "2",
        2: "3",
        3: "1",
        4: "0",
        5: "49",
    }
    assert evaluator.compute_precision_recall(
        score_thresholds=scores, iou_thresholds=ious
    )
    assert evaluator.compute_confusion_matrix(
        score_thresholds=scores, iou_thresholds=ious
    )
    assert evaluator.compute_examples(
        score_thresholds=scores, iou_thresholds=ious
    )

    cm = evaluator.compute_confusion_matrix(
        score_thresholds=scores, iou_thresholds=ious
    )
    assert len(cm) == 1
    assert cm[0].to_dict() == {
        "parameters": {
            "iou_threshold": 0.1,
            "score_threshold": 0.1,
        },
        "type": "ConfusionMatrix",
        "value": {
            "confusion_matrix": {
                "0": {
                    "0": 5,
                    "1": 0,
                    "2": 0,
                    "3": 0,
                    "4": 0,
                    "49": 0,
                },
                "1": {
                    "0": 0,
                    "1": 1,
                    "2": 0,
                    "3": 0,
                    "4": 0,
                    "49": 0,
                },
                "2": {
                    "0": 0,
                    "1": 0,
                    "2": 1,
                    "3": 1,
                    "4": 0,
                    "49": 0,
                },
                "3": {
                    "0": 0,
                    "1": 0,
                    "2": 0,
                    "3": 0,
                    "4": 0,
                    "49": 0,
                },
                "4": {
                    "0": 0,
                    "1": 0,
                    "2": 0,
                    "3": 0,
                    "4": 2,
                    "49": 0,
                },
                "49": {
                    "0": 0,
                    "1": 0,
                    "2": 0,
                    "3": 0,
                    "4": 0,
                    "49": 9,
                },
            },
            "unmatched_ground_truths": {
                "0": 0,
                "1": 0,
                "2": 0,
                "3": 0,
                "4": 0,
                "49": 1,
            },
            "unmatched_predictions": {
                "0": 0,
                "1": 0,
                "2": 0,
                "3": 0,
                "4": 0,
                "49": 0,
            },
        },
    }

    filtered = evaluator.filter(
        groundtruths=pc.field("gt_label").isin(["2", "3"]),
        predictions=pc.field("pd_label").isin(["2", "3"]),
        path=tmp_path / "filter",
    )

    assert filtered._index_to_label == {
        0: "4",
        1: "2",
        2: "3",
        3: "1",
        4: "0",
        5: "49",
    }
    assert evaluator.compute_precision_recall(
        score_thresholds=scores, iou_thresholds=ious
    )
    assert evaluator.compute_confusion_matrix(
        score_thresholds=scores, iou_thresholds=ious
    )
    assert evaluator.compute_examples(
        score_thresholds=scores, iou_thresholds=ious
    )

    cm = filtered.compute_confusion_matrix(
        score_thresholds=scores, iou_thresholds=ious
    )
    assert len(cm) == 1
    assert cm[0].to_dict() == {
        "parameters": {
            "iou_threshold": 0.1,
            "score_threshold": 0.1,
        },
        "type": "ConfusionMatrix",
        "value": {
            "confusion_matrix": {
                "0": {
                    "0": 0,
                    "1": 0,
                    "2": 0,
                    "3": 0,
                    "4": 0,
                    "49": 0,
                },
                "1": {
                    "0": 0,
                    "1": 0,
                    "2": 0,
                    "3": 0,
                    "4": 0,
                    "49": 0,
                },
                "2": {
                    "0": 0,
                    "1": 0,
                    "2": 1,
                    "3": 1,
                    "4": 0,
                    "49": 0,
                },
                "3": {
                    "0": 0,
                    "1": 0,
                    "2": 0,
                    "3": 0,
                    "4": 0,
                    "49": 0,
                },
                "4": {
                    "0": 0,
                    "1": 0,
                    "2": 0,
                    "3": 0,
                    "4": 0,
                    "49": 0,
                },
                "49": {
                    "0": 0,
                    "1": 0,
                    "2": 0,
                    "3": 0,
                    "4": 0,
                    "49": 0,
                },
            },
            "unmatched_ground_truths": {
                "0": 0,
                "1": 0,
                "2": 0,
                "3": 0,
                "4": 0,
                "49": 0,
            },
            "unmatched_predictions": {
                "0": 0,
                "1": 0,
                "2": 0,
                "3": 0,
                "4": 0,
                "49": 0,
            },
        },
    }
