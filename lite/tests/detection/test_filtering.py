from dataclasses import replace

import numpy as np
import pytest
from valor_lite.detection import BoundingBox, DataLoader, Detection, MetricType


@pytest.fixture
def one_detection(
    basic_detections_first_class: list[Detection],
) -> list[Detection]:
    return [basic_detections_first_class[0]]


@pytest.fixture
def two_detections(
    basic_detections_first_class: list[Detection],
) -> list[Detection]:
    return basic_detections_first_class


@pytest.fixture
def four_detections(
    basic_detections_first_class: list[Detection],
) -> list[Detection]:
    det1 = basic_detections_first_class[0]
    det2 = basic_detections_first_class[1]
    det3 = replace(basic_detections_first_class[0])
    det4 = replace(basic_detections_first_class[1])

    det3.uid = "uid3"
    det4.uid = "uid4"

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
            xmin,
            xmax,
            ymin,
            ymax,
            [choice(labels)],
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


def test_filtering_one_detection(one_detection: list[Detection]):
    """
    Basic object detection test.

    groundtruths
        datum uid1
            box 1 - label (k1, v1) - tp
            box 3 - label (k2, v2) - fn missing prediction

    predictions
        datum uid1
            box 1 - label (k1, v1) - score 0.3 - tp
    """

    loader = DataLoader()
    loader.add_bounding_boxes(one_detection)
    evaluator = loader.finalize()

    assert (
        evaluator._ranked_pairs
        == np.array(
            [
                [0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.3],
            ]
        )
    ).all()

    assert (
        evaluator._label_metadata_per_datum
        == np.array(
            [
                [
                    [1],
                ],
                [
                    [1],
                ],
            ]
        )
    ).all()

    assert (
        evaluator._label_metadata
        == np.array(
            [
                [
                    1,
                    1,
                ]
            ]
        )
    ).all()

    # test datum filtering
    filter_ = evaluator.create_filter(datum_uids=["uid1"])
    assert (filter_.ranked_indices == np.array([0])).all()
    assert (filter_.detailed_indices == np.array([0, 1, 2])).all()
    assert (filter_.label_metadata == np.array([[1, 1]])).all()

    with pytest.raises(KeyError) as e:
        evaluator.create_filter(datum_uids=["uid2"])
    assert "uid2" in str(e)

    # test label filtering
    filter_ = evaluator.create_filter(labels=["v1"])
    assert (filter_.ranked_indices == np.array([0])).all()
    assert (filter_.detailed_indices == np.array([0, 1])).all()
    assert (filter_.label_metadata == np.array([[1, 1]])).all()

    # v2 isn't a label in the data
    with pytest.raises(KeyError):
        evaluator.create_filter(labels=["v2"])

    # test combo
    filter_ = evaluator.create_filter(
        datum_uids=["uid1"],
        labels=["v1"],
    )
    assert (filter_.ranked_indices == np.array([0])).all()
    assert (filter_.detailed_indices == np.array([0, 1])).all()
    assert (filter_.label_metadata == np.array([[1, 1]])).all()

    # test evaluation
    filter_ = evaluator.create_filter(datum_uids=["uid1"])
    metrics = evaluator.evaluate(
        iou_thresholds=[0.5],
        filter_=filter_,
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
    ]
    for m in actual_metrics:
        assert m in expected_metrics
    for m in expected_metrics:
        assert m in actual_metrics


def test_filtering_two_detections(two_detections: list[Detection]):
    """
    Basic object detection test.

    groundtruths
        datum uid1
            box 1 - label (k1, v1) - tp
        datum uid2
            box 2 - label (k1, v1) - fn misclassification

    predictions
        datum uid1
            box 1 - label (k1, v1) - score 0.3 - tp
        datum uid2
            none - fn
    """

    loader = DataLoader()
    loader.add_bounding_boxes(two_detections)
    evaluator = loader.finalize()

    assert (
        evaluator._ranked_pairs
        == np.array([[0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.3]])
    ).all()

    assert (
        evaluator._label_metadata_per_datum
        == np.array(
            [
                [
                    [1],
                    [1],
                ],
                [
                    [1],
                    [0],
                ],
            ]
        )
    ).all()

    assert (evaluator._label_metadata == np.array([[2, 1]])).all()

    # test datum filtering
    filter_ = evaluator.create_filter(datum_uids=["uid1"])
    assert (filter_.ranked_indices == np.array([0])).all()
    assert (filter_.detailed_indices == np.array([0, 1, 2])).all()
    assert (filter_.label_metadata == np.array([[1, 1]])).all()

    filter_ = evaluator.create_filter(datum_uids=["uid2"])
    assert (filter_.ranked_indices == np.array([0])).all()
    assert (filter_.detailed_indices == np.array([3, 4])).all()
    assert (filter_.label_metadata == np.array([[1, 1]])).all()

    # test label filtering
    filter_ = evaluator.create_filter(labels=["v1"])
    assert (filter_.ranked_indices == np.array([0])).all()
    assert (filter_.detailed_indices == np.array([0, 1, 3, 4])).all()
    assert (
        filter_.label_metadata
        == np.array(
            [
                [
                    2,
                    1,
                ]
            ]
        )
    ).all()

    # test combo
    filter_ = evaluator.create_filter(
        datum_uids=["uid1"],
        labels=["v1"],
    )
    assert (filter_.ranked_indices == np.array([0])).all()
    assert (filter_.detailed_indices == np.array([0, 1])).all()
    assert (filter_.label_metadata == np.array([[1, 1]])).all()

    # test evaluation
    filter_ = evaluator.create_filter(datum_uids=["uid1"])
    metrics = evaluator.evaluate(
        iou_thresholds=[0.5],
        filter_=filter_,
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
    ]
    for m in actual_metrics:
        assert m in expected_metrics
    for m in expected_metrics:
        assert m in actual_metrics


def test_filtering_four_detections(four_detections: list[Detection]):
    """
    Basic object detection test.

    groundtruths
        datum uid1
            box 1 - label (k1, v1) - tp
        datum uid2
            box 2 - label (k1, v1) - fn misclassification
        datum uid3
            box 1 - label (k1, v1) - tp
        datum uid4
            box 2 - label (k1, v1) - fn misclassification


    predictions
        datum uid1
            box 1 - label (k1, v1) - score 0.3 - tp
        datum uid2
            none - fn
        datum uid3
            box 1 - label (k1, v1) - score 0.3 - tp
        datum uid4
            none - fn
    """

    loader = DataLoader()
    loader.add_bounding_boxes(four_detections)
    evaluator = loader.finalize()

    assert (
        evaluator._ranked_pairs
        == np.array(
            [
                [0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.3],
                [2.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.3],
            ]
        )
    ).all()

    assert (
        evaluator._label_metadata_per_datum
        == np.array(
            [[[1], [1], [1], [1]], [[1], [0], [1], [0]]], dtype=np.int32
        )
    ).all()

    assert (evaluator._label_metadata == np.array([[4, 2]])).all()

    # test datum filtering
    filter_ = evaluator.create_filter(datum_uids=["uid1"])
    assert (filter_.ranked_indices == np.array([0])).all()
    assert (filter_.detailed_indices == np.array([0, 1, 2])).all()
    assert (filter_.label_metadata == np.array([[1, 2]])).all()

    filter_ = evaluator.create_filter(datum_uids=["uid2"])
    assert (filter_.ranked_indices == np.array([0])).all()
    assert (filter_.detailed_indices == np.array([3, 4])).all()
    assert (filter_.label_metadata == np.array([[1, 2]])).all()

    # test label filtering
    filter_ = evaluator.create_filter(labels=["v1"])
    assert (filter_.ranked_indices == np.array([0, 1])).all()
    assert (
        filter_.detailed_indices == np.array([0, 1, 3, 4, 5, 6, 8, 9])
    ).all()
    assert (filter_.label_metadata == np.array([[4, 2]])).all()

    # test combo
    filter_ = evaluator.create_filter(
        datum_uids=["uid1"],
        labels=["v1"],
    )
    assert (filter_.ranked_indices == np.array([0])).all()
    assert (filter_.detailed_indices == np.array([0, 1])).all()
    assert (filter_.label_metadata == np.array([[1, 2]])).all()

    # test evaluation
    filter_ = evaluator.create_filter(datum_uids=["uid1"])

    metrics = evaluator.evaluate(
        iou_thresholds=[0.5],
        filter_=filter_,
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
    ]
    for m in actual_metrics:
        assert m in expected_metrics
    for m in expected_metrics:
        assert m in actual_metrics


def test_filtering_all_detections(four_detections: list[Detection]):
    """
    Basic object detection test.

    groundtruths
        datum uid1
            box 1 - label (k1, v1) - tp
        datum uid2
            box 2 - label (k1, v1) - fn misclassification
        datum uid3
            box 1 - label (k1, v1) - tp
        datum uid4
            box 2 - label (k1, v1) - fn misclassification


    predictions
        datum uid1
            box 1 - label (k1, v1) - score 0.3 - tp
        datum uid2
            none - fn
        datum uid3
            box 1 - label (k1, v1) - score 0.3 - tp
        datum uid4
            none - fn
    """

    loader = DataLoader()
    loader.add_bounding_boxes(four_detections)
    evaluator = loader.finalize()

    assert (
        evaluator._ranked_pairs
        == np.array(
            [
                [0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.3],
                [2.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.3],
            ]
        )
    ).all()

    assert (
        evaluator._label_metadata_per_datum
        == np.array(
            [
                [
                    [1],
                    [1],
                    [1],
                    [1],
                ],
                [
                    [1],
                    [0],
                    [1],
                    [0],
                ],
            ],
            dtype=np.int32,
        )
    ).all()

    assert (evaluator._label_metadata == np.array([[4, 2]])).all()

    # test datum filtering
    filter_ = evaluator.create_filter(datum_uids=[])
    assert (filter_.ranked_indices == np.array([])).all()
    assert (filter_.detailed_indices == np.array([])).all()
    assert (filter_.label_metadata == np.array([[0, 2]])).all()

    # test label filtering
    filter_ = evaluator.create_filter(labels=[])
    assert (filter_.ranked_indices == np.array([])).all()
    assert (filter_.detailed_indices == np.array([])).all()
    assert (filter_.label_metadata == np.array([[0, 2]])).all()

    # test combo
    filter_ = evaluator.create_filter(
        datum_uids=[],
        labels=["v1"],
    )
    assert (filter_.ranked_indices == np.array([])).all()
    assert (filter_.detailed_indices == np.array([])).all()
    assert (filter_.label_metadata == np.array([[0, 2]])).all()

    # test evaluation
    filter_ = evaluator.create_filter(datum_uids=[])

    metrics = evaluator.evaluate(
        iou_thresholds=[0.5],
        filter_=filter_,
        metrics_to_return=[
            *MetricType.base_metrics(),
            MetricType.ConfusionMatrix,
        ],
    )

    actual_metrics = [m.to_dict() for m in metrics[MetricType.AP]]
    assert len(actual_metrics) == 0


def test_filtering_random_detections():
    loader = DataLoader()
    loader.add_bounding_boxes(_generate_random_detections(13, 4, "abc"))
    evaluator = loader.finalize()
    f = evaluator.create_filter(datum_uids=["uid1"])
    evaluator.evaluate(filter_=f)
