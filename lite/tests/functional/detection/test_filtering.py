from dataclasses import replace

import numpy as np
import pytest
from valor_lite.detection import BoundingBox, DataLoader, Detection, MetricType


@pytest.fixture
def one_detection(basic_detections: list[Detection]) -> list[Detection]:
    return [basic_detections[0]]


@pytest.fixture
def two_detections(basic_detections: list[Detection]) -> list[Detection]:
    return basic_detections


@pytest.fixture
def four_detections(basic_detections: list[Detection]) -> list[Detection]:
    det1 = basic_detections[0]
    det2 = basic_detections[1]
    det3 = replace(basic_detections[0])
    det4 = replace(basic_detections[1])

    det3.uid = "uid3"
    det4.uid = "uid4"

    return [det1, det2, det3, det4]


def generate_random_detections(
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
            [("cl", choice(labels))],
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

    manager = DataLoader()
    manager.add_data(one_detection)
    evaluator = manager.finalize()

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
                    [1, 1],
                ],
                [
                    [1, 0],
                ],
            ]
        )
    ).all()

    assert (
        evaluator._label_metadata == np.array([[1, 1, 0], [1, 0, 1]])
    ).all()

    # test datum filtering

    filter_ = evaluator.create_filter(datum_uids=["uid1"])
    assert (filter_.indices == np.array([0])).all()
    assert (filter_.label_metadata == np.array([[1, 1, 0], [1, 0, 1]])).all()

    # test label filtering

    filter_ = evaluator.create_filter(labels=[("k1", "v1")])
    assert (filter_.indices == np.array([])).all()

    filter_ = evaluator.create_filter(labels=[("k2", "v2")])
    assert (filter_.indices == np.array([])).all()

    # test label key filtering

    filter_ = evaluator.create_filter(label_keys=["k1"])
    assert (filter_.indices == np.array([0])).all()

    filter_ = evaluator.create_filter(label_keys=["k2"])
    assert (filter_.indices == np.array([])).all()

    # test combo
    filter_ = evaluator.create_filter(
        datum_uids=["uid1"],
        label_keys=["k1"],
    )
    assert (filter_.indices == np.array([0])).all()

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
            "parameters": {"iou": 0.5, "label": {"key": "k1", "value": "v1"}},
        },
        {
            "type": "AP",
            "value": 0.0,
            "parameters": {"iou": 0.5, "label": {"key": "k2", "value": "v2"}},
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
            box 3 - label (k2, v2) - fn missing prediction
        datum uid2
            box 2 - label (k1, v1) - fn misclassification

    predictions
        datum uid1
            box 1 - label (k1, v1) - score 0.3 - tp
        datum uid2
            box 2 - label (k2, v2) - score 0.98 - fp
    """

    manager = DataLoader()
    manager.add_data(two_detections)
    evaluator = manager.finalize()

    assert (
        evaluator._ranked_pairs
        == np.array(
            [
                [1.0, -1.0, 0.0, 0.0, -1.0, 1.0, 0.98],
                [0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.3],
            ]
        )
    ).all()

    assert (
        evaluator._label_metadata_per_datum
        == np.array(
            [
                [
                    [1, 1],
                    [1, 0],
                ],
                [
                    [1, 0],
                    [0, 1],
                ],
            ]
        )
    ).all()

    assert (
        evaluator._label_metadata == np.array([[2, 1, 0], [1, 1, 1]])
    ).all()

    # test datum filtering

    filter_ = evaluator.create_filter(datum_uids=["uid1"])
    assert (filter_.indices == np.array([1])).all()
    assert (filter_.label_metadata == np.array([[1, 1, 0], [1, 0, 1]])).all()

    filter_ = evaluator.create_filter(datum_uids=["uid2"])
    assert (filter_.indices == np.array([0])).all()

    # test label filtering

    filter_ = evaluator.create_filter(labels=[("k1", "v1")])
    assert (filter_.indices == np.array([1])).all()

    filter_ = evaluator.create_filter(labels=[("k2", "v2")])
    assert (filter_.indices == np.array([])).all()

    # test label key filtering

    filter_ = evaluator.create_filter(label_keys=["k1"])
    assert (filter_.indices == np.array([1])).all()

    filter_ = evaluator.create_filter(label_keys=["k2"])
    assert (filter_.indices == np.array([])).all()

    # test combo
    filter_ = evaluator.create_filter(
        datum_uids=["uid1"],
        label_keys=["k1"],
    )
    assert (filter_.indices == np.array([1])).all()

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
            "parameters": {"iou": 0.5, "label": {"key": "k1", "value": "v1"}},
        },
        {
            "type": "AP",
            "value": 0.0,
            "parameters": {"iou": 0.5, "label": {"key": "k2", "value": "v2"}},
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
            box 3 - label (k2, v2) - fn missing prediction
        datum uid2
            box 2 - label (k1, v1) - fn misclassification
        datum uid3
            box 1 - label (k1, v1) - tp
            box 3 - label (k2, v2) - fn missing prediction
        datum uid4
            box 2 - label (k1, v1) - fn misclassification

    predictions
        datum uid1
            box 1 - label (k1, v1) - score 0.3 - tp
        datum uid2
            box 2 - label (k2, v2) - score 0.98 - fp
        datum uid3
            box 1 - label (k1, v1) - score 0.3 - tp
        datum uid4
            box 2 - label (k2, v2) - score 0.98 - fp
    """

    manager = DataLoader()
    manager.add_data(four_detections)
    evaluator = manager.finalize()

    assert (
        evaluator._ranked_pairs
        == np.array(
            [
                [1.0, -1.0, 0.0, 0.0, -1.0, 1.0, 0.98],
                [3.0, -1.0, 0.0, 0.0, -1.0, 1.0, 0.98],
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
                    [1, 1],
                    [1, 0],
                    [1, 1],
                    [1, 0],
                ],
                [
                    [1, 0],
                    [0, 1],
                    [1, 0],
                    [0, 1],
                ],
            ],
            dtype=np.int32,
        )
    ).all()

    assert (
        evaluator._label_metadata == np.array([[4, 2, 0], [2, 2, 1]])
    ).all()

    # test datum filtering

    filter_ = evaluator.create_filter(datum_uids=["uid1"])
    assert (filter_.indices == np.array([2])).all()
    assert (filter_.label_metadata == np.array([[1, 1, 0], [1, 0, 1]])).all()

    filter_ = evaluator.create_filter(datum_uids=["uid2"])
    assert (filter_.indices == np.array([0])).all()

    # test label filtering

    filter_ = evaluator.create_filter(labels=[("k1", "v1")])
    assert (filter_.indices == np.array([2, 3])).all()

    filter_ = evaluator.create_filter(labels=[("k2", "v2")])
    assert (filter_.indices == np.array([])).all()

    # test label key filtering

    filter_ = evaluator.create_filter(label_keys=["k1"])
    assert (filter_.indices == np.array([2, 3])).all()

    filter_ = evaluator.create_filter(label_keys=["k2"])
    assert (filter_.indices == np.array([])).all()

    # test combo
    filter_ = evaluator.create_filter(
        datum_uids=["uid1"],
        label_keys=["k1"],
    )
    assert (filter_.indices == np.array([2])).all()

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
            "parameters": {"iou": 0.5, "label": {"key": "k1", "value": "v1"}},
        },
        {
            "type": "AP",
            "value": 0.0,
            "parameters": {"iou": 0.5, "label": {"key": "k2", "value": "v2"}},
        },
    ]
    for m in actual_metrics:
        assert m in expected_metrics
    for m in expected_metrics:
        assert m in actual_metrics


def test_filtering_random_detections():
    loader = DataLoader()
    loader.add_data(generate_random_detections(13, 4, "abc"))
    evaluator = loader.finalize()
    f = evaluator.create_filter(datum_uids=["uid1"])
    evaluator.evaluate(filter_=f)
