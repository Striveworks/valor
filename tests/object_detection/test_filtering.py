from dataclasses import replace
from uuid import uuid4

import numpy as np
import pytest

from valor_lite.object_detection import (
    BoundingBox,
    DataLoader,
    Detection,
    MetricType,
)


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
    Basic object detection test that combines the labels of basic_detections_first_class and basic_detections_second_class.

    groundtruths
        datum uid1
            box 1 - label v1 - tp
            box 3 - label v2 - fn unmatched ground truths

    predictions
        datum uid1
            box 1 - label v1 - score 0.3 - tp
    """

    loader = DataLoader()
    loader.add_bounding_boxes(one_detection)
    evaluator = loader.finalize()

    assert (evaluator.label_metadata == np.array([[1, 1], [1, 0]])).all()

    # test datum filtering
    evaluator.apply_filter(datum_ids=["uid1"])
    ids, values = evaluator.detailed_pairs
    assert np.all(
        ids
        == np.array(
            [
                [0.0, 0.0, 0.0, 0.0, 0.0],
                [0.0, 1.0, -1.0, 1.0, -1.0],
            ]
        )
    )
    assert np.all(
        values
        == np.array(
            [
                [1.0, 0.3],
                [0.0, -1.0],
            ]
        )
    )
    assert (
        evaluator.label_metadata
        == np.array(
            [
                [
                    1,
                    1,
                ],
                [1, 0],
            ]
        )
    ).all()

    with pytest.raises(KeyError) as e:
        evaluator.apply_filter(datum_ids=["uid2"])
    assert "uid2" in str(e)

    # test label filtering
    evaluator.apply_filter(labels=["v1"])
    ids, values = evaluator.detailed_pairs
    assert np.all(ids == np.array([[0.0, 0.0, 0.0, 0.0, 0.0]]))
    assert np.all(values == np.array([[1.0, 0.3]]))
    assert (evaluator.label_metadata == np.array([[1, 1], [0, 0]])).all()

    evaluator.apply_filter(labels=["v2"])
    ids, values = evaluator.detailed_pairs
    assert np.all(ids == np.array([[0.0, 1.0, -1.0, 1.0, -1.0]]))
    assert np.all(values == np.array([[0.0, -1.0]]))
    assert (evaluator.label_metadata == np.array([[0, 0], [1, 0]])).all()

    # test combo
    evaluator.apply_filter(
        datum_ids=["uid1"],
        labels=["v1"],
    )
    ids, values = evaluator.detailed_pairs
    assert np.all(ids == np.array([[0.0, 0.0, 0.0, 0.0, 0.0]]))
    assert np.all(values == np.array([[1.0, 0.3]]))
    assert (evaluator.label_metadata == np.array([[1, 1], [0, 0]])).all()

    # test evaluation
    evaluator.apply_filter(datum_ids=["uid1"])
    metrics = evaluator.evaluate(iou_thresholds=[0.5])
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


def test_filtering_two_detections(two_detections: list[Detection]):
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

    loader = DataLoader()
    loader.add_bounding_boxes(two_detections)
    evaluator = loader.finalize()

    assert (evaluator.label_metadata == np.array([[2, 1], [1, 1]])).all()

    # test datum filtering
    evaluator.apply_filter(datum_ids=["uid1"])
    ids, values = evaluator.detailed_pairs
    assert np.all(
        ids
        == np.array(
            [
                [0.0, 0.0, 0.0, 0.0, 0.0],
                [0.0, 1.0, -1.0, 1.0, -1.0],
            ]
        )
    )
    assert np.all(
        values
        == np.array(
            [
                [1.0, 0.3],
                [0.0, -1.0],
            ]
        )
    )
    assert (evaluator.label_metadata == np.array([[1, 1], [1, 0]])).all()

    evaluator.apply_filter(datum_ids=["uid2"])
    ids, values = evaluator.detailed_pairs
    assert np.all(ids == np.array([[1.0, 2.0, 1.0, 0.0, 1.0]]))
    assert np.all(values == np.array([[1.0, 0.98]]))
    assert (
        evaluator.label_metadata
        == np.array(
            [
                [
                    1,
                    0,
                ],
                [
                    0,
                    1,
                ],
            ]
        )
    ).all()

    # test label filtering
    evaluator.apply_filter(labels=["v1"])
    ids, values = evaluator.detailed_pairs
    assert np.all(
        ids
        == np.array(
            [
                [1.0, 2.0, -1.0, 0.0, -1.0],
                [0.0, 0.0, 0.0, 0.0, 0.0],
            ]
        )
    )
    assert np.all(
        values
        == np.array(
            [
                [0.0, -1.0],
                [1.0, 0.3],
            ]
        )
    )
    assert (evaluator.label_metadata == np.array([[2, 1], [0, 0]])).all()

    evaluator.apply_filter(labels=["v2"])
    ids, values = evaluator.detailed_pairs
    assert np.all(
        ids
        == np.array(
            [
                [1.0, -1.0, 1.0, -1.0, 1.0],
                [0.0, 1.0, -1.0, 1.0, -1.0],
            ]
        )
    )
    assert np.all(
        values
        == np.array(
            [
                [0.0, 0.98],
                [0.0, -1.0],
            ]
        )
    )
    assert (evaluator.label_metadata == np.array([[0, 0], [1, 1]])).all()

    # test combo
    evaluator.apply_filter(
        datum_ids=["uid1"],
        labels=["v1"],
    )
    ids, values = evaluator.detailed_pairs
    assert np.all(ids == np.array([[0.0, 0.0, 0.0, 0.0, 0.0]]))
    assert np.all(values == np.array([[1.0, 0.3]]))
    assert (evaluator.label_metadata == np.array([[1, 1], [0, 0]])).all()

    # test evaluation
    evaluator.apply_filter(datum_ids=["uid1"])
    metrics = evaluator.evaluate(iou_thresholds=[0.5])

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


def test_filtering_four_detections(four_detections: list[Detection]):
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

    loader = DataLoader()
    loader.add_bounding_boxes(four_detections)
    evaluator = loader.finalize()

    assert (evaluator.label_metadata == np.array([[4, 2], [2, 2]])).all()

    # test datum filtering
    evaluator.apply_filter(datum_ids=["uid1"])
    ids, values = evaluator.detailed_pairs
    assert np.all(
        ids
        == np.array(
            [
                [0.0, 0.0, 0.0, 0.0, 0.0],
                [0.0, 1.0, -1.0, 1.0, -1.0],
            ]
        )
    )
    assert np.all(values == np.array([[1.0, 0.3], [0.0, -1.0]]))
    assert (evaluator.label_metadata == np.array([[1, 1], [1, 0]])).all()

    evaluator.apply_filter(datum_ids=["uid2"])
    ids, values = evaluator.detailed_pairs
    assert np.all(ids == np.array([[1.0, 2.0, 1.0, 0.0, 1.0]]))
    assert np.all(values == np.array([[1.0, 0.98]]))
    assert (evaluator.label_metadata == np.array([[1, 0], [0, 1]])).all()

    # test label filtering
    evaluator.apply_filter(labels=["v1"])
    ids, values = evaluator.detailed_pairs
    assert np.all(
        ids
        == np.array(
            [
                [1.0, 2.0, -1.0, 0.0, -1.0],
                [3.0, 2.0, -1.0, 0.0, -1.0],
                [
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                ],
                [2.0, 0.0, 0.0, 0.0, 0.0],
            ]
        )
    )
    assert np.all(
        values
        == np.array(
            [
                [0.0, -1.0],
                [0.0, -1.0],
                [1.0, 0.3],
                [1.0, 0.3],
            ]
        )
    )
    assert (evaluator.label_metadata == np.array([[4, 2], [0, 0]])).all()

    evaluator.apply_filter(labels=["v2"])
    ids, values = evaluator.detailed_pairs
    assert np.all(
        ids
        == np.array(
            [
                [1.0, -1.0, 1.0, -1.0, 1.0],
                [3.0, -1.0, 1.0, -1.0, 1.0],
                [0.0, 1.0, -1.0, 1.0, -1.0],
                [2.0, 1.0, -1.0, 1.0, -1.0],
            ]
        )
    )
    assert np.all(
        values
        == np.array(
            [
                [0.0, 0.98],
                [0.0, 0.98],
                [0.0, -1.0],
                [0.0, -1.0],
            ]
        )
    )
    assert (evaluator.label_metadata == np.array([[0, 0], [2, 2]])).all()

    # test combo
    evaluator.apply_filter(
        datum_ids=["uid1"],
        labels=["v1"],
    )
    ids, values = evaluator.detailed_pairs
    assert np.all(ids == np.array([[0.0, 0.0, 0.0, 0.0, 0.0]]))
    assert np.all(values == np.array([[1.0, 0.3]]))
    assert (
        evaluator.label_metadata
        == np.array(
            [
                [
                    1,
                    1,
                ],
                [
                    0,
                    0,
                ],
            ]
        )
    ).all()

    # test evaluation
    evaluator.apply_filter(datum_ids=["uid1"])
    metrics = evaluator.evaluate(iou_thresholds=[0.5])
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


def test_filtering_all_detections(four_detections: list[Detection]):
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

    loader = DataLoader()
    loader.add_bounding_boxes(four_detections)
    evaluator = loader.finalize()

    assert (evaluator.label_metadata == np.array([[4, 2], [2, 2]])).all()

    # test datum filtering
    evaluator.apply_filter(datum_ids=[])
    ids, values = evaluator.detailed_pairs
    assert np.all(ids == np.zeros((0, 5)))
    assert np.all(values == np.zeros((0, 2)))
    assert (
        evaluator.label_metadata
        == np.array(
            [
                [
                    0,
                    0,
                ],
                [
                    0,
                    0,
                ],
            ]
        )
    ).all()

    # test label filtering
    evaluator.apply_filter(labels=[])
    ids, values = evaluator.detailed_pairs
    assert np.all(ids == np.zeros((0, 5)))
    assert np.all(values == np.zeros((0, 2)))
    assert (
        evaluator.label_metadata
        == np.array(
            [
                [
                    0,
                    0,
                ],
                [
                    0,
                    0,
                ],
            ]
        )
    ).all()

    # test combo
    evaluator.apply_filter(
        datum_ids=[],
        labels=["v1"],
    )
    ids, values = evaluator.detailed_pairs
    assert np.all(ids == np.zeros((0, 5)))
    assert np.all(values == np.zeros((0, 2)))
    assert (
        evaluator.label_metadata
        == np.array(
            [
                [
                    0,
                    0,
                ],
                [0, 0],
            ]
        )
    ).all()

    # test evaluation
    evaluator.apply_filter(datum_ids=[])
    metrics = evaluator.evaluate(iou_thresholds=[0.5])
    evaluator.compute_confusion_matrix(
        iou_thresholds=[0.5],
        score_thresholds=[0.5],
    )

    actual_metrics = [m.to_dict() for m in metrics[MetricType.AP]]
    assert len(actual_metrics) == 0


def test_filtering_random_detections():
    loader = DataLoader()
    loader.add_bounding_boxes(_generate_random_detections(13, 4, "abc"))
    evaluator = loader.finalize()
    evaluator.apply_filter(datum_ids=["uid1"])
    evaluator.evaluate()
