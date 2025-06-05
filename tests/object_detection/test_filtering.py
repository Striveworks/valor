from copy import deepcopy
from uuid import uuid4

import numpy as np
import pytest

from valor_lite.exceptions import EmptyFilterError
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
    det3 = deepcopy(basic_detections[0])
    det4 = deepcopy(basic_detections[1])

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

    assert (evaluator._label_metadata == np.array([[1, 1], [1, 0]])).all()

    # test datum filtering
    filter_ = evaluator.create_filter(datums=["uid1"])
    detailed_pairs, _, label_metadata = evaluator.filter(filter_)
    assert np.all(
        detailed_pairs
        == np.array(
            [
                [0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.3],
                [0.0, 1.0, -1.0, 1.0, -1.0, 0.0, -1.0],
            ]
        )
    )
    assert (
        label_metadata
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
        filter_ = evaluator.create_filter(datums=["uid2"])
    detailed_pairs, _, label_metadata = evaluator.filter(filter_)
    assert "uid2" in str(e)

    # test label filtering
    filter_ = evaluator.create_filter(labels=["v1"])
    detailed_pairs, _, label_metadata = evaluator.filter(filter_)
    assert np.all(
        detailed_pairs == np.array([[0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.3]])
    )
    assert (label_metadata == np.array([[1, 1], [0, 0]])).all()

    filter_ = evaluator.create_filter(labels=["v2"])
    detailed_pairs, _, label_metadata = evaluator.filter(filter_)
    assert np.all(
        detailed_pairs == np.array([[0.0, 1.0, -1.0, 1.0, -1.0, 0.0, -1.0]])
    )
    assert (label_metadata == np.array([[0, 0], [1, 0]])).all()

    # test combo
    filter_ = evaluator.create_filter(
        datums=["uid1"],
        labels=["v1"],
    )
    detailed_pairs, _, label_metadata = evaluator.filter(filter_)
    assert np.all(
        detailed_pairs == np.array([[0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.3]])
    )
    assert (label_metadata == np.array([[1, 1], [0, 0]])).all()

    # test evaluation
    filter_ = evaluator.create_filter(datums=["uid1"])
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

    assert (evaluator._label_metadata == np.array([[2, 1], [1, 1]])).all()

    # test datum filtering
    filter_ = evaluator.create_filter(datums=["uid1"])
    detailed_pairs, _, label_metadata = evaluator.filter(filter_)
    assert np.all(
        detailed_pairs
        == np.array(
            [
                [0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.3],
                [0.0, 1.0, -1.0, 1.0, -1.0, 0.0, -1.0],
            ]
        )
    )
    assert (label_metadata == np.array([[1, 1], [1, 0]])).all()

    filter_ = evaluator.create_filter(datums=["uid2"])
    detailed_pairs, _, label_metadata = evaluator.filter(filter_)
    assert np.all(
        detailed_pairs == np.array([[1.0, 2.0, 1.0, 0.0, 1.0, 1.0, 0.98]])
    )
    assert (
        label_metadata
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
    filter_ = evaluator.create_filter(labels=["v1"])
    detailed_pairs, _, label_metadata = evaluator.filter(filter_)
    assert np.all(
        detailed_pairs
        == np.array(
            [
                [0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.3],
                [1.0, 2.0, -1.0, 0.0, -1.0, 0.0, -1.0],
            ]
        )
    )
    assert (label_metadata == np.array([[2, 1], [0, 0]])).all()

    filter_ = evaluator.create_filter(labels=["v2"])
    detailed_pairs, _, label_metadata = evaluator.filter(filter_)
    assert np.all(
        detailed_pairs
        == np.array(
            [
                [1.0, -1.0, 1.0, -1.0, 1.0, 0.0, 0.98],
                [0.0, 1.0, -1.0, 1.0, -1.0, 0.0, -1.0],
            ]
        )
    )
    assert (label_metadata == np.array([[0, 0], [1, 1]])).all()

    # test combo
    filter_ = evaluator.create_filter(
        datums=["uid1"],
        labels=["v1"],
    )
    detailed_pairs, _, label_metadata = evaluator.filter(filter_)
    assert np.all(
        detailed_pairs == np.array([[0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.3]])
    )
    assert (label_metadata == np.array([[1, 1], [0, 0]])).all()

    # test evaluation
    filter_ = evaluator.create_filter(datums=["uid1"])
    metrics = evaluator.evaluate(iou_thresholds=[0.5], filter_=filter_)
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

    assert (evaluator._label_metadata == np.array([[4, 2], [2, 2]])).all()

    # test datum filtering
    filter_ = evaluator.create_filter(datums=["uid1"])
    detailed_pairs, _, label_metadata = evaluator.filter(filter_)
    assert np.all(
        detailed_pairs
        == np.array(
            [
                [0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.3],
                [0.0, 1.0, -1.0, 1.0, -1.0, 0.0, -1.0],
            ]
        )
    )
    assert (label_metadata == np.array([[1, 1], [1, 0]])).all()

    filter_ = evaluator.create_filter(datums=["uid2"])
    detailed_pairs, _, label_metadata = evaluator.filter(filter_)
    assert np.all(
        detailed_pairs == np.array([[1.0, 2.0, 1.0, 0.0, 1.0, 1.0, 0.98]])
    )
    assert (label_metadata == np.array([[1, 0], [0, 1]])).all()

    # test label filtering
    filter_ = evaluator.create_filter(labels=["v1"])
    detailed_pairs, _, label_metadata = evaluator.filter(filter_)
    assert np.all(
        detailed_pairs
        == np.array(
            [
                [0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.3],
                [2.0, 3.0, 2.0, 0.0, 0.0, 1.0, 0.3],
                [1.0, 2.0, -1.0, 0.0, -1.0, 0.0, -1.0],
                [3.0, 5.0, -1.0, 0.0, -1.0, 0.0, -1.0],
            ]
        )
    )
    assert (label_metadata == np.array([[4, 2], [0, 0]])).all()

    filter_ = evaluator.create_filter(labels=["v2"])
    detailed_pairs, _, label_metadata = evaluator.filter(filter_)
    assert np.all(
        detailed_pairs
        == np.array(
            [
                [1.0, -1.0, 1.0, -1.0, 1.0, 0.0, 0.98],
                [3.0, -1.0, 3.0, -1.0, 1.0, 0.0, 0.98],
                [0.0, 1.0, -1.0, 1.0, -1.0, 0.0, -1.0],
                [2.0, 4.0, -1.0, 1.0, -1.0, 0.0, -1.0],
            ]
        )
    )
    assert (label_metadata == np.array([[0, 0], [2, 2]])).all()

    # test combo
    filter_ = evaluator.create_filter(
        datums=["uid1"],
        labels=["v1"],
    )
    detailed_pairs, _, label_metadata = evaluator.filter(filter_)
    assert np.all(
        detailed_pairs == np.array([[0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.3]])
    )
    assert (
        label_metadata
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
    filter_ = evaluator.create_filter(datums=["uid1"])
    metrics = evaluator.evaluate(iou_thresholds=[0.5], filter_=filter_)
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

    assert (evaluator._label_metadata == np.array([[4, 2], [2, 2]])).all()

    # test datum filtering
    with pytest.raises(EmptyFilterError):
        evaluator.create_filter(datums=[])

    # test ground truth annotation filtering
    with pytest.warns(UserWarning):
        filter_ = evaluator.create_filter(groundtruths=[])
    detailed_pairs, _, label_metadata = evaluator.filter(filter_)
    assert np.all(
        detailed_pairs
        == np.array(
            [
                [1.0, -1.0, 1.0, -1.0, 1.0, 0.0, 0.98],
                [3.0, -1.0, 3.0, -1.0, 1.0, 0.0, 0.98],
                [0.0, -1.0, 0.0, -1.0, 0.0, 0.0, 0.3],
                [2.0, -1.0, 2.0, -1.0, 0.0, 0.0, 0.3],
            ]
        )
    )
    assert (
        label_metadata
        == np.array(
            [
                [0, 2],
                [0, 2],
            ]
        )
    ).all()

    filter_ = evaluator.create_filter(groundtruths=["uid1_gt_0"])
    detailed_pairs, _, label_metadata = evaluator.filter(filter_)
    assert np.all(
        detailed_pairs
        == np.array(
            [
                [1.0, -1.0, 1.0, -1.0, 1.0, 0.0, 0.98],
                [3.0, -1.0, 3.0, -1.0, 1.0, 0.0, 0.98],
                [0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.3],
                [2.0, -1.0, 2.0, -1.0, 0.0, 0.0, 0.3],
            ]
        )
    )
    assert (
        label_metadata
        == np.array(
            [
                [1, 2],
                [0, 2],
            ]
        )
    ).all()

    # test prediction annotation filtering
    with pytest.warns(UserWarning):
        filter_ = evaluator.create_filter(predictions=[])
    detailed_pairs, _, label_metadata = evaluator.filter(filter_)
    assert np.all(
        detailed_pairs
        == np.array(
            [
                [0.0, 0.0, -1.0, 0.0, -1.0, 0.0, -1.0],
                [0.0, 1.0, -1.0, 1.0, -1.0, 0.0, -1.0],
                [1.0, 2.0, -1.0, 0.0, -1.0, 0.0, -1.0],
                [2.0, 3.0, -1.0, 0.0, -1.0, 0.0, -1.0],
                [2.0, 4.0, -1.0, 1.0, -1.0, 0.0, -1.0],
                [3.0, 5.0, -1.0, 0.0, -1.0, 0.0, -1.0],
            ]
        )
    )
    assert (
        label_metadata
        == np.array(
            [
                [4, 0],
                [2, 0],
            ]
        )
    ).all()

    filter_ = evaluator.create_filter(predictions=["uid1_pd_0"])
    detailed_pairs, _, label_metadata = evaluator.filter(filter_)
    assert np.all(
        detailed_pairs
        == np.array(
            [
                [0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.3],
                [0.0, 1.0, -1.0, 1.0, -1.0, 0.0, -1.0],
                [1.0, 2.0, -1.0, 0.0, -1.0, 0.0, -1.0],
                [2.0, 3.0, -1.0, 0.0, -1.0, 0.0, -1.0],
                [2.0, 4.0, -1.0, 1.0, -1.0, 0.0, -1.0],
                [3.0, 5.0, -1.0, 0.0, -1.0, 0.0, -1.0],
            ]
        )
    )
    assert (
        label_metadata
        == np.array(
            [
                [4, 1],
                [2, 0],
            ]
        )
    ).all()

    # test label filtering
    with pytest.raises(EmptyFilterError):
        filter_ = evaluator.create_filter(labels=[])

    # test combo
    with pytest.raises(EmptyFilterError):
        filter_ = evaluator.create_filter(
            datums=[],
            labels=["v1"],
        )

    # test evaluation
    with pytest.warns(UserWarning):
        filter_ = evaluator.create_filter(groundtruths=[])
    metrics = evaluator.evaluate(iou_thresholds=[0.5], filter_=filter_)
    evaluator.compute_confusion_matrix(
        iou_thresholds=[0.5],
        score_thresholds=[0.5],
        filter_=filter_,
    )
    actual_metrics = [m.to_dict() for m in metrics[MetricType.AP]]
    assert len(actual_metrics) == 0


def test_filtering_random_detections():
    loader = DataLoader()
    loader.add_bounding_boxes(_generate_random_detections(13, 4, "abc"))
    evaluator = loader.finalize()
    filter_ = evaluator.create_filter(datums=["uid1"])
    evaluator.evaluate(filter_=filter_)


def test_filter_metadata(basic_detections: list[Detection]):
    manager = DataLoader()
    manager.add_bounding_boxes(basic_detections)
    evaluator = manager.finalize()

    assert evaluator.metadata.number_of_datums == 2
    assert evaluator.metadata.number_of_ground_truths == 3
    assert evaluator.metadata.number_of_predictions == 2

    filter_ = evaluator.create_filter(datums=["uid1"])
    assert filter_.metadata.number_of_datums == 1
    assert filter_.metadata.number_of_ground_truths == 2
    assert filter_.metadata.number_of_predictions == 1

    assert (
        evaluator.metadata.number_of_labels
        == filter_.metadata.number_of_labels
    )


def test_filtering_four_detections_by_indices(
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

    loader = DataLoader()
    loader.add_bounding_boxes(four_detections)
    evaluator = loader.finalize()

    assert (evaluator._label_metadata == np.array([[4, 2], [2, 2]])).all()

    # test datum filtering
    filter_ = evaluator.create_filter(datums=np.array([0]))
    detailed_pairs, _, label_metadata = evaluator.filter(filter_)
    assert np.all(
        detailed_pairs
        == np.array(
            [
                [0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.3],
                [0.0, 1.0, -1.0, 1.0, -1.0, 0.0, -1.0],
            ]
        )
    )
    assert (label_metadata == np.array([[1, 1], [1, 0]])).all()

    filter_ = evaluator.create_filter(datums=np.array([1]))
    detailed_pairs, _, label_metadata = evaluator.filter(filter_)
    assert np.all(
        detailed_pairs == np.array([[1.0, 2.0, 1.0, 0.0, 1.0, 1.0, 0.98]])
    )
    assert (label_metadata == np.array([[1, 0], [0, 1]])).all()

    # test label filtering
    filter_ = evaluator.create_filter(labels=np.array([0]))
    detailed_pairs, _, label_metadata = evaluator.filter(filter_)
    assert np.all(
        detailed_pairs
        == np.array(
            [
                [0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.3],
                [2.0, 3.0, 2.0, 0.0, 0.0, 1.0, 0.3],
                [1.0, 2.0, -1.0, 0.0, -1.0, 0.0, -1.0],
                [3.0, 5.0, -1.0, 0.0, -1.0, 0.0, -1.0],
            ]
        )
    )
    assert (label_metadata == np.array([[4, 2], [0, 0]])).all()

    filter_ = evaluator.create_filter(labels=np.array([1]))
    detailed_pairs, _, label_metadata = evaluator.filter(filter_)
    assert np.all(
        detailed_pairs
        == np.array(
            [
                [1.0, -1.0, 1.0, -1.0, 1.0, 0.0, 0.98],
                [3.0, -1.0, 3.0, -1.0, 1.0, 0.0, 0.98],
                [0.0, 1.0, -1.0, 1.0, -1.0, 0.0, -1.0],
                [2.0, 4.0, -1.0, 1.0, -1.0, 0.0, -1.0],
            ]
        )
    )
    assert (label_metadata == np.array([[0, 0], [2, 2]])).all()

    # test combo
    filter_ = evaluator.create_filter(
        datums=np.array([0]), labels=np.array([0])
    )
    detailed_pairs, _, label_metadata = evaluator.filter(filter_)
    assert np.all(
        detailed_pairs == np.array([[0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.3]])
    )
    assert (
        label_metadata
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
    filter_ = evaluator.create_filter(datums=np.array([0]))
    metrics = evaluator.evaluate(iou_thresholds=[0.5], filter_=filter_)
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
