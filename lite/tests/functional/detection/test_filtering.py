import numpy as np
from valor_lite.detection import DataLoader, Detection, MetricType


def test_filtering(basic_detections: list[Detection]):
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
    manager.add_data(basic_detections)
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
    assert (filter_.mask == np.array([False, True])).all()
    assert (filter_.label_metadata == np.array([[1, 1, 0], [1, 0, 1]])).all()

    filter_ = evaluator.create_filter(datum_uids=["uid2"])
    assert (filter_.mask == np.array([True, False])).all()

    # test label filtering

    filter_ = evaluator.create_filter(labels=[("k1", "v1")])
    assert (filter_.mask == np.array([False, True])).all()

    filter_ = evaluator.create_filter(labels=[("k2", "v2")])
    assert (filter_.mask == np.array([False, False])).all()

    # test label key filtering

    filter_ = evaluator.create_filter(label_keys=["k1"])
    assert (filter_.mask == np.array([False, True])).all()

    filter_ = evaluator.create_filter(label_keys=["k2"])
    assert (filter_.mask == np.array([False, False])).all()

    # test combo
    filter_ = evaluator.create_filter(
        datum_uids=["uid1"],
        label_keys=["k1"],
    )
    assert (filter_.mask == np.array([False, True])).all()

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
