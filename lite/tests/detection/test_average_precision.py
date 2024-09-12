import numpy as np
from valor_lite.detection import (
    DataLoader,
    Detection,
    MetricType,
    compute_metrics,
)


def test__compute_average_precision():

    sorted_pairs = np.array(
        [
            # dt,  gt,  pd,  iou,  gl,  pl, score,
            [0.0, 0.0, 2.0, 0.25, 0.0, 0.0, 0.95],
            [0.0, 0.0, 3.0, 0.33333, 0.0, 0.0, 0.9],
            [0.0, 0.0, 4.0, 0.66667, 0.0, 0.0, 0.65],
            [0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.1],
            [0.0, 0.0, 1.0, 0.5, 0.0, 0.0, 0.01],
        ]
    )

    label_counts = np.array([[1, 5, 0]])
    iou_thresholds = np.array([0.1, 0.6])
    score_thresholds = np.array([0.0])

    (results, _, _, _,) = compute_metrics(
        sorted_pairs,
        label_counts=label_counts,
        iou_thresholds=iou_thresholds,
        score_thresholds=score_thresholds,
    )
    (
        average_precision,
        mean_average_precision,
        average_precision_averaged_over_ious,
        mean_average_precision_averaged_over_ious,
    ) = results

    expected_ap = np.array(
        [
            [1.0],  # iou = 0.1
            [1 / 3],  # iou = 0.6
        ]
    )
    assert expected_ap.shape == average_precision.shape
    assert np.isclose(average_precision, expected_ap).all()

    # since only one class, ap == map
    assert expected_ap.shape == mean_average_precision.shape
    assert np.isclose(mean_average_precision, expected_ap).all()

    expected_average = np.array([2 / 3])

    assert average_precision_averaged_over_ious.shape == expected_average.shape
    assert np.isclose(
        average_precision_averaged_over_ious, expected_average
    ).all()

    # since only one class, ap == map
    assert (
        mean_average_precision_averaged_over_ious.shape
        == expected_average.shape
    )
    assert np.isclose(
        mean_average_precision_averaged_over_ious, expected_average
    ).all()


def test_ap_metrics(basic_detections: list[Detection]):
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

    metrics = evaluator.evaluate(
        iou_thresholds=[0.1, 0.6],
    )

    assert evaluator.ignored_prediction_labels == []
    assert evaluator.missing_prediction_labels == []
    assert evaluator.n_datums == 2
    assert evaluator.n_labels == 2
    assert evaluator.n_groundtruths == 3
    assert evaluator.n_predictions == 2

    # test AP
    actual_metrics = [m.to_dict() for m in metrics[MetricType.AP]]
    expected_metrics = [
        {
            "type": "AP",
            "value": 0.504950495049505,
            "parameters": {
                "iou": 0.1,
                "label": {"key": "k1", "value": "v1"},
            },
        },
        {
            "type": "AP",
            "value": 0.504950495049505,
            "parameters": {
                "iou": 0.6,
                "label": {"key": "k1", "value": "v1"},
            },
        },
        {
            "type": "AP",
            "value": 0.0,
            "parameters": {
                "iou": 0.1,
                "label": {"key": "k2", "value": "v2"},
            },
        },
        {
            "type": "AP",
            "value": 0.0,
            "parameters": {
                "iou": 0.6,
                "label": {"key": "k2", "value": "v2"},
            },
        },
    ]
    for m in actual_metrics:
        assert m in expected_metrics
    for m in expected_metrics:
        assert m in actual_metrics

    # test mAP
    actual_metrics = [m.to_dict() for m in metrics[MetricType.mAP]]
    expected_metrics = [
        {
            "type": "mAP",
            "value": 0.504950495049505,
            "parameters": {
                "iou": 0.1,
                "label_key": "k1",
            },
        },
        {
            "type": "mAP",
            "value": 0.504950495049505,
            "parameters": {
                "iou": 0.6,
                "label_key": "k1",
            },
        },
        {
            "type": "mAP",
            "value": 0.0,
            "parameters": {
                "iou": 0.1,
                "label_key": "k2",
            },
        },
        {
            "type": "mAP",
            "value": 0.0,
            "parameters": {
                "iou": 0.6,
                "label_key": "k2",
            },
        },
    ]
    for m in actual_metrics:
        assert m in expected_metrics
    for m in expected_metrics:
        assert m in actual_metrics

    # test AP Averaged Over IoUs
    actual_metrics = [
        m.to_dict() for m in metrics[MetricType.APAveragedOverIOUs]
    ]
    expected_metrics = [
        {
            "type": "APAveragedOverIOUs",
            "value": 0.504950495049505,
            "parameters": {
                "ious": [0.1, 0.6],
                "label": {"key": "k1", "value": "v1"},
            },
        },
        {
            "type": "APAveragedOverIOUs",
            "value": 0.0,
            "parameters": {
                "ious": [0.1, 0.6],
                "label": {"key": "k2", "value": "v2"},
            },
        },
    ]
    for m in actual_metrics:
        assert m in expected_metrics
    for m in expected_metrics:
        assert m in actual_metrics

    # test mAP Averaged Over IoUs
    actual_metrics = [
        m.to_dict() for m in metrics[MetricType.mAPAveragedOverIOUs]
    ]
    expected_metrics = [
        {
            "type": "mAPAveragedOverIOUs",
            "value": 0.504950495049505,
            "parameters": {
                "ious": [0.1, 0.6],
                "label_key": "k1",
            },
        },
        {
            "type": "mAPAveragedOverIOUs",
            "value": 0.0,
            "parameters": {
                "ious": [0.1, 0.6],
                "label_key": "k2",
            },
        },
    ]
    for m in actual_metrics:
        assert m in expected_metrics
    for m in expected_metrics:
        assert m in actual_metrics


def test_ap_using_torch_metrics_example(
    torchmetrics_detections: list[Detection],
):
    """
    cf with torch metrics/pycocotools results listed here:
    https://github.com/Lightning-AI/metrics/blob/107dbfd5fb158b7ae6d76281df44bd94c836bfce/tests/unittests/detection/test_map.py#L231
    """
    manager = DataLoader()
    manager.add_data(torchmetrics_detections)
    evaluator = manager.finalize()

    assert evaluator.ignored_prediction_labels == [("class", "3")]
    assert evaluator.missing_prediction_labels == []
    assert evaluator.n_datums == 4
    assert evaluator.n_labels == 6
    assert evaluator.n_groundtruths == 20
    assert evaluator.n_predictions == 19

    metrics = evaluator.evaluate(
        iou_thresholds=[0.5, 0.75],
    )

    # test AP
    actual_metrics = [m.to_dict() for m in metrics[MetricType.AP]]
    expected_metrics = [
        {
            "type": "AP",
            "value": 1.0,
            "parameters": {
                "iou": 0.5,
                "label": {"key": "class", "value": "0"},
            },
        },
        {
            "type": "AP",
            "value": 0.7227722772277229,
            "parameters": {
                "iou": 0.75,
                "label": {"key": "class", "value": "0"},
            },
        },
        {
            "type": "AP",
            "value": 1.0,
            "parameters": {
                "iou": 0.5,
                "label": {"key": "class", "value": "1"},
            },
        },
        {
            "type": "AP",
            "value": 1.0,
            "parameters": {
                "iou": 0.75,
                "label": {"key": "class", "value": "1"},
            },
        },
        {
            "type": "AP",
            "value": 0.504950495049505,
            "parameters": {
                "iou": 0.5,
                "label": {"key": "class", "value": "2"},
            },
        },
        {
            "type": "AP",
            "value": 0.504950495049505,
            "parameters": {
                "iou": 0.75,
                "label": {"key": "class", "value": "2"},
            },
        },
        {
            "type": "AP",
            "value": 1.0,
            "parameters": {
                "iou": 0.5,
                "label": {"key": "class", "value": "4"},
            },
        },
        {
            "type": "AP",
            "value": 1.0,
            "parameters": {
                "iou": 0.75,
                "label": {"key": "class", "value": "4"},
            },
        },
        {
            "type": "AP",
            "value": 0.7909790979097909,
            "parameters": {
                "iou": 0.5,
                "label": {"key": "class", "value": "49"},
            },
        },
        {
            "type": "AP",
            "value": 0.5756718528995757,
            "parameters": {
                "iou": 0.75,
                "label": {"key": "class", "value": "49"},
            },
        },
    ]
    for m in actual_metrics:
        assert m in expected_metrics
    for m in expected_metrics:
        assert m in actual_metrics

    # test mAP
    actual_metrics = [m.to_dict() for m in metrics[MetricType.mAP]]
    expected_metrics = [
        {
            "type": "mAP",
            "value": 0.8591859185918592,
            "parameters": {
                "iou": 0.5,
                "label_key": "class",
            },
        },
        {
            "type": "mAP",
            "value": 0.7606789250353607,
            "parameters": {
                "iou": 0.75,
                "label_key": "class",
            },
        },
    ]
    for m in actual_metrics:
        assert m in expected_metrics
    for m in expected_metrics:
        assert m in actual_metrics


def test_ap_false_negatives_single_datum_baseline(
    false_negatives_single_datum_baseline_detections: list[Detection],
):
    """This is the baseline for the below test. In this case there are two predictions and
    one groundtruth, but the highest confident prediction overlaps sufficiently with the groundtruth
    so there is not a penalty for the false negative so the AP is 1
    """

    manager = DataLoader()
    manager.add_data(false_negatives_single_datum_baseline_detections)
    evaluator = manager.finalize()
    metrics = evaluator.evaluate(iou_thresholds=[0.5])

    actual_metrics = [m.to_dict() for m in metrics[MetricType.AP]]
    expected_metrics = [
        {
            "type": "AP",
            "value": 1.0,
            "parameters": {
                "iou": 0.5,
                "label": {
                    "key": "key",
                    "value": "value",
                },
            },
        }
    ]
    for m in expected_metrics:
        assert m in actual_metrics
    for m in actual_metrics:
        assert m in expected_metrics


def test_ap_false_negatives_single_datum(
    false_negatives_single_datum_detections: list[Detection],
):
    """Tests where high confidence false negative was not being penalized. The
    difference between this test and the above is that here the prediction with higher confidence
    does not sufficiently overlap the groundtruth and so is penalized and we get an AP of 0.5
    """

    manager = DataLoader()
    manager.add_data(false_negatives_single_datum_detections)
    evaluator = manager.finalize()
    metrics = evaluator.evaluate(iou_thresholds=[0.5])

    actual_metrics = [m.to_dict() for m in metrics[MetricType.AP]]
    expected_metrics = [
        {
            "type": "AP",
            "value": 0.5,
            "parameters": {
                "iou": 0.5,
                "label": {
                    "key": "key",
                    "value": "value",
                },
            },
        }
    ]
    for m in expected_metrics:
        assert m in actual_metrics
    for m in actual_metrics:
        assert m in expected_metrics


def test_ap_false_negatives_two_datums_one_empty_low_confidence_of_fp(
    false_negatives_two_datums_one_empty_low_confidence_of_fp_detections: list[
        Detection
    ],
):
    """In this test we have
        1. An image with a matching groundtruth and prediction (same class and high IOU)
        2. A second image with empty groundtruth annotation but a prediction with lower confidence
        then the prediction on the first image.

    In this case, the AP should be 1.0 since the false positive has lower confidence than the true positive

    """

    manager = DataLoader()
    manager.add_data(
        false_negatives_two_datums_one_empty_low_confidence_of_fp_detections
    )
    evaluator = manager.finalize()
    metrics = evaluator.evaluate(iou_thresholds=[0.5])

    actual_metrics = [m.to_dict() for m in metrics[MetricType.AP]]
    expected_metrics = [
        {
            "type": "AP",
            "value": 1.0,
            "parameters": {
                "iou": 0.5,
                "label": {
                    "key": "key",
                    "value": "value",
                },
            },
        }
    ]
    for m in expected_metrics:
        assert m in actual_metrics
    for m in actual_metrics:
        assert m in expected_metrics


def test_ap_false_negatives_two_datums_one_empty_high_confidence_of_fp(
    false_negatives_two_datums_one_empty_high_confidence_of_fp_detections: list[
        Detection
    ],
):
    """In this test we have
        1. An image with a matching groundtruth and prediction (same class and high IOU)
        2. A second image with empty groundtruth annotation and a prediction with higher confidence
        then the prediction on the first image.

    In this case, the AP should be 0.5 since the false positive has higher confidence than the true positive
    """

    manager = DataLoader()
    manager.add_data(
        false_negatives_two_datums_one_empty_high_confidence_of_fp_detections
    )
    evaluator = manager.finalize()
    metrics = evaluator.evaluate(iou_thresholds=[0.5])

    actual_metrics = [m.to_dict() for m in metrics[MetricType.AP]]
    expected_metrics = [
        {
            "type": "AP",
            "value": 0.5,
            "parameters": {
                "iou": 0.5,
                "label": {
                    "key": "key",
                    "value": "value",
                },
            },
        }
    ]
    for m in expected_metrics:
        assert m in actual_metrics
    for m in actual_metrics:
        assert m in expected_metrics


def test_ap_false_negatives_two_datums_one_only_with_different_class_low_confidence_of_fp(
    false_negatives_two_datums_one_only_with_different_class_low_confidence_of_fp_detections: list[
        Detection
    ],
):
    """In this test we have
        1. An image with a matching groundtruth and prediction (same class, `"value"`, and high IOU)
        2. A second image with a groundtruth annotation with class `"other value"` and a prediction with lower confidence
        then the prediction on the first image.

    In this case, the AP for class `"value"` should be 1 since the false positive has lower confidence than the true positive.
    AP for class `"other value"` should be 0 since there is no prediction for the `"other value"` groundtruth
    """
    manager = DataLoader()
    manager.add_data(
        false_negatives_two_datums_one_only_with_different_class_low_confidence_of_fp_detections
    )
    evaluator = manager.finalize()
    metrics = evaluator.evaluate(iou_thresholds=[0.5])

    actual_metrics = [m.to_dict() for m in metrics[MetricType.AP]]
    expected_metrics = [
        {
            "type": "AP",
            "value": 1.0,
            "parameters": {
                "iou": 0.5,
                "label": {
                    "key": "key",
                    "value": "value",
                },
            },
        },
        {
            "type": "AP",
            "value": 0.0,
            "parameters": {
                "iou": 0.5,
                "label": {
                    "key": "key",
                    "value": "other value",
                },
            },
        },
    ]
    for m in expected_metrics:
        assert m in actual_metrics
    for m in actual_metrics:
        assert m in expected_metrics


def test_ap_false_negatives_two_datums_one_only_with_different_class_high_confidence_of_fp(
    false_negatives_two_images_one_only_with_different_class_high_confidence_of_fp_detections: list[
        Detection
    ],
):
    """In this test we have
        1. An image with a matching groundtruth and prediction (same class, `"value"`, and high IOU)
        2. A second image with a groundtruth annotation with class `"other value"` and a prediction with higher confidence
        then the prediction on the first image.

    In this case, the AP for class `"value"` should be 0.5 since the false positive has higher confidence than the true positive.
    AP for class `"other value"` should be 0 since there is no prediction for the `"other value"` groundtruth
    """
    manager = DataLoader()
    manager.add_data(
        false_negatives_two_images_one_only_with_different_class_high_confidence_of_fp_detections
    )
    evaluator = manager.finalize()
    metrics = evaluator.evaluate(iou_thresholds=[0.5])

    actual_metrics = [m.to_dict() for m in metrics[MetricType.AP]]
    expected_metrics = [
        {
            "type": "AP",
            "value": 0.5,
            "parameters": {
                "iou": 0.5,
                "label": {
                    "key": "key",
                    "value": "value",
                },
            },
        },
        {
            "type": "AP",
            "value": 0.0,
            "parameters": {
                "iou": 0.5,
                "label": {
                    "key": "key",
                    "value": "other value",
                },
            },
        },
    ]
    for m in expected_metrics:
        assert m in actual_metrics
    for m in actual_metrics:
        assert m in expected_metrics
