import numpy as np
from valor_lite import schemas
from valor_lite.detection import (
    Manager,
    MetricType,
    _compute_average_precision,
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

    label_counts = np.array([[1, 5]])
    iou_thresholds = np.array([0.1, 0.6])

    results = _compute_average_precision(
        sorted_pairs,
        label_counts=label_counts,
        iou_thresholds=iou_thresholds,
    )

    expected = np.array(
        [
            [1.0],
            [1 / 3],
        ]
    )

    assert expected.shape == results.shape
    assert np.isclose(results, expected).all()


def test_ap_using_torch_metrics_example(
    evaluate_detection_functional_test_groundtruths,
    evaluate_detection_functional_test_predictions,
):
    """
    cf with torch metrics/pycocotools results listed here:
    https://github.com/Lightning-AI/metrics/blob/107dbfd5fb158b7ae6d76281df44bd94c836bfce/tests/unittests/detection/test_map.py#L231
    """
    manager = Manager()
    manager.add_data(
        groundtruths=evaluate_detection_functional_test_groundtruths,
        predictions=evaluate_detection_functional_test_predictions,
    )
    manager.finalize()

    assert manager.ignored_prediction_labels == [
        schemas.Label(key="class", value="3")
    ]
    assert manager.missing_prediction_labels == []
    assert manager.n_datums == 4
    assert manager.n_labels == 6
    assert manager.n_groundtruths == 20
    assert manager.n_predictions == 19

    ap_metrics = manager.evaluate(
        iou_thresholds=[0.5, 0.75],
    )[MetricType.AP]

    expected_metrics = [
        {
            "type": "AP",
            "values": {"0.5": 1.0, "0.75": 0.7227722772277229},
            "label": {"key": "class", "value": "0"},
        },
        {
            "type": "AP",
            "values": {"0.5": 1.0, "0.75": 1.0},
            "label": {"key": "class", "value": "1"},
        },
        {
            "type": "AP",
            "values": {"0.5": 0.504950495049505, "0.75": 0.504950495049505},
            "label": {"key": "class", "value": "2"},
        },
        {
            "type": "AP",
            "values": {"0.5": 1.0, "0.75": 1.0},
            "label": {"key": "class", "value": "4"},
        },
        {
            "type": "AP",
            "values": {"0.5": 0.7909790979097909, "0.75": 0.5756718528995757},
            "label": {"key": "class", "value": "49"},
        },
    ]
    actual_metrics = [m.to_dict() for m in ap_metrics]

    for m in actual_metrics:
        assert m in expected_metrics
    for m in expected_metrics:
        assert m in actual_metrics


def test_ap_metrics(
    evaluate_detection_groundtruths,
    evaluate_detection_predictions,
):
    """
    Test detection evaluations with area thresholds.

    gt_dets1
        datum 1
            - Label (k1, v1) with Annotation area = 1500
            - Label (k2, v2) with Annotation area = 57,510
        datum2
            - Label (k1, v1) with Annotation area = 1100

    pred_dets
        datum 1
            - Label (k1, v1) with Annotation area = 1500
            - Label (k2, v2) with Annotation area = 57,510
        datum2
            - Label (k1, v1) with Annotation area = 1100
    """

    manager = Manager()
    manager.add_data(
        groundtruths=evaluate_detection_groundtruths,
        predictions=evaluate_detection_predictions,
    )
    manager.finalize()

    ap_metrics = manager.evaluate(
        iou_thresholds=[0.1, 0.6],
    )[MetricType.AP]

    expected_metrics = [
        {
            "type": "AP",
            "values": {"0.1": 0.0, "0.6": 0.0},
            "label": {"key": "k2", "value": "v2"},
        },
        {
            "type": "AP",
            "values": {"0.1": 0.504950495049505, "0.6": 0.504950495049505},
            "label": {"key": "k1", "value": "v1"},
        },
    ]
    actual_metrics = [m.to_dict() for m in ap_metrics]

    for m in actual_metrics:
        assert m in expected_metrics
    for m in expected_metrics:
        assert m in actual_metrics

    assert manager.ignored_prediction_labels == []
    assert manager.missing_prediction_labels == []
    assert manager.n_datums == 2
    assert manager.n_labels == 2
    assert manager.n_groundtruths == 3
    assert manager.n_predictions == 2


def test_evaluate_detection_false_negatives_single_image_baseline(
    evaluate_detection_false_negatives_single_image_baseline_inputs: tuple,
):
    """This is the baseline for the below test. In this case there are two predictions and
    one groundtruth, but the highest confident prediction overlaps sufficiently with the groundtruth
    so there is not a penalty for the false negative so the AP is 1
    """

    (
        groundtruths,
        predictions,
    ) = evaluate_detection_false_negatives_single_image_baseline_inputs

    manager = Manager()
    manager.add_data(
        groundtruths=groundtruths,
        predictions=predictions,
    )
    manager.finalize()
    ap_metrics = manager.evaluate(iou_thresholds=[0.5])[MetricType.AP]

    assert len(ap_metrics) == 1
    assert ap_metrics[0].to_dict() == {
        "type": "AP",
        "values": {"0.5": 1.0},
        "label": {
            "key": "key",
            "value": "value",
        },
    }


def test_evaluate_detection_false_negatives_single_image(
    evaluate_detection_false_negatives_single_image_inputs: tuple,
):
    """Tests fix for a bug where high confidence false negative was not being penalized. The
    difference between this test and the above is that here the prediction with higher confidence
    does not sufficiently overlap the groundtruth and so is penalized and we get an AP of 0.5
    """

    (
        groundtruths,
        predictions,
    ) = evaluate_detection_false_negatives_single_image_inputs

    manager = Manager()
    manager.add_data(
        groundtruths=groundtruths,
        predictions=predictions,
    )
    manager.finalize()
    ap_metrics = manager.evaluate(iou_thresholds=[0.5])[MetricType.AP]

    assert len(ap_metrics) == 1
    assert ap_metrics[0].to_dict() == {
        "type": "AP",
        "values": {"0.5": 0.5},
        "label": {
            "key": "key",
            "value": "value",
        },
    }


def test_evaluate_detection_false_negatives_two_images_one_empty_low_confidence_of_fp(
    evaluate_detection_false_negatives_two_images_one_empty_low_confidence_of_fp_inputs: tuple,
):
    """In this test we have
        1. An image with a matching groundtruth and prediction (same class and high IOU)
        2. A second image with empty groundtruth annotation but a prediction with lower confidence
        then the prediction on the first image.

    In this case, the AP should be 1.0 since the false positive has lower confidence than the true positive

    """

    (
        groundtruths,
        predictions,
    ) = evaluate_detection_false_negatives_two_images_one_empty_low_confidence_of_fp_inputs

    manager = Manager()
    manager.add_data(
        groundtruths=groundtruths,
        predictions=predictions,
    )
    manager.finalize()
    ap_metrics = manager.evaluate(iou_thresholds=[0.5])[MetricType.AP]

    assert len(ap_metrics) == 1
    assert ap_metrics[0].to_dict() == {
        "type": "AP",
        "values": {"0.5": 1.0},
        "label": {
            "key": "key",
            "value": "value",
        },
    }


def test_evaluate_detection_false_negatives_two_images_one_empty_high_confidence_of_fp(
    evaluate_detection_false_negatives_two_images_one_empty_high_confidence_of_fp_inputs: tuple,
):
    """In this test we have
        1. An image with a matching groundtruth and prediction (same class and high IOU)
        2. A second image with empty groundtruth annotation and a prediction with higher confidence
        then the prediction on the first image.

    In this case, the AP should be 0.5 since the false positive has higher confidence than the true positive
    """
    (
        groundtruths,
        predictions,
    ) = evaluate_detection_false_negatives_two_images_one_empty_high_confidence_of_fp_inputs

    manager = Manager()
    manager.add_data(
        groundtruths=groundtruths,
        predictions=predictions,
    )
    manager.finalize()
    ap_metrics = manager.evaluate(iou_thresholds=[0.5])[MetricType.AP]

    assert len(ap_metrics) == 1
    assert ap_metrics[0].to_dict() == {
        "type": "AP",
        "values": {"0.5": 0.5},
        "label": {
            "key": "key",
            "value": "value",
        },
    }


def test_evaluate_detection_false_negatives_two_images_one_only_with_different_class_low_confidence_of_fp(
    evaluate_detection_false_negatives_two_images_one_only_with_different_class_low_confidence_of_fp_inputs: tuple,
):
    """In this test we have
        1. An image with a matching groundtruth and prediction (same class, `"value"`, and high IOU)
        2. A second image with a groundtruth annotation with class `"other value"` and a prediction with lower confidence
        then the prediction on the first image.

    In this case, the AP for class `"value"` should be 1 since the false positive has lower confidence than the true positive.
    AP for class `"other value"` should be 0 since there is no prediction for the `"other value"` groundtruth
    """
    (
        groundtruths,
        predictions,
    ) = evaluate_detection_false_negatives_two_images_one_only_with_different_class_low_confidence_of_fp_inputs
    manager = Manager()
    manager.add_data(
        groundtruths=groundtruths,
        predictions=predictions,
    )
    manager.finalize()
    ap_metrics = manager.evaluate(iou_thresholds=[0.5])[MetricType.AP]

    expected_metrics = [
        {
            "type": "AP",
            "values": {"0.5": 1.0},
            "label": {
                "key": "key",
                "value": "value",
            },
        },
        {
            "type": "AP",
            "values": {"0.5": 0.0},
            "label": {
                "key": "key",
                "value": "other value",
            },
        },
    ]
    actual_metrics = [m.to_dict() for m in ap_metrics]

    assert len(ap_metrics) == 2
    for m in expected_metrics:
        assert m in actual_metrics
    for m in actual_metrics:
        assert m in expected_metrics


def test_evaluate_detection_false_negatives_two_images_one_only_with_different_class_high_confidence_of_fp(
    evaluate_detection_false_negatives_two_images_one_only_with_different_class_high_confidence_of_fp_inputs: tuple,
):
    """In this test we have
        1. An image with a matching groundtruth and prediction (same class, `"value"`, and high IOU)
        2. A second image with a groundtruth annotation with clas `"other value"` and a prediction with higher confidence
        then the prediction on the first image.

    In this case, the AP for class `"value"` should be 0.5 since the false positive has higher confidence than the true positive.
    AP for class `"other value"` should be 0 since there is no prediction for the `"other value"` groundtruth
    """
    (
        groundtruths,
        predictions,
    ) = evaluate_detection_false_negatives_two_images_one_only_with_different_class_high_confidence_of_fp_inputs

    manager = Manager()
    manager.add_data(
        groundtruths=groundtruths,
        predictions=predictions,
    )
    manager.finalize()
    ap_metrics = manager.evaluate(iou_thresholds=[0.5])[MetricType.AP]

    expected_metrics = [
        {
            "type": "AP",
            "values": {"0.5": 0.5},
            "label": {
                "key": "key",
                "value": "value",
            },
        },
        {
            "type": "AP",
            "values": {"0.5": 0.0},
            "label": {
                "key": "key",
                "value": "other value",
            },
        },
    ]
    actual_metrics = [m.to_dict() for m in ap_metrics]

    assert len(ap_metrics) == 2
    for m in expected_metrics:
        assert m in actual_metrics
    for m in actual_metrics:
        assert m in expected_metrics
