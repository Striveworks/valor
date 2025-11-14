from valor_lite.object_detection import Evaluator, MetricType


def test_ap_metrics_first_class(basic_detections_first_class: Evaluator):
    """
    Basic object detection test.

    groundtruths
        datum uid1
            box 1 - label v1 - tp
        datum uid2
            box 2 - label v1 - fn unmatched ground truths

    predictions
        datum uid1
            box 1 - label v1 - score 0.3 - tp
        datum uid2
            none
    """
    evaluator = basic_detections_first_class
    assert evaluator.info.number_of_datums == 2
    assert evaluator.info.number_of_labels == 1
    assert evaluator.info.number_of_groundtruth_annotations == 2
    assert evaluator.info.number_of_prediction_annotations == 1

    # test AP
    metrics = metrics = evaluator.compute_precision_recall(
        iou_thresholds=[0.1, 0.6],
        score_thresholds=[0.5],
    )
    actual_metrics = [m.to_dict() for m in metrics[MetricType.AP]]
    expected_metrics = [
        {
            "type": "AP",
            "value": 0.504950495049505,
            "parameters": {
                "iou_threshold": 0.1,
                "label": "v1",
            },
        },
        {
            "type": "AP",
            "value": 0.504950495049505,
            "parameters": {
                "iou_threshold": 0.6,
                "label": "v1",
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
                "iou_threshold": 0.1,
            },
        },
        {
            "type": "mAP",
            "value": 0.504950495049505,
            "parameters": {
                "iou_threshold": 0.6,
            },
        },
    ]
    for m in actual_metrics:
        assert m in expected_metrics
    for m in expected_metrics:
        assert m in actual_metrics

    # test AP Averaged Over IOUs
    actual_metrics = [
        m.to_dict() for m in metrics[MetricType.APAveragedOverIOUs]
    ]
    expected_metrics = [
        {
            "type": "APAveragedOverIOUs",
            "value": 0.504950495049505,
            "parameters": {"iou_thresholds": [0.1, 0.6], "label": "v1"},
        },
    ]
    for m in actual_metrics:
        assert m in expected_metrics
    for m in expected_metrics:
        assert m in actual_metrics

    # test mAP Averaged Over IOUs
    actual_metrics = [
        m.to_dict() for m in metrics[MetricType.mAPAveragedOverIOUs]
    ]
    expected_metrics = [
        {
            "type": "mAPAveragedOverIOUs",
            "value": 0.504950495049505,
            "parameters": {
                "iou_thresholds": [0.1, 0.6],
            },
        },
    ]
    for m in actual_metrics:
        assert m in expected_metrics
    for m in expected_metrics:
        assert m in actual_metrics


def test_ap_metrics_second_class(basic_detections_second_class: Evaluator):
    """
    Basic object detection test.

    groundtruths
        datum uid1
            box 3 - label v2 - fn unmatched ground truths
        datum uid2
           none
    predictions
        datum uid1
            none
        datum uid2
            box 2 - label v2 - score 0.98 - fp
    """
    evaluator = basic_detections_second_class
    assert evaluator.info.number_of_datums == 2
    assert evaluator.info.number_of_labels == 1
    assert evaluator.info.number_of_groundtruth_annotations == 1
    assert evaluator.info.number_of_prediction_annotations == 1

    # test AP
    metrics = evaluator.compute_precision_recall(
        iou_thresholds=[0.1, 0.6],
        score_thresholds=[0.5],
    )
    actual_metrics = [m.to_dict() for m in metrics[MetricType.AP]]
    expected_metrics = [
        {
            "type": "AP",
            "value": 0.0,
            "parameters": {
                "iou_threshold": 0.1,
                "label": "v2",
            },
        },
        {
            "type": "AP",
            "value": 0.0,
            "parameters": {"iou_threshold": 0.6, "label": "v2"},
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
            "value": 0.0,
            "parameters": {
                "iou_threshold": 0.1,
            },
        },
        {
            "type": "mAP",
            "value": 0.0,
            "parameters": {
                "iou_threshold": 0.6,
            },
        },
    ]
    for m in actual_metrics:
        assert m in expected_metrics
    for m in expected_metrics:
        assert m in actual_metrics

    # test AP Averaged Over IOUs
    actual_metrics = [
        m.to_dict() for m in metrics[MetricType.APAveragedOverIOUs]
    ]
    expected_metrics = [
        {
            "type": "APAveragedOverIOUs",
            "value": 0.0,
            "parameters": {
                "iou_thresholds": [0.1, 0.6],
                "label": "v2",
            },
        },
    ]
    for m in actual_metrics:
        assert m in expected_metrics
    for m in expected_metrics:
        assert m in actual_metrics

    # test mAP Averaged Over IOUs
    actual_metrics = [
        m.to_dict() for m in metrics[MetricType.mAPAveragedOverIOUs]
    ]
    expected_metrics = [
        {
            "type": "mAPAveragedOverIOUs",
            "value": 0.0,
            "parameters": {
                "iou_thresholds": [0.1, 0.6],
            },
        },
    ]
    for m in actual_metrics:
        assert m in expected_metrics
    for m in expected_metrics:
        assert m in actual_metrics


def test_ap_basic_detections(basic_detections: Evaluator):
    evaluator = basic_detections
    metrics = evaluator.compute_precision_recall(
        iou_thresholds=[0.5],
        score_thresholds=[0.5],
    )
    actual_metrics = [m.to_dict() for m in metrics[MetricType.AP]]
    expected_metrics = [
        {
            "type": "AP",
            "value": 0.504950495049505,
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


def test_ap_using_torch_metrics_example(torchmetrics_detections: Evaluator):
    """
    cf with torch metrics/pycocotools results listed here:
    https://github.com/Lightning-AI/metrics/blob/107dbfd5fb158b7ae6d76281df44bd94c836bfce/tests/unittests/detection/test_map.py#L231
    """
    evaluator = torchmetrics_detections
    assert evaluator.info.number_of_datums == 4
    assert evaluator.info.number_of_labels == 6
    assert evaluator.info.number_of_groundtruth_annotations == 20
    assert evaluator.info.number_of_prediction_annotations == 19

    # test AP
    metrics = evaluator.compute_precision_recall(
        iou_thresholds=[0.5, 0.75],
        score_thresholds=[0.5],
    )
    actual_metrics = [m.to_dict() for m in metrics[MetricType.AP]]
    expected_metrics = [
        {
            "type": "AP",
            "value": 1.0,
            "parameters": {
                "iou_threshold": 0.5,
                "label": "0",
            },
        },
        {
            "type": "AP",
            "value": 0.7227722772277229,
            "parameters": {
                "iou_threshold": 0.75,
                "label": "0",
            },
        },
        {
            "type": "AP",
            "value": 1.0,
            "parameters": {
                "iou_threshold": 0.5,
                "label": "1",
            },
        },
        {
            "type": "AP",
            "value": 1.0,
            "parameters": {
                "iou_threshold": 0.75,
                "label": "1",
            },
        },
        {
            "type": "AP",
            "value": 0.504950495049505,
            "parameters": {
                "iou_threshold": 0.5,
                "label": "2",
            },
        },
        {
            "type": "AP",
            "value": 0.504950495049505,
            "parameters": {
                "iou_threshold": 0.75,
                "label": "2",
            },
        },
        {
            "type": "AP",
            "value": 0.0,
            "parameters": {
                "iou_threshold": 0.5,
                "label": "3",
            },
        },
        {
            "type": "AP",
            "value": 0.0,
            "parameters": {
                "iou_threshold": 0.75,
                "label": "3",
            },
        },
        {
            "type": "AP",
            "value": 1.0,
            "parameters": {
                "iou_threshold": 0.5,
                "label": "4",
            },
        },
        {
            "type": "AP",
            "value": 1.0,
            "parameters": {
                "iou_threshold": 0.75,
                "label": "4",
            },
        },
        {
            "type": "AP",
            "value": 0.7909790979097909,
            "parameters": {
                "iou_threshold": 0.5,
                "label": "49",
            },
        },
        {
            "type": "AP",
            "value": 0.5756718528995757,
            "parameters": {
                "iou_threshold": 0.75,
                "label": "49",
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
            "value": 0.715988265493216,
            "parameters": {
                "iou_threshold": 0.5,
            },
        },
        {
            "type": "mAP",
            "value": 0.6338991041961339,
            "parameters": {
                "iou_threshold": 0.75,
            },
        },
    ]
    for m in actual_metrics:
        assert m in expected_metrics
    for m in expected_metrics:
        assert m in actual_metrics


def test_ap_false_negatives_single_datum_baseline(
    false_negatives_single_datum_baseline_detections: Evaluator,
):
    """This is the baseline for the below test. In this case there are two predictions and
    one groundtruth, but the highest confident prediction overlaps sufficiently with the groundtruth
    so there is not a penalty for the false negative so the AP is 1
    """
    evaluator = false_negatives_single_datum_baseline_detections
    metrics = evaluator.compute_precision_recall(
        iou_thresholds=[0.5],
        score_thresholds=[0.5],
    )

    actual_metrics = [m.to_dict() for m in metrics[MetricType.AP]]
    expected_metrics = [
        {
            "type": "AP",
            "value": 1.0,
            "parameters": {"iou_threshold": 0.5, "label": "value"},
        }
    ]
    for m in actual_metrics:
        assert m in expected_metrics
    for m in expected_metrics:
        assert m in actual_metrics


def test_ap_false_negatives_single_datum(
    false_negatives_single_datum_detections: Evaluator,
):
    """Tests where high confidence false negative was not being penalized. The
    difference between this test and the above is that here the prediction with higher confidence
    does not sufficiently overlap the groundtruth and so is penalized and we get an AP of 0.5
    """

    evaluator = false_negatives_single_datum_detections
    metrics = evaluator.compute_precision_recall(
        iou_thresholds=[0.5],
        score_thresholds=[0.5],
    )

    actual_metrics = [m.to_dict() for m in metrics[MetricType.AP]]
    expected_metrics = [
        {
            "type": "AP",
            "value": 0.5,
            "parameters": {"iou_threshold": 0.5, "label": "value"},
        }
    ]
    for m in actual_metrics:
        assert m in expected_metrics
    for m in expected_metrics:
        assert m in actual_metrics


def test_ap_false_negatives_two_datums_one_empty_low_confidence_of_fp(
    false_negatives_two_datums_one_empty_low_confidence_of_fp_detections: Evaluator,
):
    """In this test we have
        1. An image with a matching groundtruth and prediction (same class and high IOU)
        2. A second image with empty groundtruth annotation but a prediction with lower confidence
        then the prediction on the first image.

    In this case, the AP should be 1.0 since the false positive has lower confidence than the true positive

    """

    evaluator = (
        false_negatives_two_datums_one_empty_low_confidence_of_fp_detections
    )
    metrics = evaluator.compute_precision_recall(
        iou_thresholds=[0.5],
        score_thresholds=[0.5],
    )

    actual_metrics = [m.to_dict() for m in metrics[MetricType.AP]]
    expected_metrics = [
        {
            "type": "AP",
            "value": 1.0,
            "parameters": {"iou_threshold": 0.5, "label": "value"},
        }
    ]
    for m in actual_metrics:
        assert m in expected_metrics
    for m in expected_metrics:
        assert m in actual_metrics


def test_ap_false_negatives_two_datums_one_empty_high_confidence_of_fp(
    false_negatives_two_datums_one_empty_high_confidence_of_fp_detections: Evaluator,
):
    """In this test we have
        1. An image with a matching groundtruth and prediction (same class and high IOU)
        2. A second image with empty groundtruth annotation and a prediction with higher confidence
        then the prediction on the first image.

    In this case, the AP should be 0.5 since the false positive has higher confidence than the true positive
    """

    evaluator = (
        false_negatives_two_datums_one_empty_high_confidence_of_fp_detections
    )
    metrics = evaluator.compute_precision_recall(
        iou_thresholds=[0.5], score_thresholds=[0.5]
    )

    actual_metrics = [m.to_dict() for m in metrics[MetricType.AP]]
    expected_metrics = [
        {
            "type": "AP",
            "value": 0.5,
            "parameters": {"iou_threshold": 0.5, "label": "value"},
        }
    ]
    for m in actual_metrics:
        assert m in expected_metrics
    for m in expected_metrics:
        assert m in actual_metrics


def test_ap_false_negatives_two_datums_one_only_with_different_class_low_confidence_of_fp(
    false_negatives_two_datums_one_only_with_different_class_low_confidence_of_fp_detections: Evaluator,
):
    """In this test we have
        1. An image with a matching groundtruth and prediction (same class, `"value"`, and high IOU)
        2. A second image with a groundtruth annotation with class `"other value"` and a prediction with lower confidence
        then the prediction on the first image.

    In this case, the AP for class `"value"` should be 1 since the false positive has lower confidence than the true positive.
    AP for class `"other value"` should be 0 since there is no prediction for the `"other value"` groundtruth
    """
    evaluator = false_negatives_two_datums_one_only_with_different_class_low_confidence_of_fp_detections
    metrics = evaluator.compute_precision_recall(
        iou_thresholds=[0.5], score_thresholds=[0.5]
    )

    actual_metrics = [m.to_dict() for m in metrics[MetricType.AP]]
    expected_metrics = [
        {
            "type": "AP",
            "value": 1.0,
            "parameters": {"iou_threshold": 0.5, "label": "value"},
        },
        {
            "type": "AP",
            "value": 0.0,
            "parameters": {"iou_threshold": 0.5, "label": "other value"},
        },
    ]
    for m in actual_metrics:
        assert m in expected_metrics
    for m in expected_metrics:
        assert m in actual_metrics


def test_ap_false_negatives_two_datums_one_only_with_different_class_high_confidence_of_fp(
    false_negatives_two_images_one_only_with_different_class_high_confidence_of_fp_detections: Evaluator,
):
    """In this test we have
        1. An image with a matching groundtruth and prediction (same class, `"value"`, and high IOU)
        2. A second image with a groundtruth annotation with class `"other value"` and a prediction with higher confidence
        then the prediction on the first image.

    In this case, the AP for class `"value"` should be 0.5 since the false positive has higher confidence than the true positive.
    AP for class `"other value"` should be 0 since there is no prediction for the `"other value"` groundtruth
    """
    evaluator = false_negatives_two_images_one_only_with_different_class_high_confidence_of_fp_detections
    metrics = evaluator.compute_precision_recall(
        iou_thresholds=[0.5], score_thresholds=[0.5]
    )

    actual_metrics = [m.to_dict() for m in metrics[MetricType.AP]]
    expected_metrics = [
        {
            "type": "AP",
            "value": 0.5,
            "parameters": {"iou_threshold": 0.5, "label": "value"},
        },
        {
            "type": "AP",
            "value": 0.0,
            "parameters": {"iou_threshold": 0.5, "label": "other value"},
        },
    ]
    for m in actual_metrics:
        assert m in expected_metrics
    for m in expected_metrics:
        assert m in actual_metrics


def test_ap_ranked_pair_ordering(detection_ranked_pair_ordering: Evaluator):
    evaluator = detection_ranked_pair_ordering
    assert evaluator.info.number_of_datums == 1
    assert evaluator.info.number_of_groundtruth_annotations == 3
    assert evaluator.info.number_of_labels == 4
    assert evaluator.info.number_of_prediction_annotations == 4

    metrics = evaluator.compute_precision_recall(
        iou_thresholds=[0.5, 0.75], score_thresholds=[0.5]
    )

    actual_metrics = [m.to_dict() for m in metrics[MetricType.AP]]
    expected_metrics = [
        {
            "parameters": {
                "iou_threshold": 0.5,
                "label": "label1",
            },
            "value": 1.0,
            "type": "AP",
        },
        {
            "parameters": {
                "iou_threshold": 0.75,
                "label": "label1",
            },
            "value": 1.0,
            "type": "AP",
        },
        {
            "parameters": {
                "iou_threshold": 0.5,
                "label": "label2",
            },
            "value": 1.0,
            "type": "AP",
        },
        {
            "parameters": {
                "iou_threshold": 0.75,
                "label": "label2",
            },
            "value": 1.0,
            "type": "AP",
        },
        {
            "parameters": {
                "iou_threshold": 0.5,
                "label": "label3",
            },
            "value": 0.0,
            "type": "AP",
        },
        {
            "parameters": {
                "iou_threshold": 0.75,
                "label": "label3",
            },
            "value": 0.0,
            "type": "AP",
        },
        {
            "parameters": {
                "iou_threshold": 0.5,
                "label": "label4",
            },
            "value": 0.0,
            "type": "AP",
        },
        {
            "parameters": {
                "iou_threshold": 0.75,
                "label": "label4",
            },
            "value": 0.0,
            "type": "AP",
        },
    ]
    for m in actual_metrics:
        assert m in expected_metrics
    for m in expected_metrics:
        assert m in actual_metrics

    actual_metrics = [m.to_dict() for m in metrics[MetricType.mAP]]
    expected_metrics = [
        {
            "parameters": {"iou_threshold": 0.5},
            "value": 0.5,
            "type": "mAP",
        },
        {
            "parameters": {"iou_threshold": 0.75},
            "value": 0.5,
            "type": "mAP",
        },
    ]
    for m in actual_metrics:
        assert m in expected_metrics
    for m in expected_metrics:
        assert m in actual_metrics

    actual_metrics = [
        m.to_dict() for m in metrics[MetricType.APAveragedOverIOUs]
    ]
    expected_metrics = [
        {
            "parameters": {
                "label": "label1",
                "iou_thresholds": [0.5, 0.75],
            },
            "value": 1.0,
            "type": "APAveragedOverIOUs",
        },
        {
            "parameters": {
                "iou_thresholds": [0.5, 0.75],
                "label": "label2",
            },
            "value": 1.0,
            "type": "APAveragedOverIOUs",
        },
        {
            "parameters": {
                "iou_thresholds": [0.5, 0.75],
                "label": "label3",
            },
            "value": 0.0,
            "type": "APAveragedOverIOUs",
        },
        {
            "parameters": {
                "iou_thresholds": [0.5, 0.75],
                "label": "label4",
            },
            "value": 0.0,
            "type": "APAveragedOverIOUs",
        },
    ]
    for m in actual_metrics:
        assert m in expected_metrics
    for m in expected_metrics:
        assert m in actual_metrics

    actual_metrics = [
        m.to_dict() for m in metrics[MetricType.mAPAveragedOverIOUs]
    ]
    expected_metrics = [
        {
            "parameters": {
                "iou_thresholds": [
                    0.5,
                    0.75,
                ],
            },
            "value": 0.5,
            "type": "mAPAveragedOverIOUs",
        },
    ]
    for m in actual_metrics:
        assert m in expected_metrics
    for m in expected_metrics:
        assert m in actual_metrics


def test_ap_true_positive_deassignment(
    detections_tp_deassignment_edge_case: Evaluator,
):
    evaluator = detections_tp_deassignment_edge_case

    assert evaluator.info.number_of_datums == 1
    assert evaluator.info.number_of_labels == 1
    assert evaluator.info.number_of_groundtruth_annotations == 2
    assert evaluator.info.number_of_prediction_annotations == 4

    metrics = evaluator.compute_precision_recall(
        iou_thresholds=[0.5],
        score_thresholds=[0.5],
    )
    assert len(metrics) == 13

    # test AP
    actual_metrics = [m.to_dict() for m in metrics[MetricType.AP]]
    expected_metrics = [
        {
            "type": "AP",
            "value": 0.504950495049505,
            "parameters": {"iou_threshold": 0.5, "label": "v1"},
        },
    ]
    for m in actual_metrics:
        assert m in expected_metrics
    for m in expected_metrics:
        assert m in actual_metrics
