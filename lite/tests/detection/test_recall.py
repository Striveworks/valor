from valor_lite.detection import DataLoader, Detection, MetricType


def test_recall_metrics(basic_detections: list[Detection]):
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
        score_thresholds=[0.0, 0.5],
    )

    assert evaluator.ignored_prediction_labels == []
    assert evaluator.missing_prediction_labels == []
    assert evaluator.n_datums == 2
    assert evaluator.n_labels == 2
    assert evaluator.n_groundtruths == 3
    assert evaluator.n_predictions == 2

    # test Recall
    actual_metrics = [m.to_dict() for m in metrics[MetricType.Recall]]
    expected_metrics = [
        {
            "type": "Recall",
            "value": 0.0,
            "parameters": {
                "iou": 0.1,
                "score": 0.0,
                "label": {"key": "k2", "value": "v2"},
            },
        },
        {
            "type": "Recall",
            "value": 0.0,
            "parameters": {
                "iou": 0.6,
                "score": 0.0,
                "label": {"key": "k2", "value": "v2"},
            },
        },
        {
            "type": "Recall",
            "value": 0.5,
            "parameters": {
                "iou": 0.1,
                "score": 0.0,
                "label": {"key": "k1", "value": "v1"},
            },
        },
        {
            "type": "Recall",
            "value": 0.5,
            "parameters": {
                "iou": 0.6,
                "score": 0.0,
                "label": {"key": "k1", "value": "v1"},
            },
        },
        {
            "type": "Recall",
            "value": 0.0,
            "parameters": {
                "iou": 0.1,
                "score": 0.5,
                "label": {"key": "k2", "value": "v2"},
            },
        },
        {
            "type": "Recall",
            "value": 0.0,
            "parameters": {
                "iou": 0.6,
                "score": 0.5,
                "label": {"key": "k2", "value": "v2"},
            },
        },
        {
            "type": "Recall",
            "value": 0.0,
            "parameters": {
                "iou": 0.1,
                "score": 0.5,
                "label": {"key": "k1", "value": "v1"},
            },
        },
        {
            "type": "Recall",
            "value": 0.0,
            "parameters": {
                "iou": 0.6,
                "score": 0.5,
                "label": {"key": "k1", "value": "v1"},
            },
        },
    ]
    for m in actual_metrics:
        assert m in expected_metrics
    for m in expected_metrics:
        assert m in actual_metrics


def test_recall_false_negatives_single_datum_baseline(
    false_negatives_single_datum_baseline_detections: list[Detection],
):
    """This is the baseline for the below test. In this case there are two predictions and
    one groundtruth, but the highest confident prediction overlaps sufficiently with the groundtruth
    so there is not a penalty for the false negative so the AP is 1
    """

    manager = DataLoader()
    manager.add_data(false_negatives_single_datum_baseline_detections)
    evaluator = manager.finalize()

    metrics = evaluator.evaluate(
        iou_thresholds=[0.5], score_thresholds=[0.0, 0.9]
    )

    actual_metrics = [m.to_dict() for m in metrics[MetricType.Recall]]
    expected_metrics = [
        {
            "type": "Recall",
            "value": 1.0,
            "parameters": {
                "iou": 0.5,
                "score": 0.0,
                "label": {
                    "key": "key",
                    "value": "value",
                },
            },
        },
        {
            "type": "Recall",
            "value": 0.0,
            "parameters": {
                "iou": 0.5,
                "score": 0.9,
                "label": {
                    "key": "key",
                    "value": "value",
                },
            },
        },
    ]
    for m in actual_metrics:
        assert m in expected_metrics
    for m in expected_metrics:
        assert m in actual_metrics


def test_recall_false_negatives_single_datum(
    false_negatives_single_datum_detections: list[Detection],
):
    """Tests where high confidence false negative was not being penalized. The
    difference between this test and the above is that here the prediction with higher confidence
    does not sufficiently overlap the groundtruth and so is penalized and we get an AP of 0.5
    """

    manager = DataLoader()
    manager.add_data(false_negatives_single_datum_detections)
    evaluator = manager.finalize()
    metrics = evaluator.evaluate(iou_thresholds=[0.5], score_thresholds=[0.0])

    actual_metrics = [m.to_dict() for m in metrics[MetricType.Recall]]
    expected_metrics = [
        {
            "type": "Recall",
            "value": 1.0,
            "parameters": {
                "iou": 0.5,
                "score": 0.0,
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


def test_recall_false_negatives_two_datums_one_empty_low_confidence_of_fp(
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
    metrics = evaluator.evaluate(iou_thresholds=[0.5], score_thresholds=[0.0])

    actual_metrics = [m.to_dict() for m in metrics[MetricType.Recall]]
    expected_metrics = [
        {
            "type": "Recall",
            "value": 1.0,
            "parameters": {
                "iou": 0.5,
                "score": 0.0,
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


def test_recall_false_negatives_two_datums_one_empty_high_confidence_of_fp(
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
    metrics = evaluator.evaluate(iou_thresholds=[0.5], score_thresholds=[0.0])

    actual_metrics = [m.to_dict() for m in metrics[MetricType.Recall]]
    expected_metrics = [
        {
            "type": "Recall",
            "value": 1.0,
            "parameters": {
                "iou": 0.5,
                "score": 0.0,
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


def test_recall_false_negatives_two_datums_one_only_with_different_class_low_confidence_of_fp(
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
    metrics = evaluator.evaluate(iou_thresholds=[0.5], score_thresholds=[0.0])

    actual_metrics = [m.to_dict() for m in metrics[MetricType.Recall]]
    expected_metrics = [
        {
            "type": "Recall",
            "value": 1.0,
            "parameters": {
                "iou": 0.5,
                "score": 0.0,
                "label": {
                    "key": "key",
                    "value": "value",
                },
            },
        },
        {
            "type": "Recall",
            "value": 0.0,
            "parameters": {
                "iou": 0.5,
                "score": 0.0,
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


def test_recall_false_negatives_two_datums_one_only_with_different_class_high_confidence_of_fp(
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
    metrics = evaluator.evaluate(iou_thresholds=[0.5], score_thresholds=[0.0])

    actual_metrics = [m.to_dict() for m in metrics[MetricType.Recall]]
    expected_metrics = [
        {
            "type": "Recall",
            "value": 1.0,
            "parameters": {
                "iou": 0.5,
                "score": 0.0,
                "label": {
                    "key": "key",
                    "value": "value",
                },
            },
        },
        {
            "type": "Recall",
            "value": 0.0,
            "parameters": {
                "iou": 0.5,
                "score": 0.0,
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
