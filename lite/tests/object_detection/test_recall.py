import numpy as np
from valor_lite.object_detection import DataLoader, Detection, MetricType
from valor_lite.object_detection.computation import compute_precion_recall


def test__compute_recall():

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

    label_metadata = np.array([[1, 5, 0]])
    iou_thresholds = np.array([0.1, 0.6])
    score_thresholds = np.array([0.0])

    (_, _, _, counts, _) = compute_precion_recall(
        sorted_pairs,
        label_metadata=label_metadata,
        iou_thresholds=iou_thresholds,
        score_thresholds=score_thresholds,
    )

    recall = counts[:, :, :, 4]

    # precision
    expected = np.array(
        [
            [1.0],  # iou = 0.1
            [1.0],  # iou = 0.6
        ]
    )
    assert (recall == expected).all()


def test_recall_metrics_first_class(
    basic_detections_first_class: list[Detection],
    basic_rotated_detections_first_class: list[Detection],
):
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
    for input_, method in [
        (basic_detections_first_class, DataLoader.add_bounding_boxes),
        (basic_rotated_detections_first_class, DataLoader.add_polygons),
    ]:
        loader = DataLoader()
        method(loader, input_)
        evaluator = loader.finalize()

        metrics = evaluator.evaluate(
            iou_thresholds=[0.1, 0.6],
            score_thresholds=[0.0, 0.5],
        )

        assert evaluator.ignored_prediction_labels == []
        assert evaluator.missing_prediction_labels == []
        assert evaluator.n_datums == 2
        assert evaluator.n_labels == 1
        assert evaluator.n_groundtruths == 2
        assert evaluator.n_predictions == 1

        # test Recall
        actual_metrics = [m.to_dict() for m in metrics[MetricType.Recall]]
        expected_metrics = [
            {
                "type": "Recall",
                "value": 0.5,
                "parameters": {
                    "iou_threshold": 0.1,
                    "score_threshold": 0.0,
                    "label": "v1",
                },
            },
            {
                "type": "Recall",
                "value": 0.5,
                "parameters": {
                    "iou_threshold": 0.6,
                    "score_threshold": 0.0,
                    "label": "v1",
                },
            },
            {
                "type": "Recall",
                "value": 0.0,
                "parameters": {
                    "iou_threshold": 0.1,
                    "score_threshold": 0.5,
                    "label": "v1",
                },
            },
            {
                "type": "Recall",
                "value": 0.0,
                "parameters": {
                    "iou_threshold": 0.6,
                    "score_threshold": 0.5,
                    "label": "v1",
                },
            },
        ]
        for m in actual_metrics:
            assert m in expected_metrics
        for m in expected_metrics:
            assert m in actual_metrics


def test_recall_metrics_second_class(
    basic_detections_second_class: list[Detection],
    basic_rotated_detections_second_class: list[Detection],
):
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
    for input_, method in [
        (basic_detections_second_class, DataLoader.add_bounding_boxes),
        (basic_rotated_detections_second_class, DataLoader.add_polygons),
    ]:
        loader = DataLoader()
        method(loader, input_)
        evaluator = loader.finalize()

        metrics = evaluator.evaluate(
            iou_thresholds=[0.1, 0.6],
            score_thresholds=[0.0, 0.5],
        )

        assert evaluator.ignored_prediction_labels == []
        assert evaluator.missing_prediction_labels == []
        assert evaluator.n_datums == 2
        assert evaluator.n_labels == 1
        assert evaluator.n_groundtruths == 1
        assert evaluator.n_predictions == 1

        # test Recall
        actual_metrics = [m.to_dict() for m in metrics[MetricType.Recall]]
        expected_metrics = [
            {
                "type": "Recall",
                "value": 0.0,
                "parameters": {
                    "iou_threshold": 0.1,
                    "score_threshold": 0.0,
                    "label": "v2",
                },
            },
            {
                "type": "Recall",
                "value": 0.0,
                "parameters": {
                    "iou_threshold": 0.6,
                    "score_threshold": 0.0,
                    "label": "v2",
                },
            },
            {
                "type": "Recall",
                "value": 0.0,
                "parameters": {
                    "iou_threshold": 0.1,
                    "score_threshold": 0.5,
                    "label": "v2",
                },
            },
            {
                "type": "Recall",
                "value": 0.0,
                "parameters": {
                    "iou_threshold": 0.6,
                    "score_threshold": 0.5,
                    "label": "v2",
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

    loader = DataLoader()
    loader.add_bounding_boxes(false_negatives_single_datum_baseline_detections)
    evaluator = loader.finalize()

    metrics = evaluator.evaluate(
        iou_thresholds=[0.5],
        score_thresholds=[0.0, 0.9],
    )

    actual_metrics = [m.to_dict() for m in metrics[MetricType.Recall]]
    expected_metrics = [
        {
            "type": "Recall",
            "value": 1.0,
            "parameters": {
                "iou_threshold": 0.5,
                "score_threshold": 0.0,
                "label": "value",
            },
        },
        {
            "type": "Recall",
            "value": 0.0,
            "parameters": {
                "iou_threshold": 0.5,
                "score_threshold": 0.9,
                "label": "value",
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

    loader = DataLoader()
    loader.add_bounding_boxes(false_negatives_single_datum_detections)
    evaluator = loader.finalize()
    metrics = evaluator.evaluate(
        iou_thresholds=[0.5],
        score_thresholds=[0.0],
    )

    actual_metrics = [m.to_dict() for m in metrics[MetricType.Recall]]
    expected_metrics = [
        {
            "type": "Recall",
            "value": 1.0,
            "parameters": {
                "iou_threshold": 0.5,
                "score_threshold": 0.0,
                "label": "value",
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

    loader = DataLoader()
    loader.add_bounding_boxes(
        false_negatives_two_datums_one_empty_low_confidence_of_fp_detections
    )
    evaluator = loader.finalize()
    metrics = evaluator.evaluate(
        iou_thresholds=[0.5],
        score_thresholds=[0.0],
    )

    actual_metrics = [m.to_dict() for m in metrics[MetricType.Recall]]
    expected_metrics = [
        {
            "type": "Recall",
            "value": 1.0,
            "parameters": {
                "iou_threshold": 0.5,
                "score_threshold": 0.0,
                "label": "value",
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

    loader = DataLoader()
    loader.add_bounding_boxes(
        false_negatives_two_datums_one_empty_high_confidence_of_fp_detections
    )
    evaluator = loader.finalize()
    metrics = evaluator.evaluate(
        iou_thresholds=[0.5],
        score_thresholds=[0.0],
    )

    actual_metrics = [m.to_dict() for m in metrics[MetricType.Recall]]
    expected_metrics = [
        {
            "type": "Recall",
            "value": 1.0,
            "parameters": {
                "iou_threshold": 0.5,
                "score_threshold": 0.0,
                "label": "value",
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
    loader = DataLoader()
    loader.add_bounding_boxes(
        false_negatives_two_datums_one_only_with_different_class_low_confidence_of_fp_detections
    )
    evaluator = loader.finalize()
    metrics = evaluator.evaluate(
        iou_thresholds=[0.5],
        score_thresholds=[0.0],
    )

    actual_metrics = [m.to_dict() for m in metrics[MetricType.Recall]]
    expected_metrics = [
        {
            "type": "Recall",
            "value": 1.0,
            "parameters": {
                "iou_threshold": 0.5,
                "score_threshold": 0.0,
                "label": "value",
            },
        },
        {
            "type": "Recall",
            "value": 0.0,
            "parameters": {
                "iou_threshold": 0.5,
                "score_threshold": 0.0,
                "label": "other value",
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
    loader = DataLoader()
    loader.add_bounding_boxes(
        false_negatives_two_images_one_only_with_different_class_high_confidence_of_fp_detections
    )
    evaluator = loader.finalize()
    metrics = evaluator.evaluate(
        iou_thresholds=[0.5],
        score_thresholds=[0.0],
    )

    actual_metrics = [m.to_dict() for m in metrics[MetricType.Recall]]
    expected_metrics = [
        {
            "type": "Recall",
            "value": 1.0,
            "parameters": {
                "iou_threshold": 0.5,
                "score_threshold": 0.0,
                "label": "value",
            },
        },
        {
            "type": "Recall",
            "value": 0.0,
            "parameters": {
                "iou_threshold": 0.5,
                "score_threshold": 0.0,
                "label": "other value",
            },
        },
    ]
    for m in expected_metrics:
        assert m in actual_metrics
    for m in actual_metrics:
        assert m in expected_metrics
