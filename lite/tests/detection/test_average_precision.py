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

    label_metadata = np.array([[1, 5, 0]])
    iou_thresholds = np.array([0.1, 0.6])
    score_thresholds = np.array([0.0])

    (results, _, _, _,) = compute_metrics(
        sorted_pairs,
        label_metadata=label_metadata,
        iou_thresholds=iou_thresholds,
        score_thresholds=score_thresholds,
    )
    (
        average_precision,
        mean_average_precision,
        average_precision_averaged_over_ious,
        mean_average_precision_averaged_over_ious,
    ) = results

    expected = np.array(
        [
            [1.0],  # iou = 0.1
            [1 / 3],  # iou = 0.6
        ]
    )
    assert expected.shape == average_precision.shape
    assert np.isclose(average_precision, expected).all()

    # since only one class, ap == map
    assert expected.flatten().shape == mean_average_precision.shape
    assert np.isclose(mean_average_precision, expected.flatten()).all()

    expected = np.array([2 / 3])

    assert average_precision_averaged_over_ious.shape == expected.shape
    assert np.isclose(average_precision_averaged_over_ious, expected).all()

    assert isinstance(mean_average_precision_averaged_over_ious, float)
    assert np.isclose(
        mean_average_precision_averaged_over_ious, expected.flatten()
    ).all()


def test_ap_metrics_first_class(
    basic_detections_first_class: list[Detection],
    basic_rotated_detections_first_class: list[Detection],
):
    """
    Basic object detection test.

    groundtruths
        datum uid1
            box 1 - label v1 - tp
        datum uid2
            box 2 - label v1 - fn missing prediction

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
            as_dict=True,
        )

        assert evaluator.ignored_prediction_labels == []
        assert evaluator.missing_prediction_labels == []
        assert evaluator.n_datums == 2
        assert evaluator.n_labels == 1
        assert evaluator.n_groundtruths == 2
        assert evaluator.n_predictions == 1

        # test AP
        actual_metrics = [m for m in metrics[MetricType.AP]]
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
        actual_metrics = [m for m in metrics[MetricType.mAP]]
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

        # test AP Averaged Over IoUs
        actual_metrics = [m for m in metrics[MetricType.APAveragedOverIOUs]]
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

        # test mAP Averaged Over IoUs
        actual_metrics = [m for m in metrics[MetricType.mAPAveragedOverIOUs]]
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


def test_ap_metrics_second_class(
    basic_detections_second_class: list[Detection],
    basic_rotated_detections_second_class: list[Detection],
):
    """
    Basic object detection test.

    groundtruths
        datum uid1
            box 3 - label v2 - fn missing prediction
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
        )

        assert evaluator.ignored_prediction_labels == []
        assert evaluator.missing_prediction_labels == []
        assert evaluator.n_datums == 2
        assert evaluator.n_labels == 1
        assert evaluator.n_groundtruths == 1
        assert evaluator.n_predictions == 1

        # test AP
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

        # test AP Averaged Over IoUs
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

        # test mAP Averaged Over IoUs
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


def test_ap_using_torch_metrics_example(
    torchmetrics_detections: list[Detection],
):
    """
    cf with torch metrics/pycocotools results listed here:
    https://github.com/Lightning-AI/metrics/blob/107dbfd5fb158b7ae6d76281df44bd94c836bfce/tests/unittests/detection/test_map.py#L231
    """

    loader = DataLoader()
    loader.add_bounding_boxes(torchmetrics_detections)
    evaluator = loader.finalize()

    assert evaluator.ignored_prediction_labels == ["3"]
    assert evaluator.missing_prediction_labels == []
    assert evaluator.n_datums == 4
    assert evaluator.n_labels == 6
    assert evaluator.n_groundtruths == 20
    assert evaluator.n_predictions == 19

    metrics = evaluator.evaluate(
        iou_thresholds=[0.5, 0.75],
        as_dict=True,
    )

    # test AP
    actual_metrics = [m for m in metrics[MetricType.AP]]
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
    actual_metrics = [m for m in metrics[MetricType.mAP]]
    expected_metrics = [
        {
            "type": "mAP",
            "value": 0.8591859185918592,
            "parameters": {
                "iou_threshold": 0.5,
            },
        },
        {
            "type": "mAP",
            "value": 0.7606789250353607,
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
        as_dict=True,
    )

    actual_metrics = [m for m in metrics[MetricType.AP]]
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
        as_dict=True,
    )

    actual_metrics = [m for m in metrics[MetricType.AP]]
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
        as_dict=True,
    )

    actual_metrics = [m for m in metrics[MetricType.AP]]
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
        as_dict=True,
    )

    actual_metrics = [m for m in metrics[MetricType.AP]]
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
        as_dict=True,
    )

    actual_metrics = [m for m in metrics[MetricType.AP]]
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
        as_dict=True,
    )

    actual_metrics = [m for m in metrics[MetricType.AP]]
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


def test_ap_ranked_pair_ordering(
    detection_ranked_pair_ordering: Detection,
    detection_ranked_pair_ordering_with_bitmasks: Detection,
    detection_ranked_pair_ordering_with_polygons: Detection,
):
    for input_, method in [
        (detection_ranked_pair_ordering, DataLoader.add_bounding_boxes),
        (
            detection_ranked_pair_ordering_with_bitmasks,
            DataLoader.add_bitmasks,
        ),
        (
            detection_ranked_pair_ordering_with_polygons,
            DataLoader.add_polygons,
        ),
    ]:
        loader = DataLoader()
        method(loader, detections=[input_])
        evaluator = loader.finalize()

        assert evaluator.metadata == {
            "ignored_prediction_labels": [
                "label4",
            ],
            "missing_prediction_labels": [],
            "n_datums": 1,
            "n_groundtruths": 3,
            "n_labels": 4,
            "n_predictions": 4,
        }

        metrics = evaluator.evaluate(
            iou_thresholds=[0.5, 0.75],
            as_dict=True,
        )

        actual_metrics = [m for m in metrics[MetricType.AP]]
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
        ]
        for m in actual_metrics:
            assert m in expected_metrics
        for m in expected_metrics:
            assert m in actual_metrics

        actual_metrics = [m for m in metrics[MetricType.mAP]]
        expected_metrics = [
            {
                "parameters": {"iou_threshold": 0.5},
                "value": 0.6666666666666666,
                "type": "mAP",
            },
            {
                "parameters": {"iou_threshold": 0.75},
                "value": 0.6666666666666666,
                "type": "mAP",
            },
        ]
        for m in actual_metrics:
            assert m in expected_metrics
        for m in expected_metrics:
            assert m in actual_metrics

        actual_metrics = [m for m in metrics[MetricType.APAveragedOverIOUs]]
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
        ]
        for m in actual_metrics:
            assert m in expected_metrics
        for m in expected_metrics:
            assert m in actual_metrics

        actual_metrics = [m for m in metrics[MetricType.mAPAveragedOverIOUs]]
        expected_metrics = [
            {
                "parameters": {
                    "iou_thresholds": [
                        0.5,
                        0.75,
                    ],
                },
                "value": 0.6666666666666666,
                "type": "mAPAveragedOverIOUs",
            },
        ]
        for m in actual_metrics:
            assert m in expected_metrics
        for m in expected_metrics:
            assert m in actual_metrics


def test_ap_true_positive_deassignment(
    detections_tp_deassignment_edge_case: list[Detection],
):

    loader = DataLoader()
    loader.add_bounding_boxes(detections_tp_deassignment_edge_case)
    evaluator = loader.finalize()

    assert evaluator.ignored_prediction_labels == []
    assert evaluator.missing_prediction_labels == []
    assert evaluator.n_datums == 1
    assert evaluator.n_labels == 1
    assert evaluator.n_groundtruths == 2
    assert evaluator.n_predictions == 4

    metrics = evaluator.evaluate(
        iou_thresholds=[0.5],
        score_thresholds=[0.5],
        as_dict=True,
    )

    assert len(metrics) == 14

    # test AP
    actual_metrics = [m for m in metrics[MetricType.AP]]
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
