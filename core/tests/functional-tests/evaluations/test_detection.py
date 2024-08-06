import random

import pandas as pd
import pytest
from valor_core import enums, geometry, schemas
from valor_core.detection import _calculate_101_pt_interp, evaluate_detection


def test__calculate_101_pt_interp():
    # make sure we get back 0 if we don't pass any precisions
    assert _calculate_101_pt_interp([], []) == 0

    # get back -1 if all recalls and precisions are -1
    assert _calculate_101_pt_interp([-1, -1], [-1, -1]) == -1


def test_evaluate_detection(
    evaluate_detection_groundtruths, evaluate_detection_predictions
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

    eval_job = evaluate_detection(
        groundtruths=evaluate_detection_groundtruths,
        predictions=evaluate_detection_predictions,
        iou_thresholds_to_compute=[0.1, 0.6],
        iou_thresholds_to_return=[0.1, 0.6],
        metrics_to_return=[
            enums.MetricType.AP,
            enums.MetricType.AR,
            enums.MetricType.mAP,
            enums.MetricType.APAveragedOverIOUs,
            enums.MetricType.mAR,
            enums.MetricType.mAPAveragedOverIOUs,
        ],
    )

    metrics = eval_job.metrics

    expected_metrics = [
        {
            "label": {"key": "k2", "value": "v2"},
            "parameters": {"iou": 0.1},
            "value": 0.0,
            "type": "AP",
        },
        {
            "label": {"key": "k2", "value": "v2"},
            "parameters": {"iou": 0.6},
            "value": 0.0,
            "type": "AP",
        },
        {
            "label": {"key": "k1", "value": "v1"},
            "parameters": {"iou": 0.1},
            "value": 0.504950495049505,
            "type": "AP",
        },
        {
            "label": {"key": "k1", "value": "v1"},
            "parameters": {"iou": 0.6},
            "value": 0.504950495049505,
            "type": "AP",
        },
        {
            "parameters": {"label_key": "k1", "iou": 0.1},
            "value": 0.504950495049505,
            "type": "mAP",
        },
        {
            "parameters": {"label_key": "k2", "iou": 0.1},
            "value": 0.0,
            "type": "mAP",
        },
        {
            "parameters": {"label_key": "k1", "iou": 0.6},
            "value": 0.504950495049505,
            "type": "mAP",
        },
        {
            "parameters": {"label_key": "k2", "iou": 0.6},
            "value": 0.0,
            "type": "mAP",
        },
        {
            "label": {"key": "k2", "value": "v2"},
            "parameters": {"ious": [0.1, 0.6]},
            "value": 0.0,
            "type": "APAveragedOverIOUs",
        },
        {
            "label": {"key": "k1", "value": "v1"},
            "parameters": {"ious": [0.1, 0.6]},
            "value": 0.504950495049505,
            "type": "APAveragedOverIOUs",
        },
        {
            "parameters": {"label_key": "k1", "ious": [0.1, 0.6]},
            "value": 0.504950495049505,
            "type": "mAPAveragedOverIOUs",
        },
        {
            "parameters": {"label_key": "k2", "ious": [0.1, 0.6]},
            "value": 0.0,
            "type": "mAPAveragedOverIOUs",
        },
        {
            "label": {"key": "k2", "value": "v2"},
            "parameters": {"ious": [0.1, 0.6]},
            "value": 0.0,
            "type": "AR",
        },
        {
            "label": {"key": "k1", "value": "v1"},
            "parameters": {"ious": [0.1, 0.6]},
            "value": 0.5,
            "type": "AR",
        },
        {
            "parameters": {"label_key": "k1", "ious": [0.1, 0.6]},
            "value": 0.5,
            "type": "mAR",
        },
        {
            "parameters": {"label_key": "k2", "ious": [0.1, 0.6]},
            "value": 0.0,
            "type": "mAR",
        },
    ]

    for m in metrics:
        if m["type"] not in [
            "PrecisionRecallCurve",
            "DetailedPrecisionRecallCurve",
        ]:
            assert m in expected_metrics
    for m in expected_metrics:
        assert m in metrics

    assert eval_job.ignored_pred_labels == []
    assert eval_job.missing_pred_labels == []

    result = eval_job
    result_dict = result.to_dict()

    # duration isn't deterministic, so test meta separately
    assert result_dict["meta"]["datums"] == 2
    assert result_dict["meta"]["labels"] == 2
    assert result_dict["meta"]["annotations"] == 5
    assert result_dict["meta"]["duration"] <= 5
    result_dict.pop("meta")
    result_dict.pop("metrics")

    assert result_dict == {
        "parameters": {
            "label_map": {},
            "metrics_to_return": [
                enums.MetricType.AP,
                enums.MetricType.AR,
                enums.MetricType.mAP,
                enums.MetricType.APAveragedOverIOUs,
                enums.MetricType.mAR,
                enums.MetricType.mAPAveragedOverIOUs,
            ],
            "iou_thresholds_to_compute": [0.1, 0.6],
            "iou_thresholds_to_return": [0.1, 0.6],
            "recall_score_threshold": 0.0,
            "pr_curve_iou_threshold": 0.5,
            "pr_curve_max_examples": 1,
        },
        "confusion_matrices": [],
        "ignored_pred_labels": [],
        "missing_pred_labels": [],
    }

    # {
    #     "parameters": {
    #         "iou_thresholds_to_compute": [0.1, 0.6],
    #         "iou_thresholds_to_return": [0.1, 0.6],
    #         "label_map": None,
    #         "recall_score_threshold": 0.0,
    #         "metrics_to_return": [
    #             enums.MetricType.AP,
    #             enums.MetricType.AR,
    #             enums.MetricType.mAP,
    #             enums.MetricType.APAveragedOverIOUs,
    #             enums.MetricType.mAR,
    #             enums.MetricType.mAPAveragedOverIOUs,
    #         ],
    #         "pr_curve_iou_threshold": 0.5,
    #         "pr_curve_max_examples": 1,
    #     },
    #     "confusion_matrices": [],
    #     "missing_pred_labels": [],
    #     "ignored_pred_labels": [],
    # }

    # # check that metrics arg works correctly
    selected_metrics = random.sample(
        [
            enums.MetricType.AP,
            enums.MetricType.AR,
            enums.MetricType.mAP,
            enums.MetricType.APAveragedOverIOUs,
            enums.MetricType.mAR,
            enums.MetricType.mAPAveragedOverIOUs,
            enums.MetricType.PrecisionRecallCurve,
        ],
        2,
    )
    eval_job = evaluate_detection(
        groundtruths=evaluate_detection_groundtruths,
        predictions=evaluate_detection_predictions,
        iou_thresholds_to_compute=[0.1, 0.6],
        iou_thresholds_to_return=[0.1, 0.6],
        metrics_to_return=selected_metrics,
    )

    metrics = eval_job.metrics
    assert set([metric["type"] for metric in eval_job.metrics]) == set(
        selected_metrics
    )


def test_evaluate_detection_via_pandas_df():
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
    groundtruth_df = pd.DataFrame(
        [
            {
                "datum_id": 1,
                "datum_uid": "uid1",
                "id": 1,
                "annotation_id": 1,
                "label_id": 1,
                "label_key": "k1",
                "label_value": "v1",
                "is_instance": True,
                "grouper_key": "k1",
                "polygon": geometry.Polygon.from_dict(
                    {
                        "type": "Polygon",
                        "coordinates": [
                            [[10, 10], [60, 10], [60, 40], [10, 40], [10, 10]]
                        ],
                    }
                ),
                "raster": None,
                "bounding_box": None,
            },
            {
                "datum_id": 1,
                "datum_uid": "uid1",
                "id": 2,
                "annotation_id": 2,
                "label_id": 2,
                "label_key": "k2",
                "label_value": "v2",
                "is_instance": True,
                "grouper_key": "k2",
                "polygon": geometry.Polygon.from_dict(
                    {
                        "type": "Polygon",
                        "coordinates": [
                            [
                                [87, 10],
                                [158, 10],
                                [158, 820],
                                [87, 820],
                                [87, 10],
                            ]
                        ],
                    }
                ),
                "raster": None,
                "bounding_box": None,
            },
            {
                "datum_id": 2,
                "datum_uid": "uid2",
                "id": 3,
                "annotation_id": 3,
                "label_id": 1,
                "label_key": "k1",
                "label_value": "v1",
                "is_instance": True,
                "grouper_key": "k1",
                "polygon": geometry.Polygon.from_dict(
                    {
                        "type": "Polygon",
                        "coordinates": [
                            [[15, 0], [70, 0], [70, 20], [15, 20], [15, 0]]
                        ],
                    }
                ),
                "raster": None,
                "bounding_box": None,
            },
        ]
    )
    prediction_df = pd.DataFrame(
        [
            {
                "id": 1,
                "annotation_id": 4,
                "score": 0.3,
                "datum_id": 1,
                "datum_uid": "uid1",
                "label_id": 1,
                "label_key": "k1",
                "label_value": "v1",
                "is_instance": True,
                "grouper_key": "k1",
                "polygon": geometry.Polygon.from_dict(
                    {
                        "type": "Polygon",
                        "coordinates": [
                            [[10, 10], [60, 10], [60, 40], [10, 40], [10, 10]]
                        ],
                    }
                ),
                "raster": None,
                "bounding_box": None,
            },
            {
                "id": 2,
                "annotation_id": 5,
                "score": 0.98,
                "datum_id": 2,
                "datum_uid": "uid2",
                "label_id": 2,
                "label_key": "k2",
                "label_value": "v2",
                "is_instance": True,
                "grouper_key": "k2",
                "polygon": geometry.Polygon.from_dict(
                    {
                        "type": "Polygon",
                        "coordinates": [
                            [[15, 0], [70, 0], [70, 20], [15, 20], [15, 0]]
                        ],
                    }
                ),
                "raster": None,
                "bounding_box": None,
            },
        ]
    )

    eval_job = evaluate_detection(
        groundtruths=groundtruth_df,
        predictions=prediction_df,
        iou_thresholds_to_compute=[0.1, 0.6],
        iou_thresholds_to_return=[0.1, 0.6],
        metrics_to_return=[
            enums.MetricType.AP,
            enums.MetricType.AR,
            enums.MetricType.mAP,
            enums.MetricType.APAveragedOverIOUs,
            enums.MetricType.mAR,
            enums.MetricType.mAPAveragedOverIOUs,
        ],
    )

    metrics = eval_job.metrics

    expected_metrics = [
        {
            "label": {"key": "k2", "value": "v2"},
            "parameters": {"iou": 0.1},
            "value": 0.0,
            "type": "AP",
        },
        {
            "label": {"key": "k2", "value": "v2"},
            "parameters": {"iou": 0.6},
            "value": 0.0,
            "type": "AP",
        },
        {
            "label": {"key": "k1", "value": "v1"},
            "parameters": {"iou": 0.1},
            "value": 0.504950495049505,
            "type": "AP",
        },
        {
            "label": {"key": "k1", "value": "v1"},
            "parameters": {"iou": 0.6},
            "value": 0.504950495049505,
            "type": "AP",
        },
        {
            "parameters": {"label_key": "k1", "iou": 0.1},
            "value": 0.504950495049505,
            "type": "mAP",
        },
        {
            "parameters": {"label_key": "k2", "iou": 0.1},
            "value": 0.0,
            "type": "mAP",
        },
        {
            "parameters": {"label_key": "k1", "iou": 0.6},
            "value": 0.504950495049505,
            "type": "mAP",
        },
        {
            "parameters": {"label_key": "k2", "iou": 0.6},
            "value": 0.0,
            "type": "mAP",
        },
        {
            "label": {"key": "k2", "value": "v2"},
            "parameters": {"ious": [0.1, 0.6]},
            "value": 0.0,
            "type": "APAveragedOverIOUs",
        },
        {
            "label": {"key": "k1", "value": "v1"},
            "parameters": {"ious": [0.1, 0.6]},
            "value": 0.504950495049505,
            "type": "APAveragedOverIOUs",
        },
        {
            "parameters": {"label_key": "k1", "ious": [0.1, 0.6]},
            "value": 0.504950495049505,
            "type": "mAPAveragedOverIOUs",
        },
        {
            "parameters": {"label_key": "k2", "ious": [0.1, 0.6]},
            "value": 0.0,
            "type": "mAPAveragedOverIOUs",
        },
        {
            "label": {"key": "k2", "value": "v2"},
            "parameters": {"ious": [0.1, 0.6]},
            "value": 0.0,
            "type": "AR",
        },
        {
            "label": {"key": "k1", "value": "v1"},
            "parameters": {"ious": [0.1, 0.6]},
            "value": 0.5,
            "type": "AR",
        },
        {
            "parameters": {"label_key": "k1", "ious": [0.1, 0.6]},
            "value": 0.5,
            "type": "mAR",
        },
        {
            "parameters": {"label_key": "k2", "ious": [0.1, 0.6]},
            "value": 0.0,
            "type": "mAR",
        },
    ]

    for m in metrics:
        if m["type"] not in [
            "PrecisionRecallCurve",
            "DetailedPrecisionRecallCurve",
        ]:
            assert m in expected_metrics
    for m in expected_metrics:
        assert m in metrics

    assert eval_job.ignored_pred_labels == []
    assert eval_job.missing_pred_labels == []

    result = eval_job
    result_dict = result.to_dict()

    # duration isn't deterministic, so test meta separately
    assert result_dict["meta"]["datums"] == 2
    assert result_dict["meta"]["labels"] == 2
    assert result_dict["meta"]["annotations"] == 5
    assert result_dict["meta"]["duration"] <= 5
    result_dict.pop("meta")
    result_dict.pop("metrics")

    assert result_dict == {
        "parameters": {
            "iou_thresholds_to_compute": [0.1, 0.6],
            "iou_thresholds_to_return": [0.1, 0.6],
            "label_map": {},
            "recall_score_threshold": 0.0,
            "metrics_to_return": [
                enums.MetricType.AP,
                enums.MetricType.AR,
                enums.MetricType.mAP,
                enums.MetricType.APAveragedOverIOUs,
                enums.MetricType.mAR,
                enums.MetricType.mAPAveragedOverIOUs,
            ],
            "pr_curve_iou_threshold": 0.5,
            "pr_curve_max_examples": 1,
        },
        "confusion_matrices": [],
        "missing_pred_labels": [],
        "ignored_pred_labels": [],
    }

    # # check that metrics arg works correctly
    selected_metrics = random.sample(
        [
            enums.MetricType.AP,
            enums.MetricType.AR,
            enums.MetricType.mAP,
            enums.MetricType.APAveragedOverIOUs,
            enums.MetricType.mAR,
            enums.MetricType.mAPAveragedOverIOUs,
            enums.MetricType.PrecisionRecallCurve,
        ],
        2,
    )
    eval_job = evaluate_detection(
        groundtruths=groundtruth_df,
        predictions=prediction_df,
        iou_thresholds_to_compute=[0.1, 0.6],
        iou_thresholds_to_return=[0.1, 0.6],
        metrics_to_return=selected_metrics,
    )

    metrics = eval_job.metrics
    assert set([metric["type"] for metric in eval_job.metrics]) == set(
        selected_metrics
    )


def test_evaluate_detection_with_label_maps(
    evaluate_detection_groundtruths_with_label_maps,
    evaluate_detection_predictions_with_label_maps,
):
    # for the first evaluation, don't do anything about the mismatched labels
    # we expect the evaluation to return the same expected metrics as for our standard detection tests

    baseline_expected_metrics = [
        {
            "type": "AP",
            "parameters": {"iou": 0.1},
            "value": 0.504950495049505,
            "label": {"key": "k1", "value": "v1"},
        },
        {
            "type": "AP",
            "parameters": {"iou": 0.6},
            "value": 0.504950495049505,
            "label": {"key": "k1", "value": "v1"},
        },
        {
            "type": "AP",
            "parameters": {"iou": 0.1},
            "value": 0.0,
            "label": {"key": "k2", "value": "v2"},
        },
        {
            "type": "AP",
            "parameters": {"iou": 0.6},
            "value": 0.0,
            "label": {"key": "k2", "value": "v2"},
        },
        {
            "type": "AP",
            "parameters": {"iou": 0.1},
            "value": 0.0,
            "label": {"key": "class_name", "value": "maine coon cat"},
        },
        {
            "type": "AP",
            "parameters": {"iou": 0.6},
            "value": 0.0,
            "label": {"key": "class_name", "value": "maine coon cat"},
        },
        {
            "type": "AP",
            "parameters": {"iou": 0.1},
            "value": 0.0,
            "label": {"key": "class", "value": "british shorthair"},
        },
        {
            "type": "AP",
            "parameters": {"iou": 0.6},
            "value": 0.0,
            "label": {"key": "class", "value": "british shorthair"},
        },
        {
            "type": "AP",
            "parameters": {"iou": 0.1},
            "value": 0.0,
            "label": {"key": "class", "value": "siamese cat"},
        },
        {
            "type": "AP",
            "parameters": {"iou": 0.6},
            "value": 0.0,
            "label": {"key": "class", "value": "siamese cat"},
        },
        {
            "type": "AR",
            "parameters": {"ious": [0.1, 0.6]},
            "value": 0.5,
            "label": {"key": "k1", "value": "v1"},
        },
        {
            "type": "AR",
            "parameters": {"ious": [0.1, 0.6]},
            "value": -1.0,
            "label": {"key": "class_name", "value": "cat"},
        },
        {
            "type": "AR",
            "parameters": {"ious": [0.1, 0.6]},
            "value": 0.0,
            "label": {"key": "k2", "value": "v2"},
        },
        {
            "type": "AR",
            "parameters": {"ious": [0.1, 0.6]},
            "value": 0.0,
            "label": {"key": "class_name", "value": "maine coon cat"},
        },
        {
            "type": "AR",
            "parameters": {"ious": [0.1, 0.6]},
            "value": 0.0,
            "label": {"key": "class", "value": "british shorthair"},
        },
        {
            "type": "AR",
            "parameters": {"ious": [0.1, 0.6]},
            "value": -1.0,
            "label": {"key": "class", "value": "cat"},
        },
        {
            "type": "AR",
            "parameters": {"ious": [0.1, 0.6]},
            "value": 0.0,
            "label": {"key": "class", "value": "siamese cat"},
        },
        {
            "type": "mAR",
            "parameters": {"ious": [0.1, 0.6], "label_key": "k1"},
            "value": 0.5,
        },
        {
            "type": "mAR",
            "parameters": {"ious": [0.1, 0.6], "label_key": "k2"},
            "value": 0.0,
        },
        {
            "type": "mAR",
            "parameters": {"ious": [0.1, 0.6], "label_key": "class_name"},
            "value": 0.0,
        },
        {
            "type": "mAR",
            "parameters": {"ious": [0.1, 0.6], "label_key": "class"},
            "value": 0.0,
        },
        {
            "type": "mAP",
            "parameters": {"iou": 0.1, "label_key": "class"},
            "value": 0.0,
        },
        {
            "type": "mAP",
            "parameters": {"iou": 0.6, "label_key": "class"},
            "value": 0.0,
        },
        {
            "type": "mAP",
            "parameters": {"iou": 0.1, "label_key": "class_name"},
            "value": 0.0,
        },
        {
            "type": "mAP",
            "parameters": {"iou": 0.6, "label_key": "class_name"},
            "value": 0.0,
        },
        {
            "type": "mAP",
            "parameters": {"iou": 0.1, "label_key": "k1"},
            "value": 0.504950495049505,
        },
        {
            "type": "mAP",
            "parameters": {"iou": 0.6, "label_key": "k1"},
            "value": 0.504950495049505,
        },
        {
            "type": "mAP",
            "parameters": {"iou": 0.1, "label_key": "k2"},
            "value": 0.0,
        },
        {
            "type": "mAP",
            "parameters": {"iou": 0.6, "label_key": "k2"},
            "value": 0.0,
        },
        {
            "type": "APAveragedOverIOUs",
            "parameters": {"ious": [0.1, 0.6]},
            "value": 0.504950495049505,
            "label": {"key": "k1", "value": "v1"},
        },
        {
            "type": "APAveragedOverIOUs",
            "parameters": {"ious": [0.1, 0.6]},
            "value": 0.0,
            "label": {"key": "k2", "value": "v2"},
        },
        {
            "type": "APAveragedOverIOUs",
            "parameters": {"ious": [0.1, 0.6]},
            "value": 0.0,
            "label": {"key": "class_name", "value": "maine coon cat"},
        },
        {
            "type": "APAveragedOverIOUs",
            "parameters": {"ious": [0.1, 0.6]},
            "value": 0.0,
            "label": {"key": "class", "value": "british shorthair"},
        },
        {
            "type": "APAveragedOverIOUs",
            "parameters": {"ious": [0.1, 0.6]},
            "value": 0.0,
            "label": {"key": "class", "value": "siamese cat"},
        },
        {
            "type": "mAPAveragedOverIOUs",
            "parameters": {"ious": [0.1, 0.6], "label_key": "class"},
            "value": 0.0,
        },
        {
            "type": "mAPAveragedOverIOUs",
            "parameters": {"ious": [0.1, 0.6], "label_key": "class_name"},
            "value": 0.0,
        },
        {
            "type": "mAPAveragedOverIOUs",
            "parameters": {"ious": [0.1, 0.6], "label_key": "k1"},
            "value": 0.504950495049505,
        },
        {
            "type": "mAPAveragedOverIOUs",
            "parameters": {"ious": [0.1, 0.6], "label_key": "k2"},
            "value": 0.0,
        },
    ]

    eval_job = evaluate_detection(
        groundtruths=evaluate_detection_groundtruths_with_label_maps,
        predictions=evaluate_detection_predictions_with_label_maps,
        pr_curve_max_examples=1,
        iou_thresholds_to_compute=[0.1, 0.6],
        iou_thresholds_to_return=[0.1, 0.6],
        metrics_to_return=[
            enums.MetricType.AP,
            enums.MetricType.AR,
            enums.MetricType.mAP,
            enums.MetricType.APAveragedOverIOUs,
            enums.MetricType.mAR,
            enums.MetricType.mAPAveragedOverIOUs,
            enums.MetricType.PrecisionRecallCurve,
            enums.MetricType.DetailedPrecisionRecallCurve,
        ],
    )

    assert (
        eval_job.ignored_pred_labels is not None
        and eval_job.missing_pred_labels is not None
    )
    assert (
        len(eval_job.ignored_pred_labels) == 2
    )  # we're ignoring the two "cat" model predictions
    assert (
        len(eval_job.missing_pred_labels) == 3
    )  # we're missing three gts_det_syn representing different breeds of cats

    metrics = eval_job.metrics

    pr_metrics = []
    pr_metrics = []
    detailed_pr_metrics = []
    for m in metrics:
        if m["type"] == "PrecisionRecallCurve":
            pr_metrics.append(m)
        elif m["type"] == "DetailedPrecisionRecallCurve":
            detailed_pr_metrics.append(m)
        else:
            assert m in baseline_expected_metrics

    pr_metrics.sort(key=lambda x: x["parameters"]["label_key"])
    detailed_pr_metrics.sort(key=lambda x: x["parameters"]["label_key"])

    pr_expected_answers = {
        # class
        (
            0,
            "class",
            "cat",
            "0.1",
            "fp",
        ): 1,
        (0, "class", "cat", "0.4", "fp"): 0,
        (0, "class", "siamese cat", "0.1", "fn"): 1,
        (0, "class", "british shorthair", "0.1", "fn"): 1,
        # class_name
        (1, "class_name", "cat", "0.1", "fp"): 1,
        (1, "class_name", "maine coon cat", "0.1", "fn"): 1,
        # k1
        (2, "k1", "v1", "0.1", "fn"): 1,
        (2, "k1", "v1", "0.1", "tp"): 1,
        (2, "k1", "v1", "0.4", "fn"): 2,
        # k2
        (3, "k2", "v2", "0.1", "fn"): 1,
        (3, "k2", "v2", "0.1", "fp"): 1,
    }

    for (
        index,
        key,
        value,
        threshold,
        metric,
    ), expected_value in pr_expected_answers.items():
        assert (
            pr_metrics[index]["value"][value][float(threshold)][metric]
            == expected_value
        )

    # check DetailedPrecisionRecallCurve
    detailed_pr_expected_answers = {
        # class
        (0, "cat", "0.1", "fp"): {
            "hallucinations": 1,
            "misclassifications": 0,
            "total": 1,
        },
        (0, "cat", "0.4", "fp"): {
            "hallucinations": 0,
            "misclassifications": 0,
            "total": 0,
        },
        (0, "british shorthair", "0.1", "fn"): {
            "no_predictions": 1,
            "misclassifications": 0,
            "total": 1,
        },
        # class_name
        (1, "cat", "0.4", "fp"): {
            "hallucinations": 1,
            "misclassifications": 0,
            "total": 1,
        },
        (1, "maine coon cat", "0.1", "fn"): {
            "no_predictions": 1,
            "misclassifications": 0,
            "total": 1,
        },
        # k1
        (2, "v1", "0.1", "fn"): {
            "no_predictions": 1,
            "misclassifications": 0,
            "total": 1,
        },
        (2, "v1", "0.4", "fn"): {
            "no_predictions": 2,
            "misclassifications": 0,
            "total": 2,
        },
        (2, "v1", "0.1", "tp"): {"all": 1, "total": 1},
        # k2
        (3, "v2", "0.1", "fn"): {
            "no_predictions": 1,
            "misclassifications": 0,
            "total": 1,
        },
        (3, "v2", "0.1", "fp"): {
            "hallucinations": 1,
            "misclassifications": 0,
            "total": 1,
        },
    }

    for (
        index,
        value,
        threshold,
        metric,
    ), expected_output in detailed_pr_expected_answers.items():
        model_output = detailed_pr_metrics[index]["value"][value][
            float(threshold)
        ][metric]
        assert isinstance(model_output, dict)
        assert model_output["total"] == expected_output["total"]
        assert all(
            [
                model_output["observations"][key]["count"]  # type: ignore - we know this element is a dict
                == expected_output[key]
                for key in [
                    key
                    for key in expected_output.keys()
                    if key not in ["total"]
                ]
            ]
        )

    # check that we get at most 1 example
    assert (
        len(
            detailed_pr_metrics[0]["value"]["cat"][0.4]["fp"]["observations"]["hallucinations"][  # type: ignore - we know this element is a dict
                "examples"
            ]
        )
        == 0
    )
    assert (
        len(
            detailed_pr_metrics[2]["value"]["v1"][0.4]["fn"]["observations"]["no_predictions"][  # type: ignore - we know this element is a dict
                "examples"
            ]
        )
        == 1
    )

    # now, we correct most of the mismatched labels with a label map
    cat_expected_metrics = [
        {
            "type": "AP",
            "parameters": {"iou": 0.1},
            "value": 0.33663366336633666,
            "label": {"key": "class", "value": "cat"},
        },
        {
            "type": "AP",
            "parameters": {"iou": 0.1},
            "value": 0.504950495049505,
            "label": {"key": "k1", "value": "v1"},
        },
        {
            "type": "AP",
            "parameters": {"iou": 0.1},
            "value": 0.0,
            "label": {"key": "k2", "value": "v2"},
        },
        {
            "type": "AP",
            "parameters": {"iou": 0.6},
            "value": 0.33663366336633666,
            "label": {"key": "class", "value": "cat"},
        },
        {
            "type": "AP",
            "parameters": {"iou": 0.6},
            "value": 0.504950495049505,
            "label": {"key": "k1", "value": "v1"},
        },
        {
            "type": "AP",
            "parameters": {"iou": 0.6},
            "value": 0.0,
            "label": {"key": "k2", "value": "v2"},
        },
        {
            "type": "AR",
            "parameters": {"ious": [0.1, 0.6]},
            "value": 0.5,
            "label": {"key": "k1", "value": "v1"},
        },
        {
            "type": "AR",
            "parameters": {"ious": [0.1, 0.6]},
            "value": 0.3333333333333333,
            "label": {"key": "class", "value": "cat"},
        },
        {
            "type": "AR",
            "parameters": {"ious": [0.1, 0.6]},
            "value": 0.0,
            "label": {"key": "k2", "value": "v2"},
        },
        {
            "type": "AR",
            "parameters": {"ious": [0.1, 0.6]},
            "value": -1.0,
            "label": {"key": "class_name", "value": "cat"},
        },
        {
            "type": "mAP",
            "parameters": {"iou": 0.1, "label_key": "class"},
            "value": 0.33663366336633666,
        },
        {
            "type": "mAP",
            "parameters": {"iou": 0.1, "label_key": "k1"},
            "value": 0.504950495049505,
        },
        {
            "type": "mAP",
            "parameters": {"iou": 0.1, "label_key": "k2"},
            "value": 0.0,
        },
        {
            "type": "mAP",
            "parameters": {"iou": 0.6, "label_key": "class"},
            "value": 0.33663366336633666,
        },
        {
            "type": "mAP",
            "parameters": {"iou": 0.6, "label_key": "k1"},
            "value": 0.504950495049505,
        },
        {
            "type": "mAP",
            "parameters": {"iou": 0.6, "label_key": "k2"},
            "value": 0.0,
        },
        {
            "type": "mAP",
            "parameters": {"iou": 0.1, "label_key": "class"},
            "value": 0.33663366336633666,
        },
        {
            "type": "mAP",
            "parameters": {"iou": 0.1, "label_key": "k1"},
            "value": 0.504950495049505,
        },
        {
            "type": "mAP",
            "parameters": {"iou": 0.1, "label_key": "k2"},
            "value": 0.0,
        },
        {
            "type": "mAP",
            "parameters": {"iou": 0.6, "label_key": "class"},
            "value": 0.33663366336633666,
        },
        {
            "type": "mAP",
            "parameters": {"iou": 0.6, "label_key": "k1"},
            "value": 0.504950495049505,
        },
        {
            "type": "mAP",
            "parameters": {"iou": 0.6, "label_key": "k2"},
            "value": 0.0,
        },
        {
            "type": "mAR",
            "parameters": {"ious": [0.1, 0.6], "label_key": "class"},
            "value": 0.3333333333333333,
        },
        {
            "type": "mAR",
            "parameters": {"ious": [0.1, 0.6], "label_key": "k1"},
            "value": 0.5,
        },
        {
            "type": "mAR",
            "parameters": {"ious": [0.1, 0.6], "label_key": "class_name"},
            "value": -1.0,
        },
        {
            "type": "mAR",
            "parameters": {"ious": [0.1, 0.6], "label_key": "k2"},
            "value": 0.0,
        },
        {
            "type": "APAveragedOverIOUs",
            "parameters": {"ious": [0.1, 0.6]},
            "value": 0.33663366336633666,
            "label": {"key": "class", "value": "cat"},
        },
        {
            "type": "APAveragedOverIOUs",
            "parameters": {"ious": [0.1, 0.6]},
            "value": 0.504950495049505,
            "label": {"key": "k1", "value": "v1"},
        },
        {
            "type": "APAveragedOverIOUs",
            "parameters": {"ious": [0.1, 0.6]},
            "value": 0.0,
            "label": {"key": "k2", "value": "v2"},
        },
        {
            "type": "mAPAveragedOverIOUs",
            "parameters": {"ious": [0.1, 0.6], "label_key": "k1"},
            "value": 0.504950495049505,
        },
        {
            "type": "mAPAveragedOverIOUs",
            "parameters": {"ious": [0.1, 0.6], "label_key": "class"},
            "value": 0.33663366336633666,
        },
        {
            "type": "mAPAveragedOverIOUs",
            "parameters": {"ious": [0.1, 0.6], "label_key": "k2"},
            "value": 0.0,
        },
    ]

    eval_job = evaluate_detection(
        groundtruths=evaluate_detection_groundtruths_with_label_maps,
        predictions=evaluate_detection_predictions_with_label_maps,
        label_map={
            schemas.Label(
                key="class_name", value="maine coon cat"
            ): schemas.Label(key="class", value="cat"),
            schemas.Label(key="class", value="siamese cat"): schemas.Label(
                key="class", value="cat"
            ),
            schemas.Label(
                key="class", value="british shorthair"
            ): schemas.Label(key="class", value="cat"),
        },
        iou_thresholds_to_compute=[0.1, 0.6],
        iou_thresholds_to_return=[0.1, 0.6],
        metrics_to_return=[
            enums.MetricType.AP,
            enums.MetricType.AR,
            enums.MetricType.mAP,
            enums.MetricType.APAveragedOverIOUs,
            enums.MetricType.mAR,
            enums.MetricType.mAPAveragedOverIOUs,
            enums.MetricType.PrecisionRecallCurve,
            enums.MetricType.DetailedPrecisionRecallCurve,
        ],
    )

    assert eval_job.ignored_pred_labels is not None
    assert eval_job.missing_pred_labels is not None

    assert (
        len(eval_job.ignored_pred_labels) == 1
    )  # Label(key='class_name', value='cat', score=None) is still never used
    assert len(eval_job.missing_pred_labels) == 0

    metrics = eval_job.metrics
    for m in metrics:
        if m["type"] not in [
            "PrecisionRecallCurve",
            "DetailedPrecisionRecallCurve",
        ]:
            assert m in cat_expected_metrics
    for m in cat_expected_metrics:
        assert m in metrics

    assert eval_job.parameters.label_map == {
        schemas.Label(
            key="class_name", value="maine coon cat", score=None
        ): schemas.Label(key="class", value="cat", score=None),
        schemas.Label(
            key="class", value="siamese cat", score=None
        ): schemas.Label(key="class", value="cat", score=None),
        schemas.Label(
            key="class", value="british shorthair", score=None
        ): schemas.Label(key="class", value="cat", score=None),
    }

    # next, we check that the label mapping works when the label is completely foreign
    # to both groundtruths and predictions
    foo_expected_metrics = [
        {
            "type": "AP",
            "parameters": {"iou": 0.1},
            "value": 0.6633663366336634,
            "label": {"key": "foo", "value": "bar"},
        },
        {
            "type": "AP",
            "parameters": {"iou": 0.1},
            "value": 0.0,
            "label": {"key": "k2", "value": "v2"},
        },
        {
            "type": "AP",
            "parameters": {"iou": 0.1},
            "value": 0.504950495049505,
            "label": {"key": "k1", "value": "v1"},
        },
        {
            "type": "AR",
            "parameters": {"ious": [0.1, 0.6]},
            "value": 0.6666666666666666,
            "label": {"key": "foo", "value": "bar"},
        },
        {
            "type": "AR",
            "parameters": {"ious": [0.1, 0.6]},
            "value": 0.5,
            "label": {"key": "k1", "value": "v1"},
        },
        {
            "type": "mAP",
            "parameters": {"iou": 0.6, "label_key": "foo"},
            "value": 0.6633663366336634,
        },
        {
            "type": "mAP",
            "parameters": {"iou": 0.6, "label_key": "k2"},
            "value": 0.0,
        },
        {
            "type": "mAP",
            "parameters": {"iou": 0.6, "label_key": "k1"},
            "value": 0.504950495049505,
        },
        {
            "type": "mAR",
            "parameters": {"ious": [0.1, 0.6], "label_key": "k2"},
            "value": 0.0,
        },
        {
            "type": "APAveragedOverIOUs",
            "parameters": {"ious": [0.1, 0.6]},
            "value": 0.6633663366336634,
            "label": {"key": "foo", "value": "bar"},
        },
        {
            "type": "APAveragedOverIOUs",
            "parameters": {"ious": [0.1, 0.6]},
            "value": 0.504950495049505,
            "label": {"key": "k1", "value": "v1"},
        },
        {
            "type": "mAPAveragedOverIOUs",
            "parameters": {"ious": [0.1, 0.6], "label_key": "k2"},
            "value": 0.0,
        },
        {
            "type": "AP",
            "parameters": {"iou": 0.6},
            "value": 0.6633663366336634,
            "label": {"key": "foo", "value": "bar"},
        },
        {
            "type": "AP",
            "parameters": {"iou": 0.6},
            "value": 0.0,
            "label": {"key": "k2", "value": "v2"},
        },
        {
            "type": "AP",
            "parameters": {"iou": 0.6},
            "value": 0.504950495049505,
            "label": {"key": "k1", "value": "v1"},
        },
        {
            "type": "AR",
            "parameters": {"ious": [0.1, 0.6]},
            "value": 0.0,
            "label": {"key": "k2", "value": "v2"},
        },
        {
            "type": "mAP",
            "parameters": {"iou": 0.1, "label_key": "foo"},
            "value": 0.6633663366336634,
        },
        {
            "type": "mAP",
            "parameters": {"iou": 0.1, "label_key": "k2"},
            "value": 0.0,
        },
        {
            "type": "mAP",
            "parameters": {"iou": 0.1, "label_key": "k1"},
            "value": 0.504950495049505,
        },
        {
            "type": "mAR",
            "parameters": {"ious": [0.1, 0.6], "label_key": "foo"},
            "value": 0.6666666666666666,
        },
        {
            "type": "mAR",
            "parameters": {"ious": [0.1, 0.6], "label_key": "k1"},
            "value": 0.5,
        },
        {
            "type": "APAveragedOverIOUs",
            "parameters": {"ious": [0.1, 0.6]},
            "value": 0.0,
            "label": {"key": "k2", "value": "v2"},
        },
        {
            "type": "mAPAveragedOverIOUs",
            "parameters": {"ious": [0.1, 0.6], "label_key": "foo"},
            "value": 0.6633663366336634,
        },
        {
            "type": "mAPAveragedOverIOUs",
            "parameters": {"ious": [0.1, 0.6], "label_key": "k1"},
            "value": 0.504950495049505,
        },
    ]

    label_mapping = {
        # map the ground truths
        schemas.Label(key="class_name", value="maine coon cat"): schemas.Label(
            key="foo", value="bar"
        ),
        schemas.Label(key="class", value="siamese cat"): schemas.Label(
            key="foo", value="bar"
        ),
        schemas.Label(key="class", value="british shorthair"): schemas.Label(
            key="foo", value="bar"
        ),
        # map the predictions
        schemas.Label(key="class", value="cat"): schemas.Label(
            key="foo", value="bar"
        ),
        schemas.Label(key="class_name", value="cat"): schemas.Label(
            key="foo", value="bar"
        ),
    }

    eval_job = evaluate_detection(
        groundtruths=evaluate_detection_groundtruths_with_label_maps,
        predictions=evaluate_detection_predictions_with_label_maps,
        label_map=label_mapping,
        iou_thresholds_to_compute=[0.1, 0.6],
        iou_thresholds_to_return=[0.1, 0.6],
        metrics_to_return=[
            enums.MetricType.AP,
            enums.MetricType.AR,
            enums.MetricType.mAP,
            enums.MetricType.APAveragedOverIOUs,
            enums.MetricType.mAR,
            enums.MetricType.mAPAveragedOverIOUs,
            enums.MetricType.PrecisionRecallCurve,
            enums.MetricType.DetailedPrecisionRecallCurve,
        ],
    )

    assert (
        eval_job.ignored_pred_labels is not None
        and eval_job.missing_pred_labels is not None
    )
    assert len(eval_job.ignored_pred_labels) == 0
    assert len(eval_job.missing_pred_labels) == 0

    metrics = eval_job.metrics
    for m in metrics:
        if m["type"] not in [
            "PrecisionRecallCurve",
            "DetailedPrecisionRecallCurve",
        ]:
            assert m in foo_expected_metrics
    for m in foo_expected_metrics:
        assert m in metrics

    assert eval_job.parameters.label_map == {
        schemas.Label(
            key="class_name", value="maine coon cat", score=None
        ): schemas.Label(key="foo", value="bar", score=None),
        schemas.Label(
            key="class", value="siamese cat", score=None
        ): schemas.Label(key="foo", value="bar", score=None),
        schemas.Label(
            key="class", value="british shorthair", score=None
        ): schemas.Label(key="foo", value="bar", score=None),
        schemas.Label(key="class", value="cat", score=None): schemas.Label(
            key="foo", value="bar", score=None
        ),
        schemas.Label(
            key="class_name", value="cat", score=None
        ): schemas.Label(key="foo", value="bar", score=None),
    }

    # finally, let's test using a higher recall_score_threshold
    # this new threshold will disqualify all of our predictions for img1

    foo_expected_metrics_with_higher_score_threshold = [
        {
            "type": "AP",
            "parameters": {"iou": 0.1},
            "value": 0.6633663366336634,
            "label": {"key": "foo", "value": "bar"},
        },
        {
            "type": "AP",
            "parameters": {"iou": 0.1},
            "value": 0.504950495049505,
            "label": {"key": "k1", "value": "v1"},
        },
        {
            "type": "AP",
            "parameters": {"iou": 0.1},
            "value": 0.0,
            "label": {"key": "k2", "value": "v2"},
        },
        {
            "type": "AR",
            "parameters": {"ious": [0.1, 0.6]},
            "value": 0.3333333333333333,  # two missed groundtruth on the first image, and 1 hit for the second image
            "label": {"key": "foo", "value": "bar"},
        },
        {
            "type": "AR",
            "parameters": {"ious": [0.1, 0.6]},
            "value": 0.0,
            "label": {"key": "k2", "value": "v2"},
        },
        {
            "type": "mAP",
            "parameters": {"iou": 0.6, "label_key": "foo"},
            "value": 0.6633663366336634,
        },
        {
            "type": "mAP",
            "parameters": {"iou": 0.6, "label_key": "k1"},
            "value": 0.504950495049505,
        },
        {
            "type": "AP",
            "parameters": {"iou": 0.6},
            "value": 0.6633663366336634,
            "label": {"key": "foo", "value": "bar"},
        },
        {
            "type": "AP",
            "parameters": {"iou": 0.6},
            "value": 0.504950495049505,
            "label": {"key": "k1", "value": "v1"},
        },
        {
            "type": "AP",
            "parameters": {"iou": 0.6},
            "value": 0.0,
            "label": {"key": "k2", "value": "v2"},
        },
        {
            "type": "AR",
            "parameters": {"ious": [0.1, 0.6]},
            "value": 0.0,
            "label": {"key": "k1", "value": "v1"},
        },
        {
            "type": "mAP",
            "parameters": {"iou": 0.1, "label_key": "foo"},
            "value": 0.6633663366336634,
        },
        {
            "type": "mAP",
            "parameters": {"iou": 0.1, "label_key": "k1"},
            "value": 0.504950495049505,
        },
        {
            "type": "mAP",
            "parameters": {"iou": 0.1, "label_key": "k2"},
            "value": 0.0,
        },
        {
            "type": "mAP",
            "parameters": {"iou": 0.6, "label_key": "k2"},
            "value": 0.0,
        },
        {
            "type": "mAR",
            "parameters": {"ious": [0.1, 0.6], "label_key": "foo"},
            "value": 0.3333333333333333,
        },
        {
            "type": "mAR",
            "parameters": {"ious": [0.1, 0.6], "label_key": "k1"},
            "value": 0.0,
        },
        {
            "type": "APAveragedOverIOUs",
            "parameters": {"ious": [0.1, 0.6]},
            "value": 0.6633663366336634,
            "label": {"key": "foo", "value": "bar"},
        },
        {
            "type": "APAveragedOverIOUs",
            "parameters": {"ious": [0.1, 0.6]},
            "value": 0.0,
            "label": {"key": "k2", "value": "v2"},
        },
        {
            "type": "mAPAveragedOverIOUs",
            "parameters": {"ious": [0.1, 0.6], "label_key": "k1"},
            "value": 0.504950495049505,
        },
        {
            "type": "mAR",
            "parameters": {"ious": [0.1, 0.6], "label_key": "k2"},
            "value": 0.0,
        },
        {
            "type": "APAveragedOverIOUs",
            "parameters": {"ious": [0.1, 0.6]},
            "value": 0.504950495049505,
            "label": {"key": "k1", "value": "v1"},
        },
        {
            "type": "mAPAveragedOverIOUs",
            "parameters": {"ious": [0.1, 0.6], "label_key": "foo"},
            "value": 0.6633663366336634,
        },
        {
            "type": "mAPAveragedOverIOUs",
            "parameters": {"ious": [0.1, 0.6], "label_key": "k2"},
            "value": 0.0,
        },
    ]

    eval_job = evaluate_detection(
        groundtruths=evaluate_detection_groundtruths_with_label_maps,
        predictions=evaluate_detection_predictions_with_label_maps,
        label_map=label_mapping,
        iou_thresholds_to_compute=[0.1, 0.6],
        iou_thresholds_to_return=[0.1, 0.6],
        recall_score_threshold=0.8,
        metrics_to_return=[
            enums.MetricType.AP,
            enums.MetricType.AR,
            enums.MetricType.mAP,
            enums.MetricType.APAveragedOverIOUs,
            enums.MetricType.mAR,
            enums.MetricType.mAPAveragedOverIOUs,
            enums.MetricType.PrecisionRecallCurve,
        ],
    )

    assert (
        eval_job.ignored_pred_labels is not None
        and eval_job.missing_pred_labels is not None
    )
    assert len(eval_job.ignored_pred_labels) == 0
    assert len(eval_job.missing_pred_labels) == 0

    assert eval_job.to_dict()["parameters"] == {
        "label_map": {
            schemas.Label(
                key="class_name", value="maine coon cat", score=None
            ): schemas.Label(key="foo", value="bar", score=None),
            schemas.Label(
                key="class", value="siamese cat", score=None
            ): schemas.Label(key="foo", value="bar", score=None),
            schemas.Label(
                key="class", value="british shorthair", score=None
            ): schemas.Label(key="foo", value="bar", score=None),
            schemas.Label(key="class", value="cat", score=None): schemas.Label(
                key="foo", value="bar", score=None
            ),
            schemas.Label(
                key="class_name", value="cat", score=None
            ): schemas.Label(key="foo", value="bar", score=None),
        },
        "metrics_to_return": [
            enums.MetricType.AP,
            enums.MetricType.AR,
            enums.MetricType.mAP,
            enums.MetricType.APAveragedOverIOUs,
            enums.MetricType.mAR,
            enums.MetricType.mAPAveragedOverIOUs,
            enums.MetricType.PrecisionRecallCurve,
        ],
        "iou_thresholds_to_compute": [0.1, 0.6],
        "iou_thresholds_to_return": [0.1, 0.6],
        "recall_score_threshold": 0.8,
        "pr_curve_iou_threshold": 0.5,
        "pr_curve_max_examples": 1,
    }

    metrics = eval_job.metrics

    pr_metrics = []
    for m in metrics:
        if m["type"] == "PrecisionRecallCurve":
            pr_metrics.append(m)
        elif m["type"] == "DetailedPrecisionRecallCurve":
            continue
        else:
            assert m in foo_expected_metrics_with_higher_score_threshold

    for m in foo_expected_metrics_with_higher_score_threshold:
        assert m in metrics

    pr_metrics.sort(key=lambda x: x["parameters"]["label_key"])

    pr_expected_answers = {
        # foo
        (0, "foo", "bar", "0.1", "fn"): 1,  # missed rect3
        (0, "foo", "bar", "0.1", "tp"): 2,
        (0, "foo", "bar", "0.4", "fn"): 2,
        (0, "foo", "bar", "0.4", "tp"): 1,
        # k1
        (1, "k1", "v1", "0.1", "fn"): 1,
        (1, "k1", "v1", "0.1", "tp"): 1,
        (1, "k1", "v1", "0.4", "fn"): 2,
        # k2
        (2, "k2", "v2", "0.1", "fn"): 1,
        (2, "k2", "v2", "0.1", "fp"): 1,
    }

    for (
        index,
        _,
        value,
        threshold,
        metric,
    ), expected_value in pr_expected_answers.items():
        assert (
            pr_metrics[index]["value"][value][float(threshold)][metric]
            == expected_value
        )

    assert eval_job.parameters.label_map == {
        schemas.Label(
            key="class_name", value="maine coon cat", score=None
        ): schemas.Label(key="foo", value="bar", score=None),
        schemas.Label(
            key="class", value="siamese cat", score=None
        ): schemas.Label(key="foo", value="bar", score=None),
        schemas.Label(
            key="class", value="british shorthair", score=None
        ): schemas.Label(key="foo", value="bar", score=None),
        schemas.Label(key="class", value="cat", score=None): schemas.Label(
            key="foo", value="bar", score=None
        ),
        schemas.Label(
            key="class_name", value="cat", score=None
        ): schemas.Label(key="foo", value="bar", score=None),
    }


def test_evaluate_detection_false_negatives_single_image_baseline():
    """This is the baseline for the below test. In this case there are two predictions and
    one groundtruth, but the highest confident prediction overlaps sufficiently with the groundtruth
    so there is not a penalty for the false negative so the AP is 1
    """
    groundtruths = [
        schemas.GroundTruth(
            datum=schemas.Datum(uid="uid1"),
            annotations=[
                schemas.Annotation(
                    bounding_box=geometry.Box.from_extrema(
                        xmin=10, xmax=20, ymin=10, ymax=20
                    ),
                    labels=[schemas.Label(key="key", value="value")],
                    is_instance=True,
                )
            ],
        )
    ]

    predictions = [
        schemas.Prediction(
            datum=schemas.Datum(uid="uid1"),
            annotations=[
                schemas.Annotation(
                    bounding_box=geometry.Box.from_extrema(
                        xmin=10, xmax=20, ymin=10, ymax=20
                    ),
                    labels=[
                        schemas.Label(key="key", value="value", score=0.8)
                    ],
                    is_instance=True,
                ),
                schemas.Annotation(
                    bounding_box=geometry.Box.from_extrema(
                        xmin=100, xmax=110, ymin=100, ymax=200
                    ),
                    labels=[
                        schemas.Label(key="key", value="value", score=0.7)
                    ],
                    is_instance=True,
                ),
            ],
        ),
    ]

    eval_job = evaluate_detection(
        groundtruths=groundtruths,
        predictions=predictions,
        iou_thresholds_to_compute=[0.5],
        iou_thresholds_to_return=[0.5],
    )

    ap_metric = [m for m in eval_job.metrics if m["type"] == "AP"][0]
    assert ap_metric == {
        "type": "AP",
        "parameters": {"iou": 0.5},
        "value": 1,
        "label": {"key": "key", "value": "value"},
    }


def test_evaluate_detection_false_negatives_single_image():
    """Tests fix for a bug where high confidence false negative was not being penalized. The
    difference between this test and the above is that here the prediction with higher confidence
    does not sufficiently overlap the groundtruth and so is penalized and we get an AP of 0.5
    """
    groundtruths = [
        schemas.GroundTruth(
            datum=schemas.Datum(uid="uid1"),
            annotations=[
                schemas.Annotation(
                    bounding_box=geometry.Box.from_extrema(
                        xmin=10, xmax=20, ymin=10, ymax=20
                    ),
                    labels=[schemas.Label(key="key", value="value")],
                    is_instance=True,
                )
            ],
        )
    ]
    predictions = [
        schemas.Prediction(
            datum=schemas.Datum(uid="uid1"),
            annotations=[
                schemas.Annotation(
                    bounding_box=geometry.Box.from_extrema(
                        xmin=10, xmax=20, ymin=10, ymax=20
                    ),
                    labels=[
                        schemas.Label(key="key", value="value", score=0.8)
                    ],
                    is_instance=True,
                ),
                schemas.Annotation(
                    bounding_box=geometry.Box.from_extrema(
                        xmin=100, xmax=110, ymin=100, ymax=200
                    ),
                    labels=[
                        schemas.Label(key="key", value="value", score=0.9)
                    ],
                    is_instance=True,
                ),
            ],
        ),
    ]

    eval_job = evaluate_detection(
        groundtruths=groundtruths,
        predictions=predictions,
        iou_thresholds_to_compute=[0.5],
        iou_thresholds_to_return=[0.5],
    )

    ap_metric = [m for m in eval_job.metrics if m["type"] == "AP"][0]
    assert ap_metric == {
        "type": "AP",
        "parameters": {"iou": 0.5},
        "value": 0.5,
        "label": {"key": "key", "value": "value"},
    }


def test_evaluate_detection_false_negatives_two_images_one_empty_low_confidence_of_fp():
    """In this test we have
        1. An image with a matching groundtruth and prediction (same class and high IOU)
        2. A second image with empty groundtruth annotation but a prediction with lower confidence
        then the prediction on the first image.

    In this case, the AP should be 1.0 since the false positive has lower confidence than the true positive

    """

    groundtruths = [
        schemas.GroundTruth(
            datum=schemas.Datum(uid="uid1"),
            annotations=[
                schemas.Annotation(
                    bounding_box=geometry.Box.from_extrema(
                        xmin=10, xmax=20, ymin=10, ymax=20
                    ),
                    labels=[schemas.Label(key="key", value="value")],
                    is_instance=True,
                )
            ],
        ),
        schemas.GroundTruth(
            datum=schemas.Datum(uid="uid2"),
            annotations=[schemas.Annotation(labels=[])],
        ),
    ]

    predictions = [
        schemas.Prediction(
            datum=schemas.Datum(uid="uid1"),
            annotations=[
                schemas.Annotation(
                    bounding_box=geometry.Box.from_extrema(
                        xmin=10, xmax=20, ymin=10, ymax=20
                    ),
                    labels=[
                        schemas.Label(key="key", value="value", score=0.8)
                    ],
                    is_instance=True,
                ),
            ],
        ),
        schemas.Prediction(
            datum=schemas.Datum(uid="uid2"),
            annotations=[
                schemas.Annotation(
                    bounding_box=geometry.Box.from_extrema(
                        xmin=10, xmax=20, ymin=10, ymax=20
                    ),
                    labels=[
                        schemas.Label(key="key", value="value", score=0.7)
                    ],
                    is_instance=True,
                ),
            ],
        ),
    ]

    eval_job = evaluate_detection(
        groundtruths=groundtruths,
        predictions=predictions,
        iou_thresholds_to_compute=[0.5],
        iou_thresholds_to_return=[0.5],
    )

    ap_metric = [m for m in eval_job.metrics if m["type"] == "AP"][0]
    assert ap_metric == {
        "type": "AP",
        "parameters": {"iou": 0.5},
        "value": 1.0,
        "label": {"key": "key", "value": "value"},
    }


def test_evaluate_detection_false_negatives_two_images_one_empty_high_confidence_of_fp():
    """In this test we have
        1. An image with a matching groundtruth and prediction (same class and high IOU)
        2. A second image with empty groundtruth annotation and a prediction with higher confidence
        then the prediction on the first image.

    In this case, the AP should be 0.5 since the false positive has higher confidence than the true positive
    """
    groundtruths = [
        schemas.GroundTruth(
            datum=schemas.Datum(uid="uid1"),
            annotations=[
                schemas.Annotation(
                    bounding_box=geometry.Box.from_extrema(
                        xmin=10, xmax=20, ymin=10, ymax=20
                    ),
                    labels=[schemas.Label(key="key", value="value")],
                    is_instance=True,
                )
            ],
        ),
        schemas.GroundTruth(
            datum=schemas.Datum(uid="uid2"),
            annotations=[schemas.Annotation(labels=[])],
        ),
    ]

    predictions = [
        schemas.Prediction(
            datum=schemas.Datum(uid="uid1"),
            annotations=[
                schemas.Annotation(
                    bounding_box=geometry.Box.from_extrema(
                        xmin=10, xmax=20, ymin=10, ymax=20
                    ),
                    labels=[
                        schemas.Label(key="key", value="value", score=0.8)
                    ],
                    is_instance=True,
                ),
            ],
        ),
        schemas.Prediction(
            datum=schemas.Datum(uid="uid2"),
            annotations=[
                schemas.Annotation(
                    bounding_box=geometry.Box.from_extrema(
                        xmin=10, xmax=20, ymin=10, ymax=20
                    ),
                    labels=[
                        schemas.Label(key="key", value="value", score=0.9)
                    ],
                    is_instance=True,
                ),
            ],
        ),
    ]

    eval_job = evaluate_detection(
        groundtruths=groundtruths,
        predictions=predictions,
        iou_thresholds_to_compute=[0.5],
        iou_thresholds_to_return=[0.5],
    )

    ap_metric = [m for m in eval_job.metrics if m["type"] == "AP"][0]
    assert ap_metric == {
        "type": "AP",
        "parameters": {"iou": 0.5},
        "value": 0.5,
        "label": {"key": "key", "value": "value"},
    }


def test_evaluate_detection_false_negatives_two_images_one_only_with_different_class_low_confidence_of_fp():
    """In this test we have
        1. An image with a matching groundtruth and prediction (same class, `"value"`, and high IOU)
        2. A second image with a groundtruth annotation with class `"other value"` and a prediction with lower confidence
        then the prediction on the first image.

    In this case, the AP for class `"value"` should be 1 since the false positive has lower confidence than the true positive.
    AP for class `"other value"` should be 0 since there is no prediction for the `"other value"` groundtruth
    """
    groundtruths = [
        schemas.GroundTruth(
            datum=schemas.Datum(uid="uid1"),
            annotations=[
                schemas.Annotation(
                    bounding_box=geometry.Box.from_extrema(
                        xmin=10, xmax=20, ymin=10, ymax=20
                    ),
                    labels=[schemas.Label(key="key", value="value")],
                    is_instance=True,
                )
            ],
        ),
        schemas.GroundTruth(
            datum=schemas.Datum(uid="uid2"),
            annotations=[
                schemas.Annotation(
                    bounding_box=geometry.Box.from_extrema(
                        xmin=10, xmax=20, ymin=10, ymax=20
                    ),
                    labels=[schemas.Label(key="key", value="other value")],
                    is_instance=True,
                )
            ],
        ),
    ]

    predictions = [
        schemas.Prediction(
            datum=schemas.Datum(uid="uid1"),
            annotations=[
                schemas.Annotation(
                    bounding_box=geometry.Box.from_extrema(
                        xmin=10, xmax=20, ymin=10, ymax=20
                    ),
                    labels=[
                        schemas.Label(key="key", value="value", score=0.8)
                    ],
                    is_instance=True,
                ),
            ],
        ),
        schemas.Prediction(
            datum=schemas.Datum(uid="uid2"),
            annotations=[
                schemas.Annotation(
                    bounding_box=geometry.Box.from_extrema(
                        xmin=10, xmax=20, ymin=10, ymax=20
                    ),
                    labels=[
                        schemas.Label(key="key", value="value", score=0.7)
                    ],
                    is_instance=True,
                ),
            ],
        ),
    ]

    eval_job = evaluate_detection(
        groundtruths=groundtruths,
        predictions=predictions,
        iou_thresholds_to_compute=[0.5],
        iou_thresholds_to_return=[0.5],
    )

    ap_metric1 = [
        m
        for m in eval_job.metrics
        if m["type"] == "AP" and m["label"] == {"key": "key", "value": "value"}
    ][0]
    assert ap_metric1 == {
        "type": "AP",
        "parameters": {"iou": 0.5},
        "value": 1.0,
        "label": {"key": "key", "value": "value"},
    }

    # label `"other value"` is not in the predictions so we should get an AP of 0
    ap_metric2 = [
        m
        for m in eval_job.metrics
        if m["type"] == "AP"
        and m["label"] == {"key": "key", "value": "other value"}
    ][0]
    assert ap_metric2 == {
        "type": "AP",
        "parameters": {"iou": 0.5},
        "value": 0,
        "label": {"key": "key", "value": "other value"},
    }


def test_evaluate_detection_false_negatives_two_images_one_only_with_different_class_high_confidence_of_fp():
    """In this test we have
        1. An image with a matching groundtruth and prediction (same class, `"value"`, and high IOU)
        2. A second image with a groundtruth annotation with clas `"other value"` and a prediction with higher confidence
        then the prediction on the first image.

    In this case, the AP for class `"value"` should be 0.5 since the false positive has higher confidence than the true positive.
    AP for class `"other value"` should be 0 since there is no prediction for the `"other value"` groundtruth
    """
    groundtruths = [
        schemas.GroundTruth(
            datum=schemas.Datum(uid="uid1"),
            annotations=[
                schemas.Annotation(
                    bounding_box=geometry.Box.from_extrema(
                        xmin=10, xmax=20, ymin=10, ymax=20
                    ),
                    labels=[schemas.Label(key="key", value="value")],
                    is_instance=True,
                )
            ],
        ),
        schemas.GroundTruth(
            datum=schemas.Datum(uid="uid2"),
            annotations=[
                schemas.Annotation(
                    bounding_box=geometry.Box.from_extrema(
                        xmin=10, xmax=20, ymin=10, ymax=20
                    ),
                    labels=[schemas.Label(key="key", value="other value")],
                    is_instance=True,
                )
            ],
        ),
    ]

    predictions = [
        schemas.Prediction(
            datum=schemas.Datum(uid="uid1"),
            annotations=[
                schemas.Annotation(
                    bounding_box=geometry.Box.from_extrema(
                        xmin=10, xmax=20, ymin=10, ymax=20
                    ),
                    labels=[
                        schemas.Label(key="key", value="value", score=0.8)
                    ],
                    is_instance=True,
                ),
            ],
        ),
        schemas.Prediction(
            datum=schemas.Datum(uid="uid2"),
            annotations=[
                schemas.Annotation(
                    bounding_box=geometry.Box.from_extrema(
                        xmin=10, xmax=20, ymin=10, ymax=20
                    ),
                    labels=[
                        schemas.Label(key="key", value="value", score=0.9)
                    ],
                    is_instance=True,
                ),
            ],
        ),
    ]

    eval_job = evaluate_detection(
        groundtruths=groundtruths,
        predictions=predictions,
        iou_thresholds_to_compute=[0.5],
        iou_thresholds_to_return=[0.5],
    )

    ap_metric1 = [
        m
        for m in eval_job.metrics
        if m["type"] == "AP" and m["label"] == {"key": "key", "value": "value"}
    ][0]
    assert ap_metric1 == {
        "type": "AP",
        "parameters": {"iou": 0.5},
        "value": 0.5,
        "label": {"key": "key", "value": "value"},
    }

    # label `"other value"` is not in the predictions so we should get an AP of 0
    ap_metric2 = [
        m
        for m in eval_job.metrics
        if m["type"] == "AP"
        and m["label"] == {"key": "key", "value": "other value"}
    ][0]
    assert ap_metric2 == {
        "type": "AP",
        "parameters": {"iou": 0.5},
        "value": 0,
        "label": {"key": "key", "value": "other value"},
    }


@pytest.fixture
def test_detailed_precision_recall_curve(
    evaluate_detection_detailed_pr_curve_groundtruths,
    evaluate_detection_detailed_pr_curve_predictions,
):

    eval_job = evaluate_detection(
        groundtruths=evaluate_detection_detailed_pr_curve_groundtruths,
        predictions=evaluate_detection_detailed_pr_curve_predictions,
        metrics_to_return=[enums.MetricType.DetailedPrecisionRecallCurve],
    )

    # one true positive that becomes a false negative when score > .5
    assert eval_job.metrics[0]["value"]["v1"]["0.3"]["tp"]["total"] == 1
    assert eval_job.metrics[0]["value"]["v1"]["0.55"]["tp"]["total"] == 0
    assert eval_job.metrics[0]["value"]["v1"]["0.55"]["fn"]["total"] == 1
    assert (
        eval_job.metrics[0]["value"]["v1"]["0.55"]["fn"]["observations"][
            "no_predictions"
        ]["count"]
        == 1
    )
    assert eval_job.metrics[0]["value"]["v1"]["0.05"]["fn"]["total"] == 0
    assert eval_job.metrics[0]["value"]["v1"]["0.05"]["fp"]["total"] == 0

    # one missed detection that never changes
    assert (
        eval_job.metrics[0]["value"]["missed_detection"]["0.05"]["fn"][
            "observations"
        ]["no_predictions"]["count"]
        == 1
    )
    assert (
        eval_job.metrics[0]["value"]["missed_detection"]["0.95"]["fn"][
            "observations"
        ]["no_predictions"]["count"]
        == 1
    )
    assert (
        eval_job.metrics[0]["value"]["missed_detection"]["0.05"]["tp"]["total"]
        == 0
    )
    assert (
        eval_job.metrics[0]["value"]["missed_detection"]["0.05"]["fp"]["total"]
        == 0
    )

    # one fn missed_dection that becomes a misclassification when pr_curve_iou_threshold <= .48 and score threshold <= .3
    assert (
        eval_job.metrics[0]["value"]["v2"]["0.3"]["fn"]["observations"][
            "no_predictions"
        ]["count"]
        == 1
    )
    assert (
        eval_job.metrics[0]["value"]["v2"]["0.35"]["fn"]["observations"][
            "no_predictions"
        ]["count"]
        == 1
    )
    assert eval_job.metrics[0]["value"]["v2"]["0.05"]["tp"]["total"] == 0
    assert eval_job.metrics[0]["value"]["v2"]["0.05"]["fp"]["total"] == 0

    # one fp hallucination that becomes a misclassification when pr_curve_iou_threshold <= .48 and score threshold <= .3
    assert (
        eval_job.metrics[0]["value"]["not_v2"]["0.05"]["fp"]["observations"][
            "hallucinations"
        ]["count"]
        == 1
    )
    assert (
        eval_job.metrics[0]["value"]["not_v2"]["0.05"]["fp"]["observations"][
            "misclassifications"
        ]["count"]
        == 0
    )
    assert eval_job.metrics[0]["value"]["not_v2"]["0.05"]["tp"]["total"] == 0
    assert eval_job.metrics[0]["value"]["not_v2"]["0.05"]["fn"]["total"] == 0

    # one fp hallucination that disappears when score threshold >.15
    assert (
        eval_job.metrics[0]["value"]["hallucination"]["0.05"]["fp"][
            "observations"
        ]["hallucinations"]["count"]
        == 1
    )
    assert (
        eval_job.metrics[0]["value"]["hallucination"]["0.35"]["fp"][
            "observations"
        ]["hallucinations"]["count"]
        == 0
    )
    assert (
        eval_job.metrics[0]["value"]["hallucination"]["0.05"]["tp"]["total"]
        == 0
    )
    assert (
        eval_job.metrics[0]["value"]["hallucination"]["0.05"]["fn"]["total"]
        == 0
    )

    # one missed detection and one hallucination due to low iou overlap
    assert (
        eval_job.metrics[0]["value"]["low_iou"]["0.3"]["fn"]["observations"][
            "no_predictions"
        ]["count"]
        == 1
    )
    assert (
        eval_job.metrics[0]["value"]["low_iou"]["0.95"]["fn"]["observations"][
            "no_predictions"
        ]["count"]
        == 1
    )
    assert (
        eval_job.metrics[0]["value"]["low_iou"]["0.3"]["fp"]["observations"][
            "hallucinations"
        ]["count"]
        == 1
    )
    assert (
        eval_job.metrics[0]["value"]["low_iou"]["0.55"]["fp"]["observations"][
            "hallucinations"
        ]["count"]
        == 0
    )

    # repeat tests using a lower IOU threshold
    eval_job_low_iou_threshold = evaluate_detection(
        groundtruths=evaluate_detection_detailed_pr_curve_groundtruths,
        predictions=evaluate_detection_detailed_pr_curve_predictions,
        metrics_to_return=[enums.MetricType.DetailedPrecisionRecallCurve],
        pr_curve_iou_threshold=0.45,
    )

    # one true positive that becomes a false negative when score > .5
    assert eval_job.metrics[0]["value"]["v1"]["0.3"]["tp"]["total"] == 1
    assert eval_job.metrics[0]["value"]["v1"]["0.55"]["tp"]["total"] == 0
    assert eval_job.metrics[0]["value"]["v1"]["0.55"]["fn"]["total"] == 1
    assert (
        eval_job.metrics[0]["value"]["v1"]["0.55"]["fn"]["observations"][
            "no_predictions"
        ]["count"]
        == 1
    )
    assert eval_job.metrics[0]["value"]["v1"]["0.05"]["fn"]["total"] == 0
    assert eval_job.metrics[0]["value"]["v1"]["0.05"]["fp"]["total"] == 0

    # one missed detection that never changes
    assert (
        eval_job.metrics[0]["value"]["missed_detection"]["0.05"]["fn"][
            "observations"
        ]["no_predictions"]["count"]
        == 1
    )
    assert (
        eval_job.metrics[0]["value"]["missed_detection"]["0.95"]["fn"][
            "observations"
        ]["no_predictions"]["count"]
        == 1
    )
    assert (
        eval_job.metrics[0]["value"]["missed_detection"]["0.05"]["tp"]["total"]
        == 0
    )
    assert (
        eval_job.metrics[0]["value"]["missed_detection"]["0.05"]["fp"]["total"]
        == 0
    )

    # one fn missed_dection that becomes a misclassification when pr_curve_iou_threshold <= .48 and score threshold <= .3
    assert (
        eval_job_low_iou_threshold.metrics[0]["value"]["v2"]["0.3"]["fn"][
            "observations"
        ]["misclassifications"]["count"]
        == 1
    )
    assert (
        eval_job_low_iou_threshold.metrics[0]["value"]["v2"]["0.3"]["fn"][
            "observations"
        ]["no_predictions"]["count"]
        == 0
    )
    assert (
        eval_job_low_iou_threshold.metrics[0]["value"]["v2"]["0.35"]["fn"][
            "observations"
        ]["misclassifications"]["count"]
        == 0
    )
    assert (
        eval_job_low_iou_threshold.metrics[0]["value"]["v2"]["0.35"]["fn"][
            "observations"
        ]["no_predictions"]["count"]
        == 1
    )
    assert (
        eval_job_low_iou_threshold.metrics[0]["value"]["v2"]["0.05"]["tp"][
            "total"
        ]
        == 0
    )
    assert (
        eval_job_low_iou_threshold.metrics[0]["value"]["v2"]["0.05"]["fp"][
            "total"
        ]
        == 0
    )

    # one fp hallucination that becomes a misclassification when pr_curve_iou_threshold <= .48 and score threshold <= .3
    assert (
        eval_job_low_iou_threshold.metrics[0]["value"]["not_v2"]["0.05"]["fp"][
            "observations"
        ]["hallucinations"]["count"]
        == 0
    )
    assert (
        eval_job_low_iou_threshold.metrics[0]["value"]["not_v2"]["0.05"]["fp"][
            "observations"
        ]["misclassifications"]["count"]
        == 1
    )
    assert (
        eval_job_low_iou_threshold.metrics[0]["value"]["not_v2"]["0.05"]["tp"][
            "total"
        ]
        == 0
    )
    assert (
        eval_job_low_iou_threshold.metrics[0]["value"]["not_v2"]["0.05"]["fn"][
            "total"
        ]
        == 0
    )

    # one fp hallucination that disappears when score threshold >.15
    assert (
        eval_job.metrics[0]["value"]["hallucination"]["0.05"]["fp"][
            "observations"
        ]["hallucinations"]["count"]
        == 1
    )
    assert (
        eval_job.metrics[0]["value"]["hallucination"]["0.35"]["fp"][
            "observations"
        ]["hallucinations"]["count"]
        == 0
    )
    assert (
        eval_job.metrics[0]["value"]["hallucination"]["0.05"]["tp"]["total"]
        == 0
    )
    assert (
        eval_job.metrics[0]["value"]["hallucination"]["0.05"]["fn"]["total"]
        == 0
    )

    # one missed detection and one hallucination due to low iou overlap
    assert (
        eval_job.metrics[0]["value"]["low_iou"]["0.3"]["fn"]["observations"][
            "no_predictions"
        ]["count"]
        == 1
    )
    assert (
        eval_job.metrics[0]["value"]["low_iou"]["0.95"]["fn"]["observations"][
            "no_predictions"
        ]["count"]
        == 1
    )
    assert (
        eval_job.metrics[0]["value"]["low_iou"]["0.3"]["fp"]["observations"][
            "hallucinations"
        ]["count"]
        == 1
    )
    assert (
        eval_job.metrics[0]["value"]["low_iou"]["0.55"]["fp"]["observations"][
            "hallucinations"
        ]["count"]
        == 0
    )


def test_evaluate_detection_model_with_no_predictions(
    evaluate_detection_groundtruths,
):
    """
    Test detection evaluations when the model outputs nothing.

    gt_dets1
        datum 1
            - Label (k1, v1) with Annotation area = 1500
            - Label (k2, v2) with Annotation area = 57,510
        datum2
            - Label (k1, v1) with Annotation area = 1100
    """
    predictions = []
    for gt in evaluate_detection_groundtruths:
        predictions.append(
            schemas.Prediction(
                datum=gt.datum,
                annotations=[],
            )
        )

    expected_metrics = [
        {
            "label": {
                "key": "k2",
                "value": "v2",
            },
            "parameters": {
                "iou": 0.5,
            },
            "type": "AP",
            "value": 0.0,
        },
        {
            "label": {
                "key": "k2",
                "value": "v2",
            },
            "parameters": {
                "iou": 0.75,
            },
            "type": "AP",
            "value": 0.0,
        },
        {
            "label": {
                "key": "k1",
                "value": "v1",
            },
            "parameters": {
                "iou": 0.5,
            },
            "type": "AP",
            "value": 0.0,
        },
        {
            "label": {
                "key": "k1",
                "value": "v1",
            },
            "parameters": {
                "iou": 0.75,
            },
            "type": "AP",
            "value": 0.0,
        },
        {
            "label": {
                "key": "k2",
                "value": "v2",
            },
            "parameters": {
                "ious": [
                    0.5,
                    0.55,
                    0.6,
                    0.65,
                    0.7,
                    0.75,
                    0.8,
                    0.85,
                    0.9,
                    0.95,
                ],
            },
            "type": "AR",
            "value": 0.0,
        },
        {
            "label": {
                "key": "k1",
                "value": "v1",
            },
            "parameters": {
                "ious": [
                    0.5,
                    0.55,
                    0.6,
                    0.65,
                    0.7,
                    0.75,
                    0.8,
                    0.85,
                    0.9,
                    0.95,
                ],
            },
            "type": "AR",
            "value": 0.0,
        },
        {
            "parameters": {
                "iou": 0.5,
                "label_key": "k2",
            },
            "type": "mAP",
            "value": 0.0,
        },
        {
            "parameters": {
                "iou": 0.75,
                "label_key": "k2",
            },
            "type": "mAP",
            "value": 0.0,
        },
        {
            "parameters": {
                "iou": 0.5,
                "label_key": "k1",
            },
            "type": "mAP",
            "value": 0.0,
        },
        {
            "parameters": {
                "iou": 0.75,
                "label_key": "k1",
            },
            "type": "mAP",
            "value": 0.0,
        },
        {
            "parameters": {
                "ious": [
                    0.5,
                    0.55,
                    0.6,
                    0.65,
                    0.7,
                    0.75,
                    0.8,
                    0.85,
                    0.9,
                    0.95,
                ],
                "label_key": "k2",
            },
            "type": "mAR",
            "value": 0.0,
        },
        {
            "parameters": {
                "ious": [
                    0.5,
                    0.55,
                    0.6,
                    0.65,
                    0.7,
                    0.75,
                    0.8,
                    0.85,
                    0.9,
                    0.95,
                ],
                "label_key": "k1",
            },
            "type": "mAR",
            "value": 0.0,
        },
        {
            "label": {
                "key": "k2",
                "value": "v2",
            },
            "parameters": {
                "ious": [
                    0.5,
                    0.55,
                    0.6,
                    0.65,
                    0.7,
                    0.75,
                    0.8,
                    0.85,
                    0.9,
                    0.95,
                ],
            },
            "type": "APAveragedOverIOUs",
            "value": 0.0,
        },
        {
            "label": {
                "key": "k1",
                "value": "v1",
            },
            "parameters": {
                "ious": [
                    0.5,
                    0.55,
                    0.6,
                    0.65,
                    0.7,
                    0.75,
                    0.8,
                    0.85,
                    0.9,
                    0.95,
                ],
            },
            "type": "APAveragedOverIOUs",
            "value": 0.0,
        },
        {
            "parameters": {
                "ious": [
                    0.5,
                    0.55,
                    0.6,
                    0.65,
                    0.7,
                    0.75,
                    0.8,
                    0.85,
                    0.9,
                    0.95,
                ],
                "label_key": "k2",
            },
            "type": "mAPAveragedOverIOUs",
            "value": 0.0,
        },
        {
            "parameters": {
                "ious": [
                    0.5,
                    0.55,
                    0.6,
                    0.65,
                    0.7,
                    0.75,
                    0.8,
                    0.85,
                    0.9,
                    0.95,
                ],
                "label_key": "k1",
            },
            "type": "mAPAveragedOverIOUs",
            "value": 0.0,
        },
    ]

    eval_job = evaluate_detection(
        groundtruths=evaluate_detection_groundtruths,
        predictions=predictions,
    )

    computed_metrics = eval_job.metrics

    assert all([metric["value"] == 0 for metric in computed_metrics])

    for m in expected_metrics:
        assert m in computed_metrics

    for m in computed_metrics:
        assert m in expected_metrics


def test_evaluate_detection_functional_test(
    evaluate_detection_functional_test_groundtruths,
    evaluate_detection_functional_test_predictions,
):

    eval_job = evaluate_detection(
        groundtruths=evaluate_detection_functional_test_groundtruths,
        predictions=evaluate_detection_functional_test_predictions,
        metrics_to_return=[
            enums.MetricType.AP,
            enums.MetricType.AR,
            enums.MetricType.mAP,
            enums.MetricType.APAveragedOverIOUs,
            enums.MetricType.mAR,
            enums.MetricType.mAPAveragedOverIOUs,
            enums.MetricType.PrecisionRecallCurve,
            enums.MetricType.DetailedPrecisionRecallCurve,
        ],
        pr_curve_iou_threshold=0.5,
        pr_curve_max_examples=1,
    )

    metrics = [
        m
        for m in eval_job.metrics
        if m["type"]
        not in ["PrecisionRecallCurve", "DetailedPrecisionRecallCurve"]
    ]

    # round all metrics to the third decimal place
    for i, m in enumerate(metrics):
        metrics[i]["value"] = round(m["value"], 3)

    pr_metrics = [
        m for m in eval_job.metrics if m["type"] == "PrecisionRecallCurve"
    ]
    detailed_pr_metrics = [
        m
        for m in eval_job.metrics
        if m["type"] == "DetailedPrecisionRecallCurve"
    ]

    # cf with torch metrics/pycocotools results listed here:
    # https://github.com/Lightning-AI/metrics/blob/107dbfd5fb158b7ae6d76281df44bd94c836bfce/tests/unittests/detection/test_map.py#L231
    expected_metrics = [
        {
            "label": {"key": "class", "value": "0"},
            "parameters": {"iou": 0.5},
            "value": 1.0,
            "type": "AP",
        },
        {
            "label": {"key": "class", "value": "0"},
            "parameters": {"iou": 0.75},
            "value": 0.723,
            "type": "AP",
        },
        {
            "label": {"key": "class", "value": "2"},
            "parameters": {"iou": 0.5},
            "value": 0.505,
            "type": "AP",
        },
        {
            "label": {"key": "class", "value": "2"},
            "parameters": {"iou": 0.75},
            "value": 0.505,
            "type": "AP",
        },
        {
            "label": {"key": "class", "value": "49"},
            "parameters": {"iou": 0.5},
            "value": 0.79,
            "type": "AP",
        },
        {
            "label": {"key": "class", "value": "49"},
            "parameters": {"iou": 0.75},
            "value": 0.576,
            "type": "AP",
        },
        {
            "label": {"key": "class", "value": "1"},
            "parameters": {"iou": 0.5},
            "value": 1.0,
            "type": "AP",
        },
        {
            "label": {"key": "class", "value": "1"},
            "parameters": {"iou": 0.75},
            "value": 1.0,
            "type": "AP",
        },
        {
            "label": {"key": "class", "value": "4"},
            "parameters": {"iou": 0.5},
            "value": 1.0,
            "type": "AP",
        },
        {
            "label": {"key": "class", "value": "4"},
            "parameters": {"iou": 0.75},
            "value": 1.0,
            "type": "AP",
        },
        {
            "parameters": {"label_key": "class", "iou": 0.5},
            "value": 0.859,
            "type": "mAP",
        },
        {
            "parameters": {"label_key": "class", "iou": 0.75},
            "value": 0.761,
            "type": "mAP",
        },
        {
            "label": {"key": "class", "value": "0"},
            "parameters": {
                "ious": [0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95]
            },
            "value": 0.725,
            "type": "APAveragedOverIOUs",
        },
        {
            "label": {"key": "class", "value": "2"},
            "parameters": {
                "ious": [0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95]
            },
            "value": 0.454,
            "type": "APAveragedOverIOUs",
        },
        {
            "label": {"key": "class", "value": "49"},
            "parameters": {
                "ious": [0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95]
            },
            "value": 0.555,
            "type": "APAveragedOverIOUs",
        },
        {
            "label": {"key": "class", "value": "1"},
            "parameters": {
                "ious": [0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95]
            },
            "value": 0.8,
            "type": "APAveragedOverIOUs",
        },
        {
            "label": {"key": "class", "value": "4"},
            "parameters": {
                "ious": [0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95]
            },
            "value": 0.65,
            "type": "APAveragedOverIOUs",
        },
        {
            "parameters": {
                "label_key": "class",
                "ious": [
                    0.5,
                    0.55,
                    0.6,
                    0.65,
                    0.7,
                    0.75,
                    0.8,
                    0.85,
                    0.9,
                    0.95,
                ],
            },
            "value": 0.637,
            "type": "mAPAveragedOverIOUs",
        },
        {
            "label": {"key": "class", "value": "0"},
            "parameters": {
                "ious": [0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95]
            },
            "value": 0.78,
            "type": "AR",
        },
        {
            "label": {"key": "class", "value": "2"},
            "parameters": {
                "ious": [0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95]
            },
            "value": 0.45,
            "type": "AR",
        },
        {
            "label": {"key": "class", "value": "49"},
            "parameters": {
                "ious": [0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95]
            },
            "value": 0.58,
            "type": "AR",
        },
        {
            "label": {"key": "class", "value": "3"},
            "parameters": {
                "ious": [0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95]
            },
            "value": -1.0,
            "type": "AR",
        },
        {
            "label": {"key": "class", "value": "1"},
            "parameters": {
                "ious": [0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95]
            },
            "value": 0.8,
            "type": "AR",
        },
        {
            "label": {"key": "class", "value": "4"},
            "parameters": {
                "ious": [0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95]
            },
            "value": 0.65,
            "type": "AR",
        },
        {
            "parameters": {
                "label_key": "class",
                "ious": [
                    0.5,
                    0.55,
                    0.6,
                    0.65,
                    0.7,
                    0.75,
                    0.8,
                    0.85,
                    0.9,
                    0.95,
                ],
            },
            "value": 0.652,
            "type": "mAR",
        },
    ]

    pr_expected_answers = {
        # (class, 4)
        ("class", "4", 0.05, "tp"): 2,
        ("class", "4", 0.05, "fn"): 0,
        ("class", "4", 0.25, "tp"): 1,
        ("class", "4", 0.25, "fn"): 1,
        ("class", "4", 0.55, "tp"): 0,
        ("class", "4", 0.55, "fn"): 2,
        # (class, 2)
        ("class", "2", 0.05, "tp"): 1,
        ("class", "2", 0.05, "fn"): 1,
        ("class", "2", 0.75, "tp"): 0,
        ("class", "2", 0.75, "fn"): 2,
        # (class, 49)
        ("class", "49", 0.05, "tp"): 8,
        ("class", "49", 0.3, "tp"): 5,
        ("class", "49", 0.5, "tp"): 4,
        ("class", "49", 0.85, "tp"): 1,
        # (class, 3)
        ("class", "3", 0.05, "tp"): 0,
        ("class", "3", 0.05, "fp"): 1,
        # (class, 1)
        ("class", "1", 0.05, "tp"): 1,
        ("class", "1", 0.35, "tp"): 0,
        # (class, 0)
        ("class", "0", 0.05, "tp"): 5,
        ("class", "0", 0.5, "tp"): 3,
        ("class", "0", 0.95, "tp"): 1,
        ("class", "0", 0.95, "fn"): 4,
    }

    detailed_pr_expected_answers = {
        # (class, 4)
        ("4", 0.05, "tp"): {"all": 2, "total": 2},
        ("4", 0.05, "fn"): {
            "no_predictions": 0,
            "misclassifications": 0,
            "total": 0,
        },
        # (class, 2)
        ("2", 0.05, "tp"): {"all": 1, "total": 1},
        ("2", 0.05, "fn"): {
            "no_predictions": 0,
            "misclassifications": 1,
            "total": 1,
        },
        ("2", 0.75, "tp"): {"all": 0, "total": 0},
        ("2", 0.75, "fn"): {
            "no_predictions": 2,
            "misclassifications": 0,
            "total": 2,
        },
        # (class, 49)
        ("49", 0.05, "tp"): {"all": 9, "total": 9},
        # (class, 3)
        ("3", 0.05, "tp"): {"all": 0, "total": 0},
        ("3", 0.05, "fp"): {
            "hallucinations": 0,
            "misclassifications": 1,
            "total": 1,
        },
        # (class, 1)
        ("1", 0.05, "tp"): {"all": 1, "total": 1},
        ("1", 0.8, "fn"): {
            "no_predictions": 1,
            "misclassifications": 0,
            "total": 1,
        },
        # (class, 0)
        ("0", 0.05, "tp"): {"all": 5, "total": 5},
        ("0", 0.95, "fn"): {
            "no_predictions": 4,
            "misclassifications": 0,
            "total": 4,
        },
    }

    for m in metrics:
        assert m in expected_metrics
    for m in metrics:
        assert m in eval_job.metrics

    for (
        _,
        value,
        threshold,
        metric,
    ), expected_value in pr_expected_answers.items():
        assert (
            pr_metrics[0]["value"][value][threshold][metric] == expected_value
        )

    for (
        value,
        threshold,
        metric,
    ), expected_output in detailed_pr_expected_answers.items():
        model_output = detailed_pr_metrics[0]["value"][value][threshold][
            metric
        ]
        assert isinstance(model_output, dict)
        assert model_output["total"] == expected_output["total"]
        assert all(
            [
                model_output["observations"][key]["count"]  # type: ignore - we know this element is a dict
                == expected_output[key]
                for key in [
                    key
                    for key in expected_output.keys()
                    if key not in ["total"]
                ]
            ]
        )

    # spot check number of examples
    assert (
        len(
            detailed_pr_metrics[0]["value"]["0"][0.95]["fn"]["observations"]["no_predictions"][  # type: ignore - we know this element is a dict
                "examples"
            ]
        )
        == 1
    )
    assert (
        len(
            detailed_pr_metrics[0]["value"]["49"][0.05]["tp"]["observations"]["all"][  # type: ignore - we know this element is a dict
                "examples"
            ]
        )
        == 1
    )

    # raise the iou threshold
    eval_job_higher_threshold = evaluate_detection(
        groundtruths=evaluate_detection_functional_test_groundtruths,
        predictions=evaluate_detection_functional_test_predictions,
        metrics_to_return=[
            enums.MetricType.PrecisionRecallCurve,
            enums.MetricType.DetailedPrecisionRecallCurve,
        ],
        pr_curve_iou_threshold=0.9,
        pr_curve_max_examples=1,
    )

    pr_metrics = [
        m
        for m in eval_job_higher_threshold.metrics
        if m["type"] == "PrecisionRecallCurve"
    ]
    detailed_pr_metrics = [
        m
        for m in eval_job_higher_threshold.metrics
        if m["type"] == "DetailedPrecisionRecallCurve"
    ]

    pr_expected_answers = {
        # (class, 4)
        ("class", "4", 0.05, "tp"): 0,
        ("class", "4", 0.05, "fn"): 2,
        # (class, 2)
        ("class", "2", 0.05, "tp"): 1,
        ("class", "2", 0.05, "fn"): 1,
        ("class", "2", 0.75, "tp"): 0,
        ("class", "2", 0.75, "fn"): 2,
        # (class, 49)
        ("class", "49", 0.05, "tp"): 2,
        ("class", "49", 0.3, "tp"): 2,
        ("class", "49", 0.5, "tp"): 2,
        ("class", "49", 0.85, "tp"): 1,
        # (class, 3)
        ("class", "3", 0.05, "tp"): 0,
        ("class", "3", 0.05, "fp"): 1,
        # (class, 1)
        ("class", "1", 0.05, "tp"): 0,
        ("class", "1", 0.05, "fn"): 1,
        # (class, 0)
        ("class", "0", 0.05, "tp"): 1,
        ("class", "0", 0.5, "tp"): 0,
        ("class", "0", 0.95, "fn"): 5,
    }

    detailed_pr_expected_answers = {
        # (class, 4)
        ("4", 0.05, "tp"): {"all": 0, "total": 0},
        ("4", 0.05, "fn"): {
            "no_predictions": 2,  # below IOU threshold of .9
            "misclassifications": 0,
            "total": 2,
        },
        # (class, 2)
        ("2", 0.05, "tp"): {"all": 1, "total": 1},
        ("2", 0.05, "fn"): {
            "no_predictions": 1,
            "misclassifications": 0,
            "total": 1,
        },
        ("2", 0.75, "tp"): {"all": 0, "total": 0},
        ("2", 0.75, "fn"): {
            "no_predictions": 2,
            "misclassifications": 0,
            "total": 2,
        },
        # (class, 49)
        ("49", 0.05, "tp"): {"all": 2, "total": 2},
        # (class, 3)
        ("3", 0.05, "tp"): {"all": 0, "total": 0},
        ("3", 0.05, "fp"): {
            "hallucinations": 1,
            "misclassifications": 0,
            "total": 1,
        },
        # (class, 1)
        ("1", 0.05, "tp"): {"all": 0, "total": 0},
        ("1", 0.8, "fn"): {
            "no_predictions": 1,
            "misclassifications": 0,
            "total": 1,
        },
        # (class, 0)
        ("0", 0.05, "tp"): {"all": 1, "total": 1},
        ("0", 0.95, "fn"): {
            "no_predictions": 5,
            "misclassifications": 0,
            "total": 5,
        },
    }

    for (
        key,
        value,
        threshold,
        metric,
    ), expected_count in pr_expected_answers.items():
        actual_count = pr_metrics[0]["value"][value][threshold][metric]
        assert actual_count == expected_count

    for (
        value,
        threshold,
        metric,
    ), expected_output in detailed_pr_expected_answers.items():
        model_output = detailed_pr_metrics[0]["value"][value][threshold][
            metric
        ]
        assert isinstance(model_output, dict)
        assert model_output["total"] == expected_output["total"]
        assert all(
            [
                model_output["observations"][key]["count"]  # type: ignore - we know this element is a dict
                == expected_output[key]
                for key in [
                    key
                    for key in expected_output.keys()
                    if key not in ["total"]
                ]
            ]
        )

    assert (
        len(
            detailed_pr_metrics[0]["value"]["0"][0.95]["fn"]["observations"]["no_predictions"][  # type: ignore - we know this element is a dict
                "examples"
            ]
        )
        == 1
    )
    assert (
        len(
            detailed_pr_metrics[0]["value"]["49"][0.05]["tp"]["observations"]["all"][  # type: ignore - we know this element is a dict
                "examples"
            ]
        )
        == 1
    )

    # repeat the above, but with a higher pr_max_curves_example
    eval_job_higher_threshold = evaluate_detection(
        groundtruths=evaluate_detection_functional_test_groundtruths,
        predictions=evaluate_detection_functional_test_predictions,
        metrics_to_return=[
            enums.MetricType.PrecisionRecallCurve,
            enums.MetricType.DetailedPrecisionRecallCurve,
        ],
        pr_curve_iou_threshold=0.9,
        pr_curve_max_examples=3,
    )

    pr_metrics = [
        m
        for m in eval_job_higher_threshold.metrics
        if m["type"] == "PrecisionRecallCurve"
    ]
    detailed_pr_metrics = [
        m
        for m in eval_job_higher_threshold.metrics
        if m["type"] == "DetailedPrecisionRecallCurve"
    ]

    for (
        key,
        value,
        threshold,
        metric,
    ), expected_count in pr_expected_answers.items():
        actual_count = pr_metrics[0]["value"][value][threshold][metric]
        assert actual_count == expected_count

    for (
        value,
        threshold,
        metric,
    ), expected_output in detailed_pr_expected_answers.items():
        model_output = detailed_pr_metrics[0]["value"][value][threshold][
            metric
        ]
        assert isinstance(model_output, dict)
        assert model_output["total"] == expected_output["total"]
        assert all(
            [
                model_output["observations"][key]["count"]  # type: ignore - we know this element is a dict
                == expected_output[key]
                for key in [
                    key
                    for key in expected_output.keys()
                    if key not in ["total"]
                ]
            ]
        )

    assert (
        len(
            detailed_pr_metrics[0]["value"]["0"][0.95]["fn"]["observations"]["no_predictions"][  # type: ignore - we know this element is a dict
                "examples"
            ]
        )
        == 3
    )
    assert (
        len(
            detailed_pr_metrics[0]["value"]["49"][0.05]["tp"]["observations"]["all"][  # type: ignore - we know this element is a dict
                "examples"
            ]
        )
        == 2
    )

    # test behavior if pr_curve_max_examples == 0
    eval_job_higher_threshold = evaluate_detection(
        groundtruths=evaluate_detection_functional_test_groundtruths,
        predictions=evaluate_detection_functional_test_predictions,
        metrics_to_return=[
            enums.MetricType.PrecisionRecallCurve,
            enums.MetricType.DetailedPrecisionRecallCurve,
        ],
        pr_curve_iou_threshold=0.9,
        pr_curve_max_examples=0,
    )

    pr_metrics = [
        m
        for m in eval_job_higher_threshold.metrics
        if m["type"] == "PrecisionRecallCurve"
    ]
    detailed_pr_metrics = [
        m
        for m in eval_job_higher_threshold.metrics
        if m["type"] == "DetailedPrecisionRecallCurve"
    ]

    for (
        key,
        value,
        threshold,
        metric,
    ), expected_count in pr_expected_answers.items():
        actual_count = pr_metrics[0]["value"][value][threshold][metric]
        assert actual_count == expected_count

    for (
        value,
        threshold,
        metric,
    ), expected_output in detailed_pr_expected_answers.items():
        model_output = detailed_pr_metrics[0]["value"][value][threshold][
            metric
        ]
        assert isinstance(model_output, dict)
        assert model_output["total"] == expected_output["total"]
        assert all(
            [
                model_output["observations"][key]["count"]  # type: ignore - we know this element is a dict
                == expected_output[key]
                for key in [
                    key
                    for key in expected_output.keys()
                    if key not in ["total"]
                ]
            ]
        )

    # spot check number of examples
    assert (
        len(
            detailed_pr_metrics[0]["value"]["0"][0.95]["fn"]["observations"]["no_predictions"][  # type: ignore - we know this element is a dict
                "examples"
            ]
        )
        == 0
    )
    assert (
        len(
            detailed_pr_metrics[0]["value"]["49"][0.05]["tp"]["observations"]["all"][  # type: ignore - we know this element is a dict
                "examples"
            ]
        )
        == 0
    )


def test_evaluate_detection_functional_test_with_rasters(
    evaluate_detection_functional_test_groundtruths_with_rasters,
    evaluate_detection_functional_test_predictions_with_rasters,
):
    eval_job = evaluate_detection(
        groundtruths=evaluate_detection_functional_test_groundtruths_with_rasters,
        predictions=evaluate_detection_functional_test_predictions_with_rasters,
        metrics_to_return=[
            enums.MetricType.AP,
            enums.MetricType.AR,
            enums.MetricType.mAP,
            enums.MetricType.APAveragedOverIOUs,
            enums.MetricType.mAR,
            enums.MetricType.mAPAveragedOverIOUs,
            enums.MetricType.PrecisionRecallCurve,
        ],
        pr_curve_iou_threshold=0.5,
        pr_curve_max_examples=1,
    )

    metrics = [
        m
        for m in eval_job.metrics
        if m["type"]
        not in ["PrecisionRecallCurve", "DetailedPrecisionRecallCurve"]
    ]

    # round all metrics to the third decimal place
    for i, m in enumerate(metrics):
        metrics[i]["value"] = round(m["value"], 3)

    pr_metrics = [
        m for m in eval_job.metrics if m["type"] == "PrecisionRecallCurve"
    ]

    expected_metrics = [
        {
            "label": {"key": "class", "value": "label1"},
            "parameters": {"iou": 0.5},
            "value": 1.0,
            "type": "AP",
        },
        {
            "label": {"key": "class", "value": "label1"},
            "parameters": {"iou": 0.75},
            "value": 1.0,
            "type": "AP",
        },
        {
            "label": {"key": "class", "value": "label2"},
            "parameters": {"iou": 0.5},
            "value": 1.0,
            "type": "AP",
        },
        {
            "label": {"key": "class", "value": "label2"},
            "parameters": {"iou": 0.75},
            "value": 1.0,
            "type": "AP",
        },
        {
            "label": {"key": "class", "value": "label3"},
            "parameters": {"iou": 0.5},
            "value": 0.0,
            "type": "AP",
        },
        {
            "label": {"key": "class", "value": "label3"},
            "parameters": {"iou": 0.75},
            "value": 0.0,
            "type": "AP",
        },
        {
            "parameters": {"label_key": "class", "iou": 0.5},
            "value": 0.667,
            "type": "mAP",
        },
        {
            "parameters": {"label_key": "class", "iou": 0.75},
            "value": 0.667,
            "type": "mAP",
        },
        {
            "label": {"key": "class", "value": "label1"},
            "parameters": {
                "ious": [0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95]
            },
            "value": 1.0,
            "type": "APAveragedOverIOUs",
        },
        {
            "label": {"key": "class", "value": "label2"},
            "parameters": {
                "ious": [0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95]
            },
            "value": 1.0,
            "type": "APAveragedOverIOUs",
        },
        {
            "label": {"key": "class", "value": "label3"},
            "parameters": {
                "ious": [0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95]
            },
            "value": 0.0,
            "type": "APAveragedOverIOUs",
        },
        {
            "parameters": {
                "label_key": "class",
                "ious": [
                    0.5,
                    0.55,
                    0.6,
                    0.65,
                    0.7,
                    0.75,
                    0.8,
                    0.85,
                    0.9,
                    0.95,
                ],
            },
            "value": 0.667,
            "type": "mAPAveragedOverIOUs",
        },
        {
            "label": {"key": "class", "value": "label1"},
            "parameters": {
                "ious": [0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95]
            },
            "value": 1.0,
            "type": "AR",
        },
        {
            "label": {"key": "class", "value": "label4"},
            "parameters": {
                "ious": [0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95]
            },
            "value": -1.0,
            "type": "AR",
        },
        {
            "label": {"key": "class", "value": "label2"},
            "parameters": {
                "ious": [0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95]
            },
            "value": 1.0,
            "type": "AR",
        },
        {
            "label": {"key": "class", "value": "label3"},
            "parameters": {
                "ious": [0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95]
            },
            "value": 0.0,
            "type": "AR",
        },
        {
            "parameters": {
                "label_key": "class",
                "ious": [
                    0.5,
                    0.55,
                    0.6,
                    0.65,
                    0.7,
                    0.75,
                    0.8,
                    0.85,
                    0.9,
                    0.95,
                ],
            },
            "value": 0.667,
            "type": "mAR",
        },
    ]

    for m in metrics:
        assert m in expected_metrics

    for m in expected_metrics:
        assert m in metrics

    pr_expected_answers = {
        ("class", "label1", 0.05, "tp"): 1,
        ("class", "label1", 0.35, "tp"): 0,
        ("class", "label2", 0.05, "tp"): 1,
        ("class", "label2", 0.05, "fp"): 0,
        ("class", "label2", 0.95, "fp"): 0,
        ("class", "label3", 0.05, "tp"): 0,
        ("class", "label3", 0.05, "fn"): 1,
        ("class", "label4", 0.05, "tp"): 0,
        ("class", "label4", 0.05, "fp"): 1,
    }

    for (
        _,
        value,
        threshold,
        metric,
    ), expected_value in pr_expected_answers.items():
        assert (
            pr_metrics[0]["value"][value][threshold][metric] == expected_value
        )
