import random

import pandas as pd
import pytest
from valor_core import enums, schemas
from valor_core.evaluator import DetectionManager as Manager


def test_evaluate_detection_with_manager(
    evaluate_detection_groundtruths,
    evaluate_detection_predictions,
    evaluate_detection_expected: tuple,
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

    expected_metrics, expected_metadata = evaluate_detection_expected

    manager = Manager()
    manager.add_data(
        groundtruths=evaluate_detection_groundtruths,
        predictions=evaluate_detection_predictions,
    )
    manager.finalize()

    metrics = manager.compute_ap(
        iou_thresholds=[0.1, 0.6],
    )

    converted_metrics = []
    for metric in metrics:
        metric.pop("type")
        label_key, label_value = metric.pop("label")
        ious = set(metric.keys())
        for iou in ious:
            converted_metrics.append(
                {
                    "type": "AP",
                    "parameters": {"iou": float(iou)},
                    "value": metric[iou],
                    "label": {"key": label_key, "value": label_value},
                }
            )
    metrics = converted_metrics

    for m in metrics:
        if m["type"] == "AP":
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

    assert result_dict == expected_metadata

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

    manager = Manager(
        iou_thresholds_to_compute=[0.1, 0.6],
        iou_thresholds_to_return=[0.1, 0.6],
        metrics_to_return=selected_metrics,
    )

    manager.add_data(
        groundtruths=evaluate_detection_groundtruths,
        predictions=evaluate_detection_predictions,
    )

    # check that ious have been precomputed
    assert "iou_" in manager.joint_df.columns
    assert all(
        [
            col not in ["raster", "bounding_box"]
            for col in manager.joint_df.columns
        ]
    )

    eval_job = manager.evaluate()

    metrics = eval_job.metrics
    assert set([metric["type"] for metric in eval_job.metrics]) == set(
        selected_metrics
    )


def test_evaluate_detection_via_pandas_df_with_ValorDetectionManager(
    evaluate_detection_groundtruths_df: pd.DataFrame,
    evaluate_detection_predictions_df: pd.DataFrame,
):
    """The Manager shouldn't except dataframes, so we just confirm this test throws an error here."""

    manager = Manager(
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

    with pytest.raises(ValueError) as e:
        manager.add_data(
            groundtruths=evaluate_detection_groundtruths_df,  # type: ignore - purposefully throwing error
            predictions=evaluate_detection_predictions_df,  # type: ignore - purposefully throwing error
        )
    assert (
        "groundtruths should be a non-empty list of schemas.GroundTruth objects."
        in str(e)
    )


def test_evaluate_detection_false_negatives_single_image_baseline_with_ValorDetectionManager(
    evaluate_detection_false_negatives_single_image_baseline_inputs: tuple,
    evaluate_detection_false_negatives_AP_of_1: dict,
):
    """This is the baseline for the below test. In this case there are two predictions and
    one groundtruth, but the highest confident prediction overlaps sufficiently with the groundtruth
    so there is not a penalty for the false negative so the AP is 1
    """

    (
        groundtruths,
        predictions,
    ) = evaluate_detection_false_negatives_single_image_baseline_inputs

    manager = Manager(
        iou_thresholds_to_compute=[0.5],
        iou_thresholds_to_return=[0.5],
    )

    manager.add_data(
        groundtruths=groundtruths,
        predictions=predictions,
    )

    # check that ious have been precomputed
    assert "iou_" in manager.joint_df.columns
    assert all(
        [
            col not in ["raster", "bounding_box"]
            for col in manager.joint_df.columns
        ]
    )

    eval_job = manager.evaluate()

    ap_metric = [m for m in eval_job.metrics if m["type"] == "AP"][0]
    assert ap_metric == evaluate_detection_false_negatives_AP_of_1


def test_evaluate_detection_false_negatives_single_image_with_ValorDetectionManager(
    evaluate_detection_false_negatives_single_image_inputs: tuple,
    evaluate_detection_false_negatives_AP_of_point_5: dict,
):
    """Tests fix for a bug where high confidence false negative was not being penalized. The
    difference between this test and the above is that here the prediction with higher confidence
    does not sufficiently overlap the groundtruth and so is penalized and we get an AP of 0.5
    """

    (
        groundtruths,
        predictions,
    ) = evaluate_detection_false_negatives_single_image_inputs

    manager = Manager(
        iou_thresholds_to_compute=[0.5],
        iou_thresholds_to_return=[0.5],
    )

    manager.add_data(
        groundtruths=groundtruths,
        predictions=predictions,
    )

    # check that ious have been precomputed
    assert "iou_" in manager.joint_df.columns
    assert all(
        [
            col not in ["raster", "bounding_box"]
            for col in manager.joint_df.columns
        ]
    )

    eval_job = manager.evaluate()

    ap_metric = [m for m in eval_job.metrics if m["type"] == "AP"][0]
    assert ap_metric == evaluate_detection_false_negatives_AP_of_point_5


def test_evaluate_detection_false_negatives_two_images_one_empty_low_confidence_of_fp_with_ValorDetectionManager(
    evaluate_detection_false_negatives_two_images_one_empty_low_confidence_of_fp_inputs: tuple,
    evaluate_detection_false_negatives_AP_of_1: dict,
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

    manager = Manager(
        iou_thresholds_to_compute=[0.5],
        iou_thresholds_to_return=[0.5],
    )

    manager.add_data(
        groundtruths=groundtruths,
        predictions=predictions,
    )

    # check that ious have been precomputed
    assert "iou_" in manager.joint_df.columns
    assert all(
        [
            col not in ["raster", "bounding_box"]
            for col in manager.joint_df.columns
        ]
    )

    eval_job = manager.evaluate()

    ap_metric = [m for m in eval_job.metrics if m["type"] == "AP"][0]
    assert ap_metric == evaluate_detection_false_negatives_AP_of_1


def test_evaluate_detection_false_negatives_two_images_one_empty_high_confidence_of_fp_with_ValorDetectionManager(
    evaluate_detection_false_negatives_two_images_one_empty_high_confidence_of_fp_inputs: tuple,
    evaluate_detection_false_negatives_AP_of_point_5: dict,
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

    manager = Manager(
        iou_thresholds_to_compute=[0.5],
        iou_thresholds_to_return=[0.5],
    )

    manager.add_data(
        groundtruths=groundtruths,
        predictions=predictions,
    )

    # check that ious have been precomputed
    assert "iou_" in manager.joint_df.columns
    assert all(
        [
            col not in ["raster", "bounding_box"]
            for col in manager.joint_df.columns
        ]
    )

    eval_job = manager.evaluate()

    ap_metric = [m for m in eval_job.metrics if m["type"] == "AP"][0]
    assert ap_metric == evaluate_detection_false_negatives_AP_of_point_5


def test_evaluate_detection_false_negatives_two_images_one_only_with_different_class_low_confidence_of_fp_with_ValorDetectionManager(
    evaluate_detection_false_negatives_two_images_one_only_with_different_class_low_confidence_of_fp_inputs: tuple,
    evaluate_detection_false_negatives_AP_of_1: dict,
    evaluate_detection_false_negatives_AP_of_0: dict,
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
    manager = Manager(
        iou_thresholds_to_compute=[0.5],
        iou_thresholds_to_return=[0.5],
    )

    manager.add_data(
        groundtruths=groundtruths,
        predictions=predictions,
    )

    # check that ious have been precomputed
    assert "iou_" in manager.joint_df.columns
    assert all(
        [
            col not in ["raster", "bounding_box"]
            for col in manager.joint_df.columns
        ]
    )

    eval_job = manager.evaluate()

    ap_metric1 = [
        m
        for m in eval_job.metrics
        if m["type"] == "AP" and m["label"] == {"key": "key", "value": "value"}
    ][0]
    assert ap_metric1 == evaluate_detection_false_negatives_AP_of_1

    # label `"other value"` is not in the predictions so we should get an AP of 0
    ap_metric2 = [
        m
        for m in eval_job.metrics
        if m["type"] == "AP"
        and m["label"] == {"key": "key", "value": "other value"}
    ][0]
    assert ap_metric2 == evaluate_detection_false_negatives_AP_of_0


def test_evaluate_detection_false_negatives_two_images_one_only_with_different_class_high_confidence_of_fp_with_ValorDetectionManager(
    evaluate_detection_false_negatives_two_images_one_only_with_different_class_high_confidence_of_fp_inputs: tuple,
    evaluate_detection_false_negatives_AP_of_point_5: dict,
    evaluate_detection_false_negatives_AP_of_0: dict,
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

    manager = Manager(
        iou_thresholds_to_compute=[0.5],
        iou_thresholds_to_return=[0.5],
    )

    manager.add_data(
        groundtruths=groundtruths,
        predictions=predictions,
    )

    # check that ious have been precomputed
    assert "iou_" in manager.joint_df.columns
    assert all(
        [
            col not in ["raster", "bounding_box"]
            for col in manager.joint_df.columns
        ]
    )

    eval_job = manager.evaluate()

    ap_metric1 = [
        m
        for m in eval_job.metrics
        if m["type"] == "AP" and m["label"] == {"key": "key", "value": "value"}
    ][0]
    assert ap_metric1 == evaluate_detection_false_negatives_AP_of_point_5

    # label `"other value"` is not in the predictions so we should get an AP of 0
    ap_metric2 = [
        m
        for m in eval_job.metrics
        if m["type"] == "AP"
        and m["label"] == {"key": "key", "value": "other value"}
    ][0]
    assert ap_metric2 == evaluate_detection_false_negatives_AP_of_0


@pytest.fixture
def test_detailed_precision_recall_curve_with_ValorDetectionManager(
    evaluate_detection_detailed_pr_curve_groundtruths: list,
    evaluate_detection_detailed_pr_curve_predictions: list,
    detailed_precision_recall_curve_outputs: tuple,
):

    expected_outputs, _ = detailed_precision_recall_curve_outputs

    manager = Manager(
        metrics_to_return=[enums.MetricType.DetailedPrecisionRecallCurve],
    )

    manager.add_data(
        groundtruths=evaluate_detection_detailed_pr_curve_groundtruths,
        predictions=evaluate_detection_detailed_pr_curve_predictions,
    )

    # check that ious have been precomputed
    assert "iou_" in manager.joint_df.columns
    assert all(
        [
            col not in ["raster", "bounding_box"]
            for col in manager.joint_df.columns
        ]
    )

    eval_job = manager.evaluate()
    for key, expected_value in expected_outputs.items():
        result = eval_job.metrics[0]["value"]
        for k in key:
            result = result[k]
        assert result == expected_value

    # repeat tests using a lower IOU threshold
    manager = Manager(
        metrics_to_return=[enums.MetricType.DetailedPrecisionRecallCurve],
        pr_curve_iou_threshold=0.45,
    )

    manager.add_data(
        groundtruths=evaluate_detection_detailed_pr_curve_groundtruths,
        predictions=evaluate_detection_detailed_pr_curve_predictions,
    )

    eval_job_low_iou_threshold = manager.evaluate()

    for key, expected_value in expected_outputs.items():
        result = eval_job_low_iou_threshold.metrics[0]["value"]
        for k in key:
            result = result[k]
        assert result == expected_value


def test_evaluate_detection_model_with_no_predictions_with_ValorDetectionManager(
    evaluate_detection_groundtruths: list,
    evaluate_detection_model_with_no_predictions_output: list,
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

    manager = Manager()

    # can't pass empty lists, but can pass predictions without annotations
    with pytest.raises(ValueError) as e:
        manager.add_data(
            groundtruths=evaluate_detection_groundtruths,
            predictions=[],
        )
    assert (
        "it's neither a dataframe nor a list of Valor Prediction objects"
        in str(e)
    )

    manager.add_data(
        groundtruths=evaluate_detection_groundtruths,
        predictions=predictions,
    )

    # check that ious have been precomputed
    assert "iou_" in manager.joint_df.columns
    assert all(
        [
            col not in ["raster", "bounding_box"]
            for col in manager.joint_df.columns
        ]
    )

    eval_job = manager.evaluate()

    computed_metrics = eval_job.metrics

    assert all([metric["value"] == 0 for metric in computed_metrics])

    for m in evaluate_detection_model_with_no_predictions_output:
        assert m in computed_metrics

    for m in computed_metrics:
        assert m in evaluate_detection_model_with_no_predictions_output


def test_evaluate_detection_functional_test_with_ValorDetectionManager(
    evaluate_detection_functional_test_groundtruths: list,
    evaluate_detection_functional_test_predictions: list,
    evaluate_detection_functional_test_outputs: tuple,
):

    (
        expected_metrics,
        pr_expected_answers,
        detailed_pr_expected_answers,
        higher_iou_threshold_pr_expected_answers,
        higher_iou_threshold_detailed_pr_expected_answers,
    ) = evaluate_detection_functional_test_outputs

    manager = Manager(
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

    manager.add_data(
        groundtruths=evaluate_detection_functional_test_groundtruths,
        predictions=evaluate_detection_functional_test_predictions,
    )

    # check that ious have been precomputed
    assert "iou_" in manager.joint_df.columns
    assert all(
        [
            col not in ["raster", "bounding_box"]
            for col in manager.joint_df.columns
        ]
    )

    eval_job = manager.evaluate()

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
    manager = Manager(
        metrics_to_return=[
            enums.MetricType.PrecisionRecallCurve,
            enums.MetricType.DetailedPrecisionRecallCurve,
        ],
        pr_curve_iou_threshold=0.9,
        pr_curve_max_examples=1,
    )

    manager.add_data(
        groundtruths=evaluate_detection_functional_test_groundtruths,
        predictions=evaluate_detection_functional_test_predictions,
    )

    # check that ious have been precomputed
    assert "iou_" in manager.joint_df.columns
    assert all(
        [
            col not in ["raster", "bounding_box"]
            for col in manager.joint_df.columns
        ]
    )

    eval_job_higher_threshold = manager.evaluate()

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
    ), expected_count in higher_iou_threshold_pr_expected_answers.items():
        actual_count = pr_metrics[0]["value"][value][threshold][metric]
        assert actual_count == expected_count

    for (
        value,
        threshold,
        metric,
    ), expected_output in (
        higher_iou_threshold_detailed_pr_expected_answers.items()
    ):
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
    manager = Manager(
        metrics_to_return=[
            enums.MetricType.PrecisionRecallCurve,
            enums.MetricType.DetailedPrecisionRecallCurve,
        ],
        pr_curve_iou_threshold=0.9,
        pr_curve_max_examples=3,
    )

    manager.add_data(
        groundtruths=evaluate_detection_functional_test_groundtruths,
        predictions=evaluate_detection_functional_test_predictions,
    )

    # check that ious have been precomputed
    assert "iou_" in manager.joint_df.columns
    assert all(
        [
            col not in ["raster", "bounding_box"]
            for col in manager.joint_df.columns
        ]
    )

    eval_job_higher_threshold = manager.evaluate()

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
    ), expected_count in higher_iou_threshold_pr_expected_answers.items():
        actual_count = pr_metrics[0]["value"][value][threshold][metric]
        assert actual_count == expected_count

    for (
        value,
        threshold,
        metric,
    ), expected_output in (
        higher_iou_threshold_detailed_pr_expected_answers.items()
    ):
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
    manager = Manager(
        metrics_to_return=[
            enums.MetricType.PrecisionRecallCurve,
            enums.MetricType.DetailedPrecisionRecallCurve,
        ],
        pr_curve_iou_threshold=0.9,
        pr_curve_max_examples=0,
    )

    manager.add_data(
        groundtruths=evaluate_detection_functional_test_groundtruths,
        predictions=evaluate_detection_functional_test_predictions,
    )

    # check that ious have been precomputed
    assert "iou_" in manager.joint_df.columns
    assert all(
        [
            col not in ["raster", "bounding_box"]
            for col in manager.joint_df.columns
        ]
    )

    eval_job_higher_threshold = manager.evaluate()

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
    ), expected_count in higher_iou_threshold_pr_expected_answers.items():
        actual_count = pr_metrics[0]["value"][value][threshold][metric]
        assert actual_count == expected_count

    for (
        value,
        threshold,
        metric,
    ), expected_output in (
        higher_iou_threshold_detailed_pr_expected_answers.items()
    ):
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


def test_evaluate_detection_functional_test_with_rasters_with_ValorDetectionManager(
    evaluate_detection_functional_test_groundtruths_with_rasters: list,
    evaluate_detection_functional_test_predictions_with_rasters: list,
    evaluate_detection_functional_test_with_rasters_outputs: tuple,
):

    (
        expected_metrics,
        pr_expected_answers,
    ) = evaluate_detection_functional_test_with_rasters_outputs

    manager = Manager(
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

    manager.add_data(
        groundtruths=evaluate_detection_functional_test_groundtruths_with_rasters,
        predictions=evaluate_detection_functional_test_predictions_with_rasters,
    )

    # check that ious have been precomputed
    assert "iou_" in manager.joint_df.columns
    assert all(
        [
            col not in ["raster", "bounding_box"]
            for col in manager.joint_df.columns
        ]
    )

    eval_job = manager.evaluate()

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

    for m in metrics:
        assert m in expected_metrics

    for m in expected_metrics:
        assert m in metrics

    for (
        _,
        value,
        threshold,
        metric,
    ), expected_value in pr_expected_answers.items():
        assert (
            pr_metrics[0]["value"][value][threshold][metric] == expected_value
        )

    # test that we get a NotImplementedError if we try to calculate DetailedPRCurves with rasters
    manager = Manager(
        metrics_to_return=[
            enums.MetricType.DetailedPrecisionRecallCurve,
        ],
    )

    manager.add_data(
        groundtruths=evaluate_detection_functional_test_groundtruths_with_rasters,
        predictions=evaluate_detection_functional_test_predictions_with_rasters,
    )

    with pytest.raises(NotImplementedError):
        manager.evaluate()


def test_evaluate_mixed_annotations_with_ValorDetectionManager(
    evaluate_mixed_annotations_inputs: tuple,
    evaluate_mixed_annotations_output: list,
):
    """Test the automatic conversion to rasters."""

    gts, pds = evaluate_mixed_annotations_inputs

    manager = Manager(
        iou_thresholds_to_compute=[0.1, 0.6],
        iou_thresholds_to_return=[0.1, 0.6],
        metrics_to_return=[
            enums.MetricType.AP,
        ],
    )

    # by default, valor_core should throw an error if given mixed AnnotationTypes without being explicitely told to convert to a certain type
    with pytest.raises(ValueError):
        manager.add_data(
            groundtruths=gts,
            predictions=pds,
        )

    # test conversion to raster. this should throw an error since the user is trying to convert a Box annotation to a polygon.
    manager = Manager(
        iou_thresholds_to_compute=[0.1, 0.6],
        iou_thresholds_to_return=[0.1, 0.6],
        metrics_to_return=[
            enums.MetricType.AP,
        ],
        convert_annotations_to_type=enums.AnnotationType.RASTER,
    )
    with pytest.raises(ValueError):
        manager.add_data(
            groundtruths=gts,
            predictions=pds,
        )

    # test conversion to polygon. this should throw an error since the user is trying to convert a Box annotation to a polygon.
    manager = Manager(
        iou_thresholds_to_compute=[0.1, 0.6],
        iou_thresholds_to_return=[0.1, 0.6],
        metrics_to_return=[
            enums.MetricType.AP,
        ],
        convert_annotations_to_type=enums.AnnotationType.POLYGON,
    )
    with pytest.raises(ValueError):
        manager.add_data(
            groundtruths=gts,
            predictions=pds,
        )

    # test conversion to box
    manager = Manager(
        iou_thresholds_to_compute=[0.1, 0.6],
        iou_thresholds_to_return=[0.1, 0.6],
        metrics_to_return=[
            enums.MetricType.AP,
        ],
        convert_annotations_to_type=enums.AnnotationType.BOX,
    )
    manager.add_data(
        groundtruths=gts,
        predictions=pds,
    )

    eval_job_box = manager.evaluate()

    for m in eval_job_box.metrics:
        assert m in evaluate_mixed_annotations_output
    for m in evaluate_mixed_annotations_output:
        assert m in eval_job_box.metrics


def test_evaluate_detection_rotated_bboxes_with_shapely_with_ValorDetectionManager(
    evaluate_detection_rotated_bboxes_with_shapely_inputs: tuple,
    evaluate_detection_expected: tuple,
):
    """
    Run the same test as test_evaluate_detection, but rotate all of the bounding boxes by some random numbewr of degrees to confirm we get the same outputs.
    """

    (
        groundtruths,
        predictions,
    ) = evaluate_detection_rotated_bboxes_with_shapely_inputs
    expected_metrics, expected_metadata = evaluate_detection_expected

    manager = Manager(
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
    manager.add_data(
        groundtruths=groundtruths,
        predictions=predictions,
    )

    eval_job = manager.evaluate()

    metrics = eval_job.metrics

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

    assert result_dict == expected_metadata

    #  check that metrics arg works correctly
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

    manager = Manager(
        iou_thresholds_to_compute=[0.1, 0.6],
        iou_thresholds_to_return=[0.1, 0.6],
        metrics_to_return=selected_metrics,
    )
    manager.add_data(
        groundtruths=groundtruths,
        predictions=predictions,
    )

    eval_job = manager.evaluate()

    metrics = eval_job.metrics
    assert set([metric["type"] for metric in eval_job.metrics]) == set(
        selected_metrics
    )


def test_evaluate_detection_rotated_bboxes_with_ValorDetectionManager(
    evaluate_detection_rotated_bboxes_inputs: tuple,
    evaluate_detection_expected: tuple,
):
    """
    Run the same test as test_evaluate_detection, but rotate all of the bounding boxes by 5 degrees around the origin to confirm we get the same outputs.
    """

    groundtruths, predictions = evaluate_detection_rotated_bboxes_inputs
    expected_metrics, expected_metadata = evaluate_detection_expected

    manager = Manager(
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
    manager.add_data(
        groundtruths=groundtruths,
        predictions=predictions,
    )

    eval_job = manager.evaluate()

    metrics = eval_job.metrics

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

    assert result_dict == expected_metadata

    #  check that metrics arg works correctly
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

    manager = Manager(
        iou_thresholds_to_compute=[0.1, 0.6],
        iou_thresholds_to_return=[0.1, 0.6],
        metrics_to_return=selected_metrics,
    )
    manager.add_data(
        groundtruths=groundtruths,
        predictions=predictions,
    )

    eval_job = manager.evaluate()

    metrics = eval_job.metrics
    assert set([metric["type"] for metric in eval_job.metrics]) == set(
        selected_metrics
    )


def test_evaluate_detection_with_label_maps_and_ValorDetectionManager(
    evaluate_detection_groundtruths_with_label_maps: list,
    evaluate_detection_predictions_with_label_maps: list,
    evaluate_detection_with_label_maps_expected: tuple,
):
    """This test is the same as test_evaluate_detection_with_label_maps, but we use ValorDetectionManager to pre-compute IOUs in advance"""

    (
        baseline_expected_metrics,
        baseline_pr_expected_answers,
        baseline_detailed_pr_expected_answers,
        cat_expected_metrics,
        foo_expected_metrics,
        foo_pr_expected_answers,
        foo_expected_metrics_with_higher_score_threshold,
    ) = evaluate_detection_with_label_maps_expected

    manager = Manager(
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

    manager.add_data(
        groundtruths=evaluate_detection_groundtruths_with_label_maps[:1],
        predictions=evaluate_detection_predictions_with_label_maps[:1],
    )

    # test that both fields are required
    with pytest.raises(ValueError):
        manager.add_data(
            groundtruths=[],
            predictions=evaluate_detection_predictions_with_label_maps[:2],
        )

    manager.add_data(
        groundtruths=evaluate_detection_groundtruths_with_label_maps[1:2],
        predictions=evaluate_detection_predictions_with_label_maps[1:2],
    )

    # can't add an already existing datum
    with pytest.raises(ValueError):
        manager.add_data(
            groundtruths=evaluate_detection_groundtruths_with_label_maps[1:2],
            predictions=evaluate_detection_predictions_with_label_maps[1:2],
        )

    # check that ious have been precomputed
    assert "iou_" in manager.joint_df.columns
    assert all(
        [
            col not in ["raster", "bounding_box"]
            for col in manager.joint_df.columns
        ]
    )

    eval_job = manager.evaluate()

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

    for (
        index,
        key,
        value,
        threshold,
        metric,
    ), expected_value in baseline_pr_expected_answers.items():
        assert (
            pr_metrics[index]["value"][value][float(threshold)][metric]
            == expected_value
        )

    # check DetailedPrecisionRecallCurve
    for (
        index,
        value,
        threshold,
        metric,
    ), expected_output in baseline_detailed_pr_expected_answers.items():
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
    label_map = {
        schemas.Label(key="class_name", value="maine coon cat"): schemas.Label(
            key="class", value="cat"
        ),
        schemas.Label(key="class", value="siamese cat"): schemas.Label(
            key="class", value="cat"
        ),
        schemas.Label(key="class", value="british shorthair"): schemas.Label(
            key="class", value="cat"
        ),
    }

    # test that you can't modify an instanciated manager since that will lead to apples-to-oranges iou calculations
    with pytest.raises(AttributeError):
        manager.label_map = label_map

    manager = Manager(
        label_map=label_map,
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

    manager.add_data(
        groundtruths=evaluate_detection_groundtruths_with_label_maps[:1],
        predictions=evaluate_detection_predictions_with_label_maps[:1],
    )

    manager.add_data(
        groundtruths=evaluate_detection_groundtruths_with_label_maps[1:2],
        predictions=evaluate_detection_predictions_with_label_maps[1:2],
    )

    eval_job = manager.evaluate()

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
    label_map = {
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

    manager = Manager(
        label_map=label_map,
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

    manager.add_data(
        groundtruths=evaluate_detection_groundtruths_with_label_maps[:1],
        predictions=evaluate_detection_predictions_with_label_maps[:1],
    )

    manager.add_data(
        groundtruths=evaluate_detection_groundtruths_with_label_maps[1:2],
        predictions=evaluate_detection_predictions_with_label_maps[1:2],
    )

    eval_job = manager.evaluate()

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
    label_map = {
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

    manager = Manager(
        label_map=label_map,
        pr_curve_max_examples=1,
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

    manager.add_data(
        groundtruths=evaluate_detection_groundtruths_with_label_maps[:1],
        predictions=evaluate_detection_predictions_with_label_maps[:1],
    )

    manager.add_data(
        groundtruths=evaluate_detection_groundtruths_with_label_maps[1:2],
        predictions=evaluate_detection_predictions_with_label_maps[1:2],
    )

    eval_job = manager.evaluate()

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
        "convert_annotations_to_type": None,
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

    for (
        index,
        _,
        value,
        threshold,
        metric,
    ), expected_value in foo_pr_expected_answers.items():
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
