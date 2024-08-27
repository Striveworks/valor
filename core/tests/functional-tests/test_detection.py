import random

import pandas as pd
import pytest
from valor_core import enums, schemas
from valor_core.detection import _calculate_101_pt_interp, evaluate_detection


def test__calculate_101_pt_interp():
    # make sure we get back 0 if we don't pass any precisions
    assert _calculate_101_pt_interp([], []) == 0

    # get back -1 if all recalls and precisions are -1
    assert _calculate_101_pt_interp([-1, -1], [-1, -1]) == -1


def test_evaluate_detection(
    evaluate_detection_groundtruths: list,
    evaluate_detection_predictions: list,
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

    # check that metrics arg works correctly
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


def test_evaluate_detection_via_pandas_df(
    evaluate_detection_groundtruths_df: pd.DataFrame,
    evaluate_detection_predictions_df: pd.DataFrame,
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

    eval_job = evaluate_detection(
        groundtruths=evaluate_detection_groundtruths_df,
        predictions=evaluate_detection_predictions_df,
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
    # check that metrics arg works correctly
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
        groundtruths=evaluate_detection_groundtruths_df,
        predictions=evaluate_detection_predictions_df,
        iou_thresholds_to_compute=[0.1, 0.6],
        iou_thresholds_to_return=[0.1, 0.6],
        metrics_to_return=selected_metrics,
    )

    metrics = eval_job.metrics
    assert set([metric["type"] for metric in eval_job.metrics]) == set(
        selected_metrics
    )


def test_evaluate_detection_with_label_maps(
    evaluate_detection_groundtruths_with_label_maps: list,
    evaluate_detection_predictions_with_label_maps: list,
    evaluate_detection_with_label_maps_expected: tuple,
):

    (
        baseline_expected_metrics,
        baseline_pr_expected_answers,
        baseline_detailed_pr_expected_answers,
        cat_expected_metrics,
        foo_expected_metrics,
        foo_pr_expected_answers,
        foo_expected_metrics_with_higher_score_threshold,
    ) = evaluate_detection_with_label_maps_expected

    # for the first evaluation, don't do anything about the mismatched labels
    # we expect the evaluation to return the same expected metrics as for our standard detection tests

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


def test_evaluate_detection_false_negatives_single_image_baseline(
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

    eval_job = evaluate_detection(
        groundtruths=groundtruths,
        predictions=predictions,
        iou_thresholds_to_compute=[0.5],
        iou_thresholds_to_return=[0.5],
    )

    ap_metric = [m for m in eval_job.metrics if m["type"] == "AP"][0]
    assert ap_metric == evaluate_detection_false_negatives_AP_of_1


def test_evaluate_detection_false_negatives_single_image(
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

    eval_job = evaluate_detection(
        groundtruths=groundtruths,
        predictions=predictions,
        iou_thresholds_to_compute=[0.5],
        iou_thresholds_to_return=[0.5],
    )

    ap_metric = [m for m in eval_job.metrics if m["type"] == "AP"][0]
    assert ap_metric == evaluate_detection_false_negatives_AP_of_point_5


def test_evaluate_detection_false_negatives_two_images_one_empty_low_confidence_of_fp(
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

    eval_job = evaluate_detection(
        groundtruths=groundtruths,
        predictions=predictions,
        iou_thresholds_to_compute=[0.5],
        iou_thresholds_to_return=[0.5],
    )

    ap_metric = [m for m in eval_job.metrics if m["type"] == "AP"][0]
    assert ap_metric == evaluate_detection_false_negatives_AP_of_1


def test_evaluate_detection_false_negatives_two_images_one_empty_high_confidence_of_fp(
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

    eval_job = evaluate_detection(
        groundtruths=groundtruths,
        predictions=predictions,
        iou_thresholds_to_compute=[0.5],
        iou_thresholds_to_return=[0.5],
    )

    ap_metric = [m for m in eval_job.metrics if m["type"] == "AP"][0]
    assert ap_metric == evaluate_detection_false_negatives_AP_of_point_5


def test_evaluate_detection_false_negatives_two_images_one_only_with_different_class_low_confidence_of_fp(
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
    assert ap_metric1 == evaluate_detection_false_negatives_AP_of_1

    # label `"other value"` is not in the predictions so we should get an AP of 0
    ap_metric2 = [
        m
        for m in eval_job.metrics
        if m["type"] == "AP"
        and m["label"] == {"key": "key", "value": "other value"}
    ][0]
    assert ap_metric2 == evaluate_detection_false_negatives_AP_of_0


def test_evaluate_detection_false_negatives_two_images_one_only_with_different_class_high_confidence_of_fp(
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
def test_detailed_precision_recall_curve(
    evaluate_detection_detailed_pr_curve_groundtruths: list,
    evaluate_detection_detailed_pr_curve_predictions: list,
    detailed_precision_recall_curve_outputs: tuple,
):

    expected_outputs, _ = detailed_precision_recall_curve_outputs

    eval_job = evaluate_detection(
        groundtruths=evaluate_detection_detailed_pr_curve_groundtruths,
        predictions=evaluate_detection_detailed_pr_curve_predictions,
        metrics_to_return=[enums.MetricType.DetailedPrecisionRecallCurve],
    )

    for key, expected_value in expected_outputs.items():
        result = eval_job.metrics[0]["value"]
        for k in key:
            result = result[k]
        assert result == expected_value

    # repeat tests using a lower IOU threshold
    eval_job_low_iou_threshold = evaluate_detection(
        groundtruths=evaluate_detection_detailed_pr_curve_groundtruths,
        predictions=evaluate_detection_detailed_pr_curve_predictions,
        metrics_to_return=[enums.MetricType.DetailedPrecisionRecallCurve],
        pr_curve_iou_threshold=0.45,
    )

    for key, expected_value in expected_outputs.items():
        result = eval_job_low_iou_threshold.metrics[0]["value"]
        for k in key:
            result = result[k]
        assert result == expected_value


def test_evaluate_detection_model_with_no_predictions(
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

    # can't pass empty lists, but can pass predictions without annotations
    with pytest.raises(ValueError) as e:
        eval_job = evaluate_detection(
            groundtruths=evaluate_detection_groundtruths,
            predictions=[],
        )
    assert (
        "it's neither a dataframe nor a list of Valor Prediction objects"
        in str(e)
    )

    eval_job = evaluate_detection(
        groundtruths=evaluate_detection_groundtruths,
        predictions=predictions,
    )

    computed_metrics = eval_job.metrics

    assert all([metric["value"] == 0 for metric in computed_metrics])

    for m in evaluate_detection_model_with_no_predictions_output:
        assert m in computed_metrics

    for m in computed_metrics:
        assert m in evaluate_detection_model_with_no_predictions_output


def test_evaluate_detection_functional_test(
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


def test_evaluate_detection_functional_test_with_rasters(
    evaluate_detection_functional_test_groundtruths_with_rasters: list,
    evaluate_detection_functional_test_predictions_with_rasters: list,
    evaluate_detection_functional_test_with_rasters_outputs: tuple,
):
    (
        expected_metrics,
        pr_expected_answers,
    ) = evaluate_detection_functional_test_with_rasters_outputs
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


def test_evaluate_mixed_annotations(
    evaluate_mixed_annotations_inputs: tuple,
    evaluate_mixed_annotations_output: list,
):
    """Test the automatic conversion to rasters."""
    gts, pds = evaluate_mixed_annotations_inputs

    # by default, valor_core should throw an error if given mixed AnnotationTypes without being explicitely told to convert to a certain type
    with pytest.raises(ValueError):
        _ = evaluate_detection(
            groundtruths=gts,
            predictions=pds,
            iou_thresholds_to_compute=[0.1, 0.6],
            iou_thresholds_to_return=[0.1, 0.6],
            metrics_to_return=[
                enums.MetricType.AP,
            ],
        )

    # test conversion to raster. this should throw an error since the user is trying to convert a Box annotation to a polygon.
    with pytest.raises(ValueError):
        evaluate_detection(
            groundtruths=gts,
            predictions=pds,
            iou_thresholds_to_compute=[0.1, 0.6],
            iou_thresholds_to_return=[0.1, 0.6],
            metrics_to_return=[
                enums.MetricType.AP,
            ],
            convert_annotations_to_type=enums.AnnotationType.RASTER,
        )

    # test conversion to polygon. this should throw an error since the user is trying to convert a Box annotation to a polygon.
    with pytest.raises(ValueError):
        evaluate_detection(
            groundtruths=gts,
            predictions=pds,
            iou_thresholds_to_compute=[0.1, 0.6],
            iou_thresholds_to_return=[0.1, 0.6],
            metrics_to_return=[
                enums.MetricType.AP,
            ],
            convert_annotations_to_type=enums.AnnotationType.POLYGON,
        )

    # test conversion to box
    eval_job_box = evaluate_detection(
        groundtruths=gts,
        predictions=pds,
        iou_thresholds_to_compute=[0.1, 0.6],
        iou_thresholds_to_return=[0.1, 0.6],
        metrics_to_return=[
            enums.MetricType.AP,
        ],
        convert_annotations_to_type=enums.AnnotationType.BOX,
    )

    for m in eval_job_box.metrics:
        assert m in evaluate_mixed_annotations_output
    for m in evaluate_mixed_annotations_output:
        assert m in eval_job_box.metrics


def test_evaluate_detection_rotated_bboxes_with_shapely(
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
    expected_metrics, _ = evaluate_detection_expected

    eval_job = evaluate_detection(
        groundtruths=groundtruths,
        predictions=predictions,
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
            "convert_annotations_to_type": None,
        },
        "confusion_matrices": [],
        "ignored_pred_labels": [],
        "missing_pred_labels": [],
    }

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
    eval_job = evaluate_detection(
        groundtruths=groundtruths,
        predictions=predictions,
        iou_thresholds_to_compute=[0.1, 0.6],
        iou_thresholds_to_return=[0.1, 0.6],
        metrics_to_return=selected_metrics,
    )

    metrics = eval_job.metrics
    assert set([metric["type"] for metric in eval_job.metrics]) == set(
        selected_metrics
    )


def test_evaluate_detection_rotated_bboxes(
    evaluate_detection_rotated_bboxes_inputs: tuple,
    evaluate_detection_expected: tuple,
):
    """
    Run the same test as test_evaluate_detection, but rotate all of the bounding boxes by 5 degrees around the origin to confirm we get the same outputs.
    """

    groundtruths, predictions = evaluate_detection_rotated_bboxes_inputs
    expected_metrics, expected_metadata = evaluate_detection_expected

    eval_job = evaluate_detection(
        groundtruths=groundtruths,
        predictions=predictions,
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
    eval_job = evaluate_detection(
        groundtruths=groundtruths,
        predictions=predictions,
        iou_thresholds_to_compute=[0.1, 0.6],
        iou_thresholds_to_return=[0.1, 0.6],
        metrics_to_return=selected_metrics,
    )

    metrics = eval_job.metrics
    assert set([metric["type"] for metric in eval_job.metrics]) == set(
        selected_metrics
    )


def test_two_groundtruths_one_datum(
    evaluate_detection_predictions: list[schemas.Prediction],
    two_groundtruths_one_datum_groundtruths: list,
    evaluate_detection_expected: tuple,
):
    """Same test as test_evaluate_detection, but we show that we can handle two groundtruths for a single datum"""
    expected_metrics, _ = evaluate_detection_expected

    eval_job = evaluate_detection(
        groundtruths=two_groundtruths_one_datum_groundtruths,
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


def test_evaluate_detection_pr_fp(evaluate_detection_pr_fp_inputs):

    gts, preds = evaluate_detection_pr_fp_inputs

    eval_job = evaluate_detection(
        groundtruths=gts,
        predictions=preds,
        metrics_to_return=[
            enums.MetricType.PrecisionRecallCurve,
        ],
    )

    metrics = eval_job.metrics

    assert metrics[0]["value"]["v1"][0.5] == {
        "fn": 1,  # img2
        "fp": 1,  # img2
        "tn": None,
        "tp": 1,  # img1
        "recall": 0.5,
        "accuracy": None,
        "f1_score": 0.5,
        "precision": 0.5,
    }

    # score threshold is now higher than the scores, so we should the predictions drop out such that we're only left with 2 fns (one for each image)
    assert metrics[0]["value"]["v1"][0.85] == {
        "tp": 0,
        "fp": 0,
        "fn": 2,
        "tn": None,
        "precision": 0.0,
        "recall": 0.0,
        "accuracy": None,
        "f1_score": 0.0,
    }

    # test DetailedPRCurve version
    eval_job = evaluate_detection(
        groundtruths=gts,
        predictions=preds,
        metrics_to_return=[
            enums.MetricType.DetailedPrecisionRecallCurve,
        ],
    )

    metrics = eval_job.metrics

    score_threshold = 0.5
    assert metrics[0]["value"]["v1"][score_threshold]["tp"]["total"] == 1
    assert "tn" not in metrics[0]["value"]["v1"][score_threshold]
    assert (
        metrics[0]["value"]["v1"][score_threshold]["fp"]["observations"][
            "hallucinations"
        ]["count"]
        == 1
    )
    assert (
        metrics[0]["value"]["v1"][score_threshold]["fp"]["observations"][
            "misclassifications"
        ]["count"]
        == 0
    )
    assert (
        metrics[0]["value"]["v1"][score_threshold]["fn"]["observations"][
            "no_predictions"
        ]["count"]
        == 1
    )
    assert metrics[0]["value"]["v1"][score_threshold]["tp"]["total"] == 1
    assert (
        metrics[0]["value"]["v1"][score_threshold]["fn"]["observations"][
            "misclassifications"
        ]["count"]
        == 0
    )

    # score threshold is now higher than the scores, so we should the predictions drop out such that we're only left with 2 fns (one for each image)
    score_threshold = 0.85
    assert metrics[0]["value"]["v1"][score_threshold]["tp"]["total"] == 0
    assert "tn" not in metrics[0]["value"]["v1"][score_threshold]
    assert (
        metrics[0]["value"]["v1"][score_threshold]["fp"]["observations"][
            "hallucinations"
        ]["count"]
        == 0
    )
    assert (
        metrics[0]["value"]["v1"][score_threshold]["fp"]["observations"][
            "misclassifications"
        ]["count"]
        == 0
    )
    assert (
        metrics[0]["value"]["v1"][score_threshold]["fn"]["observations"][
            "no_predictions"
        ]["count"]
        == 2
    )
    assert (
        metrics[0]["value"]["v1"][score_threshold]["fn"]["observations"][
            "misclassifications"
        ]["count"]
        == 0
    )
