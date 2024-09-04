import pytest
from valor_lite import DetectionManager as Manager
from valor_lite import enums, schemas


@pytest.fixture
def test_detailed_precision_recall_curve(
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


# def test_evaluate_detection_functional_test(
#     evaluate_detection_functional_test_groundtruths: list,
#     evaluate_detection_functional_test_predictions: list,
#     evaluate_detection_functional_test_outputs: tuple,
# ):

#     (
#         expected_metrics,
#         pr_expected_answers,
#         detailed_pr_expected_answers,
#         higher_iou_threshold_pr_expected_answers,
#         higher_iou_threshold_detailed_pr_expected_answers,
#     ) = evaluate_detection_functional_test_outputs

#     manager = Manager()
#     manager.add_data(
#         groundtruths=evaluate_detection_functional_test_groundtruths,
#         predictions=evaluate_detection_functional_test_predictions,
#     )
#     manager.finalize()

#     ap_metrics = translate_ap_metrics(
#         manager.compute_ap(
#             iou_thresholds=[0.5, 0.75]
#         )
#     )

#     pr_curves = translate_pr_curves(
#         manager.compute_pr_curve(
#             iou_thresholds=[0.5],
#             n_samples=1,
#         )
#     )

#         metrics_to_return=[
#             enums.MetricType.AP,
#             enums.MetricType.AR,
#             enums.MetricType.mAP,
#             enums.MetricType.APAveragedOverIOUs,
#             enums.MetricType.mAR,
#             enums.MetricType.mAPAveragedOverIOUs,
#             enums.MetricType.PrecisionRecallCurve,
#             enums.MetricType.DetailedPrecisionRecallCurve,
#         ],
#         pr_curve_iou_threshold=0.5,
#         pr_curve_max_examples=1,
#     )


#     metrics = [
#         m
#         for m in eval_job.metrics
#         if m["type"]
#         not in ["PrecisionRecallCurve", "DetailedPrecisionRecallCurve"]
#     ]

#     # round all metrics to the third decimal place
#     for i, m in enumerate(metrics):
#         metrics[i]["value"] = round(m["value"], 3)

#     pr_metrics = [
#         m for m in eval_job.metrics if m["type"] == "PrecisionRecallCurve"
#     ]
#     detailed_pr_metrics = [
#         m
#         for m in eval_job.metrics
#         if m["type"] == "DetailedPrecisionRecallCurve"
#     ]

#     for m in metrics:
#         assert m in expected_metrics
#     for m in metrics:
#         assert m in eval_job.metrics

#     for (
#         _,
#         value,
#         threshold,
#         metric,
#     ), expected_value in pr_expected_answers.items():
#         assert (
#             pr_metrics[0]["value"][value][threshold][metric] == expected_value
#         )

#     for (
#         value,
#         threshold,
#         metric,
#     ), expected_output in detailed_pr_expected_answers.items():
#         model_output = detailed_pr_metrics[0]["value"][value][threshold][
#             metric
#         ]
#         assert isinstance(model_output, dict)
#         assert model_output["total"] == expected_output["total"]
#         assert all(
#             [
#                 model_output["observations"][key]["count"]  # type: ignore - we know this element is a dict
#                 == expected_output[key]
#                 for key in [
#                     key
#                     for key in expected_output.keys()
#                     if key not in ["total"]
#                 ]
#             ]
#         )

#     # spot check number of examples
#     assert (
#         len(
#             detailed_pr_metrics[0]["value"]["0"][0.95]["fn"]["observations"]["no_predictions"][  # type: ignore - we know this element is a dict
#                 "examples"
#             ]
#         )
#         == 1
#     )
#     assert (
#         len(
#             detailed_pr_metrics[0]["value"]["49"][0.05]["tp"]["observations"]["all"][  # type: ignore - we know this element is a dict
#                 "examples"
#             ]
#         )
#         == 1
#     )

#     # raise the iou threshold
#     manager = Manager(
#         metrics_to_return=[
#             enums.MetricType.PrecisionRecallCurve,
#             enums.MetricType.DetailedPrecisionRecallCurve,
#         ],
#         pr_curve_iou_threshold=0.9,
#         pr_curve_max_examples=1,
#     )

#     manager.add_data(
#         groundtruths=evaluate_detection_functional_test_groundtruths,
#         predictions=evaluate_detection_functional_test_predictions,
#     )

#     # check that ious have been precomputed
#     assert "iou_" in manager.joint_df.columns
#     assert all(
#         [
#             col not in ["raster", "bounding_box"]
#             for col in manager.joint_df.columns
#         ]
#     )

#     eval_job_higher_threshold = manager.evaluate()

#     pr_metrics = [
#         m
#         for m in eval_job_higher_threshold.metrics
#         if m["type"] == "PrecisionRecallCurve"
#     ]
#     detailed_pr_metrics = [
#         m
#         for m in eval_job_higher_threshold.metrics
#         if m["type"] == "DetailedPrecisionRecallCurve"
#     ]

#     for (
#         key,
#         value,
#         threshold,
#         metric,
#     ), expected_count in higher_iou_threshold_pr_expected_answers.items():
#         actual_count = pr_metrics[0]["value"][value][threshold][metric]
#         assert actual_count == expected_count

#     for (
#         value,
#         threshold,
#         metric,
#     ), expected_output in (
#         higher_iou_threshold_detailed_pr_expected_answers.items()
#     ):
#         model_output = detailed_pr_metrics[0]["value"][value][threshold][
#             metric
#         ]
#         assert isinstance(model_output, dict)
#         assert model_output["total"] == expected_output["total"]
#         assert all(
#             [
#                 model_output["observations"][key]["count"]  # type: ignore - we know this element is a dict
#                 == expected_output[key]
#                 for key in [
#                     key
#                     for key in expected_output.keys()
#                     if key not in ["total"]
#                 ]
#             ]
#         )

#     assert (
#         len(
#             detailed_pr_metrics[0]["value"]["0"][0.95]["fn"]["observations"]["no_predictions"][  # type: ignore - we know this element is a dict
#                 "examples"
#             ]
#         )
#         == 1
#     )
#     assert (
#         len(
#             detailed_pr_metrics[0]["value"]["49"][0.05]["tp"]["observations"]["all"][  # type: ignore - we know this element is a dict
#                 "examples"
#             ]
#         )
#         == 1
#     )

#     # repeat the above, but with a higher pr_max_curves_example
#     manager = Manager(
#         metrics_to_return=[
#             enums.MetricType.PrecisionRecallCurve,
#             enums.MetricType.DetailedPrecisionRecallCurve,
#         ],
#         pr_curve_iou_threshold=0.9,
#         pr_curve_max_examples=3,
#     )

#     manager.add_data(
#         groundtruths=evaluate_detection_functional_test_groundtruths,
#         predictions=evaluate_detection_functional_test_predictions,
#     )

#     # check that ious have been precomputed
#     assert "iou_" in manager.joint_df.columns
#     assert all(
#         [
#             col not in ["raster", "bounding_box"]
#             for col in manager.joint_df.columns
#         ]
#     )

#     eval_job_higher_threshold = manager.evaluate()

#     pr_metrics = [
#         m
#         for m in eval_job_higher_threshold.metrics
#         if m["type"] == "PrecisionRecallCurve"
#     ]
#     detailed_pr_metrics = [
#         m
#         for m in eval_job_higher_threshold.metrics
#         if m["type"] == "DetailedPrecisionRecallCurve"
#     ]

#     for (
#         key,
#         value,
#         threshold,
#         metric,
#     ), expected_count in higher_iou_threshold_pr_expected_answers.items():
#         actual_count = pr_metrics[0]["value"][value][threshold][metric]
#         assert actual_count == expected_count

#     for (
#         value,
#         threshold,
#         metric,
#     ), expected_output in (
#         higher_iou_threshold_detailed_pr_expected_answers.items()
#     ):
#         model_output = detailed_pr_metrics[0]["value"][value][threshold][
#             metric
#         ]
#         assert isinstance(model_output, dict)
#         assert model_output["total"] == expected_output["total"]
#         assert all(
#             [
#                 model_output["observations"][key]["count"]  # type: ignore - we know this element is a dict
#                 == expected_output[key]
#                 for key in [
#                     key
#                     for key in expected_output.keys()
#                     if key not in ["total"]
#                 ]
#             ]
#         )

#     assert (
#         len(
#             detailed_pr_metrics[0]["value"]["0"][0.95]["fn"]["observations"]["no_predictions"][  # type: ignore - we know this element is a dict
#                 "examples"
#             ]
#         )
#         == 3
#     )
#     assert (
#         len(
#             detailed_pr_metrics[0]["value"]["49"][0.05]["tp"]["observations"]["all"][  # type: ignore - we know this element is a dict
#                 "examples"
#             ]
#         )
#         == 2
#     )

#     # test behavior if pr_curve_max_examples == 0
#     manager = Manager(
#         metrics_to_return=[
#             enums.MetricType.PrecisionRecallCurve,
#             enums.MetricType.DetailedPrecisionRecallCurve,
#         ],
#         pr_curve_iou_threshold=0.9,
#         pr_curve_max_examples=0,
#     )

#     manager.add_data(
#         groundtruths=evaluate_detection_functional_test_groundtruths,
#         predictions=evaluate_detection_functional_test_predictions,
#     )

#     # check that ious have been precomputed
#     assert "iou_" in manager.joint_df.columns
#     assert all(
#         [
#             col not in ["raster", "bounding_box"]
#             for col in manager.joint_df.columns
#         ]
#     )

#     eval_job_higher_threshold = manager.evaluate()

#     pr_metrics = [
#         m
#         for m in eval_job_higher_threshold.metrics
#         if m["type"] == "PrecisionRecallCurve"
#     ]
#     detailed_pr_metrics = [
#         m
#         for m in eval_job_higher_threshold.metrics
#         if m["type"] == "DetailedPrecisionRecallCurve"
#     ]

#     for (
#         key,
#         value,
#         threshold,
#         metric,
#     ), expected_count in higher_iou_threshold_pr_expected_answers.items():
#         actual_count = pr_metrics[0]["value"][value][threshold][metric]
#         assert actual_count == expected_count

#     for (
#         value,
#         threshold,
#         metric,
#     ), expected_output in (
#         higher_iou_threshold_detailed_pr_expected_answers.items()
#     ):
#         model_output = detailed_pr_metrics[0]["value"][value][threshold][
#             metric
#         ]
#         assert isinstance(model_output, dict)
#         assert model_output["total"] == expected_output["total"]
#         assert all(
#             [
#                 model_output["observations"][key]["count"]  # type: ignore - we know this element is a dict
#                 == expected_output[key]
#                 for key in [
#                     key
#                     for key in expected_output.keys()
#                     if key not in ["total"]
#                 ]
#             ]
#         )

#     # spot check number of examples
#     assert (
#         len(
#             detailed_pr_metrics[0]["value"]["0"][0.95]["fn"]["observations"]["no_predictions"][  # type: ignore - we know this element is a dict
#                 "examples"
#             ]
#         )
#         == 0
#     )
#     assert (
#         len(
#             detailed_pr_metrics[0]["value"]["49"][0.05]["tp"]["observations"]["all"][  # type: ignore - we know this element is a dict
#                 "examples"
#             ]
#         )
#         == 0
#     )
