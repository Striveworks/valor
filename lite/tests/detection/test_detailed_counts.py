import numpy as np
from valor_lite.detection import (
    DataLoader,
    Detection,
    Evaluator,
    compute_detailed_pr_curve,
)


def test_detailed_pr_curve_no_data():
    evaluator = Evaluator()
    curves = evaluator.compute_detailed_pr_curve()
    assert isinstance(curves, list)
    assert len(curves) == 0


def test_compute_detailed_pr_curve():
    sorted_pairs = np.array(
        [
            # dt,  gt,  pd,  iou,  gl,  pl, score,
            [0.0, 0.0, 1.0, 0.98, 0.0, 0.0, 0.95],
            [1.0, 1.0, 2.0, 0.55, 1.0, 0.0, 0.95],
            [2.0, -1.0, 3.0, 0.67, -1.0, 0.0, 0.65],
            [3.0, 4.0, 4.0, 1.0, 0.0, 0.0, 0.1],
            [4.0, 5.0, -1.0, 0.5, 0.0, -1.0, -1.0],
        ]
    )
    label_counts = np.array([[3, 4], [1, 0]])
    iou_thresholds = np.array([0.5])
    score_thresholds = np.array([score / 100.0 for score in range(1, 101)])

    results = compute_detailed_pr_curve(
        data=sorted_pairs,
        label_counts=label_counts,
        iou_thresholds=iou_thresholds,
        score_thresholds=score_thresholds,
        n_samples=0,
    )

    assert len(results) == 1
    assert results.shape == (1, 100, 2, 5)  # iou, score, label, metrics

    """
    @ iou=0.5, score<0.1
    2x tp
    1x fp misclassification
    1x fp hallucination
    0x fn misclassification
    1x fn missing prediction
    """
    assert np.isclose(results[0, :10, 0, :], np.array([2, 1, 1, 0, 1])).all()

    """
    @ iou=0.5, score=0.5
    1x tp
    1x fp misclassification
    1x fp hallucination
    1x fn misclassification
    1x fn missing prediction
    """
    assert np.isclose(results[0, 10:95, 0, :], np.array([1, 1, 1, 1, 1])).all()

    """
    @ iou=0.5, score>=0.95
    0x tp
    0x fp misclassification
    2x fp hallucination
    2x fn misclassification
    1x fn missing prediction
    """
    assert np.isclose(results[0, 95:, 0, :], np.array([0, 0, 2, 2, 1])).all()

    # compute with examples

    """

    output

    label_idx
    tp
    ... examples
    fp_misclassification
    ... examples
    fp_hallucination
    ... examples
    fn_misclassification
    ... examples
    fn_missing_prediction
    ... examples
    """

    n_samples = 2

    results = compute_detailed_pr_curve(
        data=sorted_pairs,
        label_counts=label_counts,
        iou_thresholds=iou_thresholds,
        score_thresholds=score_thresholds,
        n_samples=n_samples,
    )

    assert len(results) == 1
    assert results.shape == (1, 100, 2, 15)  # iou, score, label, metrics

    tp_idx = 0
    fp_misclf_idx = tp_idx + n_samples + 1
    fp_halluc_idx = fp_misclf_idx + n_samples + 1
    fn_misclf_idx = fp_halluc_idx + n_samples + 1
    fn_misprd_idx = fn_misclf_idx + n_samples + 1

    metric_indices = np.zeros((15,), dtype=bool)
    for index in [
        tp_idx,
        fp_misclf_idx,
        fp_halluc_idx,
        fn_misclf_idx,
        fn_misprd_idx,
    ]:
        metric_indices[index] = True

    """
    @ iou=0.5, score<0.1
    2x tp
    1x fp misclassification
    1x fp hallucination
    0x fn misclassification
    1x fn missing prediction
    """
    assert np.isclose(
        results[0, :10, 0, metric_indices],
        np.array([2, 1, 1, 0, 1])[:, np.newaxis],
    ).all()  # metrics
    assert np.isclose(
        results[0, :10, 0, tp_idx + 1 : fp_misclf_idx], np.array([0.0, 3.0])
    ).all()  # tp
    assert np.isclose(
        results[0, :10, 0, fp_misclf_idx + 1 : fp_halluc_idx],
        np.array([1.0, -1.0]),
    ).all()  # fp misclf
    assert np.isclose(
        results[0, :10, 0, fp_halluc_idx + 1 : fn_misclf_idx],
        np.array([2.0, -1.0]),
    ).all()  # fp halluc
    assert np.isclose(
        results[0, :10, 0, fn_misclf_idx + 1 : fn_misprd_idx],
        np.array([-1.0, -1.0]),
    ).all()  # fn misclf
    assert np.isclose(
        results[0, :10, 0, fn_misprd_idx + 1 :], np.array([4.0, -1.0])
    ).all()  # fn misprd

    """
    @ iou=0.5, score=0.5
    1x tp
    1x fp misclassification
    1x fp hallucination
    1x fn misclassification
    1x fn missing prediction
    """
    assert np.isclose(
        results[0, 10:95, 0, metric_indices],
        np.array([1, 1, 1, 1, 1])[:, np.newaxis],
    ).all()
    assert np.isclose(
        results[0, 10:95, 0, tp_idx + 1 : fp_misclf_idx], np.array([0.0, -1.0])
    ).all()  # tp
    assert np.isclose(
        results[0, 10:95, 0, fp_misclf_idx + 1 : fp_halluc_idx],
        np.array([1.0, -1.0]),
    ).all()  # fp misclf
    assert np.isclose(
        results[0, 10:95, 0, fp_halluc_idx + 1 : fn_misclf_idx],
        np.array([2.0, -1.0]),
    ).all()  # fp halluc
    assert np.isclose(
        results[0, 10:95, 0, fn_misclf_idx + 1 : fn_misprd_idx],
        np.array([3.0, -1.0]),
    ).all()  # fn misclf
    assert np.isclose(
        results[0, 10:95, 0, fn_misprd_idx + 1 :], np.array([4.0, -1.0])
    ).all()  # fn misprd

    """
    @ iou=0.5, score>=0.95
    0x tp
    0x fp misclassification
    2x fp hallucination
    2x fn misclassification
    1x fn missing prediction
    """
    assert np.isclose(
        results[0, 95:, 0, metric_indices],
        np.array([0, 0, 2, 2, 1])[:, np.newaxis],
    ).all()
    assert np.isclose(
        results[0, 95:, 0, tp_idx + 1 : fp_misclf_idx], np.array([-1.0, -1.0])
    ).all()  # tp
    assert np.isclose(
        results[0, 95:, 0, fp_misclf_idx + 1 : fp_halluc_idx],
        np.array([-1.0, -1.0]),
    ).all()  # fp misclf
    assert np.isclose(
        results[0, 95:, 0, fp_halluc_idx + 1 : fn_misclf_idx],
        np.array([1.0, 2.0]),
    ).all()  # fp halluc
    assert np.isclose(
        results[0, 95:, 0, fn_misclf_idx + 1 : fn_misprd_idx],
        np.array([0.0, 3.0]),
    ).all()  # fn misclf
    assert np.isclose(
        results[0, 95:, 0, fn_misprd_idx + 1 :], np.array([4.0, -1.0])
    ).all()  # fn misprd


def test_detailed_pr_curve_using_torch_metrics_example(
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

    metrics = evaluator.compute_detailed_pr_curve(
        iou_thresholds=[0.5, 0.75],
        score_thresholds=[0.25, 0.75],
        n_samples=1,
    )

    # test DetailedPrecisionRecallCurve
    actual_metrics = [m.to_dict() for m in metrics]
    expected_metrics = [
        {
            "value": [
                {
                    "score": 0.25,
                    "tp": 1.0,
                    "fp_misclassification": 0.0,
                    "fp_hallucination": 0.0,
                    "fn_misclassification": 1.0,
                    "fn_missing_prediction": 0.0,
                    "tp_examples": ["2"],
                    "fp_misclassification_examples": [],
                    "fp_hallucination_examples": [],
                    "fn_misclassification_examples": ["0"],
                    "fn_missing_prediction_examples": [],
                },
                {
                    "score": 0.75,
                    "tp": 0.0,
                    "fp_misclassification": 0.0,
                    "fp_hallucination": 6.0,
                    "fn_misclassification": 2.0,
                    "fn_missing_prediction": 4.0,
                    "tp_examples": [],
                    "fp_misclassification_examples": [],
                    "fp_hallucination_examples": ["2"],
                    "fn_misclassification_examples": ["0"],
                    "fn_missing_prediction_examples": ["2"],
                },
            ],
            "iou": 0.5,
            "label": {"key": "class", "value": "4"},
            "type": "DetailedPrecisionRecallCurve",
        },
        {
            "value": [
                {
                    "score": 0.25,
                    "tp": 1.0,
                    "fp_misclassification": 0.0,
                    "fp_hallucination": 0.0,
                    "fn_misclassification": 1.0,
                    "fn_missing_prediction": 0.0,
                    "tp_examples": ["2"],
                    "fp_misclassification_examples": [],
                    "fp_hallucination_examples": [],
                    "fn_misclassification_examples": ["0"],
                    "fn_missing_prediction_examples": [],
                },
                {
                    "score": 0.75,
                    "tp": 0.0,
                    "fp_misclassification": 0.0,
                    "fp_hallucination": 6.0,
                    "fn_misclassification": 2.0,
                    "fn_missing_prediction": 4.0,
                    "tp_examples": [],
                    "fp_misclassification_examples": [],
                    "fp_hallucination_examples": ["2"],
                    "fn_misclassification_examples": ["0"],
                    "fn_missing_prediction_examples": ["2"],
                },
            ],
            "iou": 0.75,
            "label": {"key": "class", "value": "4"},
            "type": "DetailedPrecisionRecallCurve",
        },
        {
            "value": [
                {
                    "score": 0.25,
                    "tp": 1.0,
                    "fp_misclassification": 0.0,
                    "fp_hallucination": 1.0,
                    "fn_misclassification": 0.0,
                    "fn_missing_prediction": 1.0,
                    "tp_examples": ["1"],
                    "fp_misclassification_examples": [],
                    "fp_hallucination_examples": ["1"],
                    "fn_misclassification_examples": [],
                    "fn_missing_prediction_examples": ["1"],
                },
                {
                    "score": 0.75,
                    "tp": 0.0,
                    "fp_misclassification": 0.0,
                    "fp_hallucination": 1.0,
                    "fn_misclassification": 1.0,
                    "fn_missing_prediction": 3.0,
                    "tp_examples": [],
                    "fp_misclassification_examples": [],
                    "fp_hallucination_examples": ["1"],
                    "fn_misclassification_examples": ["1"],
                    "fn_missing_prediction_examples": ["1"],
                },
            ],
            "iou": 0.5,
            "label": {"key": "class", "value": "2"},
            "type": "DetailedPrecisionRecallCurve",
        },
        {
            "value": [
                {
                    "score": 0.25,
                    "tp": 1.0,
                    "fp_misclassification": 0.0,
                    "fp_hallucination": 1.0,
                    "fn_misclassification": 0.0,
                    "fn_missing_prediction": 1.0,
                    "tp_examples": ["1"],
                    "fp_misclassification_examples": [],
                    "fp_hallucination_examples": ["1"],
                    "fn_misclassification_examples": [],
                    "fn_missing_prediction_examples": ["1"],
                },
                {
                    "score": 0.75,
                    "tp": 0.0,
                    "fp_misclassification": 0.0,
                    "fp_hallucination": 1.0,
                    "fn_misclassification": 1.0,
                    "fn_missing_prediction": 3.0,
                    "tp_examples": [],
                    "fp_misclassification_examples": [],
                    "fp_hallucination_examples": ["1"],
                    "fn_misclassification_examples": ["1"],
                    "fn_missing_prediction_examples": ["1"],
                },
            ],
            "iou": 0.75,
            "label": {"key": "class", "value": "2"},
            "type": "DetailedPrecisionRecallCurve",
        },
    ]
    for m in actual_metrics:
        assert m in expected_metrics
    for m in expected_metrics:
        assert m in actual_metrics


# @pytest.fixture
# def test_detailed_precision_recall_curve(
#     evaluate_detection_detailed_pr_curve_groundtruths: list,
#     evaluate_detection_detailed_pr_curve_predictions: list,
#     detailed_precision_recall_curve_outputs: tuple,
# ):

#     expected_outputs, _ = detailed_precision_recall_curve_outputs

#     Dataloader = Dataloader(
#         metrics_to_return=[enums.MetricType.DetailedDetailedPrecisionRecallCurve],
#     )

#     Dataloader.add_data(
#         groundtruths=evaluate_detection_detailed_pr_curve_groundtruths,
#         predictions=evaluate_detection_detailed_pr_curve_predictions,
#     )

#     # check that ious have been precomputed
#     assert "iou_" in Dataloader.joint_df.columns
#     assert all(
#         [
#             col not in ["raster", "bounding_box"]
#             for col in Dataloader.joint_df.columns
#         ]
#     )

#     eval_job = Dataloader.evaluate()
#     for key, expected_value in expected_outputs.items():
#         result = eval_job.metrics[0]["value"]
#         for k in key:
#             result = result[k]
#         assert result == expected_value

#     # repeat tests using a lower IOU threshold
#     Dataloader = Dataloader(
#         metrics_to_return=[enums.MetricType.DetailedDetailedPrecisionRecallCurve],
#         pr_curve_iou_threshold=0.45,
#     )

#     Dataloader.add_data(
#         groundtruths=evaluate_detection_detailed_pr_curve_groundtruths,
#         predictions=evaluate_detection_detailed_pr_curve_predictions,
#     )

#     eval_job_low_iou_threshold = Dataloader.evaluate()

#     for key, expected_value in expected_outputs.items():
#         result = eval_job_low_iou_threshold.metrics[0]["value"]
#         for k in key:
#             result = result[k]
#         assert result == expected_value


# def test_evaluate_detection_model_with_no_predictions(
#     evaluate_detection_groundtruths: list,
#     evaluate_detection_model_with_no_predictions_output: list,
# ):
#     """
#     Test detection evaluations when the model outputs nothing.

#     gt_dets1
#         datum 1
#             - Label (k1, v1) with Annotation area = 1500
#             - Label (k2, v2) with Annotation area = 57,510
#         datum2
#             - Label (k1, v1) with Annotation area = 1100
#     """
#     predictions = []
#     for gt in evaluate_detection_groundtruths:
#         predictions.append(
#             schemas.Prediction(
#                 datum=gt.datum,
#                 annotations=[],
#             )
#         )

#     Dataloader = Dataloader()

#     # can't pass empty lists, but can pass predictions without annotations
#     with pytest.raises(ValueError) as e:
#         Dataloader.add_data(
#             groundtruths=evaluate_detection_groundtruths,
#             predictions=[],
#         )
#     assert (
#         "it's neither a dataframe nor a list of Valor Prediction objects"
#         in str(e)
#     )

#     Dataloader.add_data(
#         groundtruths=evaluate_detection_groundtruths,
#         predictions=predictions,
#     )

#     # check that ious have been precomputed
#     assert "iou_" in Dataloader.joint_df.columns
#     assert all(
#         [
#             col not in ["raster", "bounding_box"]
#             for col in Dataloader.joint_df.columns
#         ]
#     )

#     eval_job = Dataloader.evaluate()

#     computed_metrics = eval_job.metrics

#     assert all([metric["value"] == 0 for metric in computed_metrics])

#     for m in evaluate_detection_model_with_no_predictions_output:
#         assert m in computed_metrics

#     for m in computed_metrics:
#         assert m in evaluate_detection_model_with_no_predictions_output


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

#     Dataloader = Dataloader()
#     Dataloader.add_data(
#         groundtruths=evaluate_detection_functional_test_groundtruths,
#         predictions=evaluate_detection_functional_test_predictions,
#     )
#     Dataloader.finalize()

#     ap_metrics = translate_ap_metrics(
#         Dataloader.compute_ap(
#             iou_thresholds=[0.5, 0.75]
#         )
#     )

#     pr_curves = translate_pr_curves(
#         Dataloader.compute_pr_curve(
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
#             enums.MetricType.DetailedPrecisionRecallCurve,
#             enums.MetricType.DetailedDetailedPrecisionRecallCurve,
#         ],
#         pr_curve_iou_threshold=0.5,
#         pr_curve_max_examples=1,
#     )


#     metrics = [
#         m
#         for m in eval_job.metrics
#         if m["type"]
#         not in ["DetailedPrecisionRecallCurve", "DetailedDetailedPrecisionRecallCurve"]
#     ]

#     # round all metrics to the third decimal place
#     for i, m in enumerate(metrics):
#         metrics[i]["value"] = round(m["value"], 3)

#     pr_metrics = [
#         m for m in eval_job.metrics if m["type"] == "DetailedPrecisionRecallCurve"
#     ]
#     detailed_pr_metrics = [
#         m
#         for m in eval_job.metrics
#         if m["type"] == "DetailedDetailedPrecisionRecallCurve"
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
#     Dataloader = Dataloader(
#         metrics_to_return=[
#             enums.MetricType.DetailedPrecisionRecallCurve,
#             enums.MetricType.DetailedDetailedPrecisionRecallCurve,
#         ],
#         pr_curve_iou_threshold=0.9,
#         pr_curve_max_examples=1,
#     )

#     Dataloader.add_data(
#         groundtruths=evaluate_detection_functional_test_groundtruths,
#         predictions=evaluate_detection_functional_test_predictions,
#     )

#     # check that ious have been precomputed
#     assert "iou_" in Dataloader.joint_df.columns
#     assert all(
#         [
#             col not in ["raster", "bounding_box"]
#             for col in Dataloader.joint_df.columns
#         ]
#     )

#     eval_job_higher_threshold = Dataloader.evaluate()

#     pr_metrics = [
#         m
#         for m in eval_job_higher_threshold.metrics
#         if m["type"] == "DetailedPrecisionRecallCurve"
#     ]
#     detailed_pr_metrics = [
#         m
#         for m in eval_job_higher_threshold.metrics
#         if m["type"] == "DetailedDetailedPrecisionRecallCurve"
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
#     Dataloader = Dataloader(
#         metrics_to_return=[
#             enums.MetricType.DetailedPrecisionRecallCurve,
#             enums.MetricType.DetailedDetailedPrecisionRecallCurve,
#         ],
#         pr_curve_iou_threshold=0.9,
#         pr_curve_max_examples=3,
#     )

#     Dataloader.add_data(
#         groundtruths=evaluate_detection_functional_test_groundtruths,
#         predictions=evaluate_detection_functional_test_predictions,
#     )

#     # check that ious have been precomputed
#     assert "iou_" in Dataloader.joint_df.columns
#     assert all(
#         [
#             col not in ["raster", "bounding_box"]
#             for col in Dataloader.joint_df.columns
#         ]
#     )

#     eval_job_higher_threshold = Dataloader.evaluate()

#     pr_metrics = [
#         m
#         for m in eval_job_higher_threshold.metrics
#         if m["type"] == "DetailedPrecisionRecallCurve"
#     ]
#     detailed_pr_metrics = [
#         m
#         for m in eval_job_higher_threshold.metrics
#         if m["type"] == "DetailedDetailedPrecisionRecallCurve"
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
#     Dataloader = Dataloader(
#         metrics_to_return=[
#             enums.MetricType.DetailedPrecisionRecallCurve,
#             enums.MetricType.DetailedDetailedPrecisionRecallCurve,
#         ],
#         pr_curve_iou_threshold=0.9,
#         pr_curve_max_examples=0,
#     )

#     Dataloader.add_data(
#         groundtruths=evaluate_detection_functional_test_groundtruths,
#         predictions=evaluate_detection_functional_test_predictions,
#     )

#     # check that ious have been precomputed
#     assert "iou_" in Dataloader.joint_df.columns
#     assert all(
#         [
#             col not in ["raster", "bounding_box"]
#             for col in Dataloader.joint_df.columns
#         ]
#     )

#     eval_job_higher_threshold = Dataloader.evaluate()

#     pr_metrics = [
#         m
#         for m in eval_job_higher_threshold.metrics
#         if m["type"] == "DetailedPrecisionRecallCurve"
#     ]
#     detailed_pr_metrics = [
#         m
#         for m in eval_job_higher_threshold.metrics
#         if m["type"] == "DetailedDetailedPrecisionRecallCurve"
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
