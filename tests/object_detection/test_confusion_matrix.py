from valor_lite.object_detection import DataLoader, Detection

# def test_compute_pair_classifications():

#     sorted_pairs = np.array(
#         [
#             # dt, gt, pd, gl, pl
#             [0.0, 0.0, 1.0, 0.0, 0.0, 0.98, 0.9],
#             [1.0, 1.0, 2.0, 1.0, 0.0, 0.55, 0.9],
#             [2.0, -1.0, 4.0, -1.0, 0, 0.0, 0.65],
#             [3.0, 4, 5.0, 0.0, 0.0, 1.0, 0.1],
#             [1.0, 2.0, 3.0, 0.0, 0.0, 0.55, 0.1],
#             [4.0, 5.0, -1.0, 0.0, -1.0, 0.0, -1.0],
#         ]
#     )

#     iou_thresholds = np.array([0.5])
#     score_thresholds = np.array([score / 100.0 for score in range(1, 101)])

#     results = compute_pair_classifications(
#         detailed_pairs=sorted_pairs,
#         iou_thresholds=iou_thresholds,
#         score_thresholds=score_thresholds,
#     )
#     assert results.shape == (1, 100, 6)

#     """
#     @ iou=0.5, score < 0.1
#     3x tp
#     1x fp misclassification
#     1x fp unmatched prediction
#     0x fn misclassification
#     1x fn unmatched ground truth
#     """
#     indices = slice(10)
#     np.testing.assert_allclose(
#         np.unique(results[0, indices, :], axis=0),
#         np.array(
#             [
#                 [
#                     PairClassification.TP,
#                     PairClassification.FP_FN_MISCLF,
#                     PairClassification.FP_UNMATCHED,
#                     PairClassification.TP,
#                     PairClassification.TP,
#                     PairClassification.FN_UNMATCHED,
#                 ]
#             ]
#         ),
#     )

#     """
#     @ iou=0.5, 0.1 <= score < 0.65
#     1x tp
#     1x fp misclassification
#     1x fp unmatched prediction
#     1x fn misclassification
#     3x fn unmatched ground truth
#     """
#     indices = slice(10, 65)
#     np.testing.assert_allclose(
#         np.unique(results[0, indices, :], axis=0),
#         np.array(
#             [
#                 [
#                     PairClassification.TP,
#                     PairClassification.FP_FN_MISCLF,
#                     PairClassification.FP_UNMATCHED,
#                     PairClassification.FN_UNMATCHED,
#                     PairClassification.FN_UNMATCHED,
#                     PairClassification.FN_UNMATCHED,
#                 ]
#             ]
#         ),
#     )

#     """
#     @ iou=0.5, 0.65 <= score < 0.9
#     1x tp
#     1x fp misclassification
#     0x fp unmatched prediction
#     1x fn misclassification
#     3x fn unmatched ground truth
#     """
#     indices = slice(65, 90)
#     np.testing.assert_allclose(
#         np.unique(results[0, indices, :], axis=0),
#         np.array(
#             [
#                 [
#                     PairClassification.TP.value,
#                     PairClassification.FP_FN_MISCLF.value,
#                     0,
#                     PairClassification.FN_UNMATCHED.value,
#                     PairClassification.FN_UNMATCHED.value,
#                     PairClassification.FN_UNMATCHED.value,
#                 ]
#             ]
#         ),
#     )

#     """
#     @ iou=0.5, score>=0.9
#     0x tp
#     0x fp misclassification
#     0x fp unmatched prediction
#     0x fn misclassification
#     4x fn unmatched ground truth
#     """
#     indices = slice(90, None)
#     np.testing.assert_allclose(
#         np.unique(results[0, indices, :], axis=0),
#         np.array(
#             [
#                 [
#                     PairClassification.FN_UNMATCHED.value,
#                     PairClassification.FN_UNMATCHED.value,
#                     0,
#                     PairClassification.FN_UNMATCHED.value,
#                     PairClassification.FN_UNMATCHED.value,
#                     PairClassification.FN_UNMATCHED.value,
#                 ]
#             ]
#         ),
#     )


def _filter_out_zero_counts(
    confusion_matrix: dict,
    unmatched_groundtruths: dict,
    unmatched_predictions: dict,
):
    gt_labels = list(unmatched_groundtruths.keys())
    pd_labels = list(unmatched_predictions.keys())

    for gt_label in gt_labels:
        if unmatched_groundtruths[gt_label] == 0:
            unmatched_groundtruths.pop(gt_label)
        for pd_label in pd_labels:
            if confusion_matrix[gt_label][pd_label] == 0:
                confusion_matrix[gt_label].pop(pd_label)
        if len(confusion_matrix[gt_label]) == 0:
            confusion_matrix.pop(gt_label)

    for pd_label in pd_labels:
        if unmatched_predictions[pd_label] == 0:
            unmatched_predictions.pop(pd_label)


def test_confusion_matrix(
    detections_for_detailed_counting: list[Detection],
    rect1: tuple[float, float, float, float],
    rect2: tuple[float, float, float, float],
    rect3: tuple[float, float, float, float],
    rect4: tuple[float, float, float, float],
    rect5: tuple[float, float, float, float],
):
    loader = DataLoader()
    loader.add_bounding_boxes(detections_for_detailed_counting)
    evaluator = loader.finalize()

    assert evaluator.info["number_of_datums"] == 2
    assert evaluator.info["number_of_labels"] == 6
    assert evaluator.info["number_of_groundtruth_annotations"] == 4
    assert evaluator.info["number_of_prediction_annotations"] == 4

    actual_metrics = evaluator.compute_confusion_matrix(
        iou_thresholds=[0.5],
        score_thresholds=[0.05, 0.3, 0.35, 0.45, 0.55, 0.95],
    )

    actual_metrics = [m.to_dict() for m in actual_metrics]
    expected_metrics = [
        {
            "type": "ConfusionMatrix",
            "value": {
                "confusion_matrix": {
                    "v1": {
                        "v1": 1,
                    }
                },
                "unmatched_predictions": {
                    "not_v2": 1,
                    "unmatched_prediction": 1,
                    "matched_low_iou": 1,
                },
                "unmatched_ground_truths": {
                    "unmatched_groundtruth": 1,
                    "v2": 1,
                    "matched_low_iou": 1,
                },
            },
            "parameters": {
                "score_threshold": 0.05,
                "iou_threshold": 0.5,
            },
        },
        {
            "type": "ConfusionMatrix",
            "value": {
                "confusion_matrix": {
                    "v1": {
                        "v1": 1,
                    }
                },
                "unmatched_predictions": {
                    "not_v2": 1,
                    "matched_low_iou": 1,
                },
                "unmatched_ground_truths": {
                    "unmatched_groundtruth": 1,
                    "v2": 1,
                    "matched_low_iou": 1,
                },
            },
            "parameters": {
                "score_threshold": 0.3,
                "iou_threshold": 0.5,
            },
        },
        {
            "type": "ConfusionMatrix",
            "value": {
                "confusion_matrix": {
                    "v1": {
                        "v1": 1,
                    }
                },
                "unmatched_predictions": {
                    "matched_low_iou": 1,
                },
                "unmatched_ground_truths": {
                    "unmatched_groundtruth": 1,
                    "v2": 1,
                    "matched_low_iou": 1,
                },
            },
            "parameters": {
                "score_threshold": 0.35,
                "iou_threshold": 0.5,
            },
        },
        {
            "type": "ConfusionMatrix",
            "value": {
                "confusion_matrix": {
                    "v1": {
                        "v1": 1,
                    }
                },
                "unmatched_predictions": {
                    "matched_low_iou": 1,
                },
                "unmatched_ground_truths": {
                    "unmatched_groundtruth": 1,
                    "v2": 1,
                    "matched_low_iou": 1,
                },
            },
            "parameters": {
                "score_threshold": 0.45,
                "iou_threshold": 0.5,
            },
        },
        {
            "type": "ConfusionMatrix",
            "value": {
                "confusion_matrix": {},
                "unmatched_predictions": {},
                "unmatched_ground_truths": {
                    "v1": 1,
                    "unmatched_groundtruth": 1,
                    "v2": 1,
                    "matched_low_iou": 1,
                },
            },
            "parameters": {
                "score_threshold": 0.55,
                "iou_threshold": 0.5,
            },
        },
        {
            "type": "ConfusionMatrix",
            "value": {
                "confusion_matrix": {},
                "unmatched_predictions": {},
                "unmatched_ground_truths": {
                    "v1": 1,
                    "unmatched_groundtruth": 1,
                    "v2": 1,
                    "matched_low_iou": 1,
                },
            },
            "parameters": {
                "score_threshold": 0.95,
                "iou_threshold": 0.5,
            },
        },
    ]
    for m in actual_metrics:
        _filter_out_zero_counts(
            m["value"]["confusion_matrix"],
            m["value"]["unmatched_predictions"],
            m["value"]["unmatched_ground_truths"],
        )
        assert m in expected_metrics
    for m in expected_metrics:
        assert m in actual_metrics

    # test at lower IOU threshold
    actual_metrics = evaluator.compute_confusion_matrix(
        iou_thresholds=[0.45],
        score_thresholds=[0.05, 0.3, 0.35, 0.45, 0.55, 0.95],
    )

    actual_metrics = [m.to_dict() for m in actual_metrics]
    expected_metrics = [
        {
            "type": "ConfusionMatrix",
            "value": {
                "confusion_matrix": {
                    "v1": {
                        "v1": 1,
                    },
                    "v2": {
                        "not_v2": 1,
                    },
                },
                "unmatched_predictions": {
                    "unmatched_prediction": 1,
                    "matched_low_iou": 1,
                },
                "unmatched_ground_truths": {
                    "unmatched_groundtruth": 1,
                    "matched_low_iou": 1,
                },
            },
            "parameters": {
                "score_threshold": 0.05,
                "iou_threshold": 0.45,
            },
        },
        {
            "type": "ConfusionMatrix",
            "value": {
                "confusion_matrix": {
                    "v1": {
                        "v1": 1,
                    },
                    "v2": {
                        "not_v2": 1,
                    },
                },
                "unmatched_predictions": {
                    "matched_low_iou": 1,
                },
                "unmatched_ground_truths": {
                    "unmatched_groundtruth": 1,
                    "matched_low_iou": 1,
                },
            },
            "parameters": {
                "score_threshold": 0.3,
                "iou_threshold": 0.45,
            },
        },
        {
            "type": "ConfusionMatrix",
            "value": {
                "confusion_matrix": {
                    "v1": {
                        "v1": 1,
                    }
                },
                "unmatched_predictions": {
                    "matched_low_iou": 1,
                },
                "unmatched_ground_truths": {
                    "unmatched_groundtruth": 1,
                    "v2": 1,
                    "matched_low_iou": 1,
                },
            },
            "parameters": {
                "score_threshold": 0.35,
                "iou_threshold": 0.45,
            },
        },
        {
            "type": "ConfusionMatrix",
            "value": {
                "confusion_matrix": {
                    "v1": {
                        "v1": 1,
                    }
                },
                "unmatched_predictions": {
                    "matched_low_iou": 1,
                },
                "unmatched_ground_truths": {
                    "unmatched_groundtruth": 1,
                    "v2": 1,
                    "matched_low_iou": 1,
                },
            },
            "parameters": {
                "score_threshold": 0.45,
                "iou_threshold": 0.45,
            },
        },
        {
            "type": "ConfusionMatrix",
            "value": {
                "confusion_matrix": {},
                "unmatched_predictions": {},
                "unmatched_ground_truths": {
                    "v1": 1,
                    "unmatched_groundtruth": 1,
                    "v2": 1,
                    "matched_low_iou": 1,
                },
            },
            "parameters": {
                "score_threshold": 0.55,
                "iou_threshold": 0.45,
            },
        },
        {
            "type": "ConfusionMatrix",
            "value": {
                "confusion_matrix": {},
                "unmatched_predictions": {},
                "unmatched_ground_truths": {
                    "v1": 1,
                    "unmatched_groundtruth": 1,
                    "v2": 1,
                    "matched_low_iou": 1,
                },
            },
            "parameters": {
                "score_threshold": 0.95,
                "iou_threshold": 0.45,
            },
        },
    ]

    for m in actual_metrics:
        _filter_out_zero_counts(
            m["value"]["confusion_matrix"],
            m["value"]["unmatched_predictions"],
            m["value"]["unmatched_ground_truths"],
        )
        assert m in expected_metrics
    for m in expected_metrics:
        assert m in actual_metrics


def test_confusion_matrix_using_torch_metrics_example(
    torchmetrics_detections: list[Detection],
):
    """
    cf with torch metrics/pycocotools results listed here:
    https://github.com/Lightning-AI/metrics/blob/107dbfd5fb158b7ae6d76281df44bd94c836bfce/tests/unittests/detection/test_map.py#L231
    """
    loader = DataLoader()
    loader.add_bounding_boxes(torchmetrics_detections)
    evaluator = loader.finalize()

    assert evaluator.info["number_of_datums"] == 4
    assert evaluator.info["number_of_labels"] == 6
    assert evaluator.info["number_of_groundtruth_annotations"] == 20
    assert evaluator.info["number_of_prediction_annotations"] == 19

    actual_metrics = evaluator.compute_confusion_matrix(
        iou_thresholds=[0.5, 0.9],
        score_thresholds=[0.05, 0.25, 0.35, 0.55, 0.75, 0.8, 0.85, 0.95],
    )

    assert len(actual_metrics) == 16

    actual_metrics = [m.to_dict() for m in actual_metrics]
    expected_metrics = [
        {
            "type": "ConfusionMatrix",
            "value": {
                "confusion_matrix": {
                    "4": {"4": 2},
                    "2": {
                        "2": 1,
                        "3": 1,
                    },
                    "1": {"1": 1},
                    "0": {"0": 5},
                    "49": {"49": 8},
                },
                "unmatched_predictions": {"49": 1},
                "unmatched_ground_truths": {"49": 2},
            },
            "parameters": {
                "score_threshold": 0.05,
                "iou_threshold": 0.5,
            },
        },
        {
            "type": "ConfusionMatrix",
            "value": {
                "confusion_matrix": {
                    "4": {"4": 1},
                    "2": {
                        "2": 1,
                        "3": 1,
                    },
                    "1": {"1": 1},
                    "0": {"0": 5},
                    "49": {"49": 6},
                },
                "unmatched_predictions": {},
                "unmatched_ground_truths": {
                    "4": 1,
                    "49": 4,
                },
            },
            "parameters": {
                "score_threshold": 0.25,
                "iou_threshold": 0.5,
            },
        },
        {
            "type": "ConfusionMatrix",
            "value": {
                "confusion_matrix": {
                    "4": {"4": 1},
                    "2": {"2": 1},
                    "0": {"0": 4},
                    "49": {"49": 4},
                },
                "unmatched_predictions": {},
                "unmatched_ground_truths": {
                    "4": 1,
                    "2": 1,
                    "1": 1,
                    "0": 1,
                    "49": 6,
                },
            },
            "parameters": {
                "score_threshold": 0.35,
                "iou_threshold": 0.5,
            },
        },
        {
            "type": "ConfusionMatrix",
            "value": {
                "confusion_matrix": {
                    "2": {"2": 1},
                    "0": {"0": 3},
                    "49": {"49": 3},
                },
                "unmatched_predictions": {},
                "unmatched_ground_truths": {
                    "4": 2,
                    "2": 1,
                    "1": 1,
                    "0": 2,
                    "49": 7,
                },
            },
            "parameters": {
                "score_threshold": 0.55,
                "iou_threshold": 0.5,
            },
        },
        {
            "type": "ConfusionMatrix",
            "value": {
                "confusion_matrix": {
                    "0": {"0": 2},
                    "49": {"49": 2},
                },
                "unmatched_predictions": {},
                "unmatched_ground_truths": {
                    "4": 2,
                    "2": 2,
                    "1": 1,
                    "0": 3,
                    "49": 8,
                },
            },
            "parameters": {
                "score_threshold": 0.75,
                "iou_threshold": 0.5,
            },
        },
        {
            "type": "ConfusionMatrix",
            "value": {
                "confusion_matrix": {
                    "0": {"0": 2},
                    "49": {"49": 1},
                },
                "unmatched_predictions": {},
                "unmatched_ground_truths": {
                    "4": 2,
                    "2": 2,
                    "1": 1,
                    "0": 3,
                    "49": 9,
                },
            },
            "parameters": {
                "score_threshold": 0.8,
                "iou_threshold": 0.5,
            },
        },
        {
            "type": "ConfusionMatrix",
            "value": {
                "confusion_matrix": {
                    "0": {"0": 1},
                    "49": {"49": 1},
                },
                "unmatched_predictions": {},
                "unmatched_ground_truths": {
                    "4": 2,
                    "2": 2,
                    "1": 1,
                    "0": 4,
                    "49": 9,
                },
            },
            "parameters": {
                "score_threshold": 0.85,
                "iou_threshold": 0.5,
            },
        },
        {
            "type": "ConfusionMatrix",
            "value": {
                "confusion_matrix": {"0": {"0": 1}},
                "unmatched_predictions": {},
                "unmatched_ground_truths": {
                    "4": 2,
                    "2": 2,
                    "1": 1,
                    "0": 4,
                    "49": 10,
                },
            },
            "parameters": {
                "score_threshold": 0.95,
                "iou_threshold": 0.5,
            },
        },
        {
            "type": "ConfusionMatrix",
            "value": {
                "confusion_matrix": {
                    "2": {"2": 1},
                    "0": {"0": 1},
                    "49": {"49": 2},
                },
                "unmatched_predictions": {
                    "4": 2,
                    "3": 1,
                    "1": 1,
                    "0": 4,
                    "49": 7,
                },
                "unmatched_ground_truths": {
                    "4": 2,
                    "2": 1,
                    "1": 1,
                    "0": 4,
                    "49": 8,
                },
            },
            "parameters": {
                "score_threshold": 0.05,
                "iou_threshold": 0.9,
            },
        },
        {
            "type": "ConfusionMatrix",
            "value": {
                "confusion_matrix": {
                    "2": {"2": 1},
                    "0": {"0": 1},
                    "49": {"49": 2},
                },
                "unmatched_predictions": {
                    "4": 1,
                    "3": 1,
                    "1": 1,
                    "0": 4,
                    "49": 4,
                },
                "unmatched_ground_truths": {
                    "4": 2,
                    "2": 1,
                    "1": 1,
                    "0": 4,
                    "49": 8,
                },
            },
            "parameters": {
                "score_threshold": 0.25,
                "iou_threshold": 0.9,
            },
        },
        {
            "type": "ConfusionMatrix",
            "value": {
                "confusion_matrix": {
                    "2": {"2": 1},
                    "49": {"49": 2},
                },
                "unmatched_predictions": {
                    "4": 1,
                    "0": 4,
                    "49": 2,
                },
                "unmatched_ground_truths": {
                    "4": 2,
                    "2": 1,
                    "1": 1,
                    "0": 5,
                    "49": 8,
                },
            },
            "parameters": {
                "score_threshold": 0.35,
                "iou_threshold": 0.9,
            },
        },
        {
            "type": "ConfusionMatrix",
            "value": {
                "confusion_matrix": {
                    "2": {"2": 1},
                    "49": {"49": 1},
                },
                "unmatched_predictions": {
                    "0": 3,
                    "49": 2,
                },
                "unmatched_ground_truths": {
                    "4": 2,
                    "2": 1,
                    "1": 1,
                    "0": 5,
                    "49": 9,
                },
            },
            "parameters": {
                "score_threshold": 0.55,
                "iou_threshold": 0.9,
            },
        },
        {
            "type": "ConfusionMatrix",
            "value": {
                "confusion_matrix": {"49": {"49": 1}},
                "unmatched_predictions": {
                    "0": 2,
                    "49": 1,
                },
                "unmatched_ground_truths": {
                    "4": 2,
                    "2": 2,
                    "1": 1,
                    "0": 5,
                    "49": 9,
                },
            },
            "parameters": {
                "score_threshold": 0.75,
                "iou_threshold": 0.9,
            },
        },
        {
            "type": "ConfusionMatrix",
            "value": {
                "confusion_matrix": {"49": {"49": 1}},
                "unmatched_predictions": {"0": 2},
                "unmatched_ground_truths": {
                    "4": 2,
                    "2": 2,
                    "1": 1,
                    "0": 5,
                    "49": 9,
                },
            },
            "parameters": {
                "score_threshold": 0.8,
                "iou_threshold": 0.9,
            },
        },
        {
            "type": "ConfusionMatrix",
            "value": {
                "confusion_matrix": {"49": {"49": 1}},
                "unmatched_predictions": {"0": 1},
                "unmatched_ground_truths": {
                    "4": 2,
                    "2": 2,
                    "1": 1,
                    "0": 5,
                    "49": 9,
                },
            },
            "parameters": {
                "score_threshold": 0.85,
                "iou_threshold": 0.9,
            },
        },
        {
            "type": "ConfusionMatrix",
            "value": {
                "confusion_matrix": {},
                "unmatched_predictions": {"0": 1},
                "unmatched_ground_truths": {
                    "4": 2,
                    "2": 2,
                    "1": 1,
                    "0": 5,
                    "49": 10,
                },
            },
            "parameters": {
                "score_threshold": 0.95,
                "iou_threshold": 0.9,
            },
        },
    ]
    for m in actual_metrics:
        _filter_out_zero_counts(
            m["value"]["confusion_matrix"],
            m["value"]["unmatched_predictions"],
            m["value"]["unmatched_ground_truths"],
        )
        assert m in expected_metrics
    for m in expected_metrics:
        assert m in actual_metrics


def test_confusion_matrix_fp_unmatched_prediction_edge_case(
    detections_fp_unmatched_prediction_edge_case: list[Detection],
):

    loader = DataLoader()
    loader.add_bounding_boxes(detections_fp_unmatched_prediction_edge_case)
    evaluator = loader.finalize()

    assert evaluator.info["number_of_datums"] == 2
    assert evaluator.info["number_of_labels"] == 1
    assert evaluator.info["number_of_groundtruth_annotations"] == 2
    assert evaluator.info["number_of_prediction_annotations"] == 2

    actual_metrics = evaluator.compute_confusion_matrix(
        iou_thresholds=[0.5],
        score_thresholds=[0.5, 0.85],
    )

    assert len(actual_metrics) == 2

    actual_metrics = [m.to_dict() for m in actual_metrics]
    expected_metrics = [
        {
            "type": "ConfusionMatrix",
            "value": {
                "confusion_matrix": {"v1": {"v1": 1}},
                "unmatched_predictions": {"v1": 1},
                "unmatched_ground_truths": {"v1": 1},
            },
            "parameters": {
                "score_threshold": 0.5,
                "iou_threshold": 0.5,
            },
        },
        {
            "type": "ConfusionMatrix",
            "value": {
                "confusion_matrix": {},
                "unmatched_predictions": {},
                "unmatched_ground_truths": {"v1": 2},
            },
            "parameters": {
                "score_threshold": 0.85,
                "iou_threshold": 0.5,
            },
        },
    ]
    for m in actual_metrics:
        _filter_out_zero_counts(
            m["value"]["confusion_matrix"],
            m["value"]["unmatched_predictions"],
            m["value"]["unmatched_ground_truths"],
        )
        assert m in expected_metrics
    for m in expected_metrics:
        assert m in actual_metrics


def test_confusion_matrix_ranked_pair_ordering(
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

        assert evaluator.info["number_of_datums"] == 1
        assert evaluator.info["number_of_groundtruth_annotations"] == 3
        assert evaluator.info["number_of_labels"] == 4
        assert evaluator.info["number_of_prediction_annotations"] == 4
        assert evaluator.info["number_of_rows"] == 12

        actual_metrics = evaluator.compute_confusion_matrix(
            iou_thresholds=[0.5],
            score_thresholds=[0.0],
        )

        actual_metrics = [m.to_dict() for m in actual_metrics]
        expected_metrics = [
            {
                "type": "ConfusionMatrix",
                "value": {
                    "confusion_matrix": {
                        "label1": {"label2": 1},
                        "label2": {"label1": 1},
                    },
                    "unmatched_predictions": {
                        "label3": 1,
                        "label4": 1,
                    },
                    "unmatched_ground_truths": {"label3": 1},
                },
                "parameters": {
                    "score_threshold": 0.0,
                    "iou_threshold": 0.5,
                },
            },
        ]
        for m in actual_metrics:
            _filter_out_zero_counts(
                m["value"]["confusion_matrix"],
                m["value"]["unmatched_predictions"],
                m["value"]["unmatched_ground_truths"],
            )
            assert m in expected_metrics
        for m in expected_metrics:
            assert m in actual_metrics
