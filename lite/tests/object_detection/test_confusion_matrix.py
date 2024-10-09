import numpy as np
from valor_lite.object_detection import DataLoader, Detection, Evaluator
from valor_lite.object_detection.computation import compute_confusion_matrix


def test_confusion_matrix_no_data():
    evaluator = Evaluator()
    curves = evaluator.compute_confusion_matrix(
        iou_thresholds=[0.5],
        score_thresholds=[0.5],
        number_of_examples=0,
    )
    assert isinstance(curves, list)
    assert len(curves) == 0


def _test_compute_confusion_matrix(
    n_examples: int,
):
    sorted_pairs = np.array(
        [
            # dt,  gt,  pd,  iou,  gl,  pl, score,
            [0.0, 0.0, 1.0, 0.98, 0.0, 0.0, 0.9],
            [1.0, 1.0, 2.0, 0.55, 1.0, 0.0, 0.9],
            [2.0, -1.0, 4.0, 0.0, -1.0, 0.0, 0.65],
            [3.0, 4.0, 5.0, 1.0, 0.0, 0.0, 0.1],
            [1.0, 2.0, 3.0, 0.55, 0.0, 0.0, 0.1],
            [4.0, 5.0, -1.0, 0.0, 0.0, -1.0, -1.0],
        ]
    )
    label_metadata = np.array([[3, 4], [1, 0]])
    iou_thresholds = np.array([0.5])
    score_thresholds = np.array([score / 100.0 for score in range(1, 101)])

    (
        confusion_matrix,
        hallucinations,
        missing_predictions,
    ) = compute_confusion_matrix(
        data=sorted_pairs,
        label_metadata=label_metadata,
        iou_thresholds=iou_thresholds,
        score_thresholds=score_thresholds,
        n_examples=n_examples,
    )

    assert confusion_matrix.shape == (
        1,
        100,
        2,
        2,
        1 + n_examples * 4,
    )  # iou, score, gt label, pd label, metrics
    assert hallucinations.shape == (
        1,
        100,
        2,
        1 + n_examples * 3,
    )  # iou, score, pd label, metrics
    assert missing_predictions.shape == (
        1,
        100,
        2,
        1 + n_examples * 2,
    )  # iou, score, gt label, metrics

    return (confusion_matrix, hallucinations, missing_predictions)


def test_compute_confusion_matrix():

    (
        confusion_matrix,
        hallucinations,
        missing_predictions,
    ) = _test_compute_confusion_matrix(n_examples=0)

    """
    @ iou=0.5, score<0.1
    3x tp
    1x fp misclassification
    1x fp hallucination
    0x fn misclassification
    1x fn missing prediction
    """

    indices = slice(10)

    cm_gt0_pd0 = np.array([3.0])
    cm_gt0_pd1 = np.array([-1.0])
    cm_gt1_pd0 = np.array([1.0])
    cm_gt1_pd1 = np.array([-1.0])
    expected_cm = np.array(
        [[cm_gt0_pd0, cm_gt0_pd1], [cm_gt1_pd0, cm_gt1_pd1]]
    )
    assert np.isclose(confusion_matrix[0, indices, :, :, :], expected_cm).all()

    hal_pd0 = np.array([1.0])
    hal_pd1 = np.array([-1.0])
    expected_hallucinations = np.array([hal_pd0, hal_pd1])
    assert np.isclose(
        hallucinations[0, indices, :, :], expected_hallucinations
    ).all()

    misprd_gt0 = np.array([1.0])
    misprd_gt1 = np.array([-1.0])
    expected_missing_predictions = np.array([misprd_gt0, misprd_gt1])
    assert np.isclose(
        missing_predictions[0, indices, :, :], expected_missing_predictions
    ).all()

    """
    @ iou=0.5, 0.1 <= score < 0.65
    1x tp
    1x fp misclassification
    1x fp hallucination
    1x fn misclassification
    3x fn missing prediction
    """

    indices = slice(10, 65)

    cm_gt0_pd0 = np.array([1.0])
    cm_gt0_pd1 = np.array([-1.0])
    cm_gt1_pd0 = np.array([1.0])
    cm_gt1_pd1 = np.array([-1.0])
    expected_cm = np.array(
        [[cm_gt0_pd0, cm_gt0_pd1], [cm_gt1_pd0, cm_gt1_pd1]]
    )
    assert np.isclose(confusion_matrix[0, indices, :, :, :], expected_cm).all()

    hal_pd0 = np.array([1.0])
    hal_pd1 = np.array([-1.0])
    expected_hallucinations = np.array([hal_pd0, hal_pd1])
    assert np.isclose(
        hallucinations[0, indices, :, :], expected_hallucinations
    ).all()

    misprd_gt0 = np.array([3.0])
    misprd_gt1 = np.array([-1.0])
    expected_missing_predictions = np.array([misprd_gt0, misprd_gt1])
    assert np.isclose(
        missing_predictions[0, indices, :, :], expected_missing_predictions
    ).all()

    """
    @ iou=0.5, 0.65 <= score < 0.9
    1x tp
    1x fp misclassification
    0x fp hallucination
    1x fn misclassification
    3x fn missing prediction
    """

    indices = slice(65, 90)

    cm_gt0_pd0 = np.array([1.0])
    cm_gt0_pd1 = np.array([-1.0])
    cm_gt1_pd0 = np.array([1.0])
    cm_gt1_pd1 = np.array([-1.0])
    expected_cm = np.array(
        [[cm_gt0_pd0, cm_gt0_pd1], [cm_gt1_pd0, cm_gt1_pd1]]
    )
    assert np.isclose(confusion_matrix[0, indices, :, :, :], expected_cm).all()

    hal_pd0 = np.array([-1.0])
    hal_pd1 = np.array([-1.0])
    expected_hallucinations = np.array([hal_pd0, hal_pd1])
    assert np.isclose(
        hallucinations[0, indices, :, :], expected_hallucinations
    ).all()

    misprd_gt0 = np.array([3.0])
    misprd_gt1 = np.array([-1.0])
    expected_missing_predictions = np.array([misprd_gt0, misprd_gt1])
    assert np.isclose(
        missing_predictions[0, indices, :, :], expected_missing_predictions
    ).all()

    """
    @ iou=0.5, score>=0.9
    0x tp
    0x fp misclassification
    0x fp hallucination
    0x fn misclassification
    4x fn missing prediction
    """

    indices = slice(90, None)

    cm_gt0_pd0 = np.array([-1.0])
    cm_gt0_pd1 = np.array([-1.0])
    cm_gt1_pd0 = np.array([-1.0])
    cm_gt1_pd1 = np.array([-1.0])
    expected_cm = np.array(
        [[cm_gt0_pd0, cm_gt0_pd1], [cm_gt1_pd0, cm_gt1_pd1]]
    )
    assert np.isclose(confusion_matrix[0, indices, :, :, :], expected_cm).all()

    hal_pd0 = np.array([-1.0])
    hal_pd1 = np.array([-1.0])
    expected_hallucinations = np.array([hal_pd0, hal_pd1])
    assert np.isclose(
        hallucinations[0, indices, :, :], expected_hallucinations
    ).all()

    misprd_gt0 = np.array([4.0])
    misprd_gt1 = np.array([1.0])
    expected_missing_predictions = np.array([misprd_gt0, misprd_gt1])
    assert np.isclose(
        missing_predictions[0, indices, :, :], expected_missing_predictions
    ).all()


def test_compute_confusion_matrix_with_examples():

    (
        confusion_matrix,
        hallucinations,
        missing_predictions,
    ) = _test_compute_confusion_matrix(n_examples=2)

    """
    @ iou=0.5, score<0.1
    3x tp
    1x fp misclassification
    1x fp hallucination
    0x fn misclassification
    1x fn missing prediction
    """

    indices = slice(10)

    # total count, datum 0, gt 0, pd 0, score 0, datum 1, gt 1, pd 1, score 1
    cm_gt0_pd0 = np.array([3.0, 0.0, 0.0, 1.0, 0.9, 1.0, 2.0, 3.0, 0.1])
    cm_gt0_pd1 = np.array(
        [-1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0]
    )
    cm_gt1_pd0 = np.array([1.0, 1.0, 1.0, 2.0, 0.9, -1.0, -1.0, -1.0, -1.0])
    cm_gt1_pd1 = np.array(
        [-1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0]
    )
    expected_cm = np.array(
        [[cm_gt0_pd0, cm_gt0_pd1], [cm_gt1_pd0, cm_gt1_pd1]]
    )
    assert np.isclose(confusion_matrix[0, indices, :, :, :], expected_cm).all()

    # total count, datum 0, pd 0, score 0, datum 1, pd 1, score 1
    hal_pd0 = np.array([1.0, 2.0, 4.0, 0.65, -1.0, -1.0, -1.0])
    hal_pd1 = np.array([-1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0])
    expected_hallucinations = np.array([hal_pd0, hal_pd1])
    assert np.isclose(
        hallucinations[0, indices, :, :], expected_hallucinations
    ).all()

    # total count, datum 0, gt 0, datum1, gt 1
    misprd_gt0 = np.array([1.0, 4.0, 5.0, -1.0, -1.0])
    misprd_gt1 = np.array([-1.0, -1.0, -1.0, -1.0, -1.0])
    expected_missing_predictions = np.array([misprd_gt0, misprd_gt1])
    assert np.isclose(
        missing_predictions[0, indices, :, :], expected_missing_predictions
    ).all()

    """
    @ iou=0.5, 0.1 <= score < 0.65
    1x tp
    1x fp misclassification
    1x fp hallucination
    1x fn misclassification
    2x fn missing prediction
    """

    indices = slice(10, 65)

    # total count, datum 0, gt 0, pd 0, score 0, datum 1, gt 1, pd 1, score 1
    cm_gt0_pd0 = np.array([1.0, 0.0, 0.0, 1.0, 0.9, -1.0, -1.0, -1.0, -1.0])
    cm_gt0_pd1 = np.array(
        [-1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0]
    )
    cm_gt1_pd0 = np.array([1.0, 1.0, 1.0, 2.0, 0.9, -1.0, -1.0, -1.0, -1.0])
    cm_gt1_pd1 = np.array(
        [-1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0]
    )
    expected_cm = np.array(
        [[cm_gt0_pd0, cm_gt0_pd1], [cm_gt1_pd0, cm_gt1_pd1]]
    )
    assert np.isclose(confusion_matrix[0, indices, :, :, :], expected_cm).all()

    # total count, datum 0, pd 0, score 0, datum 1, pd 1, score 1
    hal_pd0 = np.array([1.0, 2.0, 4.0, 0.65, -1.0, -1.0, -1.0])
    hal_pd1 = np.array([-1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0])
    expected_hallucinations = np.array([hal_pd0, hal_pd1])
    assert np.isclose(
        hallucinations[0, indices, :, :], expected_hallucinations
    ).all()

    # total count, datum 0, gt 0, datum1, gt 1
    misprd_gt0 = np.array([3.0, 1.0, 2.0, 3.0, 4.0])
    misprd_gt1 = np.array([-1.0, -1.0, -1.0, -1.0, -1.0])
    expected_missing_predictions = np.array([misprd_gt0, misprd_gt1])
    assert np.isclose(
        missing_predictions[0, indices, :, :], expected_missing_predictions
    ).all()

    """
    @ iou=0.5, 0.65 <= score < 0.9
    1x tp
    1x fp misclassification
    0x fp hallucination
    1x fn misclassification
    2x fn missing prediction
    """

    indices = slice(65, 90)

    # total count, datum 0, gt 0, pd 0, score 0, datum 1, gt 1, pd 1, score 1
    cm_gt0_pd0 = np.array([1.0, 0.0, 0.0, 1.0, 0.9, -1.0, -1.0, -1.0, -1.0])
    cm_gt0_pd1 = np.array(
        [-1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0]
    )
    cm_gt1_pd0 = np.array([1.0, 1.0, 1.0, 2.0, 0.9, -1.0, -1.0, -1.0, -1.0])
    cm_gt1_pd1 = np.array(
        [-1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0]
    )
    expected_cm = np.array(
        [[cm_gt0_pd0, cm_gt0_pd1], [cm_gt1_pd0, cm_gt1_pd1]]
    )
    assert np.isclose(confusion_matrix[0, indices, :, :, :], expected_cm).all()

    # total count, datum 0, pd 0, score 0, datum 1, pd 1, score 1
    hal_pd0 = np.array([-1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0])
    hal_pd1 = np.array([-1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0])
    expected_hallucinations = np.array([hal_pd0, hal_pd1])
    assert np.isclose(
        hallucinations[0, indices, :, :], expected_hallucinations
    ).all()

    # total count, datum 0, gt 0, datum1, gt 1
    misprd_gt0 = np.array([3.0, 1.0, 2.0, 3.0, 4.0])
    misprd_gt1 = np.array([-1.0, -1.0, -1.0, -1.0, -1.0])
    expected_missing_predictions = np.array([misprd_gt0, misprd_gt1])
    assert np.isclose(
        missing_predictions[0, indices, :, :], expected_missing_predictions
    ).all()

    """
    @ iou=0.5, score>=0.9
    0x tp
    0x fp misclassification
    0x fp hallucination
    0x fn misclassification
    4x fn missing prediction
    """

    indices = slice(90, None)

    # total count, datum 0, gt 0, pd 0, score 0, datum 1, gt 1, pd 1, score 1
    cm_gt0_pd0 = np.array(
        [-1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0]
    )
    cm_gt0_pd1 = np.array(
        [-1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0]
    )
    cm_gt1_pd0 = np.array(
        [-1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0]
    )
    cm_gt1_pd1 = np.array(
        [-1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0]
    )
    expected_cm = np.array(
        [[cm_gt0_pd0, cm_gt0_pd1], [cm_gt1_pd0, cm_gt1_pd1]]
    )
    assert np.isclose(confusion_matrix[0, indices, :, :, :], expected_cm).all()

    # total count, datum 0, pd 0, score 0, datum 1, pd 1, score 1
    hal_pd0 = np.array([-1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0])
    hal_pd1 = np.array([-1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0])
    expected_hallucinations = np.array([hal_pd0, hal_pd1])
    assert np.isclose(
        hallucinations[0, indices, :, :], expected_hallucinations
    ).all()

    # total count, datum 0, gt 0, datum1, gt 1
    misprd_gt0 = np.array([4.0, 0.0, 0.0, 1.0, 2.0])
    misprd_gt1 = np.array([1.0, 1.0, 1.0, -1.0, -1.0])
    expected_missing_predictions = np.array([misprd_gt0, misprd_gt1])
    assert np.isclose(
        missing_predictions[0, indices, :, :], expected_missing_predictions
    ).all()


def _filter_out_zero_counts(cm: dict, hl: dict, mp: dict):
    gt_labels = list(mp.keys())
    pd_labels = list(hl.keys())

    for gt_label in gt_labels:
        for pd_label in pd_labels:
            if cm[gt_label][pd_label]["count"] == 0:
                cm[gt_label].pop(pd_label)
        if len(cm[gt_label]) == 0:
            cm.pop(gt_label)

    for pd_label in pd_labels:
        if hl[pd_label]["count"] == 0:
            hl.pop(pd_label)

    for gt_label in gt_labels:
        if mp[gt_label]["count"] == 0:
            mp.pop(gt_label)


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

    assert evaluator.ignored_prediction_labels == [
        "not_v2",
        "hallucination",
    ]
    assert evaluator.missing_prediction_labels == [
        "missed_detection",
        "v2",
    ]
    assert evaluator.n_datums == 2
    assert evaluator.n_labels == 6
    assert evaluator.n_groundtruths == 4
    assert evaluator.n_predictions == 4

    actual_metrics = evaluator.compute_confusion_matrix(
        iou_thresholds=[0.5],
        score_thresholds=[0.05, 0.3, 0.35, 0.45, 0.55, 0.95],
        number_of_examples=1,
        as_dict=True,
    )

    rect1_dict = evaluator._convert_example_to_dict(np.array(rect1))
    rect2_dict = evaluator._convert_example_to_dict(np.array(rect2))
    rect3_dict = evaluator._convert_example_to_dict(np.array(rect3))
    rect4_dict = evaluator._convert_example_to_dict(np.array(rect4))
    rect5_dict = evaluator._convert_example_to_dict(np.array(rect5))

    expected_metrics = [
        {
            "type": "ConfusionMatrix",
            "value": {
                "confusion_matrix": {
                    "v1": {
                        "v1": {
                            "count": 1,
                            "examples": [
                                {
                                    "datum": "uid1",
                                    "groundtruth": rect1_dict,
                                    "prediction": rect1_dict,
                                    "score": 0.5,
                                }
                            ],
                        }
                    }
                },
                "hallucinations": {
                    "not_v2": {
                        "count": 1,
                        "examples": [
                            {
                                "datum": "uid1",
                                "prediction": rect5_dict,
                                "score": 0.30000001192092896,
                            }
                        ],
                    },
                    "hallucination": {
                        "count": 1,
                        "examples": [
                            {
                                "datum": "uid1",
                                "prediction": rect4_dict,
                                "score": 0.10000000149011612,
                            }
                        ],
                    },
                    "low_iou": {
                        "count": 1,
                        "examples": [
                            {
                                "datum": "uid2",
                                "prediction": rect2_dict,
                                "score": 0.5,
                            }
                        ],
                    },
                },
                "missing_predictions": {
                    "missed_detection": {
                        "count": 1,
                        "examples": [
                            {"datum": "uid1", "groundtruth": rect2_dict}
                        ],
                    },
                    "v2": {
                        "count": 1,
                        "examples": [
                            {"datum": "uid1", "groundtruth": rect3_dict}
                        ],
                    },
                    "low_iou": {
                        "count": 1,
                        "examples": [
                            {"datum": "uid2", "groundtruth": rect1_dict}
                        ],
                    },
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
                        "v1": {
                            "count": 1,
                            "examples": [
                                {
                                    "datum": "uid1",
                                    "groundtruth": rect1_dict,
                                    "prediction": rect1_dict,
                                    "score": 0.5,
                                }
                            ],
                        }
                    }
                },
                "hallucinations": {
                    "not_v2": {
                        "count": 1,
                        "examples": [
                            {
                                "datum": "uid1",
                                "prediction": rect5_dict,
                                "score": 0.30000001192092896,
                            }
                        ],
                    },
                    "low_iou": {
                        "count": 1,
                        "examples": [
                            {
                                "datum": "uid2",
                                "prediction": rect2_dict,
                                "score": 0.5,
                            }
                        ],
                    },
                },
                "missing_predictions": {
                    "missed_detection": {
                        "count": 1,
                        "examples": [
                            {"datum": "uid1", "groundtruth": rect2_dict}
                        ],
                    },
                    "v2": {
                        "count": 1,
                        "examples": [
                            {"datum": "uid1", "groundtruth": rect3_dict}
                        ],
                    },
                    "low_iou": {
                        "count": 1,
                        "examples": [
                            {"datum": "uid2", "groundtruth": rect1_dict}
                        ],
                    },
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
                        "v1": {
                            "count": 1,
                            "examples": [
                                {
                                    "datum": "uid1",
                                    "groundtruth": rect1_dict,
                                    "prediction": rect1_dict,
                                    "score": 0.5,
                                }
                            ],
                        }
                    }
                },
                "hallucinations": {
                    "low_iou": {
                        "count": 1,
                        "examples": [
                            {
                                "datum": "uid2",
                                "prediction": rect2_dict,
                                "score": 0.5,
                            }
                        ],
                    }
                },
                "missing_predictions": {
                    "missed_detection": {
                        "count": 1,
                        "examples": [
                            {"datum": "uid1", "groundtruth": rect2_dict}
                        ],
                    },
                    "v2": {
                        "count": 1,
                        "examples": [
                            {"datum": "uid1", "groundtruth": rect3_dict}
                        ],
                    },
                    "low_iou": {
                        "count": 1,
                        "examples": [
                            {"datum": "uid2", "groundtruth": rect1_dict}
                        ],
                    },
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
                        "v1": {
                            "count": 1,
                            "examples": [
                                {
                                    "datum": "uid1",
                                    "groundtruth": rect1_dict,
                                    "prediction": rect1_dict,
                                    "score": 0.5,
                                }
                            ],
                        }
                    }
                },
                "hallucinations": {
                    "low_iou": {
                        "count": 1,
                        "examples": [
                            {
                                "datum": "uid2",
                                "prediction": rect2_dict,
                                "score": 0.5,
                            }
                        ],
                    }
                },
                "missing_predictions": {
                    "missed_detection": {
                        "count": 1,
                        "examples": [
                            {"datum": "uid1", "groundtruth": rect2_dict}
                        ],
                    },
                    "v2": {
                        "count": 1,
                        "examples": [
                            {"datum": "uid1", "groundtruth": rect3_dict}
                        ],
                    },
                    "low_iou": {
                        "count": 1,
                        "examples": [
                            {"datum": "uid2", "groundtruth": rect1_dict}
                        ],
                    },
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
                "hallucinations": {},
                "missing_predictions": {
                    "v1": {
                        "count": 1,
                        "examples": [
                            {"datum": "uid1", "groundtruth": rect1_dict}
                        ],
                    },
                    "missed_detection": {
                        "count": 1,
                        "examples": [
                            {"datum": "uid1", "groundtruth": rect2_dict}
                        ],
                    },
                    "v2": {
                        "count": 1,
                        "examples": [
                            {"datum": "uid1", "groundtruth": rect3_dict}
                        ],
                    },
                    "low_iou": {
                        "count": 1,
                        "examples": [
                            {"datum": "uid2", "groundtruth": rect1_dict}
                        ],
                    },
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
                "hallucinations": {},
                "missing_predictions": {
                    "v1": {
                        "count": 1,
                        "examples": [
                            {"datum": "uid1", "groundtruth": rect1_dict}
                        ],
                    },
                    "missed_detection": {
                        "count": 1,
                        "examples": [
                            {"datum": "uid1", "groundtruth": rect2_dict}
                        ],
                    },
                    "v2": {
                        "count": 1,
                        "examples": [
                            {"datum": "uid1", "groundtruth": rect3_dict}
                        ],
                    },
                    "low_iou": {
                        "count": 1,
                        "examples": [
                            {"datum": "uid2", "groundtruth": rect1_dict}
                        ],
                    },
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
            m["value"]["hallucinations"],
            m["value"]["missing_predictions"],
        )
        assert m in expected_metrics
    for m in expected_metrics:
        assert m in actual_metrics

    # test at lower IoU threshold

    actual_metrics = evaluator.compute_confusion_matrix(
        iou_thresholds=[0.45],
        score_thresholds=[0.05, 0.3, 0.35, 0.45, 0.55, 0.95],
        number_of_examples=1,
        as_dict=True,
    )

    expected_metrics = [
        {
            "type": "ConfusionMatrix",
            "value": {
                "confusion_matrix": {
                    "v1": {
                        "v1": {
                            "count": 1,
                            "examples": [
                                {
                                    "datum": "uid1",
                                    "groundtruth": rect1_dict,
                                    "prediction": rect1_dict,
                                    "score": 0.5,
                                }
                            ],
                        }
                    },
                    "v2": {
                        "not_v2": {
                            "count": 1,
                            "examples": [
                                {
                                    "datum": "uid1",
                                    "groundtruth": rect3_dict,
                                    "prediction": rect5_dict,
                                    "score": 0.30000001192092896,
                                }
                            ],
                        }
                    },
                },
                "hallucinations": {
                    "hallucination": {
                        "count": 1,
                        "examples": [
                            {
                                "datum": "uid1",
                                "prediction": rect4_dict,
                                "score": 0.10000000149011612,
                            }
                        ],
                    },
                    "low_iou": {
                        "count": 1,
                        "examples": [
                            {
                                "datum": "uid2",
                                "prediction": rect2_dict,
                                "score": 0.5,
                            }
                        ],
                    },
                },
                "missing_predictions": {
                    "missed_detection": {
                        "count": 1,
                        "examples": [
                            {"datum": "uid1", "groundtruth": rect2_dict}
                        ],
                    },
                    "low_iou": {
                        "count": 1,
                        "examples": [
                            {
                                "datum": "uid2",
                                "groundtruth": rect1_dict,
                            }
                        ],
                    },
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
                        "v1": {
                            "count": 1,
                            "examples": [
                                {
                                    "datum": "uid1",
                                    "groundtruth": rect1_dict,
                                    "prediction": rect1_dict,
                                    "score": 0.5,
                                }
                            ],
                        }
                    },
                    "v2": {
                        "not_v2": {
                            "count": 1,
                            "examples": [
                                {
                                    "datum": "uid1",
                                    "groundtruth": rect3_dict,
                                    "prediction": rect5_dict,
                                    "score": 0.30000001192092896,
                                }
                            ],
                        }
                    },
                },
                "hallucinations": {
                    "low_iou": {
                        "count": 1,
                        "examples": [
                            {
                                "datum": "uid2",
                                "prediction": rect2_dict,
                                "score": 0.5,
                            }
                        ],
                    }
                },
                "missing_predictions": {
                    "missed_detection": {
                        "count": 1,
                        "examples": [
                            {"datum": "uid1", "groundtruth": rect2_dict}
                        ],
                    },
                    "low_iou": {
                        "count": 1,
                        "examples": [
                            {
                                "datum": "uid2",
                                "groundtruth": rect1_dict,
                            }
                        ],
                    },
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
                        "v1": {
                            "count": 1,
                            "examples": [
                                {
                                    "datum": "uid1",
                                    "groundtruth": rect1_dict,
                                    "prediction": rect1_dict,
                                    "score": 0.5,
                                }
                            ],
                        }
                    }
                },
                "hallucinations": {
                    "low_iou": {
                        "count": 1,
                        "examples": [
                            {
                                "datum": "uid2",
                                "prediction": rect2_dict,
                                "score": 0.5,
                            }
                        ],
                    }
                },
                "missing_predictions": {
                    "missed_detection": {
                        "count": 1,
                        "examples": [
                            {"datum": "uid1", "groundtruth": rect2_dict}
                        ],
                    },
                    "v2": {
                        "count": 1,
                        "examples": [
                            {"datum": "uid1", "groundtruth": rect3_dict}
                        ],
                    },
                    "low_iou": {
                        "count": 1,
                        "examples": [
                            {
                                "datum": "uid2",
                                "groundtruth": rect1_dict,
                            }
                        ],
                    },
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
                        "v1": {
                            "count": 1,
                            "examples": [
                                {
                                    "datum": "uid1",
                                    "groundtruth": rect1_dict,
                                    "prediction": rect1_dict,
                                    "score": 0.5,
                                }
                            ],
                        }
                    }
                },
                "hallucinations": {
                    "low_iou": {
                        "count": 1,
                        "examples": [
                            {
                                "datum": "uid2",
                                "prediction": rect2_dict,
                                "score": 0.5,
                            }
                        ],
                    }
                },
                "missing_predictions": {
                    "missed_detection": {
                        "count": 1,
                        "examples": [
                            {"datum": "uid1", "groundtruth": rect2_dict}
                        ],
                    },
                    "v2": {
                        "count": 1,
                        "examples": [
                            {"datum": "uid1", "groundtruth": rect3_dict}
                        ],
                    },
                    "low_iou": {
                        "count": 1,
                        "examples": [
                            {
                                "datum": "uid2",
                                "groundtruth": rect1_dict,
                            }
                        ],
                    },
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
                "hallucinations": {},
                "missing_predictions": {
                    "v1": {
                        "count": 1,
                        "examples": [
                            {
                                "datum": "uid1",
                                "groundtruth": rect1_dict,
                            }
                        ],
                    },
                    "missed_detection": {
                        "count": 1,
                        "examples": [
                            {"datum": "uid1", "groundtruth": rect2_dict}
                        ],
                    },
                    "v2": {
                        "count": 1,
                        "examples": [
                            {"datum": "uid1", "groundtruth": rect3_dict}
                        ],
                    },
                    "low_iou": {
                        "count": 1,
                        "examples": [
                            {
                                "datum": "uid2",
                                "groundtruth": rect1_dict,
                            }
                        ],
                    },
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
                "hallucinations": {},
                "missing_predictions": {
                    "v1": {
                        "count": 1,
                        "examples": [
                            {
                                "datum": "uid1",
                                "groundtruth": rect1_dict,
                            }
                        ],
                    },
                    "missed_detection": {
                        "count": 1,
                        "examples": [
                            {"datum": "uid1", "groundtruth": rect2_dict}
                        ],
                    },
                    "v2": {
                        "count": 1,
                        "examples": [
                            {"datum": "uid1", "groundtruth": rect3_dict}
                        ],
                    },
                    "low_iou": {
                        "count": 1,
                        "examples": [
                            {
                                "datum": "uid2",
                                "groundtruth": rect1_dict,
                            }
                        ],
                    },
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
            m["value"]["hallucinations"],
            m["value"]["missing_predictions"],
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

    assert evaluator.ignored_prediction_labels == ["3"]
    assert evaluator.missing_prediction_labels == []
    assert evaluator.n_datums == 4
    assert evaluator.n_labels == 6
    assert evaluator.n_groundtruths == 20
    assert evaluator.n_predictions == 19

    actual_metrics = evaluator.compute_confusion_matrix(
        iou_thresholds=[0.5, 0.9],
        score_thresholds=[0.05, 0.25, 0.35, 0.55, 0.75, 0.8, 0.85, 0.95],
        number_of_examples=0,
        as_dict=True,
    )

    assert len(actual_metrics) == 16

    expected_metrics = [
        {
            "type": "ConfusionMatrix",
            "value": {
                "confusion_matrix": {
                    "4": {"4": {"count": 2, "examples": []}},
                    "2": {
                        "2": {"count": 1, "examples": []},
                        "3": {"count": 1, "examples": []},
                    },
                    "1": {"1": {"count": 1, "examples": []}},
                    "0": {"0": {"count": 5, "examples": []}},
                    "49": {"49": {"count": 9, "examples": []}},
                },
                "hallucinations": {},
                "missing_predictions": {"49": {"count": 1, "examples": []}},
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
                    "4": {"4": {"count": 1, "examples": []}},
                    "2": {
                        "2": {"count": 1, "examples": []},
                        "3": {"count": 1, "examples": []},
                    },
                    "1": {"1": {"count": 1, "examples": []}},
                    "0": {"0": {"count": 5, "examples": []}},
                    "49": {"49": {"count": 6, "examples": []}},
                },
                "hallucinations": {},
                "missing_predictions": {
                    "4": {"count": 1, "examples": []},
                    "49": {"count": 4, "examples": []},
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
                    "4": {"4": {"count": 1, "examples": []}},
                    "2": {"2": {"count": 1, "examples": []}},
                    "0": {"0": {"count": 4, "examples": []}},
                    "49": {"49": {"count": 4, "examples": []}},
                },
                "hallucinations": {},
                "missing_predictions": {
                    "4": {"count": 1, "examples": []},
                    "2": {"count": 1, "examples": []},
                    "1": {"count": 1, "examples": []},
                    "0": {"count": 1, "examples": []},
                    "49": {"count": 6, "examples": []},
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
                    "2": {"2": {"count": 1, "examples": []}},
                    "0": {"0": {"count": 3, "examples": []}},
                    "49": {"49": {"count": 3, "examples": []}},
                },
                "hallucinations": {},
                "missing_predictions": {
                    "4": {"count": 2, "examples": []},
                    "2": {"count": 1, "examples": []},
                    "1": {"count": 1, "examples": []},
                    "0": {"count": 2, "examples": []},
                    "49": {"count": 7, "examples": []},
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
                    "0": {"0": {"count": 2, "examples": []}},
                    "49": {"49": {"count": 2, "examples": []}},
                },
                "hallucinations": {},
                "missing_predictions": {
                    "4": {"count": 2, "examples": []},
                    "2": {"count": 2, "examples": []},
                    "1": {"count": 1, "examples": []},
                    "0": {"count": 3, "examples": []},
                    "49": {"count": 8, "examples": []},
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
                    "0": {"0": {"count": 2, "examples": []}},
                    "49": {"49": {"count": 1, "examples": []}},
                },
                "hallucinations": {},
                "missing_predictions": {
                    "4": {"count": 2, "examples": []},
                    "2": {"count": 2, "examples": []},
                    "1": {"count": 1, "examples": []},
                    "0": {"count": 3, "examples": []},
                    "49": {"count": 9, "examples": []},
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
                    "0": {"0": {"count": 1, "examples": []}},
                    "49": {"49": {"count": 1, "examples": []}},
                },
                "hallucinations": {},
                "missing_predictions": {
                    "4": {"count": 2, "examples": []},
                    "2": {"count": 2, "examples": []},
                    "1": {"count": 1, "examples": []},
                    "0": {"count": 4, "examples": []},
                    "49": {"count": 9, "examples": []},
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
                "confusion_matrix": {"0": {"0": {"count": 1, "examples": []}}},
                "hallucinations": {},
                "missing_predictions": {
                    "4": {"count": 2, "examples": []},
                    "2": {"count": 2, "examples": []},
                    "1": {"count": 1, "examples": []},
                    "0": {"count": 4, "examples": []},
                    "49": {"count": 10, "examples": []},
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
                    "2": {"2": {"count": 1, "examples": []}},
                    "0": {"0": {"count": 1, "examples": []}},
                    "49": {"49": {"count": 2, "examples": []}},
                },
                "hallucinations": {
                    "4": {"count": 2, "examples": []},
                    "3": {"count": 1, "examples": []},
                    "1": {"count": 1, "examples": []},
                    "0": {"count": 4, "examples": []},
                    "49": {"count": 7, "examples": []},
                },
                "missing_predictions": {
                    "4": {"count": 2, "examples": []},
                    "2": {"count": 1, "examples": []},
                    "1": {"count": 1, "examples": []},
                    "0": {"count": 4, "examples": []},
                    "49": {"count": 8, "examples": []},
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
                    "2": {"2": {"count": 1, "examples": []}},
                    "0": {"0": {"count": 1, "examples": []}},
                    "49": {"49": {"count": 2, "examples": []}},
                },
                "hallucinations": {
                    "4": {"count": 1, "examples": []},
                    "3": {"count": 1, "examples": []},
                    "1": {"count": 1, "examples": []},
                    "0": {"count": 4, "examples": []},
                    "49": {"count": 4, "examples": []},
                },
                "missing_predictions": {
                    "4": {"count": 2, "examples": []},
                    "2": {"count": 1, "examples": []},
                    "1": {"count": 1, "examples": []},
                    "0": {"count": 4, "examples": []},
                    "49": {"count": 8, "examples": []},
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
                    "2": {"2": {"count": 1, "examples": []}},
                    "49": {"49": {"count": 2, "examples": []}},
                },
                "hallucinations": {
                    "4": {"count": 1, "examples": []},
                    "0": {"count": 4, "examples": []},
                    "49": {"count": 2, "examples": []},
                },
                "missing_predictions": {
                    "4": {"count": 2, "examples": []},
                    "2": {"count": 1, "examples": []},
                    "1": {"count": 1, "examples": []},
                    "0": {"count": 5, "examples": []},
                    "49": {"count": 8, "examples": []},
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
                    "2": {"2": {"count": 1, "examples": []}},
                    "49": {"49": {"count": 1, "examples": []}},
                },
                "hallucinations": {
                    "0": {"count": 3, "examples": []},
                    "49": {"count": 2, "examples": []},
                },
                "missing_predictions": {
                    "4": {"count": 2, "examples": []},
                    "2": {"count": 1, "examples": []},
                    "1": {"count": 1, "examples": []},
                    "0": {"count": 5, "examples": []},
                    "49": {"count": 9, "examples": []},
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
                "confusion_matrix": {
                    "49": {"49": {"count": 1, "examples": []}}
                },
                "hallucinations": {
                    "0": {"count": 2, "examples": []},
                    "49": {"count": 1, "examples": []},
                },
                "missing_predictions": {
                    "4": {"count": 2, "examples": []},
                    "2": {"count": 2, "examples": []},
                    "1": {"count": 1, "examples": []},
                    "0": {"count": 5, "examples": []},
                    "49": {"count": 9, "examples": []},
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
                "confusion_matrix": {
                    "49": {"49": {"count": 1, "examples": []}}
                },
                "hallucinations": {"0": {"count": 2, "examples": []}},
                "missing_predictions": {
                    "4": {"count": 2, "examples": []},
                    "2": {"count": 2, "examples": []},
                    "1": {"count": 1, "examples": []},
                    "0": {"count": 5, "examples": []},
                    "49": {"count": 9, "examples": []},
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
                "confusion_matrix": {
                    "49": {"49": {"count": 1, "examples": []}}
                },
                "hallucinations": {"0": {"count": 1, "examples": []}},
                "missing_predictions": {
                    "4": {"count": 2, "examples": []},
                    "2": {"count": 2, "examples": []},
                    "1": {"count": 1, "examples": []},
                    "0": {"count": 5, "examples": []},
                    "49": {"count": 9, "examples": []},
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
                "hallucinations": {"0": {"count": 1, "examples": []}},
                "missing_predictions": {
                    "4": {"count": 2, "examples": []},
                    "2": {"count": 2, "examples": []},
                    "1": {"count": 1, "examples": []},
                    "0": {"count": 5, "examples": []},
                    "49": {"count": 10, "examples": []},
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
            m["value"]["hallucinations"],
            m["value"]["missing_predictions"],
        )
        assert m in expected_metrics
    for m in expected_metrics:
        assert m in actual_metrics


def test_confusion_matrix_fp_hallucination_edge_case(
    detections_fp_hallucination_edge_case: list[Detection],
):

    loader = DataLoader()
    loader.add_bounding_boxes(detections_fp_hallucination_edge_case)
    evaluator = loader.finalize()

    assert evaluator.ignored_prediction_labels == []
    assert evaluator.missing_prediction_labels == []
    assert evaluator.n_datums == 2
    assert evaluator.n_labels == 1
    assert evaluator.n_groundtruths == 2
    assert evaluator.n_predictions == 2

    actual_metrics = evaluator.compute_confusion_matrix(
        iou_thresholds=[0.5],
        score_thresholds=[0.5, 0.85],
        number_of_examples=1,
        as_dict=True,
    )

    assert len(actual_metrics) == 2

    expected_metrics = [
        {
            "type": "ConfusionMatrix",
            "value": {
                "confusion_matrix": {
                    "v1": {
                        "v1": {
                            "count": 1,
                            "examples": [
                                {
                                    "datum": "uid1",
                                    "groundtruth": {
                                        "xmin": 0.0,
                                        "xmax": 5.0,
                                        "ymin": 0.0,
                                        "ymax": 5.0,
                                    },
                                    "prediction": {
                                        "xmin": 0.0,
                                        "xmax": 5.0,
                                        "ymin": 0.0,
                                        "ymax": 5.0,
                                    },
                                    "score": 0.800000011920929,
                                }
                            ],
                        }
                    }
                },
                "hallucinations": {
                    "v1": {
                        "count": 1,
                        "examples": [
                            {
                                "datum": "uid2",
                                "prediction": {
                                    "xmin": 10.0,
                                    "xmax": 20.0,
                                    "ymin": 10.0,
                                    "ymax": 20.0,
                                },
                                "score": 0.800000011920929,
                            }
                        ],
                    }
                },
                "missing_predictions": {
                    "v1": {
                        "count": 1,
                        "examples": [
                            {
                                "datum": "uid2",
                                "groundtruth": {
                                    "xmin": 0.0,
                                    "xmax": 5.0,
                                    "ymin": 0.0,
                                    "ymax": 5.0,
                                },
                            }
                        ],
                    }
                },
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
                "hallucinations": {},
                "missing_predictions": {
                    "v1": {
                        "count": 2,
                        "examples": [
                            {
                                "datum": "uid1",
                                "groundtruth": {
                                    "xmin": 0.0,
                                    "xmax": 5.0,
                                    "ymin": 0.0,
                                    "ymax": 5.0,
                                },
                            }
                        ],
                    }
                },
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
            m["value"]["hallucinations"],
            m["value"]["missing_predictions"],
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

        actual_metrics = evaluator.compute_confusion_matrix(
            iou_thresholds=[0.5],
            score_thresholds=[0.0],
            number_of_examples=0,
            as_dict=True,
        )

        expected_metrics = [
            {
                "type": "ConfusionMatrix",
                "value": {
                    "confusion_matrix": {
                        "label1": {"label1": {"count": 1, "examples": []}},
                        "label2": {"label2": {"count": 1, "examples": []}},
                    },
                    "hallucinations": {
                        "label3": {"count": 1, "examples": []},
                        "label4": {"count": 1, "examples": []},
                    },
                    "missing_predictions": {
                        "label3": {"count": 1, "examples": []}
                    },
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
                m["value"]["hallucinations"],
                m["value"]["missing_predictions"],
            )
            assert m in expected_metrics
        for m in expected_metrics:
            assert m in actual_metrics
