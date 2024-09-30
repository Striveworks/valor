import numpy as np
from valor_lite.detection import DataLoader, Detection, Evaluator, MetricType
from valor_lite.detection.computation import compute_confusion_matrix


def test_confusion_matrix_no_data():
    evaluator = Evaluator()
    curves = evaluator._compute_confusion_matrix()
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
    cm_gt0_pd0 = np.array([3.0, 0.0, 0.0, 1.0, 0.9, 3.0, 4.0, 5.0, 0.1])
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
    misprd_gt0 = np.array([3.0, 3.0, 4.0, 1.0, 2.0])
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
    misprd_gt0 = np.array([3.0, 3.0, 4.0, 1.0, 2.0])
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
    misprd_gt0 = np.array([4.0, 0.0, 0.0, 3.0, 4.0])
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
        ("k1", "not_v2"),
        ("k1", "hallucination"),
    ]
    assert evaluator.missing_prediction_labels == [
        ("k1", "missed_detection"),
        ("k1", "v2"),
    ]
    assert evaluator.n_datums == 2
    assert evaluator.n_labels == 6
    assert evaluator.n_groundtruths == 4
    assert evaluator.n_predictions == 4

    metrics = evaluator.evaluate(
        iou_thresholds=[0.5],
        score_thresholds=[0.05, 0.3, 0.35, 0.45, 0.55, 0.95],
        number_of_examples=1,
        metrics_to_return=[MetricType.ConfusionMatrix],
    )

    # test ConfusionMatrix
    actual_metrics = [m.to_dict() for m in metrics[MetricType.ConfusionMatrix]]
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
                                    "groundtruth": rect1,
                                    "prediction": rect1,
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
                                "prediction": rect5,
                                "score": 0.30000001192092896,
                            }
                        ],
                    },
                    "hallucination": {
                        "count": 1,
                        "examples": [
                            {
                                "datum": "uid1",
                                "prediction": rect4,
                                "score": 0.10000000149011612,
                            }
                        ],
                    },
                    "low_iou": {
                        "count": 1,
                        "examples": [
                            {
                                "datum": "uid2",
                                "prediction": rect2,
                                "score": 0.5,
                            }
                        ],
                    },
                },
                "missing_predictions": {
                    "missed_detection": {
                        "count": 1,
                        "examples": [{"datum": "uid1", "groundtruth": rect2}],
                    },
                    "v2": {
                        "count": 1,
                        "examples": [{"datum": "uid1", "groundtruth": rect3}],
                    },
                    "low_iou": {
                        "count": 1,
                        "examples": [{"datum": "uid2", "groundtruth": rect1}],
                    },
                },
            },
            "parameters": {
                "score_threshold": 0.05,
                "iou_threshold": 0.5,
                "label_key": "k1",
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
                                    "groundtruth": rect1,
                                    "prediction": rect1,
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
                                "prediction": rect5,
                                "score": 0.30000001192092896,
                            }
                        ],
                    },
                    "low_iou": {
                        "count": 1,
                        "examples": [
                            {
                                "datum": "uid2",
                                "prediction": rect2,
                                "score": 0.5,
                            }
                        ],
                    },
                },
                "missing_predictions": {
                    "missed_detection": {
                        "count": 1,
                        "examples": [{"datum": "uid1", "groundtruth": rect2}],
                    },
                    "v2": {
                        "count": 1,
                        "examples": [{"datum": "uid1", "groundtruth": rect3}],
                    },
                    "low_iou": {
                        "count": 1,
                        "examples": [{"datum": "uid2", "groundtruth": rect1}],
                    },
                },
            },
            "parameters": {
                "score_threshold": 0.3,
                "iou_threshold": 0.5,
                "label_key": "k1",
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
                                    "groundtruth": rect1,
                                    "prediction": rect1,
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
                                "prediction": rect2,
                                "score": 0.5,
                            }
                        ],
                    }
                },
                "missing_predictions": {
                    "missed_detection": {
                        "count": 1,
                        "examples": [{"datum": "uid1", "groundtruth": rect2}],
                    },
                    "v2": {
                        "count": 1,
                        "examples": [{"datum": "uid1", "groundtruth": rect3}],
                    },
                    "low_iou": {
                        "count": 1,
                        "examples": [{"datum": "uid2", "groundtruth": rect1}],
                    },
                },
            },
            "parameters": {
                "score_threshold": 0.35,
                "iou_threshold": 0.5,
                "label_key": "k1",
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
                                    "groundtruth": rect1,
                                    "prediction": rect1,
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
                                "prediction": rect2,
                                "score": 0.5,
                            }
                        ],
                    }
                },
                "missing_predictions": {
                    "missed_detection": {
                        "count": 1,
                        "examples": [{"datum": "uid1", "groundtruth": rect2}],
                    },
                    "v2": {
                        "count": 1,
                        "examples": [{"datum": "uid1", "groundtruth": rect3}],
                    },
                    "low_iou": {
                        "count": 1,
                        "examples": [{"datum": "uid2", "groundtruth": rect1}],
                    },
                },
            },
            "parameters": {
                "score_threshold": 0.45,
                "iou_threshold": 0.5,
                "label_key": "k1",
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
                        "examples": [{"datum": "uid1", "groundtruth": rect1}],
                    },
                    "missed_detection": {
                        "count": 1,
                        "examples": [{"datum": "uid1", "groundtruth": rect2}],
                    },
                    "v2": {
                        "count": 1,
                        "examples": [{"datum": "uid1", "groundtruth": rect3}],
                    },
                    "low_iou": {
                        "count": 1,
                        "examples": [{"datum": "uid2", "groundtruth": rect1}],
                    },
                },
            },
            "parameters": {
                "score_threshold": 0.55,
                "iou_threshold": 0.5,
                "label_key": "k1",
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
                        "examples": [{"datum": "uid1", "groundtruth": rect1}],
                    },
                    "missed_detection": {
                        "count": 1,
                        "examples": [{"datum": "uid1", "groundtruth": rect2}],
                    },
                    "v2": {
                        "count": 1,
                        "examples": [{"datum": "uid1", "groundtruth": rect3}],
                    },
                    "low_iou": {
                        "count": 1,
                        "examples": [{"datum": "uid2", "groundtruth": rect1}],
                    },
                },
            },
            "parameters": {
                "score_threshold": 0.95,
                "iou_threshold": 0.5,
                "label_key": "k1",
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

    metrics = evaluator.evaluate(
        iou_thresholds=[0.45],
        score_thresholds=[0.05, 0.3, 0.35, 0.45, 0.55, 0.95],
        number_of_examples=1,
        metrics_to_return=[MetricType.ConfusionMatrix],
    )

    # test ConfusionMatrix
    actual_metrics = [m.to_dict() for m in metrics[MetricType.ConfusionMatrix]]
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
                                    "groundtruth": rect1,
                                    "prediction": rect1,
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
                                    "groundtruth": rect3,
                                    "prediction": rect5,
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
                                "prediction": rect4,
                                "score": 0.10000000149011612,
                            }
                        ],
                    },
                    "low_iou": {
                        "count": 1,
                        "examples": [
                            {
                                "datum": "uid2",
                                "prediction": rect2,
                                "score": 0.5,
                            }
                        ],
                    },
                },
                "missing_predictions": {
                    "missed_detection": {
                        "count": 1,
                        "examples": [{"datum": "uid1", "groundtruth": rect2}],
                    },
                    "low_iou": {
                        "count": 1,
                        "examples": [
                            {
                                "datum": "uid2",
                                "groundtruth": rect1,
                            }
                        ],
                    },
                },
            },
            "parameters": {
                "score_threshold": 0.05,
                "iou_threshold": 0.45,
                "label_key": "k1",
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
                                    "groundtruth": rect1,
                                    "prediction": rect1,
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
                                    "groundtruth": rect3,
                                    "prediction": rect5,
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
                                "prediction": rect2,
                                "score": 0.5,
                            }
                        ],
                    }
                },
                "missing_predictions": {
                    "missed_detection": {
                        "count": 1,
                        "examples": [{"datum": "uid1", "groundtruth": rect2}],
                    },
                    "low_iou": {
                        "count": 1,
                        "examples": [
                            {
                                "datum": "uid2",
                                "groundtruth": rect1,
                            }
                        ],
                    },
                },
            },
            "parameters": {
                "score_threshold": 0.3,
                "iou_threshold": 0.45,
                "label_key": "k1",
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
                                    "groundtruth": rect1,
                                    "prediction": rect1,
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
                                "prediction": rect2,
                                "score": 0.5,
                            }
                        ],
                    }
                },
                "missing_predictions": {
                    "missed_detection": {
                        "count": 1,
                        "examples": [{"datum": "uid1", "groundtruth": rect2}],
                    },
                    "v2": {
                        "count": 1,
                        "examples": [{"datum": "uid1", "groundtruth": rect3}],
                    },
                    "low_iou": {
                        "count": 1,
                        "examples": [
                            {
                                "datum": "uid2",
                                "groundtruth": rect1,
                            }
                        ],
                    },
                },
            },
            "parameters": {
                "score_threshold": 0.35,
                "iou_threshold": 0.45,
                "label_key": "k1",
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
                                    "groundtruth": rect1,
                                    "prediction": rect1,
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
                                "prediction": rect2,
                                "score": 0.5,
                            }
                        ],
                    }
                },
                "missing_predictions": {
                    "missed_detection": {
                        "count": 1,
                        "examples": [{"datum": "uid1", "groundtruth": rect2}],
                    },
                    "v2": {
                        "count": 1,
                        "examples": [{"datum": "uid1", "groundtruth": rect3}],
                    },
                    "low_iou": {
                        "count": 1,
                        "examples": [
                            {
                                "datum": "uid2",
                                "groundtruth": rect1,
                            }
                        ],
                    },
                },
            },
            "parameters": {
                "score_threshold": 0.45,
                "iou_threshold": 0.45,
                "label_key": "k1",
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
                                "groundtruth": rect1,
                            }
                        ],
                    },
                    "missed_detection": {
                        "count": 1,
                        "examples": [{"datum": "uid1", "groundtruth": rect2}],
                    },
                    "v2": {
                        "count": 1,
                        "examples": [{"datum": "uid1", "groundtruth": rect3}],
                    },
                    "low_iou": {
                        "count": 1,
                        "examples": [
                            {
                                "datum": "uid2",
                                "groundtruth": rect1,
                            }
                        ],
                    },
                },
            },
            "parameters": {
                "score_threshold": 0.55,
                "iou_threshold": 0.45,
                "label_key": "k1",
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
                                "groundtruth": rect1,
                            }
                        ],
                    },
                    "missed_detection": {
                        "count": 1,
                        "examples": [{"datum": "uid1", "groundtruth": rect2}],
                    },
                    "v2": {
                        "count": 1,
                        "examples": [{"datum": "uid1", "groundtruth": rect3}],
                    },
                    "low_iou": {
                        "count": 1,
                        "examples": [
                            {
                                "datum": "uid2",
                                "groundtruth": rect1,
                            }
                        ],
                    },
                },
            },
            "parameters": {
                "score_threshold": 0.95,
                "iou_threshold": 0.45,
                "label_key": "k1",
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

    assert evaluator.ignored_prediction_labels == [("class", "3")]
    assert evaluator.missing_prediction_labels == []
    assert evaluator.n_datums == 4
    assert evaluator.n_labels == 6
    assert evaluator.n_groundtruths == 20
    assert evaluator.n_predictions == 19

    metrics = evaluator.evaluate(
        iou_thresholds=[0.5, 0.9],
        score_thresholds=[0.05, 0.25, 0.35, 0.55, 0.75, 0.8, 0.85, 0.95],
        number_of_examples=1,
        metrics_to_return=[MetricType.ConfusionMatrix],
    )

    assert len(metrics[MetricType.ConfusionMatrix]) == 16

    uid0_gt_0 = (214.125, 562.5, 41.28125, 285.0)
    uid1_gt_0 = (13.0, 549.0, 22.75, 632.5)
    uid2_gt_1 = (2.75, 162.125, 3.66015625, 316.0)
    uid2_gt_2 = (295.5, 314.0, 93.9375, 152.75)
    uid2_gt_4 = (356.5, 372.25, 95.5, 147.5)
    uid3_gt_0 = (72.9375, 91.25, 45.96875, 80.5625)
    uid3_gt_1 = (50.15625, 71.25, 45.34375, 79.8125)
    uid3_gt_5 = (56.375, 75.6875, 21.65625, 45.53125)

    uid0_pd_0 = (258.25, 606.5, 41.28125, 285.0)
    uid1_pd_0 = (61.0, 565.0, 22.75, 632.5)
    uid1_pd_1 = (12.65625, 281.25, 3.3203125, 275.25)
    uid2_pd_0 = (87.875, 384.25, 276.25, 379.5)
    uid2_pd_1 = (0.0, 142.125, 3.66015625, 316.0)
    uid2_pd_2 = (296.5, 315.0, 93.9375, 152.75)
    uid2_pd_3 = (329.0, 342.5, 97.0625, 123.0)
    uid2_pd_4 = (356.5, 372.25, 95.5, 147.5)
    uid2_pd_5 = (464.0, 495.75, 105.0625, 147.0)
    uid2_pd_6 = (276.0, 291.5, 103.8125, 150.75)
    uid3_pd_0 = (72.9375, 91.25, 45.96875, 80.5625)
    uid3_pd_1 = (45.15625, 66.25, 45.34375, 79.8125)
    uid3_pd_2 = (82.25, 99.6875, 47.03125, 78.5)
    uid3_pd_4 = (75.3125, 91.875, 23.015625, 50.84375)

    # test ConfusionMatrix
    actual_metrics = [m.to_dict() for m in metrics[MetricType.ConfusionMatrix]]
    expected_metrics = [
        {
            "type": "ConfusionMatrix",
            "value": {
                "confusion_matrix": {
                    "4": {
                        "4": {
                            "count": 2,
                            "examples": [
                                {
                                    "datum": "0",
                                    "groundtruth": uid0_gt_0,
                                    "prediction": uid0_pd_0,
                                    "score": 0.23600000143051147,
                                }
                            ],
                        }
                    },
                    "2": {
                        "2": {
                            "count": 1,
                            "examples": [
                                {
                                    "datum": "1",
                                    "groundtruth": [
                                        1.66015625,
                                        270.25,
                                        3.3203125,
                                        275.25,
                                    ],
                                    "prediction": uid1_pd_1,
                                    "score": 0.7260000109672546,
                                }
                            ],
                        },
                        "3": {
                            "count": 1,
                            "examples": [
                                {
                                    "datum": "1",
                                    "groundtruth": uid1_gt_0,
                                    "prediction": uid1_pd_0,
                                    "score": 0.3179999887943268,
                                }
                            ],
                        },
                    },
                    "1": {
                        "1": {
                            "count": 1,
                            "examples": [
                                {
                                    "datum": "2",
                                    "groundtruth": uid2_gt_2,
                                    "prediction": uid2_pd_2,
                                    "score": 0.40700000524520874,
                                }
                            ],
                        }
                    },
                    "0": {
                        "0": {
                            "count": 5,
                            "examples": [
                                {
                                    "datum": "3",
                                    "groundtruth": uid3_gt_0,
                                    "prediction": uid3_pd_0,
                                    "score": 0.5320000052452087,
                                }
                            ],
                        }
                    },
                    "49": {"49": {"count": 11, "examples": []}},
                },
                "hallucinations": {},
                "missing_predictions": {
                    "49": {
                        "count": 4,
                        "examples": [
                            {
                                "datum": "3",
                                "groundtruth": uid3_gt_5,
                            }
                        ],
                    }
                },
            },
            "parameters": {
                "score_threshold": 0.05,
                "iou_threshold": 0.5,
                "label_key": "class",
            },
        },
        {
            "type": "ConfusionMatrix",
            "value": {
                "confusion_matrix": {
                    "4": {
                        "4": {
                            "count": 1,
                            "examples": [
                                {
                                    "datum": "1",
                                    "groundtruth": [
                                        1.66015625,
                                        270.25,
                                        3.3203125,
                                        275.25,
                                    ],
                                    "prediction": uid1_pd_1,
                                    "score": 0.7260000109672546,
                                }
                            ],
                        }
                    },
                    "2": {
                        "2": {
                            "count": 1,
                            "examples": [
                                {
                                    "datum": "2",
                                    "groundtruth": [
                                        61.875,
                                        358.25,
                                        276.25,
                                        379.5,
                                    ],
                                    "prediction": uid2_pd_0,
                                    "score": 0.5460000038146973,
                                }
                            ],
                        },
                        "3": {
                            "count": 1,
                            "examples": [
                                {
                                    "datum": "1",
                                    "groundtruth": uid1_gt_0,
                                    "prediction": uid1_pd_0,
                                    "score": 0.3179999887943268,
                                }
                            ],
                        },
                    },
                    "1": {
                        "1": {
                            "count": 1,
                            "examples": [
                                {
                                    "datum": "2",
                                    "groundtruth": uid2_gt_1,
                                    "prediction": uid2_pd_1,
                                    "score": 0.30000001192092896,
                                }
                            ],
                        }
                    },
                    "0": {
                        "0": {
                            "count": 5,
                            "examples": [
                                {
                                    "datum": "2",
                                    "groundtruth": uid2_gt_2,
                                    "prediction": uid2_pd_2,
                                    "score": 0.40700000524520874,
                                }
                            ],
                        }
                    },
                    "49": {"49": {"count": 6, "examples": []}},
                },
                "hallucinations": {},
                "missing_predictions": {
                    "4": {
                        "count": 1,
                        "examples": [
                            {
                                "datum": "0",
                                "groundtruth": uid0_gt_0,
                            }
                        ],
                    },
                    "49": {
                        "count": 21,
                        "examples": [
                            {
                                "datum": "3",
                                "groundtruth": [
                                    63.96875,
                                    84.375,
                                    46.15625,
                                    80.5,
                                ],
                            }
                        ],
                    },
                },
            },
            "parameters": {
                "score_threshold": 0.25,
                "iou_threshold": 0.5,
                "label_key": "class",
            },
        },
        {
            "type": "ConfusionMatrix",
            "value": {
                "confusion_matrix": {
                    "4": {
                        "4": {
                            "count": 1,
                            "examples": [
                                {
                                    "datum": "1",
                                    "groundtruth": [
                                        1.66015625,
                                        270.25,
                                        3.3203125,
                                        275.25,
                                    ],
                                    "prediction": uid1_pd_1,
                                    "score": 0.7260000109672546,
                                }
                            ],
                        }
                    },
                    "2": {
                        "2": {
                            "count": 1,
                            "examples": [
                                {
                                    "datum": "2",
                                    "groundtruth": [
                                        61.875,
                                        358.25,
                                        276.25,
                                        379.5,
                                    ],
                                    "prediction": uid2_pd_0,
                                    "score": 0.5460000038146973,
                                }
                            ],
                        }
                    },
                    "0": {
                        "0": {
                            "count": 4,
                            "examples": [
                                {
                                    "datum": "2",
                                    "groundtruth": uid2_gt_2,
                                    "prediction": uid2_pd_2,
                                    "score": 0.40700000524520874,
                                }
                            ],
                        }
                    },
                    "49": {"49": {"count": 4, "examples": []}},
                },
                "hallucinations": {},
                "missing_predictions": {
                    "4": {
                        "count": 1,
                        "examples": [
                            {
                                "datum": "0",
                                "groundtruth": uid0_gt_0,
                            }
                        ],
                    },
                    "2": {
                        "count": 2,
                        "examples": [
                            {
                                "datum": "1",
                                "groundtruth": uid1_gt_0,
                            }
                        ],
                    },
                    "1": {
                        "count": 2,
                        "examples": [
                            {
                                "datum": "2",
                                "groundtruth": uid2_gt_4,
                            }
                        ],
                    },
                    "0": {"count": 1, "examples": []},
                    "49": {"count": 28, "examples": []},
                },
            },
            "parameters": {
                "score_threshold": 0.35,
                "iou_threshold": 0.5,
                "label_key": "class",
            },
        },
        {
            "type": "ConfusionMatrix",
            "value": {
                "confusion_matrix": {
                    "2": {
                        "2": {
                            "count": 1,
                            "examples": [
                                {
                                    "datum": "1",
                                    "groundtruth": [
                                        1.66015625,
                                        270.25,
                                        3.3203125,
                                        275.25,
                                    ],
                                    "prediction": uid1_pd_1,
                                    "score": 0.7260000109672546,
                                }
                            ],
                        }
                    },
                    "0": {
                        "0": {
                            "count": 3,
                            "examples": [
                                {
                                    "datum": "2",
                                    "groundtruth": [
                                        327.0,
                                        340.5,
                                        97.0625,
                                        123.0,
                                    ],
                                    "prediction": uid2_pd_3,
                                    "score": 0.6110000014305115,
                                }
                            ],
                        }
                    },
                    "49": {"49": {"count": 3, "examples": []}},
                },
                "hallucinations": {},
                "missing_predictions": {
                    "4": {
                        "count": 3,
                        "examples": [
                            {
                                "datum": "0",
                                "groundtruth": uid0_gt_0,
                            }
                        ],
                    },
                    "2": {
                        "count": 2,
                        "examples": [
                            {
                                "datum": "1",
                                "groundtruth": uid1_gt_0,
                            }
                        ],
                    },
                    "1": {
                        "count": 2,
                        "examples": [
                            {
                                "datum": "3",
                                "groundtruth": uid3_gt_0,
                            }
                        ],
                    },
                    "0": {"count": 2, "examples": []},
                    "49": {"count": 33, "examples": []},
                },
            },
            "parameters": {
                "score_threshold": 0.55,
                "iou_threshold": 0.5,
                "label_key": "class",
            },
        },
        {
            "type": "ConfusionMatrix",
            "value": {
                "confusion_matrix": {
                    "0": {
                        "0": {
                            "count": 2,
                            "examples": [
                                {
                                    "datum": "2",
                                    "groundtruth": [
                                        462.0,
                                        493.75,
                                        105.0625,
                                        147.0,
                                    ],
                                    "prediction": uid2_pd_5,
                                    "score": 0.8050000071525574,
                                }
                            ],
                        }
                    },
                    "49": {"49": {"count": 2, "examples": []}},
                },
                "hallucinations": {},
                "missing_predictions": {
                    "4": {
                        "count": 3,
                        "examples": [
                            {
                                "datum": "0",
                                "groundtruth": uid0_gt_0,
                            }
                        ],
                    },
                    "2": {
                        "count": 4,
                        "examples": [
                            {
                                "datum": "1",
                                "groundtruth": uid1_gt_0,
                            }
                        ],
                    },
                    "1": {"count": 2, "examples": []},
                    "0": {"count": 3, "examples": []},
                    "49": {"count": 38, "examples": []},
                },
            },
            "parameters": {
                "score_threshold": 0.75,
                "iou_threshold": 0.5,
                "label_key": "class",
            },
        },
        {
            "type": "ConfusionMatrix",
            "value": {
                "confusion_matrix": {
                    "0": {
                        "0": {
                            "count": 2,
                            "examples": [
                                {
                                    "datum": "2",
                                    "groundtruth": [
                                        462.0,
                                        493.75,
                                        105.0625,
                                        147.0,
                                    ],
                                    "prediction": uid2_pd_5,
                                    "score": 0.8050000071525574,
                                }
                            ],
                        }
                    },
                    "49": {"49": {"count": 1, "examples": []}},
                },
                "hallucinations": {},
                "missing_predictions": {
                    "4": {
                        "count": 3,
                        "examples": [
                            {
                                "datum": "0",
                                "groundtruth": uid0_gt_0,
                            }
                        ],
                    },
                    "2": {
                        "count": 4,
                        "examples": [
                            {
                                "datum": "1",
                                "groundtruth": uid1_gt_0,
                            }
                        ],
                    },
                    "1": {"count": 2, "examples": []},
                    "0": {"count": 3, "examples": []},
                    "49": {"count": 41, "examples": []},
                },
            },
            "parameters": {
                "score_threshold": 0.8,
                "iou_threshold": 0.5,
                "label_key": "class",
            },
        },
        {
            "type": "ConfusionMatrix",
            "value": {
                "confusion_matrix": {
                    "0": {
                        "0": {
                            "count": 1,
                            "examples": [
                                {
                                    "datum": "2",
                                    "groundtruth": [
                                        277.0,
                                        292.5,
                                        103.8125,
                                        150.75,
                                    ],
                                    "prediction": uid2_pd_6,
                                    "score": 0.953000009059906,
                                }
                            ],
                        }
                    },
                    "49": {
                        "49": {
                            "count": 1,
                            "examples": [
                                {
                                    "datum": "3",
                                    "groundtruth": [
                                        75.3125,
                                        91.875,
                                        23.015625,
                                        50.84375,
                                    ],
                                    "prediction": uid3_pd_4,
                                    "score": 0.8830000162124634,
                                }
                            ],
                        }
                    },
                },
                "hallucinations": {},
                "missing_predictions": {
                    "4": {
                        "count": 3,
                        "examples": [
                            {
                                "datum": "0",
                                "groundtruth": uid0_gt_0,
                            }
                        ],
                    },
                    "2": {
                        "count": 4,
                        "examples": [
                            {
                                "datum": "1",
                                "groundtruth": uid1_gt_0,
                            }
                        ],
                    },
                    "1": {"count": 2, "examples": []},
                    "0": {"count": 4, "examples": []},
                    "49": {"count": 41, "examples": []},
                },
            },
            "parameters": {
                "score_threshold": 0.85,
                "iou_threshold": 0.5,
                "label_key": "class",
            },
        },
        {
            "type": "ConfusionMatrix",
            "value": {
                "confusion_matrix": {
                    "0": {
                        "0": {
                            "count": 1,
                            "examples": [
                                {
                                    "datum": "2",
                                    "groundtruth": [
                                        277.0,
                                        292.5,
                                        103.8125,
                                        150.75,
                                    ],
                                    "prediction": uid2_pd_6,
                                    "score": 0.953000009059906,
                                }
                            ],
                        }
                    }
                },
                "hallucinations": {},
                "missing_predictions": {
                    "4": {
                        "count": 3,
                        "examples": [
                            {
                                "datum": "0",
                                "groundtruth": uid0_gt_0,
                            }
                        ],
                    },
                    "2": {
                        "count": 4,
                        "examples": [
                            {
                                "datum": "1",
                                "groundtruth": uid1_gt_0,
                            }
                        ],
                    },
                    "1": {"count": 2, "examples": []},
                    "0": {"count": 4, "examples": []},
                    "49": {"count": 46, "examples": []},
                },
            },
            "parameters": {
                "score_threshold": 0.95,
                "iou_threshold": 0.5,
                "label_key": "class",
            },
        },
        {
            "type": "ConfusionMatrix",
            "value": {
                "confusion_matrix": {
                    "2": {
                        "2": {
                            "count": 1,
                            "examples": [
                                {
                                    "datum": "1",
                                    "groundtruth": [
                                        1.66015625,
                                        270.25,
                                        3.3203125,
                                        275.25,
                                    ],
                                    "prediction": uid1_pd_1,
                                    "score": 0.7260000109672546,
                                }
                            ],
                        }
                    },
                    "0": {
                        "0": {
                            "count": 1,
                            "examples": [
                                {
                                    "datum": "2",
                                    "groundtruth": uid2_gt_4,
                                    "prediction": uid2_pd_4,
                                    "score": 0.33500000834465027,
                                }
                            ],
                        }
                    },
                    "49": {
                        "49": {
                            "count": 2,
                            "examples": [
                                {
                                    "datum": "3",
                                    "groundtruth": uid3_gt_0,
                                    "prediction": uid3_pd_0,
                                    "score": 0.5320000052452087,
                                }
                            ],
                        }
                    },
                },
                "hallucinations": {
                    "4": {
                        "count": 3,
                        "examples": [
                            {
                                "datum": "0",
                                "prediction": [258.25, 606.5, 41.28125, 285.0],
                                "score": 0.23600000143051147,
                            }
                        ],
                    },
                    "3": {
                        "count": 2,
                        "examples": [
                            {
                                "datum": "1",
                                "prediction": [61.0, 565.0, 22.75, 632.5],
                                "score": 0.3179999887943268,
                            }
                        ],
                    },
                    "1": {"count": 2, "examples": []},
                    "0": {"count": 4, "examples": []},
                    "49": {"count": 35, "examples": []},
                },
                "missing_predictions": {
                    "4": {
                        "count": 3,
                        "examples": [
                            {
                                "datum": "0",
                                "groundtruth": uid0_gt_0,
                            }
                        ],
                    },
                    "2": {
                        "count": 2,
                        "examples": [
                            {
                                "datum": "1",
                                "groundtruth": uid1_gt_0,
                            }
                        ],
                    },
                    "1": {
                        "count": 2,
                        "examples": [
                            {
                                "datum": "3",
                                "groundtruth": [
                                    81.25,
                                    98.6875,
                                    47.03125,
                                    78.5,
                                ],
                            }
                        ],
                    },
                    "0": {"count": 4, "examples": []},
                    "49": {"count": 36, "examples": []},
                },
            },
            "parameters": {
                "score_threshold": 0.05,
                "iou_threshold": 0.9,
                "label_key": "class",
            },
        },
        {
            "type": "ConfusionMatrix",
            "value": {
                "confusion_matrix": {
                    "2": {
                        "2": {
                            "count": 1,
                            "examples": [
                                {
                                    "datum": "1",
                                    "groundtruth": [
                                        1.66015625,
                                        270.25,
                                        3.3203125,
                                        275.25,
                                    ],
                                    "prediction": uid1_pd_1,
                                    "score": 0.7260000109672546,
                                }
                            ],
                        }
                    },
                    "0": {
                        "0": {
                            "count": 1,
                            "examples": [
                                {
                                    "datum": "2",
                                    "groundtruth": uid2_gt_4,
                                    "prediction": uid2_pd_4,
                                    "score": 0.33500000834465027,
                                }
                            ],
                        }
                    },
                    "49": {
                        "49": {
                            "count": 2,
                            "examples": [
                                {
                                    "datum": "3",
                                    "groundtruth": uid3_gt_0,
                                    "prediction": uid3_pd_0,
                                    "score": 0.5320000052452087,
                                }
                            ],
                        }
                    },
                },
                "hallucinations": {
                    "4": {
                        "count": 2,
                        "examples": [
                            {
                                "datum": "2",
                                "prediction": [
                                    0.0,
                                    142.125,
                                    3.66015625,
                                    316.0,
                                ],
                                "score": 0.30000001192092896,
                            }
                        ],
                    },
                    "3": {
                        "count": 2,
                        "examples": [
                            {
                                "datum": "2",
                                "prediction": uid2_pd_0,
                                "score": 0.5460000038146973,
                            }
                        ],
                    },
                    "1": {
                        "count": 2,
                        "examples": [
                            {
                                "datum": "3",
                                "prediction": uid3_pd_2,
                                "score": 0.7820000052452087,
                            }
                        ],
                    },
                    "0": {"count": 4, "examples": []},
                    "49": {"count": 19, "examples": []},
                },
                "missing_predictions": {
                    "4": {
                        "count": 3,
                        "examples": [
                            {
                                "datum": "0",
                                "groundtruth": uid0_gt_0,
                            }
                        ],
                    },
                    "2": {
                        "count": 2,
                        "examples": [
                            {
                                "datum": "1",
                                "groundtruth": uid1_gt_0,
                            }
                        ],
                    },
                    "1": {
                        "count": 2,
                        "examples": [
                            {
                                "datum": "3",
                                "groundtruth": [
                                    81.25,
                                    98.6875,
                                    47.03125,
                                    78.5,
                                ],
                            }
                        ],
                    },
                    "0": {"count": 4, "examples": []},
                    "49": {"count": 36, "examples": []},
                },
            },
            "parameters": {
                "score_threshold": 0.25,
                "iou_threshold": 0.9,
                "label_key": "class",
            },
        },
        {
            "type": "ConfusionMatrix",
            "value": {
                "confusion_matrix": {
                    "2": {
                        "2": {
                            "count": 1,
                            "examples": [
                                {
                                    "datum": "1",
                                    "groundtruth": [
                                        1.66015625,
                                        270.25,
                                        3.3203125,
                                        275.25,
                                    ],
                                    "prediction": uid1_pd_1,
                                    "score": 0.7260000109672546,
                                }
                            ],
                        }
                    },
                    "49": {
                        "49": {
                            "count": 2,
                            "examples": [
                                {
                                    "datum": "3",
                                    "groundtruth": uid3_gt_0,
                                    "prediction": uid3_pd_0,
                                    "score": 0.5320000052452087,
                                }
                            ],
                        }
                    },
                },
                "hallucinations": {
                    "4": {
                        "count": 2,
                        "examples": [
                            {
                                "datum": "2",
                                "prediction": uid2_pd_0,
                                "score": 0.5460000038146973,
                            }
                        ],
                    },
                    "0": {
                        "count": 4,
                        "examples": [
                            {
                                "datum": "3",
                                "prediction": uid3_pd_2,
                                "score": 0.7820000052452087,
                            }
                        ],
                    },
                    "49": {"count": 10, "examples": []},
                },
                "missing_predictions": {
                    "4": {
                        "count": 3,
                        "examples": [
                            {
                                "datum": "0",
                                "groundtruth": uid0_gt_0,
                            }
                        ],
                    },
                    "2": {
                        "count": 2,
                        "examples": [
                            {
                                "datum": "1",
                                "groundtruth": uid1_gt_0,
                            }
                        ],
                    },
                    "1": {
                        "count": 2,
                        "examples": [
                            {
                                "datum": "3",
                                "groundtruth": [
                                    81.25,
                                    98.6875,
                                    47.03125,
                                    78.5,
                                ],
                            }
                        ],
                    },
                    "0": {"count": 5, "examples": []},
                    "49": {"count": 36, "examples": []},
                },
            },
            "parameters": {
                "score_threshold": 0.35,
                "iou_threshold": 0.9,
                "label_key": "class",
            },
        },
        {
            "type": "ConfusionMatrix",
            "value": {
                "confusion_matrix": {
                    "2": {
                        "2": {
                            "count": 1,
                            "examples": [
                                {
                                    "datum": "1",
                                    "groundtruth": [
                                        1.66015625,
                                        270.25,
                                        3.3203125,
                                        275.25,
                                    ],
                                    "prediction": uid1_pd_1,
                                    "score": 0.7260000109672546,
                                }
                            ],
                        }
                    },
                    "49": {
                        "49": {
                            "count": 1,
                            "examples": [
                                {
                                    "datum": "3",
                                    "groundtruth": [
                                        75.3125,
                                        91.875,
                                        23.015625,
                                        50.84375,
                                    ],
                                    "prediction": uid3_pd_4,
                                    "score": 0.8830000162124634,
                                }
                            ],
                        }
                    },
                },
                "hallucinations": {
                    "0": {
                        "count": 3,
                        "examples": [
                            {
                                "datum": "2",
                                "prediction": uid2_pd_3,
                                "score": 0.6110000014305115,
                            }
                        ],
                    },
                    "49": {"count": 10, "examples": []},
                },
                "missing_predictions": {
                    "4": {
                        "count": 3,
                        "examples": [
                            {
                                "datum": "0",
                                "groundtruth": uid0_gt_0,
                            }
                        ],
                    },
                    "2": {
                        "count": 2,
                        "examples": [
                            {
                                "datum": "1",
                                "groundtruth": uid1_gt_0,
                            }
                        ],
                    },
                    "1": {
                        "count": 2,
                        "examples": [
                            {
                                "datum": "3",
                                "groundtruth": uid3_gt_0,
                            }
                        ],
                    },
                    "0": {"count": 5, "examples": []},
                    "49": {"count": 41, "examples": []},
                },
            },
            "parameters": {
                "score_threshold": 0.55,
                "iou_threshold": 0.9,
                "label_key": "class",
            },
        },
        {
            "type": "ConfusionMatrix",
            "value": {
                "confusion_matrix": {
                    "49": {
                        "49": {
                            "count": 1,
                            "examples": [
                                {
                                    "datum": "3",
                                    "groundtruth": [
                                        75.3125,
                                        91.875,
                                        23.015625,
                                        50.84375,
                                    ],
                                    "prediction": uid3_pd_4,
                                    "score": 0.8830000162124634,
                                }
                            ],
                        }
                    }
                },
                "hallucinations": {
                    "0": {
                        "count": 2,
                        "examples": [
                            {
                                "datum": "2",
                                "prediction": uid2_pd_5,
                                "score": 0.8050000071525574,
                            }
                        ],
                    },
                    "49": {"count": 4, "examples": []},
                },
                "missing_predictions": {
                    "4": {
                        "count": 3,
                        "examples": [
                            {
                                "datum": "0",
                                "groundtruth": uid0_gt_0,
                            }
                        ],
                    },
                    "2": {
                        "count": 4,
                        "examples": [
                            {
                                "datum": "1",
                                "groundtruth": uid1_gt_0,
                            }
                        ],
                    },
                    "1": {"count": 2, "examples": []},
                    "0": {"count": 5, "examples": []},
                    "49": {"count": 41, "examples": []},
                },
            },
            "parameters": {
                "score_threshold": 0.75,
                "iou_threshold": 0.9,
                "label_key": "class",
            },
        },
        {
            "type": "ConfusionMatrix",
            "value": {
                "confusion_matrix": {
                    "49": {
                        "49": {
                            "count": 1,
                            "examples": [
                                {
                                    "datum": "3",
                                    "groundtruth": [
                                        75.3125,
                                        91.875,
                                        23.015625,
                                        50.84375,
                                    ],
                                    "prediction": uid3_pd_4,
                                    "score": 0.8830000162124634,
                                }
                            ],
                        }
                    }
                },
                "hallucinations": {
                    "0": {
                        "count": 2,
                        "examples": [
                            {
                                "datum": "2",
                                "prediction": uid2_pd_5,
                                "score": 0.8050000071525574,
                            }
                        ],
                    }
                },
                "missing_predictions": {
                    "4": {
                        "count": 3,
                        "examples": [
                            {
                                "datum": "0",
                                "groundtruth": uid0_gt_0,
                            }
                        ],
                    },
                    "2": {
                        "count": 4,
                        "examples": [
                            {
                                "datum": "1",
                                "groundtruth": uid1_gt_0,
                            }
                        ],
                    },
                    "1": {"count": 2, "examples": []},
                    "0": {"count": 5, "examples": []},
                    "49": {"count": 41, "examples": []},
                },
            },
            "parameters": {
                "score_threshold": 0.8,
                "iou_threshold": 0.9,
                "label_key": "class",
            },
        },
        {
            "type": "ConfusionMatrix",
            "value": {
                "confusion_matrix": {
                    "49": {
                        "49": {
                            "count": 1,
                            "examples": [
                                {
                                    "datum": "3",
                                    "groundtruth": [
                                        75.3125,
                                        91.875,
                                        23.015625,
                                        50.84375,
                                    ],
                                    "prediction": uid3_pd_4,
                                    "score": 0.8830000162124634,
                                }
                            ],
                        }
                    }
                },
                "hallucinations": {
                    "0": {
                        "count": 1,
                        "examples": [
                            {
                                "datum": "2",
                                "prediction": uid2_pd_6,
                                "score": 0.953000009059906,
                            }
                        ],
                    }
                },
                "missing_predictions": {
                    "4": {
                        "count": 3,
                        "examples": [
                            {
                                "datum": "0",
                                "groundtruth": uid0_gt_0,
                            }
                        ],
                    },
                    "2": {
                        "count": 4,
                        "examples": [
                            {
                                "datum": "1",
                                "groundtruth": uid1_gt_0,
                            }
                        ],
                    },
                    "1": {"count": 2, "examples": []},
                    "0": {"count": 5, "examples": []},
                    "49": {"count": 41, "examples": []},
                },
            },
            "parameters": {
                "score_threshold": 0.85,
                "iou_threshold": 0.9,
                "label_key": "class",
            },
        },
        {
            "type": "ConfusionMatrix",
            "value": {
                "confusion_matrix": {},
                "hallucinations": {
                    "0": {
                        "count": 1,
                        "examples": [
                            {
                                "datum": "2",
                                "prediction": uid2_pd_6,
                                "score": 0.953000009059906,
                            }
                        ],
                    }
                },
                "missing_predictions": {
                    "4": {
                        "count": 3,
                        "examples": [
                            {
                                "datum": "0",
                                "groundtruth": uid0_gt_0,
                            }
                        ],
                    },
                    "2": {
                        "count": 4,
                        "examples": [
                            {
                                "datum": "1",
                                "groundtruth": uid1_gt_0,
                            }
                        ],
                    },
                    "1": {"count": 2, "examples": []},
                    "0": {"count": 5, "examples": []},
                    "49": {"count": 46, "examples": []},
                },
            },
            "parameters": {
                "score_threshold": 0.95,
                "iou_threshold": 0.9,
                "label_key": "class",
            },
        },
    ]

    for m in actual_metrics:
        _filter_out_zero_counts(
            m["value"]["confusion_matrix"],
            m["value"]["hallucinations"],
            m["value"]["missing_predictions"],
        )
        import json

        print(json.dumps(m, indent=4) + ",")
        # assert m in expected_metrics
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

    metrics = evaluator.evaluate(
        iou_thresholds=[0.5],
        score_thresholds=[0.5, 0.85],
        number_of_examples=1,
        metrics_to_return=[MetricType.ConfusionMatrix],
    )

    assert len(metrics[MetricType.ConfusionMatrix]) == 1

    # test ConfusionMatrix
    actual_metrics = [m.to_dict() for m in metrics[MetricType.ConfusionMatrix]]
    expected_metrics = [
        {
            "type": "ConfusionMatrix",
            "value": {
                "tp": [1, 0],
                "fp_misclassification": [0, 0],
                "fp_hallucination": [1, 0],
                "fn_misclassification": [0, 0],
                "fn_missing_prediction": [1, 2],
                "tp_examples": [[("uid1", (0.0, 5.0, 0.0, 5.0))], []],
                "fp_misclassification_examples": [[], []],
                "fp_hallucination_examples": [
                    [("uid2", (10.0, 20.0, 10.0, 20.0))],
                    [],
                ],
                "fn_misclassification_examples": [[], []],
                "fn_missing_prediction_examples": [
                    [("uid2", (0.0, 5.0, 0.0, 5.0))],
                    [("uid1", (0.0, 5.0, 0.0, 5.0))],
                ],
            },
            "parameters": {
                "score_thresholds": [0.5, 0.85],
                "iou_threshold": 0.5,
                "label": {"key": "k1", "value": "v1"},
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
                ("class", "label4"),
            ],
            "missing_prediction_labels": [],
            "n_datums": 1,
            "n_groundtruths": 3,
            "n_labels": 4,
            "n_predictions": 4,
        }

        metrics = evaluator.evaluate(
            iou_thresholds=[0.5],
            score_thresholds=[0.0],
            number_of_examples=0,
            metrics_to_return=[MetricType.ConfusionMatrix],
        )

        actual_metrics = [
            m.to_dict() for m in metrics[MetricType.ConfusionMatrix]
        ]
        expected_metrics = [
            {
                "type": "ConfusionMatrix",
                "value": {
                    "tp": [1],
                    "fp_misclassification": [0],
                    "fp_hallucination": [0],
                    "fn_misclassification": [0],
                    "fn_missing_prediction": [0],
                    "tp_examples": [[]],
                    "fp_misclassification_examples": [[]],
                    "fp_hallucination_examples": [[]],
                    "fn_misclassification_examples": [[]],
                    "fn_missing_prediction_examples": [[]],
                },
                "parameters": {
                    "score_thresholds": [0.0],
                    "iou_threshold": 0.5,
                    "label": {"key": "class", "value": "label1"},
                },
            },
            {
                "type": "ConfusionMatrix",
                "value": {
                    "tp": [1],
                    "fp_misclassification": [0],
                    "fp_hallucination": [0],
                    "fn_misclassification": [0],
                    "fn_missing_prediction": [0],
                    "tp_examples": [[]],
                    "fp_misclassification_examples": [[]],
                    "fp_hallucination_examples": [[]],
                    "fn_misclassification_examples": [[]],
                    "fn_missing_prediction_examples": [[]],
                },
                "parameters": {
                    "score_thresholds": [0.0],
                    "iou_threshold": 0.5,
                    "label": {"key": "class", "value": "label2"},
                },
            },
            {
                "type": "ConfusionMatrix",
                "value": {
                    "tp": [0],
                    "fp_misclassification": [0],
                    "fp_hallucination": [1],
                    "fn_misclassification": [1],
                    "fn_missing_prediction": [0],
                    "tp_examples": [[]],
                    "fp_misclassification_examples": [[]],
                    "fp_hallucination_examples": [[]],
                    "fn_misclassification_examples": [[]],
                    "fn_missing_prediction_examples": [[]],
                },
                "parameters": {
                    "score_thresholds": [0.0],
                    "iou_threshold": 0.5,
                    "label": {"key": "class", "value": "label3"},
                },
            },
            {
                "type": "ConfusionMatrix",
                "value": {
                    "tp": [0],
                    "fp_misclassification": [0],
                    "fp_hallucination": [1],
                    "fn_misclassification": [0],
                    "fn_missing_prediction": [0],
                    "tp_examples": [[]],
                    "fp_misclassification_examples": [[]],
                    "fp_hallucination_examples": [[]],
                    "fn_misclassification_examples": [[]],
                    "fn_missing_prediction_examples": [[]],
                },
                "parameters": {
                    "score_thresholds": [0.0],
                    "iou_threshold": 0.5,
                    "label": {"key": "class", "value": "label4"},
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
