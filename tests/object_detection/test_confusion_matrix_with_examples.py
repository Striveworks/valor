from valor_lite.object_detection import Evaluator


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


def test_confusion_matrix_with_examples(
    detections_for_detailed_counting: Evaluator,
):
    evaluator = detections_for_detailed_counting
    assert evaluator.info.number_of_datums == 2
    assert evaluator.info.number_of_labels == 6
    assert evaluator.info.number_of_groundtruth_annotations == 4
    assert evaluator.info.number_of_prediction_annotations == 4

    actual_metrics = evaluator.compute_confusion_matrix_with_examples(
        iou_thresholds=[0.5],
        score_thresholds=[0.05, 0.3, 0.35, 0.45, 0.55, 0.95],
    )

    actual_metrics = [m.to_dict() for m in actual_metrics]
    expected_metrics = [
        {
            "type": "ConfusionMatrixWithExamples",
            "value": {
                "confusion_matrix": {
                    "v1": {
                        "v1": {
                            "count": 1,
                            "examples": [
                                {
                                    "datum_id": "uid1",
                                    "ground_truth_id": "uid_1_gt_0",
                                    "prediction_id": "uid_1_pd_0",
                                }
                            ],
                        }
                    }
                },
                "unmatched_predictions": {
                    "not_v2": {
                        "count": 1,
                        "examples": [
                            {
                                "datum_id": "uid1",
                                "prediction_id": "uid_1_pd_1",
                            },
                        ],
                    },
                    "unmatched_prediction": {
                        "count": 1,
                        "examples": [
                            {
                                "datum_id": "uid1",
                                "prediction_id": "uid_1_pd_2",
                            },
                        ],
                    },
                    "matched_low_iou": {
                        "count": 1,
                        "examples": [
                            {
                                "datum_id": "uid2",
                                "prediction_id": "uid_2_pd_0",
                            }
                        ],
                    },
                },
                "unmatched_ground_truths": {
                    "unmatched_groundtruth": {
                        "count": 1,
                        "examples": [
                            {
                                "datum_id": "uid1",
                                "ground_truth_id": "uid_1_gt_1",
                            }
                        ],
                    },
                    "v2": {
                        "count": 1,
                        "examples": [
                            {
                                "datum_id": "uid1",
                                "ground_truth_id": "uid_1_gt_2",
                            }
                        ],
                    },
                    "matched_low_iou": {
                        "count": 1,
                        "examples": [
                            {
                                "datum_id": "uid2",
                                "ground_truth_id": "uid_2_gt_0",
                            }
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
            "type": "ConfusionMatrixWithExamples",
            "value": {
                "confusion_matrix": {
                    "v1": {
                        "v1": {
                            "count": 1,
                            "examples": [
                                {
                                    "datum_id": "uid1",
                                    "ground_truth_id": "uid_1_gt_0",
                                    "prediction_id": "uid_1_pd_0",
                                }
                            ],
                        }
                    }
                },
                "unmatched_predictions": {
                    "not_v2": {
                        "count": 1,
                        "examples": [
                            {
                                "datum_id": "uid1",
                                "prediction_id": "uid_1_pd_1",
                            }
                        ],
                    },
                    "matched_low_iou": {
                        "count": 1,
                        "examples": [
                            {
                                "datum_id": "uid2",
                                "prediction_id": "uid_2_pd_0",
                            }
                        ],
                    },
                },
                "unmatched_ground_truths": {
                    "unmatched_groundtruth": {
                        "count": 1,
                        "examples": [
                            {
                                "datum_id": "uid1",
                                "ground_truth_id": "uid_1_gt_1",
                            }
                        ],
                    },
                    "v2": {
                        "count": 1,
                        "examples": [
                            {
                                "datum_id": "uid1",
                                "ground_truth_id": "uid_1_gt_2",
                            }
                        ],
                    },
                    "matched_low_iou": {
                        "count": 1,
                        "examples": [
                            {
                                "datum_id": "uid2",
                                "ground_truth_id": "uid_2_gt_0",
                            }
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
            "type": "ConfusionMatrixWithExamples",
            "value": {
                "confusion_matrix": {
                    "v1": {
                        "v1": {
                            "count": 1,
                            "examples": [
                                {
                                    "datum_id": "uid1",
                                    "ground_truth_id": "uid_1_gt_0",
                                    "prediction_id": "uid_1_pd_0",
                                }
                            ],
                        }
                    }
                },
                "unmatched_predictions": {
                    "matched_low_iou": {
                        "count": 1,
                        "examples": [
                            {
                                "datum_id": "uid2",
                                "prediction_id": "uid_2_pd_0",
                            }
                        ],
                    }
                },
                "unmatched_ground_truths": {
                    "unmatched_groundtruth": {
                        "count": 1,
                        "examples": [
                            {
                                "datum_id": "uid1",
                                "ground_truth_id": "uid_1_gt_1",
                            }
                        ],
                    },
                    "v2": {
                        "count": 1,
                        "examples": [
                            {
                                "datum_id": "uid1",
                                "ground_truth_id": "uid_1_gt_2",
                            }
                        ],
                    },
                    "matched_low_iou": {
                        "count": 1,
                        "examples": [
                            {
                                "datum_id": "uid2",
                                "ground_truth_id": "uid_2_gt_0",
                            }
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
            "type": "ConfusionMatrixWithExamples",
            "value": {
                "confusion_matrix": {
                    "v1": {
                        "v1": {
                            "count": 1,
                            "examples": [
                                {
                                    "datum_id": "uid1",
                                    "ground_truth_id": "uid_1_gt_0",
                                    "prediction_id": "uid_1_pd_0",
                                }
                            ],
                        }
                    }
                },
                "unmatched_predictions": {
                    "matched_low_iou": {
                        "count": 1,
                        "examples": [
                            {
                                "datum_id": "uid2",
                                "prediction_id": "uid_2_pd_0",
                            }
                        ],
                    }
                },
                "unmatched_ground_truths": {
                    "unmatched_groundtruth": {
                        "count": 1,
                        "examples": [
                            {
                                "datum_id": "uid1",
                                "ground_truth_id": "uid_1_gt_1",
                            }
                        ],
                    },
                    "v2": {
                        "count": 1,
                        "examples": [
                            {
                                "datum_id": "uid1",
                                "ground_truth_id": "uid_1_gt_2",
                            }
                        ],
                    },
                    "matched_low_iou": {
                        "count": 1,
                        "examples": [
                            {
                                "datum_id": "uid2",
                                "ground_truth_id": "uid_2_gt_0",
                            }
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
            "type": "ConfusionMatrixWithExamples",
            "value": {
                "confusion_matrix": {},
                "unmatched_predictions": {},
                "unmatched_ground_truths": {
                    "v1": {
                        "count": 1,
                        "examples": [
                            {
                                "datum_id": "uid1",
                                "ground_truth_id": "uid_1_gt_0",
                            }
                        ],
                    },
                    "unmatched_groundtruth": {
                        "count": 1,
                        "examples": [
                            {
                                "datum_id": "uid1",
                                "ground_truth_id": "uid_1_gt_1",
                            }
                        ],
                    },
                    "v2": {
                        "count": 1,
                        "examples": [
                            {
                                "datum_id": "uid1",
                                "ground_truth_id": "uid_1_gt_2",
                            }
                        ],
                    },
                    "matched_low_iou": {
                        "count": 1,
                        "examples": [
                            {
                                "datum_id": "uid2",
                                "ground_truth_id": "uid_2_gt_0",
                            }
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
            "type": "ConfusionMatrixWithExamples",
            "value": {
                "confusion_matrix": {},
                "unmatched_predictions": {},
                "unmatched_ground_truths": {
                    "v1": {
                        "count": 1,
                        "examples": [
                            {
                                "datum_id": "uid1",
                                "ground_truth_id": "uid_1_gt_0",
                            }
                        ],
                    },
                    "unmatched_groundtruth": {
                        "count": 1,
                        "examples": [
                            {
                                "datum_id": "uid1",
                                "ground_truth_id": "uid_1_gt_1",
                            }
                        ],
                    },
                    "v2": {
                        "count": 1,
                        "examples": [
                            {
                                "datum_id": "uid1",
                                "ground_truth_id": "uid_1_gt_2",
                            }
                        ],
                    },
                    "matched_low_iou": {
                        "count": 1,
                        "examples": [
                            {
                                "datum_id": "uid2",
                                "ground_truth_id": "uid_2_gt_0",
                            }
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
            m["value"]["unmatched_predictions"],
            m["value"]["unmatched_ground_truths"],
        )
        assert m in expected_metrics
    for m in expected_metrics:
        assert m in actual_metrics

    # test at lower IOU threshold
    actual_metrics = evaluator.compute_confusion_matrix_with_examples(
        iou_thresholds=[0.45],
        score_thresholds=[0.05, 0.3, 0.35, 0.45, 0.55, 0.95],
    )

    actual_metrics = [m.to_dict() for m in actual_metrics]
    expected_metrics = [
        {
            "type": "ConfusionMatrixWithExamples",
            "value": {
                "confusion_matrix": {
                    "v1": {
                        "v1": {
                            "count": 1,
                            "examples": [
                                {
                                    "datum_id": "uid1",
                                    "ground_truth_id": "uid_1_gt_0",
                                    "prediction_id": "uid_1_pd_0",
                                }
                            ],
                        }
                    },
                    "v2": {
                        "not_v2": {
                            "count": 1,
                            "examples": [
                                {
                                    "datum_id": "uid1",
                                    "ground_truth_id": "uid_1_gt_2",
                                    "prediction_id": "uid_1_pd_1",
                                }
                            ],
                        }
                    },
                },
                "unmatched_predictions": {
                    "unmatched_prediction": {
                        "count": 1,
                        "examples": [
                            {
                                "datum_id": "uid1",
                                "prediction_id": "uid_1_pd_2",
                            }
                        ],
                    },
                    "matched_low_iou": {
                        "count": 1,
                        "examples": [
                            {
                                "datum_id": "uid2",
                                "prediction_id": "uid_2_pd_0",
                            }
                        ],
                    },
                },
                "unmatched_ground_truths": {
                    "unmatched_groundtruth": {
                        "count": 1,
                        "examples": [
                            {
                                "datum_id": "uid1",
                                "ground_truth_id": "uid_1_gt_1",
                            }
                        ],
                    },
                    "matched_low_iou": {
                        "count": 1,
                        "examples": [
                            {
                                "datum_id": "uid2",
                                "ground_truth_id": "uid_2_gt_0",
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
            "type": "ConfusionMatrixWithExamples",
            "value": {
                "confusion_matrix": {
                    "v1": {
                        "v1": {
                            "count": 1,
                            "examples": [
                                {
                                    "datum_id": "uid1",
                                    "ground_truth_id": "uid_1_gt_0",
                                    "prediction_id": "uid_1_pd_0",
                                }
                            ],
                        }
                    },
                    "v2": {
                        "not_v2": {
                            "count": 1,
                            "examples": [
                                {
                                    "datum_id": "uid1",
                                    "ground_truth_id": "uid_1_gt_2",
                                    "prediction_id": "uid_1_pd_1",
                                }
                            ],
                        }
                    },
                },
                "unmatched_predictions": {
                    "matched_low_iou": {
                        "count": 1,
                        "examples": [
                            {
                                "datum_id": "uid2",
                                "prediction_id": "uid_2_pd_0",
                            }
                        ],
                    }
                },
                "unmatched_ground_truths": {
                    "unmatched_groundtruth": {
                        "count": 1,
                        "examples": [
                            {
                                "datum_id": "uid1",
                                "ground_truth_id": "uid_1_gt_1",
                            }
                        ],
                    },
                    "matched_low_iou": {
                        "count": 1,
                        "examples": [
                            {
                                "datum_id": "uid2",
                                "ground_truth_id": "uid_2_gt_0",
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
            "type": "ConfusionMatrixWithExamples",
            "value": {
                "confusion_matrix": {
                    "v1": {
                        "v1": {
                            "count": 1,
                            "examples": [
                                {
                                    "datum_id": "uid1",
                                    "ground_truth_id": "uid_1_gt_0",
                                    "prediction_id": "uid_1_pd_0",
                                }
                            ],
                        }
                    }
                },
                "unmatched_predictions": {
                    "matched_low_iou": {
                        "count": 1,
                        "examples": [
                            {
                                "datum_id": "uid2",
                                "prediction_id": "uid_2_pd_0",
                            }
                        ],
                    }
                },
                "unmatched_ground_truths": {
                    "unmatched_groundtruth": {
                        "count": 1,
                        "examples": [
                            {
                                "datum_id": "uid1",
                                "ground_truth_id": "uid_1_gt_1",
                            }
                        ],
                    },
                    "v2": {
                        "count": 1,
                        "examples": [
                            {
                                "datum_id": "uid1",
                                "ground_truth_id": "uid_1_gt_2",
                            }
                        ],
                    },
                    "matched_low_iou": {
                        "count": 1,
                        "examples": [
                            {
                                "datum_id": "uid2",
                                "ground_truth_id": "uid_2_gt_0",
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
            "type": "ConfusionMatrixWithExamples",
            "value": {
                "confusion_matrix": {
                    "v1": {
                        "v1": {
                            "count": 1,
                            "examples": [
                                {
                                    "datum_id": "uid1",
                                    "ground_truth_id": "uid_1_gt_0",
                                    "prediction_id": "uid_1_pd_0",
                                }
                            ],
                        }
                    }
                },
                "unmatched_predictions": {
                    "matched_low_iou": {
                        "count": 1,
                        "examples": [
                            {
                                "datum_id": "uid2",
                                "prediction_id": "uid_2_pd_0",
                            }
                        ],
                    }
                },
                "unmatched_ground_truths": {
                    "unmatched_groundtruth": {
                        "count": 1,
                        "examples": [
                            {
                                "datum_id": "uid1",
                                "ground_truth_id": "uid_1_gt_1",
                            }
                        ],
                    },
                    "v2": {
                        "count": 1,
                        "examples": [
                            {
                                "datum_id": "uid1",
                                "ground_truth_id": "uid_1_gt_2",
                            }
                        ],
                    },
                    "matched_low_iou": {
                        "count": 1,
                        "examples": [
                            {
                                "datum_id": "uid2",
                                "ground_truth_id": "uid_2_gt_0",
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
            "type": "ConfusionMatrixWithExamples",
            "value": {
                "confusion_matrix": {},
                "unmatched_predictions": {},
                "unmatched_ground_truths": {
                    "v1": {
                        "count": 1,
                        "examples": [
                            {
                                "datum_id": "uid1",
                                "ground_truth_id": "uid_1_gt_0",
                            }
                        ],
                    },
                    "unmatched_groundtruth": {
                        "count": 1,
                        "examples": [
                            {
                                "datum_id": "uid1",
                                "ground_truth_id": "uid_1_gt_1",
                            }
                        ],
                    },
                    "v2": {
                        "count": 1,
                        "examples": [
                            {
                                "datum_id": "uid1",
                                "ground_truth_id": "uid_1_gt_2",
                            }
                        ],
                    },
                    "matched_low_iou": {
                        "count": 1,
                        "examples": [
                            {
                                "datum_id": "uid2",
                                "ground_truth_id": "uid_2_gt_0",
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
            "type": "ConfusionMatrixWithExamples",
            "value": {
                "confusion_matrix": {},
                "unmatched_predictions": {},
                "unmatched_ground_truths": {
                    "v1": {
                        "count": 1,
                        "examples": [
                            {
                                "datum_id": "uid1",
                                "ground_truth_id": "uid_1_gt_0",
                            }
                        ],
                    },
                    "unmatched_groundtruth": {
                        "count": 1,
                        "examples": [
                            {
                                "datum_id": "uid1",
                                "ground_truth_id": "uid_1_gt_1",
                            }
                        ],
                    },
                    "v2": {
                        "count": 1,
                        "examples": [
                            {
                                "datum_id": "uid1",
                                "ground_truth_id": "uid_1_gt_2",
                            }
                        ],
                    },
                    "matched_low_iou": {
                        "count": 1,
                        "examples": [
                            {
                                "datum_id": "uid2",
                                "ground_truth_id": "uid_2_gt_0",
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
            m["value"]["unmatched_predictions"],
            m["value"]["unmatched_ground_truths"],
        )
        assert m in expected_metrics
    for m in expected_metrics:
        assert m in actual_metrics


def _filter_out_examples_and_zero_counts(cm: dict, hl: dict, mp: dict):
    gt_labels = list(mp.keys())
    pd_labels = list(hl.keys())

    for gt_label in gt_labels:
        for pd_label in pd_labels:
            if cm[gt_label][pd_label]["count"] == 0:
                cm[gt_label].pop(pd_label)
            else:
                cm[gt_label][pd_label]["examples"] = list()
        if len(cm[gt_label]) == 0:
            cm.pop(gt_label)

    for pd_label in pd_labels:
        if hl[pd_label]["count"] == 0:
            hl.pop(pd_label)
        else:
            hl[pd_label]["examples"] = list()

    for gt_label in gt_labels:
        if mp[gt_label]["count"] == 0:
            mp.pop(gt_label)
        else:
            mp[gt_label]["examples"] = list()


def test_confusion_matrix_with_examples_using_torch_metrics_example(
    torchmetrics_evaluator: Evaluator,
):
    """
    cf with torch metrics/pycocotools results listed here:
    https://github.com/Lightning-AI/metrics/blob/107dbfd5fb158b7ae6d76281df44bd94c836bfce/tests/unittests/detection/test_map.py#L231
    """
    evaluator = torchmetrics_evaluator
    assert evaluator.info.number_of_datums == 4
    assert evaluator.info.number_of_labels == 6
    assert evaluator.info.number_of_groundtruth_annotations == 20
    assert evaluator.info.number_of_prediction_annotations == 19

    actual_metrics = evaluator.compute_confusion_matrix_with_examples(
        iou_thresholds=[0.5, 0.9],
        score_thresholds=[0.05, 0.25, 0.35, 0.55, 0.75, 0.8, 0.85, 0.95],
    )

    assert len(actual_metrics) == 16

    actual_metrics = [m.to_dict() for m in actual_metrics]
    expected_metrics = [
        {
            "type": "ConfusionMatrixWithExamples",
            "value": {
                "confusion_matrix": {
                    "4": {"4": {"count": 2, "examples": []}},
                    "2": {
                        "2": {"count": 1, "examples": []},
                        "3": {"count": 1, "examples": []},
                    },
                    "1": {"1": {"count": 1, "examples": []}},
                    "0": {"0": {"count": 5, "examples": []}},
                    "49": {"49": {"count": 8, "examples": []}},
                },
                "unmatched_predictions": {"49": {"count": 1, "examples": []}},
                "unmatched_ground_truths": {
                    "49": {"count": 2, "examples": []}
                },
            },
            "parameters": {
                "score_threshold": 0.05,
                "iou_threshold": 0.5,
            },
        },
        {
            "type": "ConfusionMatrixWithExamples",
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
                "unmatched_predictions": {},
                "unmatched_ground_truths": {
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
            "type": "ConfusionMatrixWithExamples",
            "value": {
                "confusion_matrix": {
                    "4": {"4": {"count": 1, "examples": []}},
                    "2": {"2": {"count": 1, "examples": []}},
                    "0": {"0": {"count": 4, "examples": []}},
                    "49": {"49": {"count": 4, "examples": []}},
                },
                "unmatched_predictions": {},
                "unmatched_ground_truths": {
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
            "type": "ConfusionMatrixWithExamples",
            "value": {
                "confusion_matrix": {
                    "2": {"2": {"count": 1, "examples": []}},
                    "0": {"0": {"count": 3, "examples": []}},
                    "49": {"49": {"count": 3, "examples": []}},
                },
                "unmatched_predictions": {},
                "unmatched_ground_truths": {
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
            "type": "ConfusionMatrixWithExamples",
            "value": {
                "confusion_matrix": {
                    "0": {"0": {"count": 2, "examples": []}},
                    "49": {"49": {"count": 2, "examples": []}},
                },
                "unmatched_predictions": {},
                "unmatched_ground_truths": {
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
            "type": "ConfusionMatrixWithExamples",
            "value": {
                "confusion_matrix": {
                    "0": {"0": {"count": 2, "examples": []}},
                    "49": {"49": {"count": 1, "examples": []}},
                },
                "unmatched_predictions": {},
                "unmatched_ground_truths": {
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
            "type": "ConfusionMatrixWithExamples",
            "value": {
                "confusion_matrix": {
                    "0": {"0": {"count": 1, "examples": []}},
                    "49": {"49": {"count": 1, "examples": []}},
                },
                "unmatched_predictions": {},
                "unmatched_ground_truths": {
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
            "type": "ConfusionMatrixWithExamples",
            "value": {
                "confusion_matrix": {"0": {"0": {"count": 1, "examples": []}}},
                "unmatched_predictions": {},
                "unmatched_ground_truths": {
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
            "type": "ConfusionMatrixWithExamples",
            "value": {
                "confusion_matrix": {
                    "2": {"2": {"count": 1, "examples": []}},
                    "0": {"0": {"count": 1, "examples": []}},
                    "49": {"49": {"count": 2, "examples": []}},
                },
                "unmatched_predictions": {
                    "4": {"count": 2, "examples": []},
                    "3": {"count": 1, "examples": []},
                    "1": {"count": 1, "examples": []},
                    "0": {"count": 4, "examples": []},
                    "49": {"count": 7, "examples": []},
                },
                "unmatched_ground_truths": {
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
            "type": "ConfusionMatrixWithExamples",
            "value": {
                "confusion_matrix": {
                    "2": {"2": {"count": 1, "examples": []}},
                    "0": {"0": {"count": 1, "examples": []}},
                    "49": {"49": {"count": 2, "examples": []}},
                },
                "unmatched_predictions": {
                    "4": {"count": 1, "examples": []},
                    "3": {"count": 1, "examples": []},
                    "1": {"count": 1, "examples": []},
                    "0": {"count": 4, "examples": []},
                    "49": {"count": 4, "examples": []},
                },
                "unmatched_ground_truths": {
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
            "type": "ConfusionMatrixWithExamples",
            "value": {
                "confusion_matrix": {
                    "2": {"2": {"count": 1, "examples": []}},
                    "49": {"49": {"count": 2, "examples": []}},
                },
                "unmatched_predictions": {
                    "4": {"count": 1, "examples": []},
                    "0": {"count": 4, "examples": []},
                    "49": {"count": 2, "examples": []},
                },
                "unmatched_ground_truths": {
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
            "type": "ConfusionMatrixWithExamples",
            "value": {
                "confusion_matrix": {
                    "2": {"2": {"count": 1, "examples": []}},
                    "49": {"49": {"count": 1, "examples": []}},
                },
                "unmatched_predictions": {
                    "0": {"count": 3, "examples": []},
                    "49": {"count": 2, "examples": []},
                },
                "unmatched_ground_truths": {
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
            "type": "ConfusionMatrixWithExamples",
            "value": {
                "confusion_matrix": {
                    "49": {"49": {"count": 1, "examples": []}}
                },
                "unmatched_predictions": {
                    "0": {"count": 2, "examples": []},
                    "49": {"count": 1, "examples": []},
                },
                "unmatched_ground_truths": {
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
            "type": "ConfusionMatrixWithExamples",
            "value": {
                "confusion_matrix": {
                    "49": {"49": {"count": 1, "examples": []}}
                },
                "unmatched_predictions": {"0": {"count": 2, "examples": []}},
                "unmatched_ground_truths": {
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
            "type": "ConfusionMatrixWithExamples",
            "value": {
                "confusion_matrix": {
                    "49": {"49": {"count": 1, "examples": []}}
                },
                "unmatched_predictions": {"0": {"count": 1, "examples": []}},
                "unmatched_ground_truths": {
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
            "type": "ConfusionMatrixWithExamples",
            "value": {
                "confusion_matrix": {},
                "unmatched_predictions": {"0": {"count": 1, "examples": []}},
                "unmatched_ground_truths": {
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
        _filter_out_examples_and_zero_counts(
            m["value"]["confusion_matrix"],
            m["value"]["unmatched_predictions"],
            m["value"]["unmatched_ground_truths"],
        )
        assert m in expected_metrics
    for m in expected_metrics:
        assert m in actual_metrics


def test_confusion_matrix_with_examples_fp_unmatched_prediction_edge_case(
    detections_fp_unmatched_prediction_edge_case: Evaluator,
):
    evaluator = detections_fp_unmatched_prediction_edge_case
    assert evaluator.info.number_of_datums == 2
    assert evaluator.info.number_of_labels == 1
    assert evaluator.info.number_of_groundtruth_annotations == 2
    assert evaluator.info.number_of_prediction_annotations == 2

    actual_metrics = evaluator.compute_confusion_matrix_with_examples(
        iou_thresholds=[0.5],
        score_thresholds=[0.5, 0.85],
    )

    assert len(actual_metrics) == 2

    actual_metrics = [m.to_dict() for m in actual_metrics]
    expected_metrics = [
        {
            "type": "ConfusionMatrixWithExamples",
            "value": {
                "confusion_matrix": {
                    "v1": {
                        "v1": {
                            "count": 1,
                            "examples": [
                                {
                                    "datum_id": "uid1",
                                    "ground_truth_id": "uid1_gt0",
                                    "prediction_id": "uid1_pd0",
                                }
                            ],
                        }
                    }
                },
                "unmatched_predictions": {
                    "v1": {
                        "count": 1,
                        "examples": [
                            {
                                "datum_id": "uid2",
                                "prediction_id": "uid2_pd0",
                            }
                        ],
                    }
                },
                "unmatched_ground_truths": {
                    "v1": {
                        "count": 1,
                        "examples": [
                            {
                                "datum_id": "uid2",
                                "ground_truth_id": "uid2_gt0",
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
            "type": "ConfusionMatrixWithExamples",
            "value": {
                "confusion_matrix": {},
                "unmatched_predictions": {},
                "unmatched_ground_truths": {
                    "v1": {
                        "count": 2,
                        "examples": [
                            {
                                "datum_id": "uid1",
                                "ground_truth_id": "uid1_gt0",
                            },
                            {
                                "datum_id": "uid2",
                                "ground_truth_id": "uid2_gt0",
                            },
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
            m["value"]["unmatched_predictions"],
            m["value"]["unmatched_ground_truths"],
        )
        assert m in expected_metrics
    for m in expected_metrics:
        assert m in actual_metrics


def test_confusion_matrix_with_examples_ranked_pair_ordering(
    detection_ranked_pair_ordering: Evaluator,
):
    evaluator = detection_ranked_pair_ordering
    assert evaluator.info.number_of_datums == 1
    assert evaluator.info.number_of_labels == 4
    assert evaluator.info.number_of_groundtruth_annotations == 3
    assert evaluator.info.number_of_prediction_annotations == 4

    actual_metrics = evaluator.compute_confusion_matrix_with_examples(
        iou_thresholds=[0.5],
        score_thresholds=[0.0],
    )

    actual_metrics = [m.to_dict() for m in actual_metrics]
    expected_metrics = [
        {
            "type": "ConfusionMatrixWithExamples",
            "value": {
                "confusion_matrix": {
                    "label1": {
                        "label2": {
                            "count": 1,
                            "examples": [
                                {
                                    "datum_id": "uid1",
                                    "ground_truth_id": "gt_0",
                                    "prediction_id": "pd_1",
                                }
                            ],
                        }
                    },
                    "label2": {
                        "label1": {
                            "count": 1,
                            "examples": [
                                {
                                    "datum_id": "uid1",
                                    "ground_truth_id": "gt_1",
                                    "prediction_id": "pd_0",
                                }
                            ],
                        }
                    },
                },
                "unmatched_predictions": {
                    "label3": {
                        "count": 1,
                        "examples": [
                            {"datum_id": "uid1", "prediction_id": "pd_2"}
                        ],
                    },
                    "label4": {
                        "count": 1,
                        "examples": [
                            {"datum_id": "uid1", "prediction_id": "pd_3"}
                        ],
                    },
                },
                "unmatched_ground_truths": {
                    "label3": {
                        "count": 1,
                        "examples": [
                            {"datum_id": "uid1", "ground_truth_id": "gt_2"}
                        ],
                    }
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
            m["value"]["unmatched_predictions"],
            m["value"]["unmatched_ground_truths"],
        )
        assert m in expected_metrics
    for m in expected_metrics:
        assert m in actual_metrics
