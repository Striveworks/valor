from valor_lite.object_detection import Evaluator


def test_examples(detections_for_detailed_counting: Evaluator):
    evaluator = detections_for_detailed_counting
    assert evaluator.info.number_of_datums == 2
    assert evaluator.info.number_of_labels == 6
    assert evaluator.info.number_of_groundtruth_annotations == 4
    assert evaluator.info.number_of_prediction_annotations == 4

    actual_metrics = evaluator.compute_examples(
        iou_thresholds=[0.5],
        score_thresholds=[0.05, 0.3, 0.35, 0.45, 0.55, 0.95],
    )

    actual_metrics = [m.to_dict() for m in actual_metrics]
    expected_metrics = [
        {
            "type": "Examples",
            "value": {
                "datum_id": "uid1",
                "true_positives": [("uid_1_gt_0", "uid_1_pd_0")],
                "false_positives": ["uid_1_pd_1", "uid_1_pd_2"],
                "false_negatives": ["uid_1_gt_1", "uid_1_gt_2"],
            },
            "parameters": {"score_threshold": 0.05, "iou_threshold": 0.5},
        },
        {
            "type": "Examples",
            "value": {
                "datum_id": "uid1",
                "true_positives": [("uid_1_gt_0", "uid_1_pd_0")],
                "false_positives": ["uid_1_pd_1"],
                "false_negatives": ["uid_1_gt_1", "uid_1_gt_2"],
            },
            "parameters": {
                "score_threshold": 0.3,
                "iou_threshold": 0.5,
            },
        },
        {
            "type": "Examples",
            "value": {
                "datum_id": "uid1",
                "true_positives": [("uid_1_gt_0", "uid_1_pd_0")],
                "false_positives": [],
                "false_negatives": ["uid_1_gt_1", "uid_1_gt_2"],
            },
            "parameters": {"score_threshold": 0.35, "iou_threshold": 0.5},
        },
        {
            "type": "Examples",
            "value": {
                "datum_id": "uid1",
                "true_positives": [("uid_1_gt_0", "uid_1_pd_0")],
                "false_positives": [],
                "false_negatives": ["uid_1_gt_1", "uid_1_gt_2"],
            },
            "parameters": {"score_threshold": 0.45, "iou_threshold": 0.5},
        },
        {
            "type": "Examples",
            "value": {
                "datum_id": "uid1",
                "true_positives": [],
                "false_positives": [],
                "false_negatives": ["uid_1_gt_0", "uid_1_gt_1", "uid_1_gt_2"],
            },
            "parameters": {"score_threshold": 0.55, "iou_threshold": 0.5},
        },
        {
            "type": "Examples",
            "value": {
                "datum_id": "uid1",
                "true_positives": [],
                "false_positives": [],
                "false_negatives": ["uid_1_gt_0", "uid_1_gt_1", "uid_1_gt_2"],
            },
            "parameters": {"score_threshold": 0.95, "iou_threshold": 0.5},
        },
        {
            "type": "Examples",
            "value": {
                "datum_id": "uid2",
                "true_positives": [],
                "false_positives": ["uid_2_pd_0"],
                "false_negatives": ["uid_2_gt_0"],
            },
            "parameters": {"score_threshold": 0.05, "iou_threshold": 0.5},
        },
        {
            "type": "Examples",
            "value": {
                "datum_id": "uid2",
                "true_positives": [],
                "false_positives": ["uid_2_pd_0"],
                "false_negatives": ["uid_2_gt_0"],
            },
            "parameters": {"score_threshold": 0.3, "iou_threshold": 0.5},
        },
        {
            "type": "Examples",
            "value": {
                "datum_id": "uid2",
                "true_positives": [],
                "false_positives": ["uid_2_pd_0"],
                "false_negatives": ["uid_2_gt_0"],
            },
            "parameters": {"score_threshold": 0.35, "iou_threshold": 0.5},
        },
        {
            "type": "Examples",
            "value": {
                "datum_id": "uid2",
                "true_positives": [],
                "false_positives": ["uid_2_pd_0"],
                "false_negatives": ["uid_2_gt_0"],
            },
            "parameters": {"score_threshold": 0.45, "iou_threshold": 0.5},
        },
        {
            "type": "Examples",
            "value": {
                "datum_id": "uid2",
                "true_positives": [],
                "false_positives": [],
                "false_negatives": ["uid_2_gt_0"],
            },
            "parameters": {"score_threshold": 0.55, "iou_threshold": 0.5},
        },
        {
            "type": "Examples",
            "value": {
                "datum_id": "uid2",
                "true_positives": [],
                "false_positives": [],
                "false_negatives": ["uid_2_gt_0"],
            },
            "parameters": {"score_threshold": 0.95, "iou_threshold": 0.5},
        },
    ]

    assert len(actual_metrics) == len(expected_metrics)
    for m in actual_metrics:
        assert m in expected_metrics
    for m in expected_metrics:
        assert m in actual_metrics

    # test at lower IOU threshold
    actual_metrics = evaluator.compute_examples(
        iou_thresholds=[0.45],
        score_thresholds=[0.05, 0.3, 0.35, 0.45, 0.55, 0.95],
    )

    actual_metrics = [m.to_dict() for m in actual_metrics]
    expected_metrics = [
        {
            "type": "Examples",
            "value": {
                "datum_id": "uid1",
                "true_positives": [("uid_1_gt_0", "uid_1_pd_0")],
                "false_positives": ["uid_1_pd_1", "uid_1_pd_2"],
                "false_negatives": ["uid_1_gt_1", "uid_1_gt_2"],
            },
            "parameters": {"score_threshold": 0.05, "iou_threshold": 0.45},
        },
        {
            "type": "Examples",
            "value": {
                "datum_id": "uid1",
                "true_positives": [("uid_1_gt_0", "uid_1_pd_0")],
                "false_positives": ["uid_1_pd_1"],
                "false_negatives": ["uid_1_gt_1", "uid_1_gt_2"],
            },
            "parameters": {"score_threshold": 0.3, "iou_threshold": 0.45},
        },
        {
            "type": "Examples",
            "value": {
                "datum_id": "uid1",
                "true_positives": [("uid_1_gt_0", "uid_1_pd_0")],
                "false_positives": [],
                "false_negatives": ["uid_1_gt_1", "uid_1_gt_2"],
            },
            "parameters": {"score_threshold": 0.35, "iou_threshold": 0.45},
        },
        {
            "type": "Examples",
            "value": {
                "datum_id": "uid1",
                "true_positives": [("uid_1_gt_0", "uid_1_pd_0")],
                "false_positives": [],
                "false_negatives": ["uid_1_gt_1", "uid_1_gt_2"],
            },
            "parameters": {"score_threshold": 0.45, "iou_threshold": 0.45},
        },
        {
            "type": "Examples",
            "value": {
                "datum_id": "uid1",
                "true_positives": [],
                "false_positives": [],
                "false_negatives": ["uid_1_gt_0", "uid_1_gt_1", "uid_1_gt_2"],
            },
            "parameters": {"score_threshold": 0.55, "iou_threshold": 0.45},
        },
        {
            "type": "Examples",
            "value": {
                "datum_id": "uid1",
                "true_positives": [],
                "false_positives": [],
                "false_negatives": ["uid_1_gt_0", "uid_1_gt_1", "uid_1_gt_2"],
            },
            "parameters": {"score_threshold": 0.95, "iou_threshold": 0.45},
        },
        {
            "type": "Examples",
            "value": {
                "datum_id": "uid2",
                "true_positives": [],
                "false_positives": ["uid_2_pd_0"],
                "false_negatives": ["uid_2_gt_0"],
            },
            "parameters": {"score_threshold": 0.05, "iou_threshold": 0.45},
        },
        {
            "type": "Examples",
            "value": {
                "datum_id": "uid2",
                "true_positives": [],
                "false_positives": ["uid_2_pd_0"],
                "false_negatives": ["uid_2_gt_0"],
            },
            "parameters": {"score_threshold": 0.3, "iou_threshold": 0.45},
        },
        {
            "type": "Examples",
            "value": {
                "datum_id": "uid2",
                "true_positives": [],
                "false_positives": ["uid_2_pd_0"],
                "false_negatives": ["uid_2_gt_0"],
            },
            "parameters": {"score_threshold": 0.35, "iou_threshold": 0.45},
        },
        {
            "type": "Examples",
            "value": {
                "datum_id": "uid2",
                "true_positives": [],
                "false_positives": ["uid_2_pd_0"],
                "false_negatives": ["uid_2_gt_0"],
            },
            "parameters": {"score_threshold": 0.45, "iou_threshold": 0.45},
        },
        {
            "type": "Examples",
            "value": {
                "datum_id": "uid2",
                "true_positives": [],
                "false_positives": [],
                "false_negatives": ["uid_2_gt_0"],
            },
            "parameters": {"score_threshold": 0.55, "iou_threshold": 0.45},
        },
        {
            "type": "Examples",
            "value": {
                "datum_id": "uid2",
                "true_positives": [],
                "false_positives": [],
                "false_negatives": ["uid_2_gt_0"],
            },
            "parameters": {"score_threshold": 0.95, "iou_threshold": 0.45},
        },
    ]

    for m in actual_metrics:
        assert m in expected_metrics
    for m in expected_metrics:
        assert m in actual_metrics


def test_examples_using_torch_metrics_example(
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

    iou_thresholds = [0.5, 0.9]
    score_thresholds = [0.05, 0.25, 0.35, 0.55, 0.75, 0.8, 0.85, 0.95]
    actual_metrics = evaluator.compute_examples(
        iou_thresholds=iou_thresholds,
        score_thresholds=score_thresholds,
    )

    actual_metrics = [m.to_dict() for m in actual_metrics]
    expected_metrics = [
        # datum 0
        {
            "type": "Examples",
            "value": {
                "datum_id": "0",
                "true_positives": [("uid_0_gt_0", "uid_0_pd_0")],
                "false_positives": [],
                "false_negatives": [],
            },
            "parameters": {"score_threshold": 0.05, "iou_threshold": 0.5},
        },
        {
            "type": "Examples",
            "value": {
                "datum_id": "0",
                "true_positives": [],
                "false_positives": ["uid_0_pd_0"],
                "false_negatives": ["uid_0_gt_0"],
            },
            "parameters": {"score_threshold": 0.05, "iou_threshold": 0.9},
        },
        *[
            {
                "type": "Examples",
                "value": {
                    "datum_id": "0",
                    "true_positives": [],
                    "false_positives": [],
                    "false_negatives": ["uid_0_gt_0"],
                },
                "parameters": {
                    "score_threshold": score_thresh,
                    "iou_threshold": iou_thresh,
                },
            }
            for iou_thresh in iou_thresholds
            for score_thresh in score_thresholds[1:]
        ],
        # datum 1
        *[
            {
                "type": "Examples",
                "value": {
                    "datum_id": "1",
                    "true_positives": [("uid_1_gt_1", "uid_1_pd_1")],
                    "false_positives": ["uid_1_pd_0"],
                    "false_negatives": ["uid_1_gt_0"],
                },
                "parameters": {
                    "score_threshold": score_thresh,
                    "iou_threshold": iou_thresh,
                },
            }
            for iou_thresh in iou_thresholds
            for score_thresh in score_thresholds[:2]
        ],
        *[
            {
                "type": "Examples",
                "value": {
                    "datum_id": "1",
                    "true_positives": [("uid_1_gt_1", "uid_1_pd_1")],
                    "false_positives": [],
                    "false_negatives": ["uid_1_gt_0"],
                },
                "parameters": {
                    "score_threshold": score_thresh,
                    "iou_threshold": iou_thresh,
                },
            }
            for iou_thresh in iou_thresholds
            for score_thresh in score_thresholds[2:4]
        ],
        *[
            {
                "type": "Examples",
                "value": {
                    "datum_id": "1",
                    "true_positives": [],
                    "false_positives": [],
                    "false_negatives": ["uid_1_gt_0", "uid_1_gt_1"],
                },
                "parameters": {
                    "score_threshold": score_thresh,
                    "iou_threshold": iou_thresh,
                },
            }
            for iou_thresh in iou_thresholds
            for score_thresh in score_thresholds[4:]
        ],
        # datum 2
        *[
            {
                "type": "Examples",
                "value": {
                    "datum_id": "2",
                    "true_positives": [
                        ("uid_2_gt_0", "uid_2_pd_0"),
                        ("uid_2_gt_1", "uid_2_pd_1"),
                        ("uid_2_gt_2", "uid_2_pd_2"),
                        ("uid_2_gt_3", "uid_2_pd_3"),
                        ("uid_2_gt_4", "uid_2_pd_4"),
                        ("uid_2_gt_5", "uid_2_pd_5"),
                        ("uid_2_gt_6", "uid_2_pd_6"),
                    ],
                    "false_positives": [],
                    "false_negatives": [],
                },
                "parameters": {
                    "score_threshold": score_thresh,
                    "iou_threshold": 0.5,
                },
            }
            for score_thresh in score_thresholds[:2]
        ],
        {
            "type": "Examples",
            "value": {
                "datum_id": "2",
                "true_positives": [
                    ("uid_2_gt_0", "uid_2_pd_0"),
                    ("uid_2_gt_2", "uid_2_pd_2"),
                    ("uid_2_gt_3", "uid_2_pd_3"),
                    ("uid_2_gt_5", "uid_2_pd_5"),
                    ("uid_2_gt_6", "uid_2_pd_6"),
                ],
                "false_positives": [],
                "false_negatives": ["uid_2_gt_1", "uid_2_gt_4"],
            },
            "parameters": {"score_threshold": 0.35, "iou_threshold": 0.5},
        },
        {
            "type": "Examples",
            "value": {
                "datum_id": "2",
                "true_positives": [
                    ("uid_2_gt_3", "uid_2_pd_3"),
                    ("uid_2_gt_5", "uid_2_pd_5"),
                    ("uid_2_gt_6", "uid_2_pd_6"),
                ],
                "false_positives": [],
                "false_negatives": [
                    "uid_2_gt_0",
                    "uid_2_gt_1",
                    "uid_2_gt_2",
                    "uid_2_gt_4",
                ],
            },
            "parameters": {"score_threshold": 0.55, "iou_threshold": 0.5},
        },
        *[
            {
                "type": "Examples",
                "value": {
                    "datum_id": "2",
                    "true_positives": [
                        ("uid_2_gt_5", "uid_2_pd_5"),
                        ("uid_2_gt_6", "uid_2_pd_6"),
                    ],
                    "false_positives": [],
                    "false_negatives": [
                        "uid_2_gt_0",
                        "uid_2_gt_1",
                        "uid_2_gt_2",
                        "uid_2_gt_3",
                        "uid_2_gt_4",
                    ],
                },
                "parameters": {
                    "score_threshold": score_thresh,
                    "iou_threshold": 0.5,
                },
            }
            for score_thresh in score_thresholds[4:6]
        ],
        *[
            {
                "type": "Examples",
                "value": {
                    "datum_id": "2",
                    "true_positives": [("uid_2_gt_6", "uid_2_pd_6")],
                    "false_positives": [],
                    "false_negatives": [
                        "uid_2_gt_0",
                        "uid_2_gt_1",
                        "uid_2_gt_2",
                        "uid_2_gt_3",
                        "uid_2_gt_4",
                        "uid_2_gt_5",
                    ],
                },
                "parameters": {
                    "score_threshold": score_thresh,
                    "iou_threshold": 0.5,
                },
            }
            for score_thresh in score_thresholds[6:]
        ],
        *[
            {
                "type": "Examples",
                "value": {
                    "datum_id": "2",
                    "true_positives": [("uid_2_gt_4", "uid_2_pd_4")],
                    "false_positives": [
                        "uid_2_pd_0",
                        "uid_2_pd_1",
                        "uid_2_pd_2",
                        "uid_2_pd_3",
                        "uid_2_pd_5",
                        "uid_2_pd_6",
                    ],
                    "false_negatives": [
                        "uid_2_gt_0",
                        "uid_2_gt_1",
                        "uid_2_gt_2",
                        "uid_2_gt_3",
                        "uid_2_gt_5",
                        "uid_2_gt_6",
                    ],
                },
                "parameters": {
                    "score_threshold": score_thresh,
                    "iou_threshold": 0.9,
                },
            }
            for score_thresh in score_thresholds[:2]
        ],
        {
            "type": "Examples",
            "value": {
                "datum_id": "2",
                "true_positives": [],
                "false_positives": [
                    "uid_2_pd_0",
                    "uid_2_pd_2",
                    "uid_2_pd_3",
                    "uid_2_pd_5",
                    "uid_2_pd_6",
                ],
                "false_negatives": [
                    "uid_2_gt_0",
                    "uid_2_gt_1",
                    "uid_2_gt_2",
                    "uid_2_gt_3",
                    "uid_2_gt_4",
                    "uid_2_gt_5",
                    "uid_2_gt_6",
                ],
            },
            "parameters": {"score_threshold": 0.35, "iou_threshold": 0.9},
        },
        {
            "type": "Examples",
            "value": {
                "datum_id": "2",
                "true_positives": [],
                "false_positives": ["uid_2_pd_3", "uid_2_pd_5", "uid_2_pd_6"],
                "false_negatives": [
                    "uid_2_gt_0",
                    "uid_2_gt_1",
                    "uid_2_gt_2",
                    "uid_2_gt_3",
                    "uid_2_gt_4",
                    "uid_2_gt_5",
                    "uid_2_gt_6",
                ],
            },
            "parameters": {"score_threshold": 0.55, "iou_threshold": 0.9},
        },
        *[
            {
                "type": "Examples",
                "value": {
                    "datum_id": "2",
                    "true_positives": [],
                    "false_positives": ["uid_2_pd_5", "uid_2_pd_6"],
                    "false_negatives": [
                        "uid_2_gt_0",
                        "uid_2_gt_1",
                        "uid_2_gt_2",
                        "uid_2_gt_3",
                        "uid_2_gt_4",
                        "uid_2_gt_5",
                        "uid_2_gt_6",
                    ],
                },
                "parameters": {
                    "score_threshold": score_thresh,
                    "iou_threshold": 0.9,
                },
            }
            for score_thresh in [0.75, 0.8]
        ],
        *[
            {
                "type": "Examples",
                "value": {
                    "datum_id": "2",
                    "true_positives": [],
                    "false_positives": ["uid_2_pd_6"],
                    "false_negatives": [
                        "uid_2_gt_0",
                        "uid_2_gt_1",
                        "uid_2_gt_2",
                        "uid_2_gt_3",
                        "uid_2_gt_4",
                        "uid_2_gt_5",
                        "uid_2_gt_6",
                    ],
                },
                "parameters": {
                    "score_threshold": score_thresh,
                    "iou_threshold": 0.9,
                },
            }
            for score_thresh in [0.85, 0.95]
        ],
        # datum 3
        {
            "type": "Examples",
            "value": {
                "datum_id": "3",
                "true_positives": [
                    ("uid_3_gt_0", "uid_3_pd_0"),
                    ("uid_3_gt_2", "uid_3_pd_2"),
                    ("uid_3_gt_3", "uid_3_pd_3"),
                    ("uid_3_gt_4", "uid_3_pd_4"),
                    ("uid_3_gt_6", "uid_3_pd_5"),
                    ("uid_3_gt_7", "uid_3_pd_6"),
                    ("uid_3_gt_8", "uid_3_pd_1"),
                    ("uid_3_gt_9", "uid_3_pd_8"),
                ],
                "false_positives": ["uid_3_pd_7"],
                "false_negatives": ["uid_3_gt_1", "uid_3_gt_5"],
            },
            "parameters": {"score_threshold": 0.05, "iou_threshold": 0.5},
        },
        {
            "type": "Examples",
            "value": {
                "datum_id": "3",
                "true_positives": [
                    ("uid_3_gt_0", "uid_3_pd_0"),
                    ("uid_3_gt_2", "uid_3_pd_2"),
                    ("uid_3_gt_4", "uid_3_pd_4"),
                    ("uid_3_gt_6", "uid_3_pd_5"),
                    ("uid_3_gt_7", "uid_3_pd_6"),
                    ("uid_3_gt_9", "uid_3_pd_8"),
                ],
                "false_positives": [],
                "false_negatives": [
                    "uid_3_gt_1",
                    "uid_3_gt_3",
                    "uid_3_gt_5",
                    "uid_3_gt_8",
                ],
            },
            "parameters": {"score_threshold": 0.25, "iou_threshold": 0.5},
        },
        {
            "type": "Examples",
            "value": {
                "datum_id": "3",
                "true_positives": [
                    ("uid_3_gt_0", "uid_3_pd_0"),
                    ("uid_3_gt_2", "uid_3_pd_2"),
                    ("uid_3_gt_4", "uid_3_pd_4"),
                    ("uid_3_gt_7", "uid_3_pd_6"),
                ],
                "false_positives": [],
                "false_negatives": [
                    "uid_3_gt_1",
                    "uid_3_gt_3",
                    "uid_3_gt_5",
                    "uid_3_gt_6",
                    "uid_3_gt_8",
                    "uid_3_gt_9",
                ],
            },
            "parameters": {"score_threshold": 0.35, "iou_threshold": 0.5},
        },
        {
            "type": "Examples",
            "value": {
                "datum_id": "3",
                "true_positives": [
                    ("uid_3_gt_2", "uid_3_pd_2"),
                    ("uid_3_gt_4", "uid_3_pd_4"),
                    ("uid_3_gt_7", "uid_3_pd_6"),
                ],
                "false_positives": [],
                "false_negatives": [
                    "uid_3_gt_0",
                    "uid_3_gt_1",
                    "uid_3_gt_3",
                    "uid_3_gt_5",
                    "uid_3_gt_6",
                    "uid_3_gt_8",
                    "uid_3_gt_9",
                ],
            },
            "parameters": {"score_threshold": 0.55, "iou_threshold": 0.5},
        },
        {
            "type": "Examples",
            "value": {
                "datum_id": "3",
                "true_positives": [
                    ("uid_3_gt_2", "uid_3_pd_2"),
                    ("uid_3_gt_4", "uid_3_pd_4"),
                ],
                "false_positives": [],
                "false_negatives": [
                    "uid_3_gt_0",
                    "uid_3_gt_1",
                    "uid_3_gt_3",
                    "uid_3_gt_5",
                    "uid_3_gt_6",
                    "uid_3_gt_7",
                    "uid_3_gt_8",
                    "uid_3_gt_9",
                ],
            },
            "parameters": {"score_threshold": 0.75, "iou_threshold": 0.5},
        },
        *[
            {
                "type": "Examples",
                "value": {
                    "datum_id": "3",
                    "true_positives": [("uid_3_gt_4", "uid_3_pd_4")],
                    "false_positives": [],
                    "false_negatives": [
                        "uid_3_gt_0",
                        "uid_3_gt_1",
                        "uid_3_gt_2",
                        "uid_3_gt_3",
                        "uid_3_gt_5",
                        "uid_3_gt_6",
                        "uid_3_gt_7",
                        "uid_3_gt_8",
                        "uid_3_gt_9",
                    ],
                },
                "parameters": {
                    "score_threshold": score_thresh,
                    "iou_threshold": 0.5,
                },
            }
            for score_thresh in [0.8, 0.85]
        ],
        {
            "type": "Examples",
            "value": {
                "datum_id": "3",
                "true_positives": [],
                "false_positives": [],
                "false_negatives": [
                    "uid_3_gt_0",
                    "uid_3_gt_1",
                    "uid_3_gt_2",
                    "uid_3_gt_3",
                    "uid_3_gt_4",
                    "uid_3_gt_5",
                    "uid_3_gt_6",
                    "uid_3_gt_7",
                    "uid_3_gt_8",
                    "uid_3_gt_9",
                ],
            },
            "parameters": {"score_threshold": 0.95, "iou_threshold": 0.5},
        },
        {
            "type": "Examples",
            "value": {
                "datum_id": "3",
                "true_positives": [
                    ("uid_3_gt_0", "uid_3_pd_0"),
                    ("uid_3_gt_4", "uid_3_pd_4"),
                ],
                "false_positives": [
                    "uid_3_pd_1",
                    "uid_3_pd_2",
                    "uid_3_pd_3",
                    "uid_3_pd_5",
                    "uid_3_pd_6",
                    "uid_3_pd_7",
                    "uid_3_pd_8",
                ],
                "false_negatives": [
                    "uid_3_gt_1",
                    "uid_3_gt_2",
                    "uid_3_gt_3",
                    "uid_3_gt_5",
                    "uid_3_gt_6",
                    "uid_3_gt_7",
                    "uid_3_gt_8",
                    "uid_3_gt_9",
                ],
            },
            "parameters": {"score_threshold": 0.05, "iou_threshold": 0.9},
        },
        {
            "type": "Examples",
            "value": {
                "datum_id": "3",
                "true_positives": [
                    ("uid_3_gt_0", "uid_3_pd_0"),
                    ("uid_3_gt_4", "uid_3_pd_4"),
                ],
                "false_positives": [
                    "uid_3_pd_2",
                    "uid_3_pd_5",
                    "uid_3_pd_6",
                    "uid_3_pd_8",
                ],
                "false_negatives": [
                    "uid_3_gt_1",
                    "uid_3_gt_2",
                    "uid_3_gt_3",
                    "uid_3_gt_5",
                    "uid_3_gt_6",
                    "uid_3_gt_7",
                    "uid_3_gt_8",
                    "uid_3_gt_9",
                ],
            },
            "parameters": {"score_threshold": 0.25, "iou_threshold": 0.9},
        },
        {
            "type": "Examples",
            "value": {
                "datum_id": "3",
                "true_positives": [
                    ("uid_3_gt_0", "uid_3_pd_0"),
                    ("uid_3_gt_4", "uid_3_pd_4"),
                ],
                "false_positives": ["uid_3_pd_2", "uid_3_pd_6"],
                "false_negatives": [
                    "uid_3_gt_1",
                    "uid_3_gt_2",
                    "uid_3_gt_3",
                    "uid_3_gt_5",
                    "uid_3_gt_6",
                    "uid_3_gt_7",
                    "uid_3_gt_8",
                    "uid_3_gt_9",
                ],
            },
            "parameters": {"score_threshold": 0.35, "iou_threshold": 0.9},
        },
        {
            "type": "Examples",
            "value": {
                "datum_id": "3",
                "true_positives": [("uid_3_gt_4", "uid_3_pd_4")],
                "false_positives": ["uid_3_pd_2", "uid_3_pd_6"],
                "false_negatives": [
                    "uid_3_gt_0",
                    "uid_3_gt_1",
                    "uid_3_gt_2",
                    "uid_3_gt_3",
                    "uid_3_gt_5",
                    "uid_3_gt_6",
                    "uid_3_gt_7",
                    "uid_3_gt_8",
                    "uid_3_gt_9",
                ],
            },
            "parameters": {"score_threshold": 0.55, "iou_threshold": 0.9},
        },
        {
            "type": "Examples",
            "value": {
                "datum_id": "3",
                "true_positives": [("uid_3_gt_4", "uid_3_pd_4")],
                "false_positives": ["uid_3_pd_2"],
                "false_negatives": [
                    "uid_3_gt_0",
                    "uid_3_gt_1",
                    "uid_3_gt_2",
                    "uid_3_gt_3",
                    "uid_3_gt_5",
                    "uid_3_gt_6",
                    "uid_3_gt_7",
                    "uid_3_gt_8",
                    "uid_3_gt_9",
                ],
            },
            "parameters": {"score_threshold": 0.75, "iou_threshold": 0.9},
        },
        {
            "type": "Examples",
            "value": {
                "datum_id": "3",
                "true_positives": [("uid_3_gt_4", "uid_3_pd_4")],
                "false_positives": [],
                "false_negatives": [
                    "uid_3_gt_0",
                    "uid_3_gt_1",
                    "uid_3_gt_2",
                    "uid_3_gt_3",
                    "uid_3_gt_5",
                    "uid_3_gt_6",
                    "uid_3_gt_7",
                    "uid_3_gt_8",
                    "uid_3_gt_9",
                ],
            },
            "parameters": {"score_threshold": 0.8, "iou_threshold": 0.9},
        },
        {
            "type": "Examples",
            "value": {
                "datum_id": "3",
                "true_positives": [("uid_3_gt_4", "uid_3_pd_4")],
                "false_positives": [],
                "false_negatives": [
                    "uid_3_gt_0",
                    "uid_3_gt_1",
                    "uid_3_gt_2",
                    "uid_3_gt_3",
                    "uid_3_gt_5",
                    "uid_3_gt_6",
                    "uid_3_gt_7",
                    "uid_3_gt_8",
                    "uid_3_gt_9",
                ],
            },
            "parameters": {"score_threshold": 0.85, "iou_threshold": 0.9},
        },
        {
            "type": "Examples",
            "value": {
                "datum_id": "3",
                "true_positives": [],
                "false_positives": [],
                "false_negatives": [
                    "uid_3_gt_0",
                    "uid_3_gt_1",
                    "uid_3_gt_2",
                    "uid_3_gt_3",
                    "uid_3_gt_4",
                    "uid_3_gt_5",
                    "uid_3_gt_6",
                    "uid_3_gt_7",
                    "uid_3_gt_8",
                    "uid_3_gt_9",
                ],
            },
            "parameters": {"score_threshold": 0.95, "iou_threshold": 0.9},
        },
    ]
    for m in actual_metrics:
        assert m in expected_metrics
    for m in expected_metrics:
        assert m in actual_metrics


def test_examples_fp_unmatched_prediction_edge_case(
    detections_fp_unmatched_prediction_edge_case: Evaluator,
):

    evaluator = detections_fp_unmatched_prediction_edge_case
    assert evaluator.info.number_of_datums == 2
    assert evaluator.info.number_of_labels == 1
    assert evaluator.info.number_of_groundtruth_annotations == 2
    assert evaluator.info.number_of_prediction_annotations == 2

    actual_metrics = evaluator.compute_examples(
        iou_thresholds=[0.5],
        score_thresholds=[0.5, 0.85],
    )

    actual_metrics = [m.to_dict() for m in actual_metrics]
    expected_metrics = [
        {
            "type": "Examples",
            "value": {
                "datum_id": "uid1",
                "true_positives": [("uid1_gt0", "uid1_pd0")],
                "false_positives": [],
                "false_negatives": [],
            },
            "parameters": {
                "score_threshold": 0.5,
                "iou_threshold": 0.5,
            },
        },
        {
            "type": "Examples",
            "value": {
                "datum_id": "uid2",
                "true_positives": [],
                "false_positives": ["uid2_pd0"],
                "false_negatives": ["uid2_gt0"],
            },
            "parameters": {
                "score_threshold": 0.5,
                "iou_threshold": 0.5,
            },
        },
        {
            "type": "Examples",
            "value": {
                "datum_id": "uid1",
                "true_positives": [],
                "false_positives": [],
                "false_negatives": ["uid1_gt0"],
            },
            "parameters": {
                "score_threshold": 0.85,
                "iou_threshold": 0.5,
            },
        },
        {
            "type": "Examples",
            "value": {
                "datum_id": "uid2",
                "true_positives": [],
                "false_positives": [],
                "false_negatives": ["uid2_gt0"],
            },
            "parameters": {
                "score_threshold": 0.85,
                "iou_threshold": 0.5,
            },
        },
    ]
    for m in actual_metrics:
        assert m in expected_metrics
    for m in expected_metrics:
        assert m in actual_metrics


def test_examples_ranked_pair_ordering(
    detection_ranked_pair_ordering: Evaluator,
):

    evaluator = detection_ranked_pair_ordering
    assert evaluator.info.number_of_datums == 1
    assert evaluator.info.number_of_groundtruth_annotations == 3
    assert evaluator.info.number_of_labels == 4
    assert evaluator.info.number_of_prediction_annotations == 4
    assert evaluator.info.number_of_rows == 12

    actual_metrics = evaluator.compute_examples(
        iou_thresholds=[0.5],
        score_thresholds=[0.0],
    )

    actual_metrics = [m.to_dict() for m in actual_metrics]
    expected_metrics = [
        {
            "type": "Examples",
            "value": {
                "datum_id": "uid1",
                "true_positives": [],
                "false_positives": ["pd_0", "pd_1", "pd_2", "pd_3"],
                "false_negatives": ["gt_0", "gt_1", "gt_2"],
            },
            "parameters": {
                "score_threshold": 0.0,
                "iou_threshold": 0.5,
            },
        },
    ]
    for m in actual_metrics:
        assert m in expected_metrics
    for m in expected_metrics:
        assert m in actual_metrics


def test_examples_using_torch_metrics_example_paginated(
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

    iou_thresholds = [0.5, 0.9]
    score_thresholds = [0.05, 0.25, 0.35, 0.55, 0.75, 0.8, 0.85, 0.95]
    actual_metrics = evaluator.compute_examples(
        iou_thresholds=iou_thresholds,
        score_thresholds=score_thresholds,
        offset=1,
        limit=2,
    )

    actual_metrics = [m.to_dict() for m in actual_metrics]
    expected_metrics = [
        # datum 1
        *[
            {
                "type": "Examples",
                "value": {
                    "datum_id": "1",
                    "true_positives": [("uid_1_gt_1", "uid_1_pd_1")],
                    "false_positives": ["uid_1_pd_0"],
                    "false_negatives": ["uid_1_gt_0"],
                },
                "parameters": {
                    "score_threshold": score_thresh,
                    "iou_threshold": iou_thresh,
                },
            }
            for iou_thresh in iou_thresholds
            for score_thresh in score_thresholds[:2]
        ],
        *[
            {
                "type": "Examples",
                "value": {
                    "datum_id": "1",
                    "true_positives": [("uid_1_gt_1", "uid_1_pd_1")],
                    "false_positives": [],
                    "false_negatives": ["uid_1_gt_0"],
                },
                "parameters": {
                    "score_threshold": score_thresh,
                    "iou_threshold": iou_thresh,
                },
            }
            for iou_thresh in iou_thresholds
            for score_thresh in score_thresholds[2:4]
        ],
        *[
            {
                "type": "Examples",
                "value": {
                    "datum_id": "1",
                    "true_positives": [],
                    "false_positives": [],
                    "false_negatives": ["uid_1_gt_0", "uid_1_gt_1"],
                },
                "parameters": {
                    "score_threshold": score_thresh,
                    "iou_threshold": iou_thresh,
                },
            }
            for iou_thresh in iou_thresholds
            for score_thresh in score_thresholds[4:]
        ],
        # datum 2
        *[
            {
                "type": "Examples",
                "value": {
                    "datum_id": "2",
                    "true_positives": [
                        ("uid_2_gt_0", "uid_2_pd_0"),
                        ("uid_2_gt_1", "uid_2_pd_1"),
                        ("uid_2_gt_2", "uid_2_pd_2"),
                        ("uid_2_gt_3", "uid_2_pd_3"),
                        ("uid_2_gt_4", "uid_2_pd_4"),
                        ("uid_2_gt_5", "uid_2_pd_5"),
                        ("uid_2_gt_6", "uid_2_pd_6"),
                    ],
                    "false_positives": [],
                    "false_negatives": [],
                },
                "parameters": {
                    "score_threshold": score_thresh,
                    "iou_threshold": 0.5,
                },
            }
            for score_thresh in score_thresholds[:2]
        ],
        {
            "type": "Examples",
            "value": {
                "datum_id": "2",
                "true_positives": [
                    ("uid_2_gt_0", "uid_2_pd_0"),
                    ("uid_2_gt_2", "uid_2_pd_2"),
                    ("uid_2_gt_3", "uid_2_pd_3"),
                    ("uid_2_gt_5", "uid_2_pd_5"),
                    ("uid_2_gt_6", "uid_2_pd_6"),
                ],
                "false_positives": [],
                "false_negatives": ["uid_2_gt_1", "uid_2_gt_4"],
            },
            "parameters": {"score_threshold": 0.35, "iou_threshold": 0.5},
        },
        {
            "type": "Examples",
            "value": {
                "datum_id": "2",
                "true_positives": [
                    ("uid_2_gt_3", "uid_2_pd_3"),
                    ("uid_2_gt_5", "uid_2_pd_5"),
                    ("uid_2_gt_6", "uid_2_pd_6"),
                ],
                "false_positives": [],
                "false_negatives": [
                    "uid_2_gt_0",
                    "uid_2_gt_1",
                    "uid_2_gt_2",
                    "uid_2_gt_4",
                ],
            },
            "parameters": {"score_threshold": 0.55, "iou_threshold": 0.5},
        },
        *[
            {
                "type": "Examples",
                "value": {
                    "datum_id": "2",
                    "true_positives": [
                        ("uid_2_gt_5", "uid_2_pd_5"),
                        ("uid_2_gt_6", "uid_2_pd_6"),
                    ],
                    "false_positives": [],
                    "false_negatives": [
                        "uid_2_gt_0",
                        "uid_2_gt_1",
                        "uid_2_gt_2",
                        "uid_2_gt_3",
                        "uid_2_gt_4",
                    ],
                },
                "parameters": {
                    "score_threshold": score_thresh,
                    "iou_threshold": 0.5,
                },
            }
            for score_thresh in score_thresholds[4:6]
        ],
        *[
            {
                "type": "Examples",
                "value": {
                    "datum_id": "2",
                    "true_positives": [("uid_2_gt_6", "uid_2_pd_6")],
                    "false_positives": [],
                    "false_negatives": [
                        "uid_2_gt_0",
                        "uid_2_gt_1",
                        "uid_2_gt_2",
                        "uid_2_gt_3",
                        "uid_2_gt_4",
                        "uid_2_gt_5",
                    ],
                },
                "parameters": {
                    "score_threshold": score_thresh,
                    "iou_threshold": 0.5,
                },
            }
            for score_thresh in score_thresholds[6:]
        ],
        *[
            {
                "type": "Examples",
                "value": {
                    "datum_id": "2",
                    "true_positives": [("uid_2_gt_4", "uid_2_pd_4")],
                    "false_positives": [
                        "uid_2_pd_0",
                        "uid_2_pd_1",
                        "uid_2_pd_2",
                        "uid_2_pd_3",
                        "uid_2_pd_5",
                        "uid_2_pd_6",
                    ],
                    "false_negatives": [
                        "uid_2_gt_0",
                        "uid_2_gt_1",
                        "uid_2_gt_2",
                        "uid_2_gt_3",
                        "uid_2_gt_5",
                        "uid_2_gt_6",
                    ],
                },
                "parameters": {
                    "score_threshold": score_thresh,
                    "iou_threshold": 0.9,
                },
            }
            for score_thresh in score_thresholds[:2]
        ],
        {
            "type": "Examples",
            "value": {
                "datum_id": "2",
                "true_positives": [],
                "false_positives": [
                    "uid_2_pd_0",
                    "uid_2_pd_2",
                    "uid_2_pd_3",
                    "uid_2_pd_5",
                    "uid_2_pd_6",
                ],
                "false_negatives": [
                    "uid_2_gt_0",
                    "uid_2_gt_1",
                    "uid_2_gt_2",
                    "uid_2_gt_3",
                    "uid_2_gt_4",
                    "uid_2_gt_5",
                    "uid_2_gt_6",
                ],
            },
            "parameters": {"score_threshold": 0.35, "iou_threshold": 0.9},
        },
        {
            "type": "Examples",
            "value": {
                "datum_id": "2",
                "true_positives": [],
                "false_positives": ["uid_2_pd_3", "uid_2_pd_5", "uid_2_pd_6"],
                "false_negatives": [
                    "uid_2_gt_0",
                    "uid_2_gt_1",
                    "uid_2_gt_2",
                    "uid_2_gt_3",
                    "uid_2_gt_4",
                    "uid_2_gt_5",
                    "uid_2_gt_6",
                ],
            },
            "parameters": {"score_threshold": 0.55, "iou_threshold": 0.9},
        },
        *[
            {
                "type": "Examples",
                "value": {
                    "datum_id": "2",
                    "true_positives": [],
                    "false_positives": ["uid_2_pd_5", "uid_2_pd_6"],
                    "false_negatives": [
                        "uid_2_gt_0",
                        "uid_2_gt_1",
                        "uid_2_gt_2",
                        "uid_2_gt_3",
                        "uid_2_gt_4",
                        "uid_2_gt_5",
                        "uid_2_gt_6",
                    ],
                },
                "parameters": {
                    "score_threshold": score_thresh,
                    "iou_threshold": 0.9,
                },
            }
            for score_thresh in [0.75, 0.8]
        ],
        *[
            {
                "type": "Examples",
                "value": {
                    "datum_id": "2",
                    "true_positives": [],
                    "false_positives": ["uid_2_pd_6"],
                    "false_negatives": [
                        "uid_2_gt_0",
                        "uid_2_gt_1",
                        "uid_2_gt_2",
                        "uid_2_gt_3",
                        "uid_2_gt_4",
                        "uid_2_gt_5",
                        "uid_2_gt_6",
                    ],
                },
                "parameters": {
                    "score_threshold": score_thresh,
                    "iou_threshold": 0.9,
                },
            }
            for score_thresh in [0.85, 0.95]
        ],
    ]
    for m in actual_metrics:
        assert m in expected_metrics
    for m in expected_metrics:
        assert m in actual_metrics
