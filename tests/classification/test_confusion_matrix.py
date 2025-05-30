import numpy as np

from valor_lite.classification import Classification, DataLoader, Evaluator
from valor_lite.classification.computation import (
    PairClassification,
    compute_confusion_matrix,
)


def test_compute_confusion_matrix():

    # groundtruth, prediction, score
    data = np.array(
        [
            # datum 0
            [0, 0, 0, 1.0, 1.0],  # tp
            [0, 0, 1, 0.0, 0.0],  # tn
            [0, 0, 2, 0.0, 0.0],  # tn
            [0, 0, 3, 0.0, 0.0],  # tn
            # datum 1
            [1, 0, 0, 0.0, 0.0],  # fn
            [1, 0, 1, 0.0, 0.0],  # tn
            [1, 0, 2, 1.0, 1.0],  # fp
            [1, 0, 3, 0.0, 0.0],  # tn
            # datum 2
            [2, 3, 0, 0.0, 0.0],  # tn
            [2, 3, 1, 0.0, 0.0],  # tn
            [2, 3, 2, 0.0, 0.0],  # tn
            [2, 3, 3, 0.3, 1.0],  # fn for score threshold > 0.3
        ],
        dtype=np.float64,
    )
    score_thresholds = np.array([0.25, 0.75], dtype=np.float64)

    result = compute_confusion_matrix(
        detailed_pairs=data,
        score_thresholds=score_thresholds,
        hardmax=True,
    )

    assert result.shape == (2, 12)
    assert np.all(
        result
        == np.array(
            [
                [
                    PairClassification.TP,
                    0,
                    0,
                    0,
                    0,
                    0,
                    PairClassification.FP_FN_MISCLF,
                    0,
                    0,
                    0,
                    0,
                    PairClassification.TP,
                ],
                [
                    PairClassification.TP,
                    0,
                    0,
                    0,
                    0,
                    0,
                    PairClassification.FP_FN_MISCLF,
                    0,
                    PairClassification.FN_UNMATCHED,
                    PairClassification.FN_UNMATCHED,
                    PairClassification.FN_UNMATCHED,
                    PairClassification.FN_UNMATCHED,
                ],
            ],
            dtype=np.uint8,
        ),
    )


def test_compute_confusion_matrix_empty_pairs():
    evaluator = Evaluator()
    metrics = evaluator.compute_confusion_matrix()
    assert metrics == []


def _filter_elements_with_zero_count(cm: dict, mp: dict):
    labels = list(mp.keys())

    for gt_label in labels:
        if mp[gt_label]["count"] == 0:
            mp.pop(gt_label)
        for pd_label in labels:
            if cm[gt_label][pd_label]["count"] == 0:
                cm[gt_label].pop(pd_label)
        if len(cm[gt_label]) == 0:
            cm.pop(gt_label)


def test_confusion_matrix_basic(basic_classifications: list[Classification]):
    loader = DataLoader()
    loader.add_data(basic_classifications)
    evaluator = loader.finalize()

    assert evaluator.ignored_prediction_labels == ["1", "2"]
    assert evaluator.missing_prediction_labels == []
    assert evaluator.metadata.to_dict() == {
        "number_of_datums": 3,
        "number_of_ground_truths": 3,
        "number_of_predictions": 12,
        "number_of_labels": 4,
    }

    actual_metrics = evaluator.compute_confusion_matrix(
        score_thresholds=[0.25, 0.75],
    )

    actual_metrics = [m.to_dict() for m in actual_metrics]
    expected_metrics = [
        {
            "type": "ConfusionMatrix",
            "value": {
                "confusion_matrix": {
                    "0": {
                        "0": {
                            "count": 1,
                            "examples": [{"datum_id": "uid0", "score": 1.0}],
                        },
                        "2": {
                            "count": 1,
                            "examples": [{"datum_id": "uid1", "score": 1.0}],
                        },
                    },
                    "3": {
                        "3": {
                            "count": 1,
                            "examples": [
                                {
                                    "datum_id": "uid2",
                                    "score": 0.3,
                                }
                            ],
                        }
                    },
                },
                "unmatched_ground_truths": {},
            },
            "parameters": {
                "score_threshold": 0.25,
            },
        },
        {
            "type": "ConfusionMatrix",
            "value": {
                "confusion_matrix": {
                    "0": {
                        "0": {
                            "count": 1,
                            "examples": [{"datum_id": "uid0", "score": 1.0}],
                        },
                        "2": {
                            "count": 1,
                            "examples": [{"datum_id": "uid1", "score": 1.0}],
                        },
                    }
                },
                "unmatched_ground_truths": {
                    "3": {"count": 1, "examples": [{"datum_id": "uid2"}]}
                },
            },
            "parameters": {
                "score_threshold": 0.75,
            },
        },
    ]
    for m in actual_metrics:
        _filter_elements_with_zero_count(
            cm=m["value"]["confusion_matrix"],
            mp=m["value"]["unmatched_ground_truths"],
        )
        assert m in expected_metrics
    for m in expected_metrics:
        assert m in actual_metrics


def test_confusion_matrix_unit(
    classifications_from_api_unit_tests: list[Classification],
):

    loader = DataLoader()
    loader.add_data(classifications_from_api_unit_tests)
    evaluator = loader.finalize()

    actual_metrics = evaluator.compute_confusion_matrix(
        score_thresholds=[0.5],
    )

    actual_metrics = [m.to_dict() for m in actual_metrics]
    expected_metrics = [
        {
            "type": "ConfusionMatrix",
            "value": {
                "confusion_matrix": {
                    "0": {
                        "0": {
                            "count": 1,
                            "examples": [
                                {"datum_id": "uid0", "score": 1.0},
                            ],
                        },
                        "1": {
                            "count": 1,
                            "examples": [
                                {"datum_id": "uid1", "score": 1.0},
                            ],
                        },
                        "2": {
                            "count": 1,
                            "examples": [
                                {"datum_id": "uid2", "score": 1.0},
                            ],
                        },
                    },
                    "1": {
                        "1": {
                            "count": 1,
                            "examples": [{"datum_id": "uid3", "score": 1.0}],
                        }
                    },
                    "2": {
                        "1": {
                            "count": 2,
                            "examples": [
                                {"datum_id": "uid4", "score": 1.0},
                                {"datum_id": "uid5", "score": 1.0},
                            ],
                        }
                    },
                },
                "unmatched_ground_truths": {},
            },
            "parameters": {
                "score_threshold": 0.5,
            },
        },
    ]
    for m in actual_metrics:
        _filter_elements_with_zero_count(
            cm=m["value"]["confusion_matrix"],
            mp=m["value"]["unmatched_ground_truths"],
        )
        assert m in expected_metrics
    for m in expected_metrics:
        assert m in actual_metrics


def test_confusion_matrix_with_animal_example(
    classifications_animal_example: list[Classification],
):

    loader = DataLoader()
    loader.add_data(classifications_animal_example)
    evaluator = loader.finalize()

    actual_metrics = evaluator.compute_confusion_matrix(
        score_thresholds=[0.5],
    )

    actual_metrics = [m.to_dict() for m in actual_metrics]
    expected_metrics = [
        {
            "type": "ConfusionMatrix",
            "value": {
                "confusion_matrix": {
                    "bird": {
                        "bird": {
                            "count": 1,
                            "examples": [
                                {
                                    "datum_id": "uid0",
                                    "score": 0.6,
                                }
                            ],
                        },
                        "dog": {
                            "count": 1,
                            "examples": [{"datum_id": "uid3", "score": 0.75}],
                        },
                        "cat": {
                            "count": 1,
                            "examples": [
                                {
                                    "datum_id": "uid2",
                                    "score": 0.8,
                                }
                            ],
                        },
                    },
                    "dog": {
                        "cat": {
                            "count": 1,
                            "examples": [
                                {
                                    "datum_id": "uid1",
                                    "score": 0.9,
                                }
                            ],
                        }
                    },
                    "cat": {
                        "cat": {
                            "count": 1,
                            "examples": [{"datum_id": "uid4", "score": 1.0}],
                        }
                    },
                },
                "unmatched_ground_truths": {
                    "dog": {"count": 1, "examples": [{"datum_id": "uid5"}]}
                },
            },
            "parameters": {
                "score_threshold": 0.5,
            },
        },
    ]
    for m in actual_metrics:
        _filter_elements_with_zero_count(
            cm=m["value"]["confusion_matrix"],
            mp=m["value"]["unmatched_ground_truths"],
        )
        assert m in expected_metrics
    for m in expected_metrics:
        assert m in actual_metrics


def test_confusion_matrix_with_color_example(
    classifications_color_example: list[Classification],
):

    loader = DataLoader()
    loader.add_data(classifications_color_example)
    evaluator = loader.finalize()

    actual_metrics = evaluator.compute_confusion_matrix(score_thresholds=[0.5])

    actual_metrics = [m.to_dict() for m in actual_metrics]
    expected_metrics = [
        {
            "type": "ConfusionMatrix",
            "value": {
                "confusion_matrix": {
                    "white": {
                        "white": {
                            "count": 1,
                            "examples": [
                                {
                                    "datum_id": "uid0",
                                    "score": 0.65,
                                }
                            ],
                        },
                        "blue": {
                            "count": 1,
                            "examples": [{"datum_id": "uid1", "score": 0.5}],
                        },
                    },
                    "red": {
                        "red": {
                            "count": 1,
                            "examples": [
                                {
                                    "datum_id": "uid5",
                                    "score": 0.9,
                                }
                            ],
                        }
                    },
                    "blue": {
                        "white": {
                            "count": 1,
                            "examples": [{"datum_id": "uid3", "score": 1.0}],
                        }
                    },
                    "black": {
                        "red": {
                            "count": 1,
                            "examples": [
                                {
                                    "datum_id": "uid4",
                                    "score": 0.8,
                                }
                            ],
                        }
                    },
                },
                "unmatched_ground_truths": {
                    "red": {"count": 1, "examples": [{"datum_id": "uid2"}]}
                },
            },
            "parameters": {
                "score_threshold": 0.5,
            },
        },
    ]
    for m in actual_metrics:
        _filter_elements_with_zero_count(
            cm=m["value"]["confusion_matrix"],
            mp=m["value"]["unmatched_ground_truths"],
        )
        assert m in expected_metrics
    for m in expected_metrics:
        assert m in actual_metrics


def test_confusion_matrix_multiclass(
    classifications_multiclass: list[Classification],
):
    loader = DataLoader()
    loader.add_data(classifications_multiclass)
    evaluator = loader.finalize()

    assert evaluator.ignored_prediction_labels == []
    assert evaluator.missing_prediction_labels == []
    assert evaluator.metadata.to_dict() == {
        "number_of_datums": 5,
        "number_of_ground_truths": 5,
        "number_of_labels": 3,
        "number_of_predictions": 15,
    }

    actual_metrics = evaluator.compute_confusion_matrix(
        score_thresholds=[0.05, 0.5, 0.85],
    )

    actual_metrics = [m.to_dict() for m in actual_metrics]
    expected_metrics = [
        {
            "type": "ConfusionMatrix",
            "value": {
                "confusion_matrix": {
                    "cat": {
                        "cat": {
                            "count": 1,
                            "examples": [
                                {
                                    "datum_id": "uid0",
                                    "score": 0.44598543489942505,
                                }
                            ],
                        },
                        "bee": {
                            "count": 1,
                            "examples": [
                                {
                                    "datum_id": "uid2",
                                    "score": 0.4076893257212283,
                                }
                            ],
                        },
                    },
                    "dog": {
                        "dog": {
                            "count": 1,
                            "examples": [
                                {
                                    "datum_id": "uid4",
                                    "score": 0.5890646197236098,
                                }
                            ],
                        }
                    },
                    "bee": {
                        "bee": {
                            "count": 2,
                            "examples": [
                                {
                                    "datum_id": "uid1",
                                    "score": 0.4445060886392194,
                                },
                                {
                                    "datum_id": "uid3",
                                    "score": 0.5510573702493565,
                                },
                            ],
                        }
                    },
                },
                "unmatched_ground_truths": {},
            },
            "parameters": {
                "score_threshold": 0.05,
            },
        },
        {
            "type": "ConfusionMatrix",
            "value": {
                "confusion_matrix": {
                    "dog": {
                        "dog": {
                            "count": 1,
                            "examples": [
                                {
                                    "datum_id": "uid4",
                                    "score": 0.5890646197236098,
                                }
                            ],
                        }
                    },
                    "bee": {
                        "bee": {
                            "count": 1,
                            "examples": [
                                {
                                    "datum_id": "uid3",
                                    "score": 0.5510573702493565,
                                }
                            ],
                        }
                    },
                },
                "unmatched_ground_truths": {
                    "cat": {
                        "count": 2,
                        "examples": [
                            {"datum_id": "uid0"},
                            {"datum_id": "uid2"},
                        ],
                    },
                    "bee": {"count": 1, "examples": [{"datum_id": "uid1"}]},
                },
            },
            "parameters": {
                "score_threshold": 0.5,
            },
        },
        {
            "type": "ConfusionMatrix",
            "value": {
                "confusion_matrix": {},
                "unmatched_ground_truths": {
                    "cat": {
                        "count": 2,
                        "examples": [
                            {"datum_id": "uid0"},
                            {"datum_id": "uid2"},
                        ],
                    },
                    "dog": {"count": 1, "examples": [{"datum_id": "uid4"}]},
                    "bee": {
                        "count": 2,
                        "examples": [
                            {"datum_id": "uid1"},
                            {"datum_id": "uid3"},
                        ],
                    },
                },
            },
            "parameters": {
                "score_threshold": 0.85,
            },
        },
    ]
    for m in actual_metrics:
        _filter_elements_with_zero_count(
            cm=m["value"]["confusion_matrix"],
            mp=m["value"]["unmatched_ground_truths"],
        )
        assert m in expected_metrics
    for m in expected_metrics:
        assert m in actual_metrics


def test_confusion_matrix_without_hardmax_animal_example(
    classifications_multiclass_true_negatives_check: list[Classification],
):
    loader = DataLoader()
    loader.add_data(classifications_multiclass_true_negatives_check)
    evaluator = loader.finalize()

    assert evaluator.ignored_prediction_labels == ["bee", "cat"]
    assert evaluator.missing_prediction_labels == []
    assert evaluator.metadata.to_dict() == {
        "number_of_datums": 1,
        "number_of_ground_truths": 1,
        "number_of_predictions": 3,
        "number_of_labels": 3,
    }

    actual_metrics = evaluator.compute_confusion_matrix(
        score_thresholds=[0.05, 0.4, 0.5],
        hardmax=False,
    )

    actual_metrics = [m.to_dict() for m in actual_metrics]
    expected_metrics = [
        {
            "type": "ConfusionMatrix",
            "value": {
                "confusion_matrix": {
                    "ant": {
                        "ant": {
                            "count": 1,
                            "examples": [
                                {
                                    "datum_id": "uid1",
                                    "score": 0.15,
                                }
                            ],
                        },
                        "bee": {
                            "count": 1,
                            "examples": [
                                {
                                    "datum_id": "uid1",
                                    "score": 0.48,
                                }
                            ],
                        },
                        "cat": {
                            "count": 1,
                            "examples": [
                                {
                                    "datum_id": "uid1",
                                    "score": 0.37,
                                }
                            ],
                        },
                    }
                },
                "unmatched_ground_truths": {},
            },
            "parameters": {
                "score_threshold": 0.05,
            },
        },
        {
            "type": "ConfusionMatrix",
            "value": {
                "confusion_matrix": {
                    "ant": {
                        "bee": {
                            "count": 1,
                            "examples": [
                                {
                                    "datum_id": "uid1",
                                    "score": 0.48,
                                }
                            ],
                        }
                    }
                },
                "unmatched_ground_truths": {},
            },
            "parameters": {
                "score_threshold": 0.4,
            },
        },
        {
            "type": "ConfusionMatrix",
            "value": {
                "confusion_matrix": {},
                "unmatched_ground_truths": {
                    "ant": {
                        "count": 1,
                        "examples": [
                            {
                                "datum_id": "uid1",
                            }
                        ],
                    }
                },
            },
            "parameters": {
                "score_threshold": 0.5,
            },
        },
    ]
    for m in actual_metrics:
        _filter_elements_with_zero_count(
            cm=m["value"]["confusion_matrix"],
            mp=m["value"]["unmatched_ground_truths"],
        )
        assert m in expected_metrics
    for m in expected_metrics:
        assert m in actual_metrics
