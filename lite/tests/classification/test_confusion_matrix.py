import numpy as np
from valor_lite.classification import Classification, DataLoader
from valor_lite.classification.computation import compute_confusion_matrix


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

    # groundtruth count, prediction count, label key
    label_metadata = np.array(
        [
            [2, 1, 0],
            [0, 2, 0],
            [1, 0, 0],
            [1, 1, 0],
        ],
        dtype=np.int32,
    )

    score_thresholds = np.array([0.25, 0.75], dtype=np.float64)

    confusion_matrix, missing_predictions = compute_confusion_matrix(
        data=data,
        label_metadata=label_metadata,
        score_thresholds=score_thresholds,
        hardmax=True,
        n_examples=0,
    )

    assert confusion_matrix.shape == (2, 4, 4, 1)
    assert (
        # score >= 0.25
        confusion_matrix[0, :, :, 0]
        == np.array(
            [
                [1.0, -1.0, 1.0, -1.0],
                [-1.0, -1.0, -1.0, -1.0],
                [-1.0, -1.0, -1.0, -1.0],
                [-1.0, -1.0, -1.0, 1.0],
            ]
        )
    ).all()
    assert (
        # score >= 0.75
        confusion_matrix[1, :, :, 0]
        == np.array(
            [
                [1.0, -1.0, 1.0, -1.0],
                [-1.0, -1.0, -1.0, -1.0],
                [-1.0, -1.0, -1.0, -1.0],
                [-1.0, -1.0, -1.0, -1.0],
            ]
        )
    ).all()

    assert missing_predictions.shape == (2, 4, 1)
    assert (
        # score >= 0.25
        missing_predictions[0, :, 0]
        == np.array([-1.0, -1.0, -1.0, -1.0])
    ).all()
    assert (
        # score >= 0.75
        missing_predictions[1, :, 0]
        == np.array([-1.0, -1.0, -1.0, 1.0])
    ).all()


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

    assert evaluator.metadata == {
        "n_datums": 3,
        "n_groundtruths": 3,
        "n_predictions": 12,
        "n_labels": 4,
        "ignored_prediction_labels": ["1", "2"],
        "missing_prediction_labels": [],
    }

    actual_metrics = evaluator.compute_confusion_matrix(
        score_thresholds=[0.25, 0.75],
        number_of_examples=1,
        as_dict=True,
    )

    expected_metrics = [
        {
            "type": "ConfusionMatrix",
            "value": {
                "confusion_matrix": {
                    "0": {
                        "0": {
                            "count": 1,
                            "examples": [{"datum": "uid0", "score": 1.0}],
                        },
                        "2": {
                            "count": 1,
                            "examples": [{"datum": "uid1", "score": 1.0}],
                        },
                    },
                    "3": {
                        "3": {
                            "count": 1,
                            "examples": [
                                {"datum": "uid2", "score": 0.30000001192092896}
                            ],
                        }
                    },
                },
                "missing_predictions": {},
            },
            "parameters": {"score_threshold": 0.25},
        },
        {
            "type": "ConfusionMatrix",
            "value": {
                "confusion_matrix": {
                    "0": {
                        "0": {
                            "count": 1,
                            "examples": [{"datum": "uid0", "score": 1.0}],
                        },
                        "2": {
                            "count": 1,
                            "examples": [{"datum": "uid1", "score": 1.0}],
                        },
                    }
                },
                "missing_predictions": {
                    "3": {"count": 1, "examples": [{"datum": "uid2"}]}
                },
            },
            "parameters": {"score_threshold": 0.75},
        },
    ]
    for m in actual_metrics:
        _filter_elements_with_zero_count(
            cm=m["value"]["confusion_matrix"],
            mp=m["value"]["missing_predictions"],
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
        as_dict=True,
    )

    expected_metrics = [
        {
            "type": "ConfusionMatrix",
            "value": {
                "confusion_matrix": {
                    "0": {
                        "0": {"count": 1, "examples": []},
                        "1": {"count": 1, "examples": []},
                        "2": {"count": 1, "examples": []},
                    },
                    "1": {"1": {"count": 1, "examples": []}},
                    "2": {"1": {"count": 2, "examples": []}},
                },
                "missing_predictions": {},
            },
            "parameters": {"score_threshold": 0.5},
        },
    ]
    for m in actual_metrics:
        _filter_elements_with_zero_count(
            cm=m["value"]["confusion_matrix"],
            mp=m["value"]["missing_predictions"],
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
        number_of_examples=6,
        as_dict=True,
    )

    expected_metrics = [
        {
            "type": "ConfusionMatrix",
            "value": {
                "confusion_matrix": {
                    "bird": {
                        "bird": {
                            "count": 1,
                            "examples": [
                                {"datum": "uid0", "score": 0.6000000238418579}
                            ],
                        },
                        "dog": {
                            "count": 1,
                            "examples": [{"datum": "uid3", "score": 0.75}],
                        },
                        "cat": {
                            "count": 1,
                            "examples": [
                                {"datum": "uid2", "score": 0.800000011920929}
                            ],
                        },
                    },
                    "dog": {
                        "cat": {
                            "count": 1,
                            "examples": [
                                {"datum": "uid1", "score": 0.8999999761581421}
                            ],
                        }
                    },
                    "cat": {
                        "cat": {
                            "count": 1,
                            "examples": [{"datum": "uid4", "score": 1.0}],
                        }
                    },
                },
                "missing_predictions": {
                    "dog": {"count": 1, "examples": [{"datum": "uid5"}]}
                },
            },
            "parameters": {"score_threshold": 0.5},
        },
    ]
    for m in actual_metrics:
        _filter_elements_with_zero_count(
            cm=m["value"]["confusion_matrix"],
            mp=m["value"]["missing_predictions"],
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

    actual_metrics = evaluator.compute_confusion_matrix(
        score_thresholds=[0.5],
        number_of_examples=6,
        as_dict=True,
    )

    expected_metrics = [
        {
            "type": "ConfusionMatrix",
            "value": {
                "confusion_matrix": {
                    "white": {
                        "white": {
                            "count": 1,
                            "examples": [
                                {"datum": "uid0", "score": 0.6499999761581421}
                            ],
                        },
                        "blue": {
                            "count": 1,
                            "examples": [{"datum": "uid1", "score": 0.5}],
                        },
                    },
                    "red": {
                        "red": {
                            "count": 1,
                            "examples": [
                                {"datum": "uid5", "score": 0.8999999761581421}
                            ],
                        }
                    },
                    "blue": {
                        "white": {
                            "count": 1,
                            "examples": [{"datum": "uid3", "score": 1.0}],
                        }
                    },
                    "black": {
                        "red": {
                            "count": 1,
                            "examples": [
                                {"datum": "uid4", "score": 0.800000011920929}
                            ],
                        }
                    },
                },
                "missing_predictions": {
                    "red": {"count": 1, "examples": [{"datum": "uid2"}]}
                },
            },
            "parameters": {"score_threshold": 0.5},
        },
    ]
    for m in actual_metrics:
        _filter_elements_with_zero_count(
            cm=m["value"]["confusion_matrix"],
            mp=m["value"]["missing_predictions"],
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

    assert evaluator.metadata == {
        "ignored_prediction_labels": [],
        "missing_prediction_labels": [],
        "n_datums": 5,
        "n_groundtruths": 5,
        "n_labels": 3,
        "n_predictions": 15,
    }

    actual_metrics = evaluator.compute_confusion_matrix(
        score_thresholds=[0.05, 0.5, 0.85],
        number_of_examples=5,
        as_dict=True,
    )

    expected_metrics = [
        {
            "type": "ConfusionMatrix",
            "value": {
                "confusion_matrix": {
                    "cat": {
                        "cat": {
                            "count": 1,
                            "examples": [
                                {"datum": "uid0", "score": 0.44598543643951416}
                            ],
                        },
                        "bee": {
                            "count": 1,
                            "examples": [
                                {"datum": "uid2", "score": 0.4026564359664917}
                            ],
                        },
                    },
                    "dog": {
                        "dog": {
                            "count": 1,
                            "examples": [
                                {"datum": "uid4", "score": 0.5890645980834961}
                            ],
                        }
                    },
                    "bee": {
                        "bee": {
                            "count": 2,
                            "examples": [
                                {
                                    "datum": "uid1",
                                    "score": 0.44450607895851135,
                                },
                                {"datum": "uid3", "score": 0.5510573983192444},
                            ],
                        }
                    },
                },
                "missing_predictions": {},
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
                                {"datum": "uid4", "score": 0.5890645980834961}
                            ],
                        }
                    },
                    "bee": {
                        "bee": {
                            "count": 1,
                            "examples": [
                                {"datum": "uid3", "score": 0.5510573983192444}
                            ],
                        }
                    },
                },
                "missing_predictions": {
                    "cat": {
                        "count": 2,
                        "examples": [{"datum": "uid0"}, {"datum": "uid2"}],
                    },
                    "bee": {"count": 1, "examples": [{"datum": "uid1"}]},
                },
            },
            "parameters": {"score_threshold": 0.5},
        },
        {
            "type": "ConfusionMatrix",
            "value": {
                "confusion_matrix": {},
                "missing_predictions": {
                    "cat": {
                        "count": 2,
                        "examples": [{"datum": "uid0"}, {"datum": "uid2"}],
                    },
                    "dog": {"count": 1, "examples": [{"datum": "uid4"}]},
                    "bee": {
                        "count": 2,
                        "examples": [{"datum": "uid1"}, {"datum": "uid3"}],
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
            mp=m["value"]["missing_predictions"],
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

    assert evaluator.metadata == {
        "n_datums": 1,
        "n_groundtruths": 1,
        "n_predictions": 3,
        "n_labels": 3,
        "ignored_prediction_labels": ["bee", "cat"],
        "missing_prediction_labels": [],
    }

    actual_metrics = evaluator.compute_confusion_matrix(
        score_thresholds=[0.05, 0.4, 0.5],
        number_of_examples=6,
        hardmax=False,
        as_dict=True,
    )

    expected_metrics = [
        {
            "type": "ConfusionMatrix",
            "value": {
                "confusion_matrix": {
                    "ant": {
                        "ant": {
                            "count": 1,
                            "examples": [
                                {"datum": "uid1", "score": 0.15000000596046448}
                            ],
                        },
                        "bee": {
                            "count": 1,
                            "examples": [
                                {"datum": "uid1", "score": 0.47999998927116394}
                            ],
                        },
                        "cat": {
                            "count": 1,
                            "examples": [
                                {"datum": "uid1", "score": 0.3700000047683716}
                            ],
                        },
                    }
                },
                "missing_predictions": {},
            },
            "parameters": {"score_threshold": 0.05},
        },
        {
            "type": "ConfusionMatrix",
            "value": {
                "confusion_matrix": {
                    "ant": {
                        "bee": {
                            "count": 1,
                            "examples": [
                                {"datum": "uid1", "score": 0.47999998927116394}
                            ],
                        }
                    }
                },
                "missing_predictions": {},
            },
            "parameters": {"score_threshold": 0.4},
        },
        {
            "type": "ConfusionMatrix",
            "value": {
                "confusion_matrix": {},
                "missing_predictions": {
                    "ant": {
                        "count": 1,
                        "examples": [
                            {
                                "datum": "uid1",
                            }
                        ],
                    }
                },
            },
            "parameters": {"score_threshold": 0.5},
        },
    ]
    for m in actual_metrics:
        _filter_elements_with_zero_count(
            cm=m["value"]["confusion_matrix"],
            mp=m["value"]["missing_predictions"],
        )
        assert m in expected_metrics
    for m in expected_metrics:
        assert m in actual_metrics
