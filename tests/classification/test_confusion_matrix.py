from pathlib import Path

from valor_lite.classification import Classification
from valor_lite.classification.loader import Loader


def _filter_elements_with_zero_count(cm: dict, mp: dict):
    labels = list(mp.keys())

    for gt_label in labels:
        if mp[gt_label] == 0:
            mp.pop(gt_label)
        for pd_label in labels:
            if cm[gt_label][pd_label] == 0:
                cm[gt_label].pop(pd_label)
        if len(cm[gt_label]) == 0:
            cm.pop(gt_label)


def test_confusion_matrix_basic(
    tmp_path: Path, basic_classifications: list[Classification]
):
    loader = Loader.create(tmp_path)
    loader.add_data(basic_classifications)
    evaluator = loader.finalize()

    assert evaluator.info.number_of_datums == 3
    assert evaluator.info.number_of_labels == 4

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
                        "0": 1,
                        "2": 1,
                    },
                    "3": {
                        "3": 1,
                    },
                },
                "unmatched_ground_truths": {},
            },
            "parameters": {
                "score_threshold": 0.25,
                "hardmax": True,
            },
        },
        {
            "type": "ConfusionMatrix",
            "value": {
                "confusion_matrix": {
                    "0": {
                        "0": 1,
                        "2": 1,
                    }
                },
                "unmatched_ground_truths": {
                    "3": 1,
                },
            },
            "parameters": {
                "score_threshold": 0.75,
                "hardmax": True,
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
    tmp_path: Path,
    classifications_from_api_unit_tests: list[Classification],
):

    loader = Loader.create(tmp_path)
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
                        "0": 1,
                        "1": 1,
                        "2": 1,
                    },
                    "1": {
                        "1": 1,
                    },
                    "2": {
                        "1": 2,
                    },
                },
                "unmatched_ground_truths": {},
            },
            "parameters": {
                "score_threshold": 0.5,
                "hardmax": True,
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
    tmp_path: Path,
    classifications_animal_example: list[Classification],
):

    loader = Loader.create(tmp_path)
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
                        "bird": 1,
                        "dog": 1,
                        "cat": 1,
                    },
                    "dog": {
                        "cat": 1,
                    },
                    "cat": {
                        "cat": 1,
                    },
                },
                "unmatched_ground_truths": {
                    "dog": 1,
                },
            },
            "parameters": {
                "score_threshold": 0.5,
                "hardmax": True,
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
    tmp_path: Path,
    classifications_color_example: list[Classification],
):

    loader = Loader.create(tmp_path)
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
                        "white": 1,
                        "blue": 1,
                    },
                    "red": {
                        "red": 1,
                    },
                    "blue": {
                        "white": 1,
                    },
                    "black": {
                        "red": 1,
                    },
                },
                "unmatched_ground_truths": {
                    "red": 1,
                },
            },
            "parameters": {
                "score_threshold": 0.5,
                "hardmax": True,
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
    tmp_path: Path,
    classifications_multiclass: list[Classification],
):
    loader = Loader.create(tmp_path)
    loader.add_data(classifications_multiclass)
    evaluator = loader.finalize()

    assert evaluator.info.number_of_datums == 5
    assert evaluator.info.number_of_labels == 3

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
                        "cat": 1,
                        "bee": 1,
                    },
                    "dog": {
                        "dog": 1,
                    },
                    "bee": {
                        "bee": 2,
                    },
                },
                "unmatched_ground_truths": {},
            },
            "parameters": {
                "score_threshold": 0.05,
                "hardmax": True,
            },
        },
        {
            "type": "ConfusionMatrix",
            "value": {
                "confusion_matrix": {
                    "dog": {
                        "dog": 1,
                    },
                    "bee": {
                        "bee": 1,
                    },
                },
                "unmatched_ground_truths": {
                    "cat": 2,
                    "bee": 1,
                },
            },
            "parameters": {
                "score_threshold": 0.5,
                "hardmax": True,
            },
        },
        {
            "type": "ConfusionMatrix",
            "value": {
                "confusion_matrix": {},
                "unmatched_ground_truths": {
                    "cat": 2,
                    "dog": 1,
                    "bee": 2,
                },
            },
            "parameters": {
                "score_threshold": 0.85,
                "hardmax": True,
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
    tmp_path: Path,
    classifications_multiclass_true_negatives_check: list[Classification],
):
    loader = Loader.create(tmp_path)
    loader.add_data(classifications_multiclass_true_negatives_check)
    evaluator = loader.finalize()

    assert evaluator.info.number_of_datums == 1
    assert evaluator.info.number_of_labels == 3

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
                        "ant": 1,
                        "bee": 1,
                        "cat": 1,
                    }
                },
                "unmatched_ground_truths": {},
            },
            "parameters": {
                "score_threshold": 0.05,
                "hardmax": False,
            },
        },
        {
            "type": "ConfusionMatrix",
            "value": {
                "confusion_matrix": {
                    "ant": {
                        "bee": 1,
                    }
                },
                "unmatched_ground_truths": {},
            },
            "parameters": {
                "score_threshold": 0.4,
                "hardmax": False,
            },
        },
        {
            "type": "ConfusionMatrix",
            "value": {
                "confusion_matrix": {},
                "unmatched_ground_truths": {
                    "ant": 1,
                },
            },
            "parameters": {
                "score_threshold": 0.5,
                "hardmax": False,
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
