from pathlib import Path

from valor_lite.classification import Classification
from valor_lite.classification.loader import Loader


def test_examples_basic(
    tmp_path: Path, basic_classifications: list[Classification]
):
    loader = Loader.create(tmp_path)
    loader.add_data(basic_classifications)
    evaluator = loader.finalize()

    assert evaluator.info.number_of_datums == 3
    assert evaluator.info.number_of_labels == 4

    actual_metrics = evaluator.compute_examples(
        score_thresholds=[0.25, 0.75],
    )

    actual_metrics = [m.to_dict() for m in actual_metrics]
    expected_metrics = [
        # score threshold = 0.25
        {
            "type": "Examples",
            "value": {
                "datum_id": "uid0",
                "true_positives": ["0"],
                "false_positives": [],
                "false_negatives": [],
            },
            "parameters": {
                "score_threshold": 0.25,
                "hardmax": True,
            },
        },
        {
            "type": "Examples",
            "value": {
                "datum_id": "uid1",
                "true_positives": [],
                "false_positives": ["2"],
                "false_negatives": ["0"],
            },
            "parameters": {
                "score_threshold": 0.25,
                "hardmax": True,
            },
        },
        {
            "type": "Examples",
            "value": {
                "datum_id": "uid2",
                "true_positives": ["3"],
                "false_positives": [],
                "false_negatives": [],
            },
            "parameters": {
                "score_threshold": 0.25,
                "hardmax": True,
            },
        },
        # score threshold = 0.75
        {
            "type": "Examples",
            "value": {
                "datum_id": "uid0",
                "true_positives": ["0"],
                "false_positives": [],
                "false_negatives": [],
            },
            "parameters": {
                "score_threshold": 0.75,
                "hardmax": True,
            },
        },
        {
            "type": "Examples",
            "value": {
                "datum_id": "uid1",
                "true_positives": [],
                "false_positives": ["2"],
                "false_negatives": ["0"],
            },
            "parameters": {
                "score_threshold": 0.75,
                "hardmax": True,
            },
        },
        {
            "type": "Examples",
            "value": {
                "datum_id": "uid2",
                "true_positives": [],
                "false_positives": [],
                "false_negatives": ["3"],
            },
            "parameters": {
                "score_threshold": 0.75,
                "hardmax": True,
            },
        },
    ]
    for m in actual_metrics:
        assert m in expected_metrics
    for m in expected_metrics:
        assert m in actual_metrics


def test_examples_unit(
    tmp_path: Path,
    classifications_from_api_unit_tests: list[Classification],
):

    loader = Loader.create(tmp_path)
    loader.add_data(classifications_from_api_unit_tests)
    evaluator = loader.finalize()

    actual_metrics = evaluator.compute_examples(
        score_thresholds=[0.5],
    )

    actual_metrics = [m.to_dict() for m in actual_metrics]
    expected_metrics = [
        {
            "type": "Examples",
            "value": {
                "datum_id": "uid0",
                "true_positives": ["0"],
                "false_positives": [],
                "false_negatives": [],
            },
            "parameters": {
                "score_threshold": 0.5,
                "hardmax": True,
            },
        },
        {
            "type": "Examples",
            "value": {
                "datum_id": "uid1",
                "true_positives": [],
                "false_positives": ["1"],
                "false_negatives": ["0"],
            },
            "parameters": {
                "score_threshold": 0.5,
                "hardmax": True,
            },
        },
        {
            "type": "Examples",
            "value": {
                "datum_id": "uid2",
                "true_positives": [],
                "false_positives": ["2"],
                "false_negatives": ["0"],
            },
            "parameters": {
                "score_threshold": 0.5,
                "hardmax": True,
            },
        },
        {
            "type": "Examples",
            "value": {
                "datum_id": "uid3",
                "true_positives": ["1"],
                "false_positives": [],
                "false_negatives": [],
            },
            "parameters": {
                "score_threshold": 0.5,
                "hardmax": True,
            },
        },
        {
            "type": "Examples",
            "value": {
                "datum_id": "uid4",
                "true_positives": [],
                "false_positives": ["1"],
                "false_negatives": ["2"],
            },
            "parameters": {
                "score_threshold": 0.5,
                "hardmax": True,
            },
        },
        {
            "type": "Examples",
            "value": {
                "datum_id": "uid5",
                "true_positives": [],
                "false_positives": ["1"],
                "false_negatives": ["2"],
            },
            "parameters": {
                "score_threshold": 0.5,
                "hardmax": True,
            },
        },
    ]
    for m in actual_metrics:
        assert m in expected_metrics
    for m in expected_metrics:
        assert m in actual_metrics


def test_examples_with_animal_example(
    tmp_path: Path,
    classifications_animal_example: list[Classification],
):

    loader = Loader.create(tmp_path)
    loader.add_data(classifications_animal_example)
    evaluator = loader.finalize()

    actual_metrics = evaluator.compute_examples(
        score_thresholds=[0.5],
    )

    actual_metrics = [m.to_dict() for m in actual_metrics]
    expected_metrics = [
        {
            "type": "Examples",
            "value": {
                "datum_id": "uid0",
                "true_positives": ["bird"],
                "false_positives": [],
                "false_negatives": [],
            },
            "parameters": {
                "score_threshold": 0.5,
                "hardmax": True,
            },
        },
        {
            "type": "Examples",
            "value": {
                "datum_id": "uid1",
                "true_positives": [],
                "false_positives": ["cat"],
                "false_negatives": ["dog"],
            },
            "parameters": {
                "score_threshold": 0.5,
                "hardmax": True,
            },
        },
        {
            "type": "Examples",
            "value": {
                "datum_id": "uid2",
                "true_positives": [],
                "false_positives": ["cat"],
                "false_negatives": ["bird"],
            },
            "parameters": {
                "score_threshold": 0.5,
                "hardmax": True,
            },
        },
        {
            "type": "Examples",
            "value": {
                "datum_id": "uid3",
                "true_positives": [],
                "false_positives": ["dog"],
                "false_negatives": ["bird"],
            },
            "parameters": {
                "score_threshold": 0.5,
                "hardmax": True,
            },
        },
        {
            "type": "Examples",
            "value": {
                "datum_id": "uid4",
                "true_positives": ["cat"],
                "false_positives": [],
                "false_negatives": [],
            },
            "parameters": {
                "score_threshold": 0.5,
                "hardmax": True,
            },
        },
        {
            "type": "Examples",
            "value": {
                "datum_id": "uid5",
                "true_positives": [],
                "false_positives": [],
                "false_negatives": ["dog"],
            },
            "parameters": {
                "score_threshold": 0.5,
                "hardmax": True,
            },
        },
    ]
    for m in actual_metrics:
        assert m in expected_metrics
    for m in expected_metrics:
        assert m in actual_metrics


def test_examples_with_color_example(
    tmp_path: Path,
    classifications_color_example: list[Classification],
):

    loader = Loader.create(tmp_path)
    loader.add_data(classifications_color_example)
    evaluator = loader.finalize()

    actual_metrics = evaluator.compute_examples(score_thresholds=[0.5])

    actual_metrics = [m.to_dict() for m in actual_metrics]
    expected_metrics = [
        {
            "type": "Examples",
            "value": {
                "datum_id": "uid0",
                "true_positives": ["white"],
                "false_positives": [],
                "false_negatives": [],
            },
            "parameters": {
                "score_threshold": 0.5,
                "hardmax": True,
            },
        },
        {
            "type": "Examples",
            "value": {
                "datum_id": "uid1",
                "true_positives": [],
                "false_positives": ["blue"],
                "false_negatives": ["white"],
            },
            "parameters": {
                "score_threshold": 0.5,
                "hardmax": True,
            },
        },
        {
            "type": "Examples",
            "value": {
                "datum_id": "uid2",
                "true_positives": [],
                "false_positives": [],
                "false_negatives": ["red"],
            },
            "parameters": {
                "score_threshold": 0.5,
                "hardmax": True,
            },
        },
        {
            "type": "Examples",
            "value": {
                "datum_id": "uid3",
                "true_positives": [],
                "false_positives": ["white"],
                "false_negatives": ["blue"],
            },
            "parameters": {
                "score_threshold": 0.5,
                "hardmax": True,
            },
        },
        {
            "type": "Examples",
            "value": {
                "datum_id": "uid4",
                "true_positives": [],
                "false_positives": ["red"],
                "false_negatives": ["black"],
            },
            "parameters": {
                "score_threshold": 0.5,
                "hardmax": True,
            },
        },
        {
            "type": "Examples",
            "value": {
                "datum_id": "uid5",
                "true_positives": ["red"],
                "false_positives": [],
                "false_negatives": [],
            },
            "parameters": {
                "score_threshold": 0.5,
                "hardmax": True,
            },
        },
    ]
    for m in actual_metrics:
        assert m in expected_metrics
    for m in expected_metrics:
        assert m in actual_metrics


def test_examples_multiclass(
    tmp_path: Path,
    classifications_multiclass: list[Classification],
):
    loader = Loader.create(tmp_path)
    loader.add_data(classifications_multiclass)
    evaluator = loader.finalize()

    assert evaluator.info.number_of_datums == 5
    assert evaluator.info.number_of_labels == 3

    actual_metrics = evaluator.compute_examples(
        score_thresholds=[0.05, 0.5, 0.85],
    )

    actual_metrics = [m.to_dict() for m in actual_metrics]
    expected_metrics = [
        # score threshold = 0.05
        {
            "type": "Examples",
            "value": {
                "datum_id": "uid0",
                "true_positives": ["cat"],
                "false_positives": [],
                "false_negatives": [],
            },
            "parameters": {
                "score_threshold": 0.05,
                "hardmax": True,
            },
        },
        {
            "type": "Examples",
            "value": {
                "datum_id": "uid1",
                "true_positives": ["bee"],
                "false_positives": [],
                "false_negatives": [],
            },
            "parameters": {
                "score_threshold": 0.05,
                "hardmax": True,
            },
        },
        {
            "type": "Examples",
            "value": {
                "datum_id": "uid2",
                "true_positives": [],
                "false_positives": ["bee"],
                "false_negatives": ["cat"],
            },
            "parameters": {
                "score_threshold": 0.05,
                "hardmax": True,
            },
        },
        {
            "type": "Examples",
            "value": {
                "datum_id": "uid3",
                "true_positives": ["bee"],
                "false_positives": [],
                "false_negatives": [],
            },
            "parameters": {
                "score_threshold": 0.05,
                "hardmax": True,
            },
        },
        {
            "type": "Examples",
            "value": {
                "datum_id": "uid4",
                "true_positives": ["dog"],
                "false_positives": [],
                "false_negatives": [],
            },
            "parameters": {
                "score_threshold": 0.05,
                "hardmax": True,
            },
        },
        # score threshold = 0.5
        {
            "type": "Examples",
            "value": {
                "datum_id": "uid0",
                "true_positives": [],
                "false_positives": [],
                "false_negatives": ["cat"],
            },
            "parameters": {
                "score_threshold": 0.5,
                "hardmax": True,
            },
        },
        {
            "type": "Examples",
            "value": {
                "datum_id": "uid1",
                "true_positives": [],
                "false_positives": [],
                "false_negatives": ["bee"],
            },
            "parameters": {
                "score_threshold": 0.5,
                "hardmax": True,
            },
        },
        {
            "type": "Examples",
            "value": {
                "datum_id": "uid2",
                "true_positives": [],
                "false_positives": [],
                "false_negatives": ["cat"],
            },
            "parameters": {
                "score_threshold": 0.5,
                "hardmax": True,
            },
        },
        {
            "type": "Examples",
            "value": {
                "datum_id": "uid3",
                "true_positives": ["bee"],
                "false_positives": [],
                "false_negatives": [],
            },
            "parameters": {
                "score_threshold": 0.5,
                "hardmax": True,
            },
        },
        {
            "type": "Examples",
            "value": {
                "datum_id": "uid4",
                "true_positives": ["dog"],
                "false_positives": [],
                "false_negatives": [],
            },
            "parameters": {
                "score_threshold": 0.5,
                "hardmax": True,
            },
        },
        # score threshold = 0.85
        {
            "type": "Examples",
            "value": {
                "datum_id": "uid0",
                "true_positives": [],
                "false_positives": [],
                "false_negatives": ["cat"],
            },
            "parameters": {
                "score_threshold": 0.85,
                "hardmax": True,
            },
        },
        {
            "type": "Examples",
            "value": {
                "datum_id": "uid1",
                "true_positives": [],
                "false_positives": [],
                "false_negatives": ["bee"],
            },
            "parameters": {
                "score_threshold": 0.85,
                "hardmax": True,
            },
        },
        {
            "type": "Examples",
            "value": {
                "datum_id": "uid2",
                "true_positives": [],
                "false_positives": [],
                "false_negatives": ["cat"],
            },
            "parameters": {
                "score_threshold": 0.85,
                "hardmax": True,
            },
        },
        {
            "type": "Examples",
            "value": {
                "datum_id": "uid3",
                "true_positives": [],
                "false_positives": [],
                "false_negatives": ["bee"],
            },
            "parameters": {
                "score_threshold": 0.85,
                "hardmax": True,
            },
        },
        {
            "type": "Examples",
            "value": {
                "datum_id": "uid4",
                "true_positives": [],
                "false_positives": [],
                "false_negatives": ["dog"],
            },
            "parameters": {
                "score_threshold": 0.85,
                "hardmax": True,
            },
        },
    ]
    for m in actual_metrics:
        assert m in expected_metrics
    for m in expected_metrics:
        assert m in actual_metrics


def test_examples_without_hardmax_animal_example(
    tmp_path: Path,
    classifications_multiclass_true_negatives_check: list[Classification],
):
    loader = Loader.create(tmp_path)
    loader.add_data(classifications_multiclass_true_negatives_check)
    evaluator = loader.finalize()

    assert evaluator.info.number_of_datums == 1
    assert evaluator.info.number_of_labels == 3

    actual_metrics = evaluator.compute_examples(
        score_thresholds=[0.05, 0.4, 0.5],
        hardmax=False,
    )

    actual_metrics = [m.to_dict() for m in actual_metrics]
    expected_metrics = [
        {
            "type": "Examples",
            "value": {
                "datum_id": "uid1",
                "true_positives": ["ant"],
                "false_positives": ["bee", "cat"],
                "false_negatives": [],
            },
            "parameters": {
                "score_threshold": 0.05,
                "hardmax": False,
            },
        },
        {
            "type": "Examples",
            "value": {
                "datum_id": "uid1",
                "true_positives": [],
                "false_positives": ["bee"],
                "false_negatives": ["ant"],
            },
            "parameters": {
                "score_threshold": 0.4,
                "hardmax": False,
            },
        },
        {
            "type": "Examples",
            "value": {
                "datum_id": "uid1",
                "true_positives": [],
                "false_positives": [],
                "false_negatives": ["ant"],
            },
            "parameters": {
                "score_threshold": 0.5,
                "hardmax": False,
            },
        },
    ]
    for m in actual_metrics:
        assert m in expected_metrics
    for m in expected_metrics:
        assert m in actual_metrics
