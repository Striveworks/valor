from pathlib import Path

from valor_lite.classification import Classification, DataLoader, MetricType


def test_accuracy_basic(
    tmp_path: Path, basic_classifications: list[Classification]
):
    loader = DataLoader.create(tmp_path)
    loader.add_data(basic_classifications)
    evaluator = loader.finalize()

    assert evaluator.metadata.number_of_datums == 3
    assert evaluator.metadata.number_of_ground_truths == 3
    assert evaluator.metadata.number_of_predictions == 12
    assert evaluator.metadata.number_of_labels == 4

    metrics = evaluator.evaluate(score_thresholds=[0.25, 0.75])

    actual_metrics = [m.to_dict() for m in metrics[MetricType.Accuracy]]
    expected_metrics = [
        {
            "type": "Accuracy",
            "value": 2 / 3,
            "parameters": {
                "score_threshold": 0.25,
                "hardmax": True,
            },
        },
        {
            "type": "Accuracy",
            "value": 1 / 3,
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


def test_accuracy_with_animal_example(
    tmp_path: Path,
    classifications_animal_example: list[Classification],
):

    loader = DataLoader.create(tmp_path)
    loader.add_data(classifications_animal_example)
    evaluator = loader.finalize()

    metrics = evaluator.evaluate(score_thresholds=[0.5])

    actual_metrics = [m.to_dict() for m in metrics[MetricType.Accuracy]]
    expected_metrics = [
        {
            "type": "Accuracy",
            "value": 2 / 6,
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


def test_accuracy_color_example(
    tmp_path: Path,
    classifications_color_example: list[Classification],
):

    loader = DataLoader.create(tmp_path)
    loader.add_data(classifications_color_example)
    evaluator = loader.finalize()

    metrics = evaluator.evaluate(score_thresholds=[0.5])

    actual_metrics = [m.to_dict() for m in metrics[MetricType.Accuracy]]
    expected_metrics = [
        {
            "type": "Accuracy",
            "value": 2 / 6,
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


def test_accuracy_with_image_example(
    tmp_path: Path,
    classifications_image_example: list[Classification],
):
    loader = DataLoader.create(tmp_path)
    loader.add_data(classifications_image_example)
    evaluator = loader.finalize()

    assert evaluator.metadata.number_of_datums == 2
    assert evaluator.metadata.number_of_ground_truths == 2
    assert evaluator.metadata.number_of_predictions == 4
    assert evaluator.metadata.number_of_labels == 4

    metrics = evaluator.evaluate()

    actual_metrics = [m.to_dict() for m in metrics[MetricType.Accuracy]]
    expected_metrics = [
        {
            "type": "Accuracy",
            "value": 0.5,
            "parameters": {
                "score_threshold": 0.0,
                "hardmax": True,
            },
        },
    ]
    for m in actual_metrics:
        assert m in expected_metrics
    for m in expected_metrics:
        assert m in actual_metrics


def test_accuracy_with_tabular_example(
    tmp_path: Path,
    classifications_tabular_example: list[Classification],
):
    loader = DataLoader.create(tmp_path)
    loader.add_data(classifications_tabular_example)
    evaluator = loader.finalize()

    assert evaluator.metadata.number_of_datums == 10
    assert evaluator.metadata.number_of_ground_truths == 10
    assert evaluator.metadata.number_of_predictions == 30
    assert evaluator.metadata.number_of_labels == 3

    metrics = evaluator.evaluate()

    actual_metrics = [m.to_dict() for m in metrics[MetricType.Accuracy]]
    expected_metrics = [
        {
            "type": "Accuracy",
            "value": 0.5,
            "parameters": {
                "score_threshold": 0.0,
                "hardmax": True,
            },
        },
    ]
    for m in actual_metrics:
        assert m in expected_metrics
    for m in expected_metrics:
        assert m in actual_metrics
