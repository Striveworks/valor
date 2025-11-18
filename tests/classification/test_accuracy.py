from valor_lite.classification import Classification, Loader, MetricType


def test_accuracy_basic(
    loader: Loader, basic_classifications: list[Classification]
):
    loader.add_data(basic_classifications)
    evaluator = loader.finalize()

    assert evaluator.info.number_of_datums == 3
    assert evaluator.info.number_of_labels == 4
    assert evaluator.info.number_of_rows == 12

    metrics = evaluator.compute_precision_recall(score_thresholds=[0.25, 0.75])

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
    loader: Loader,
    classifications_animal_example: list[Classification],
):

    loader.add_data(classifications_animal_example)
    evaluator = loader.finalize()

    metrics = evaluator.compute_precision_recall(score_thresholds=[0.5])

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
    loader: Loader,
    classifications_color_example: list[Classification],
):

    loader.add_data(classifications_color_example)
    evaluator = loader.finalize()

    metrics = evaluator.compute_precision_recall(score_thresholds=[0.5])

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
    loader: Loader,
    classifications_image_example: list[Classification],
):
    loader.add_data(classifications_image_example)
    evaluator = loader.finalize()

    assert evaluator.info.number_of_datums == 2
    assert evaluator.info.number_of_labels == 4
    assert evaluator.info.number_of_rows == 4

    metrics = evaluator.compute_precision_recall()

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
    loader: Loader,
    classifications_tabular_example: list[Classification],
):
    loader.add_data(classifications_tabular_example)
    evaluator = loader.finalize()

    assert evaluator.info.number_of_datums == 10
    assert evaluator.info.number_of_labels == 3
    assert evaluator.info.number_of_rows == 30

    metrics = evaluator.compute_precision_recall()

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
