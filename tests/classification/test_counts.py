from valor_lite.classification import Classification, Loader, MetricType


def test_counts_basic(
    loader: Loader, basic_classifications: list[Classification]
):
    loader.add_data(basic_classifications)
    evaluator = loader.finalize()

    assert evaluator.info.number_of_datums == 3
    assert evaluator.info.number_of_labels == 4
    assert evaluator.info.number_of_rows == 12

    metrics = evaluator.compute_precision_recall(
        score_thresholds=[0.25, 0.75],
    )

    actual_metrics = [m.to_dict() for m in metrics[MetricType.Counts]]
    expected_metrics = [
        # score >= 0.25
        {
            "type": "Counts",
            "value": {
                "tp": 1,
                "fp": 0,
                "fn": 1,
                "tn": 1,
            },
            "parameters": {
                "score_threshold": 0.25,
                "hardmax": True,
                "label": "0",
            },
        },
        {
            "type": "Counts",
            "value": {
                "tp": 0,
                "fp": 0,
                "fn": 0,
                "tn": 3,
            },
            "parameters": {
                "score_threshold": 0.25,
                "hardmax": True,
                "label": "1",
            },
        },
        {
            "type": "Counts",
            "value": {
                "tp": 0,
                "fp": 1,
                "fn": 0,
                "tn": 2,
            },
            "parameters": {
                "score_threshold": 0.25,
                "hardmax": True,
                "label": "2",
            },
        },
        {
            "type": "Counts",
            "value": {
                "tp": 1,
                "fp": 0,
                "fn": 0,
                "tn": 2,
            },
            "parameters": {
                "score_threshold": 0.25,
                "hardmax": True,
                "label": "3",
            },
        },
        # score >= 0.75
        {
            "type": "Counts",
            "value": {
                "tp": 1,
                "fp": 0,
                "fn": 1,
                "tn": 1,
            },
            "parameters": {
                "score_threshold": 0.75,
                "hardmax": True,
                "label": "0",
            },
        },
        {
            "type": "Counts",
            "value": {
                "tp": 0,
                "fp": 0,
                "fn": 0,
                "tn": 3,
            },
            "parameters": {
                "score_threshold": 0.75,
                "hardmax": True,
                "label": "1",
            },
        },
        {
            "type": "Counts",
            "value": {
                "tp": 0,
                "fp": 1,
                "fn": 0,
                "tn": 2,
            },
            "parameters": {
                "score_threshold": 0.75,
                "hardmax": True,
                "label": "2",
            },
        },
        {
            "type": "Counts",
            "value": {
                "tp": 0,
                "fp": 0,
                "fn": 1,
                "tn": 2,
            },
            "parameters": {
                "score_threshold": 0.75,
                "hardmax": True,
                "label": "3",
            },
        },
    ]
    for m in actual_metrics:
        assert m in expected_metrics
    for m in expected_metrics:
        assert m in actual_metrics


def test_counts_unit(
    loader: Loader,
    classifications_from_api_unit_tests: list[Classification],
):

    loader.add_data(classifications_from_api_unit_tests)
    evaluator = loader.finalize()

    metrics = evaluator.compute_precision_recall(
        score_thresholds=[0.5],
    )

    actual_metrics = [m.to_dict() for m in metrics[MetricType.Counts]]
    expected_metrics = [
        {
            "type": "Counts",
            "value": {
                "tp": 1,
                "fp": 0,
                "fn": 2,
                "tn": 3,
            },
            "parameters": {
                "score_threshold": 0.5,
                "hardmax": True,
                "label": "0",
            },
        },
        {
            "type": "Counts",
            "value": {
                "tp": 1,
                "fp": 3,
                "fn": 0,
                "tn": 2,
            },
            "parameters": {
                "score_threshold": 0.5,
                "hardmax": True,
                "label": "1",
            },
        },
        {
            "type": "Counts",
            "value": {
                "tp": 0,
                "fp": 1,
                "fn": 2,
                "tn": 3,
            },
            "parameters": {
                "score_threshold": 0.5,
                "hardmax": True,
                "label": "2",
            },
        },
    ]
    for m in actual_metrics:
        assert m in expected_metrics
    for m in expected_metrics:
        assert m in actual_metrics


def test_counts_with_animal_example(
    loader: Loader,
    classifications_animal_example: list[Classification],
):

    loader.add_data(classifications_animal_example)
    evaluator = loader.finalize()

    metrics = evaluator.compute_precision_recall(
        score_thresholds=[0.05, 0.5, 0.95],
    )

    actual_metrics = [m.to_dict() for m in metrics[MetricType.Counts]]
    expected_metrics = [
        # score >= 0.05
        {
            "type": "Counts",
            "value": {
                "tp": 1,
                "fp": 0,
                "fn": 2,
                "tn": 3,
            },
            "parameters": {
                "score_threshold": 0.05,
                "hardmax": True,
                "label": "bird",
            },
        },
        {
            "type": "Counts",
            "value": {
                "tp": 0,
                "fp": 1,
                "fn": 2,
                "tn": 3,
            },
            "parameters": {
                "score_threshold": 0.05,
                "hardmax": True,
                "label": "dog",
            },
        },
        {
            "type": "Counts",
            "value": {
                "tp": 1,
                "fp": 3,
                "fn": 0,
                "tn": 2,
            },
            "parameters": {
                "score_threshold": 0.05,
                "hardmax": True,
                "label": "cat",
            },
        },
        # score >= 0.5
        {
            "type": "Counts",
            "value": {
                "tp": 1,
                "fp": 0,
                "fn": 2,
                "tn": 3,
            },
            "parameters": {
                "score_threshold": 0.5,
                "hardmax": True,
                "label": "bird",
            },
        },
        {
            "type": "Counts",
            "value": {
                "tp": 0,
                "fp": 1,
                "fn": 2,
                "tn": 3,
            },
            "parameters": {
                "score_threshold": 0.5,
                "hardmax": True,
                "label": "dog",
            },
        },
        {
            "type": "Counts",
            "value": {
                "tp": 1,
                "fp": 2,
                "fn": 0,
                "tn": 3,
            },
            "parameters": {
                "score_threshold": 0.5,
                "hardmax": True,
                "label": "cat",
            },
        },
        # score >= 0.95
        {
            "type": "Counts",
            "value": {
                "tp": 0,
                "fp": 0,
                "fn": 3,
                "tn": 3,
            },
            "parameters": {
                "score_threshold": 0.95,
                "hardmax": True,
                "label": "bird",
            },
        },
        {
            "type": "Counts",
            "value": {
                "tp": 0,
                "fp": 0,
                "fn": 2,
                "tn": 4,
            },
            "parameters": {
                "score_threshold": 0.95,
                "hardmax": True,
                "label": "dog",
            },
        },
        {
            "type": "Counts",
            "value": {
                "tp": 1,
                "fp": 0,
                "fn": 0,
                "tn": 5,
            },
            "parameters": {
                "score_threshold": 0.95,
                "hardmax": True,
                "label": "cat",
            },
        },
    ]
    for m in actual_metrics:
        assert m in expected_metrics
    for m in expected_metrics:
        assert m in actual_metrics


def test_counts_with_color_example(
    loader: Loader,
    classifications_color_example: list[Classification],
):

    loader.add_data(classifications_color_example)
    evaluator = loader.finalize()

    metrics = evaluator.compute_precision_recall(
        score_thresholds=[0.05, 0.5, 0.95],
    )

    actual_metrics = [m.to_dict() for m in metrics[MetricType.Counts]]
    expected_metrics = [
        # score >= 0.05
        {
            "type": "Counts",
            "value": {
                "tp": 1,
                "fp": 1,
                "fn": 1,
                "tn": 3,
            },
            "parameters": {
                "score_threshold": 0.05,
                "hardmax": True,
                "label": "white",
            },
        },
        {
            "type": "Counts",
            "value": {
                "tp": 2,
                "fp": 1,
                "fn": 0,
                "tn": 3,
            },
            "parameters": {
                "score_threshold": 0.05,
                "hardmax": True,
                "label": "red",
            },
        },
        {
            "type": "Counts",
            "value": {
                "tp": 0,
                "fp": 1,
                "fn": 1,
                "tn": 4,
            },
            "parameters": {
                "score_threshold": 0.05,
                "hardmax": True,
                "label": "blue",
            },
        },
        {
            "type": "Counts",
            "value": {
                "tp": 0,
                "fp": 0,
                "fn": 1,
                "tn": 5,
            },
            "parameters": {
                "score_threshold": 0.05,
                "hardmax": True,
                "label": "black",
            },
        },
        # score >= 0.5
        {
            "type": "Counts",
            "value": {
                "tp": 1,
                "fp": 1,
                "fn": 1,
                "tn": 3,
            },
            "parameters": {
                "score_threshold": 0.5,
                "hardmax": True,
                "label": "white",
            },
        },
        {
            "type": "Counts",
            "value": {
                "tp": 1,
                "fp": 1,
                "fn": 1,
                "tn": 3,
            },
            "parameters": {
                "score_threshold": 0.5,
                "hardmax": True,
                "label": "red",
            },
        },
        {
            "type": "Counts",
            "value": {
                "tp": 0,
                "fp": 1,
                "fn": 1,
                "tn": 4,
            },
            "parameters": {
                "score_threshold": 0.5,
                "hardmax": True,
                "label": "blue",
            },
        },
        {
            "type": "Counts",
            "value": {
                "tp": 0,
                "fp": 0,
                "fn": 1,
                "tn": 5,
            },
            "parameters": {
                "score_threshold": 0.5,
                "hardmax": True,
                "label": "black",
            },
        },
        # score >= 0.95
        {
            "type": "Counts",
            "value": {
                "tp": 0,
                "fp": 1,
                "fn": 2,
                "tn": 3,
            },
            "parameters": {
                "score_threshold": 0.95,
                "hardmax": True,
                "label": "white",
            },
        },
        {
            "type": "Counts",
            "value": {
                "tp": 0,
                "fp": 0,
                "fn": 2,
                "tn": 4,
            },
            "parameters": {
                "score_threshold": 0.95,
                "hardmax": True,
                "label": "red",
            },
        },
        {
            "type": "Counts",
            "value": {
                "tp": 0,
                "fp": 0,
                "fn": 1,
                "tn": 5,
            },
            "parameters": {
                "score_threshold": 0.95,
                "hardmax": True,
                "label": "blue",
            },
        },
        {
            "type": "Counts",
            "value": {
                "tp": 0,
                "fp": 0,
                "fn": 1,
                "tn": 5,
            },
            "parameters": {
                "score_threshold": 0.95,
                "hardmax": True,
                "label": "black",
            },
        },
    ]
    for m in actual_metrics:
        assert m in expected_metrics
    for m in expected_metrics:
        assert m in actual_metrics


def test_counts_with_image_example(
    loader: Loader,
    classifications_image_example: list[Classification],
):
    loader.add_data(classifications_image_example)
    evaluator = loader.finalize()

    assert evaluator.info.number_of_datums == 2
    assert evaluator.info.number_of_labels == 4
    assert evaluator.info.number_of_rows == 4

    metrics = evaluator.compute_precision_recall()

    actual_metrics = [m.to_dict() for m in metrics[MetricType.Counts]]
    expected_metrics = [
        {
            "type": "Counts",
            "value": {
                "tp": 0,
                "fp": 0,
                "fn": 0,
                "tn": 1,
            },
            "parameters": {
                "score_threshold": 0.0,
                "hardmax": True,
                "label": "v1",
            },
        },
        {
            "type": "Counts",
            "value": {
                "tp": 1,
                "fp": 0,
                "fn": 1,
                "tn": 0,
            },
            "parameters": {
                "score_threshold": 0.0,
                "hardmax": True,
                "label": "v4",
            },
        },
        {
            "type": "Counts",
            "value": {
                "tp": 0,
                "fp": 0,
                "fn": 0,
                "tn": 1,
            },
            "parameters": {
                "score_threshold": 0.0,
                "hardmax": True,
                "label": "v5",
            },
        },
        {
            "type": "Counts",
            "value": {
                "tp": 0,
                "fp": 1,
                "fn": 0,
                "tn": 0,
            },
            "parameters": {
                "score_threshold": 0.0,
                "hardmax": True,
                "label": "v8",
            },
        },
    ]
    for m in actual_metrics:
        assert m in expected_metrics
    for m in expected_metrics:
        assert m in actual_metrics


def test_counts_with_tabular_example(
    loader: Loader,
    classifications_tabular_example: list[Classification],
):
    loader.add_data(classifications_tabular_example)
    evaluator = loader.finalize()

    assert evaluator.info.number_of_datums == 10
    assert evaluator.info.number_of_labels == 3
    assert evaluator.info.number_of_rows == 30

    metrics = evaluator.compute_precision_recall()

    actual_metrics = [m.to_dict() for m in metrics[MetricType.Counts]]
    expected_metrics = [
        {
            "type": "Counts",
            "value": {
                "tp": 3,
                "fp": 3,
                "fn": 0,
                "tn": 4,
            },
            "parameters": {
                "score_threshold": 0.0,
                "hardmax": True,
                "label": "0",
            },
        },
        {
            "type": "Counts",
            "value": {
                "tp": 2,
                "fp": 1,
                "fn": 4,
                "tn": 3,
            },
            "parameters": {
                "score_threshold": 0.0,
                "hardmax": True,
                "label": "1",
            },
        },
        {
            "type": "Counts",
            "value": {
                "tp": 0,
                "fp": 1,
                "fn": 1,
                "tn": 8,
            },
            "parameters": {
                "score_threshold": 0.0,
                "hardmax": True,
                "label": "2",
            },
        },
    ]
    for m in actual_metrics:
        assert m in expected_metrics
    for m in expected_metrics:
        assert m in actual_metrics


def test_counts_multiclass(
    loader: Loader,
    classifications_multiclass: list[Classification],
):
    loader.add_data(classifications_multiclass)
    evaluator = loader.finalize()

    assert evaluator.info.number_of_datums == 5
    assert evaluator.info.number_of_labels == 3
    assert evaluator.info.number_of_rows == 15

    metrics = evaluator.compute_precision_recall(
        score_thresholds=[0.05, 0.1, 0.3, 0.85],
    )

    actual_metrics = [m.to_dict() for m in metrics[MetricType.Counts]]
    expected_metrics = [
        # score >= 0.05
        {
            "value": {
                "tp": 1,
                "fp": 0,
                "fn": 1,
                "tn": 3,
            },
            "parameters": {
                "score_threshold": 0.05,
                "hardmax": True,
                "label": "cat",
            },
            "type": "Counts",
        },
        {
            "value": {
                "tp": 1,
                "fp": 0,
                "fn": 0,
                "tn": 4,
            },
            "parameters": {
                "score_threshold": 0.05,
                "hardmax": True,
                "label": "dog",
            },
            "type": "Counts",
        },
        {
            "value": {
                "tp": 2,
                "fp": 1,
                "fn": 0,
                "tn": 2,
            },
            "parameters": {
                "score_threshold": 0.05,
                "hardmax": True,
                "label": "bee",
            },
            "type": "Counts",
        },
        # score >= 0.1
        {
            "value": {
                "tp": 1,
                "fp": 0,
                "fn": 1,
                "tn": 3,
            },
            "parameters": {
                "score_threshold": 0.1,
                "hardmax": True,
                "label": "cat",
            },
            "type": "Counts",
        },
        {
            "value": {
                "tp": 1,
                "fp": 0,
                "fn": 0,
                "tn": 4,
            },
            "parameters": {
                "score_threshold": 0.1,
                "hardmax": True,
                "label": "dog",
            },
            "type": "Counts",
        },
        {
            "value": {
                "tp": 2,
                "fp": 1,
                "fn": 0,
                "tn": 2,
            },
            "parameters": {
                "score_threshold": 0.1,
                "hardmax": True,
                "label": "bee",
            },
            "type": "Counts",
        },
        # score >= 0.3
        {
            "value": {
                "tp": 1,
                "fp": 0,
                "fn": 1,
                "tn": 3,
            },
            "parameters": {
                "score_threshold": 0.3,
                "hardmax": True,
                "label": "cat",
            },
            "type": "Counts",
        },
        {
            "value": {
                "tp": 1,
                "fp": 0,
                "fn": 0,
                "tn": 4,
            },
            "parameters": {
                "score_threshold": 0.3,
                "hardmax": True,
                "label": "dog",
            },
            "type": "Counts",
        },
        {
            "value": {
                "tp": 2,
                "fp": 1,
                "fn": 0,
                "tn": 2,
            },
            "parameters": {
                "score_threshold": 0.3,
                "hardmax": True,
                "label": "bee",
            },
            "type": "Counts",
        },
        # score >= 0.85
        {
            "value": {
                "tp": 0,
                "fp": 0,
                "fn": 2,
                "tn": 3,
            },
            "parameters": {
                "score_threshold": 0.85,
                "hardmax": True,
                "label": "cat",
            },
            "type": "Counts",
        },
        {
            "value": {
                "tp": 0,
                "fp": 0,
                "fn": 1,
                "tn": 4,
            },
            "parameters": {
                "score_threshold": 0.85,
                "hardmax": True,
                "label": "dog",
            },
            "type": "Counts",
        },
        {
            "value": {
                "tp": 0,
                "fp": 0,
                "fn": 2,
                "tn": 3,
            },
            "parameters": {
                "score_threshold": 0.85,
                "hardmax": True,
                "label": "bee",
            },
            "type": "Counts",
        },
    ]
    for m in actual_metrics:
        assert m in expected_metrics
    for m in expected_metrics:
        assert m in actual_metrics


def test_counts_true_negatives_check_animals(
    loader: Loader,
    classifications_multiclass_true_negatives_check: list[Classification],
):
    loader.add_data(classifications_multiclass_true_negatives_check)
    evaluator = loader.finalize()

    assert evaluator.info.number_of_datums == 1
    assert evaluator.info.number_of_labels == 3
    assert evaluator.info.number_of_rows == 3

    metrics = evaluator.compute_precision_recall(
        score_thresholds=[0.05, 0.15, 0.95],
    )

    actual_metrics = [m.to_dict() for m in metrics[MetricType.Counts]]
    expected_metrics = [
        # score >= 0.05
        {
            "value": {
                "tp": 0,
                "fp": 0,
                "fn": 1,
                "tn": 0,
            },
            "parameters": {
                "score_threshold": 0.05,
                "hardmax": True,
                "label": "ant",
            },
            "type": "Counts",
        },
        {
            "value": {
                "tp": 0,
                "fp": 1,
                "fn": 0,
                "tn": 0,
            },
            "parameters": {
                "score_threshold": 0.05,
                "hardmax": True,
                "label": "bee",
            },
            "type": "Counts",
        },
        {
            "value": {
                "tp": 0,
                "fp": 0,
                "fn": 0,
                "tn": 1,
            },
            "parameters": {
                "score_threshold": 0.05,
                "hardmax": True,
                "label": "cat",
            },
            "type": "Counts",
        },
        # score >= 0.15
        {
            "value": {
                "tp": 0,
                "fp": 0,
                "fn": 1,
                "tn": 0,
            },
            "parameters": {
                "score_threshold": 0.15,
                "hardmax": True,
                "label": "ant",
            },
            "type": "Counts",
        },
        {
            "value": {
                "tp": 0,
                "fp": 1,
                "fn": 0,
                "tn": 0,
            },
            "parameters": {
                "score_threshold": 0.15,
                "hardmax": True,
                "label": "bee",
            },
            "type": "Counts",
        },
        {
            "value": {
                "tp": 0,
                "fp": 0,
                "fn": 0,
                "tn": 1,
            },
            "parameters": {
                "score_threshold": 0.15,
                "hardmax": True,
                "label": "cat",
            },
            "type": "Counts",
        },
        # score >= 0.95
        {
            "value": {
                "tp": 0,
                "fp": 0,
                "fn": 1,
                "tn": 0,
            },
            "parameters": {
                "score_threshold": 0.95,
                "hardmax": True,
                "label": "ant",
            },
            "type": "Counts",
        },
        {
            "value": {
                "tp": 0,
                "fp": 0,
                "fn": 0,
                "tn": 1,
            },
            "parameters": {
                "score_threshold": 0.95,
                "hardmax": True,
                "label": "bee",
            },
            "type": "Counts",
        },
        {
            "value": {
                "tp": 0,
                "fp": 0,
                "fn": 0,
                "tn": 1,
            },
            "parameters": {
                "score_threshold": 0.95,
                "hardmax": True,
                "label": "cat",
            },
            "type": "Counts",
        },
    ]
    for m in actual_metrics:
        assert m in expected_metrics
    for m in expected_metrics:
        assert m in actual_metrics


def test_counts_zero_count_check(
    loader: Loader,
    classifications_multiclass_zero_count: list[Classification],
):

    loader.add_data(classifications_multiclass_zero_count)
    evaluator = loader.finalize()

    assert evaluator.info.number_of_datums == 1
    assert evaluator.info.number_of_labels == 3
    assert evaluator.info.number_of_rows == 3

    metrics = evaluator.compute_precision_recall(
        score_thresholds=[0.05, 0.2, 0.95],
    )

    actual_metrics = [m.to_dict() for m in metrics[MetricType.Counts]]
    expected_metrics = [
        # score >= 0.05
        {
            "value": {
                "tp": 0,
                "fp": 0,
                "fn": 1,
                "tn": 0,
            },
            "parameters": {
                "score_threshold": 0.05,
                "hardmax": True,
                "label": "ant",
            },
            "type": "Counts",
        },
        {
            "value": {
                "tp": 0,
                "fp": 1,
                "fn": 0,
                "tn": 0,
            },
            "parameters": {
                "score_threshold": 0.05,
                "hardmax": True,
                "label": "bee",
            },
            "type": "Counts",
        },
        {
            "value": {
                "tp": 0,
                "fp": 0,
                "fn": 0,
                "tn": 1,
            },
            "parameters": {
                "score_threshold": 0.05,
                "hardmax": True,
                "label": "cat",
            },
            "type": "Counts",
        },
        # score >= 0.2
        {
            "value": {
                "tp": 0,
                "fp": 0,
                "fn": 1,
                "tn": 0,
            },
            "parameters": {
                "score_threshold": 0.2,
                "hardmax": True,
                "label": "ant",
            },
            "type": "Counts",
        },
        {
            "value": {
                "tp": 0,
                "fp": 1,
                "fn": 0,
                "tn": 0,
            },
            "parameters": {
                "score_threshold": 0.2,
                "hardmax": True,
                "label": "bee",
            },
            "type": "Counts",
        },
        {
            "value": {
                "tp": 0,
                "fp": 0,
                "fn": 0,
                "tn": 1,
            },
            "parameters": {
                "score_threshold": 0.2,
                "hardmax": True,
                "label": "cat",
            },
            "type": "Counts",
        },
        # score >= 0.95
        {
            "value": {
                "tp": 0,
                "fp": 0,
                "fn": 1,
                "tn": 0,
            },
            "parameters": {
                "score_threshold": 0.95,
                "hardmax": True,
                "label": "ant",
            },
            "type": "Counts",
        },
        {
            "value": {
                "tp": 0,
                "fp": 0,
                "fn": 0,
                "tn": 1,
            },
            "parameters": {
                "score_threshold": 0.95,
                "hardmax": True,
                "label": "bee",
            },
            "type": "Counts",
        },
        {
            "value": {
                "tp": 0,
                "fp": 0,
                "fn": 0,
                "tn": 1,
            },
            "parameters": {
                "score_threshold": 0.95,
                "hardmax": True,
                "label": "cat",
            },
            "type": "Counts",
        },
    ]
    for m in actual_metrics:
        assert m in expected_metrics
    for m in expected_metrics:
        assert m in actual_metrics
