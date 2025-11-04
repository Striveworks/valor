from valor_lite.classification import Classification, MetricType
from valor_lite.classification.loader import Loader


def test_rocauc_with_animal_example(
    loader: Loader,
    classifications_animal_example: list[Classification],
):

    loader.add_data(classifications_animal_example)
    evaluator = loader.finalize()

    metrics = evaluator.compute_rocauc()

    # test ROCAUC
    actual_metrics = [m.to_dict() for m in metrics[MetricType.ROCAUC]]
    expected_metrics = [
        {
            "type": "ROCAUC",
            "value": 0.7777777777777778,
            "parameters": {
                "label": "bird",
            },
        },
        {
            "type": "ROCAUC",
            "value": 0.625,
            "parameters": {
                "label": "dog",
            },
        },
        {
            "type": "ROCAUC",
            "value": 1.0,
            "parameters": {
                "label": "cat",
            },
        },
    ]
    for m in actual_metrics:
        assert m in expected_metrics
    for m in expected_metrics:
        assert m in actual_metrics

    # test mROCAUC
    actual_metrics = [m.to_dict() for m in metrics[MetricType.mROCAUC]]
    expected_metrics = [
        {"type": "mROCAUC", "value": 0.8009259259259259, "parameters": {}},
    ]
    for m in actual_metrics:
        assert m in expected_metrics
    for m in expected_metrics:
        assert m in actual_metrics


def test_rocauc_with_color_example(
    loader: Loader,
    classifications_color_example: list[Classification],
):

    loader.add_data(classifications_color_example)
    evaluator = loader.finalize()

    metrics = evaluator.compute_rocauc()

    # test ROCAUC
    actual_metrics = [m.to_dict() for m in metrics[MetricType.ROCAUC]]
    expected_metrics = [
        {
            "type": "ROCAUC",
            "value": 0.75,
            "parameters": {
                "label": "white",
            },
        },
        {
            "type": "ROCAUC",
            "value": 0.875,
            "parameters": {
                "label": "red",
            },
        },
        {
            "type": "ROCAUC",
            "value": 0.0,
            "parameters": {
                "label": "blue",
            },
        },
        {
            "type": "ROCAUC",
            "value": 0.09999999999999998,
            "parameters": {
                "label": "black",
            },
        },
    ]
    for m in actual_metrics:
        assert m in expected_metrics
    for m in expected_metrics:
        assert m in actual_metrics

    # test mROCAUC
    actual_metrics = [m.to_dict() for m in metrics[MetricType.mROCAUC]]
    expected_metrics = [
        {"type": "mROCAUC", "value": 0.43125, "parameters": {}},
    ]
    for m in actual_metrics:
        assert m in expected_metrics
    for m in expected_metrics:
        assert m in actual_metrics


def test_rocauc_with_image_example(
    loader: Loader,
    classifications_image_example: list[Classification],
):
    loader.add_data(classifications_image_example)
    loader.finalize()
    evaluator = loader.finalize()

    metrics = evaluator.compute_rocauc()

    actual_metrics = [m.to_dict() for m in metrics[MetricType.ROCAUC]]
    expected_metrics = [
        {
            "type": "ROCAUC",
            "value": 0.0,
            "parameters": {"label": "v1"},
        },
        {
            "type": "ROCAUC",
            "value": 0.0,
            "parameters": {"label": "v4"},
        },
        {
            "type": "ROCAUC",
            "value": 0.0,
            "parameters": {"label": "v5"},
        },
        {
            "type": "ROCAUC",
            "value": 0.0,
            "parameters": {"label": "v8"},
        },
    ]
    for m in actual_metrics:
        assert m in expected_metrics
    for m in expected_metrics:
        assert m in actual_metrics

    actual_metrics = [m.to_dict() for m in metrics[MetricType.mROCAUC]]
    expected_metrics = [
        {"type": "mROCAUC", "value": 0.0, "parameters": {}},
    ]
    for m in actual_metrics:
        assert m in expected_metrics
    for m in expected_metrics:
        assert m in actual_metrics


def test_rocauc_with_tabular_example(
    loader: Loader,
    classifications_tabular_example: list[Classification],
):
    loader.add_data(classifications_tabular_example)
    loader.finalize()
    evaluator = loader.finalize()

    metrics = evaluator.compute_rocauc()

    actual_metrics = [m.to_dict() for m in metrics[MetricType.ROCAUC]]
    expected_metrics = [
        {
            "type": "ROCAUC",
            "value": 0.75,
            "parameters": {"label": "1"},
        },
        {
            "type": "ROCAUC",
            "value": 1.0,
            "parameters": {"label": "0"},
        },
        {
            "type": "ROCAUC",
            "value": 0.5555555555555556,
            "parameters": {"label": "2"},
        },
    ]
    for m in actual_metrics:
        assert m in expected_metrics
    for m in expected_metrics:
        assert m in actual_metrics

    actual_metrics = [m.to_dict() for m in metrics[MetricType.mROCAUC]]
    expected_metrics = [
        {
            "type": "mROCAUC",
            "value": 0.7685185185185185,
            "parameters": {},
        }
    ]
    for m in actual_metrics:
        assert m in expected_metrics
    for m in expected_metrics:
        assert m in actual_metrics
