import numpy as np
from valor_lite.classification import Classification, DataLoader, MetricType
from valor_lite.classification.computation import compute_metrics


def test_compute_rocauc():
    """
    Test ROC auc computation. This agrees with scikit-learn: the code (whose data
    comes from classification_test_data)

    animal_gts = ["bird", "dog", "bird", "bird", "cat", "dog"]
    animal_preds = [
        {"bird": 0.6, "dog": 0.2, "cat": 0.2},
        {"cat": 0.9, "dog": 0.1, "bird": 0.0},
        {"cat": 0.8, "dog": 0.05, "bird": 0.15},
        {"dog": 0.75, "cat": 0.1, "bird": 0.15},
        {"cat": 1.0, "dog": 0.0, "bird": 0.0},
        {"cat": 0.4, "dog": 0.4, "bird": 0.2},
    ]

    color_gts = ["white", "white", "red", "blue", "black", "red"]
    color_preds = [
        {"white": 0.65, "red": 0.1, "blue": 0.2, "black": 0.05},
        {"blue": 0.5, "white": 0.3, "red": 0.0, "black": 0.2},
        {"red": 0.4, "white": 0.2, "blue": 0.1, "black": 0.3},
        {"white": 1.0, "red": 0.0, "blue": 0.0, "black": 0.0},
        {"red": 0.8, "white": 0.0, "blue": 0.2, "black": 0.0},
        {"red": 0.9, "white": 0.06, "blue": 0.01, "black": 0.03},
    ]

    ```
    from sklearn.metrics import roc_auc_score

    # for the "animal" label key
    y_true = [0, 2, 0, 0, 1, 2]
    y_score = [
        [0.6, 0.2, 0.2],
        [0.0, 0.9, 0.1],
        [0.15, 0.8, 0.05],
        [0.15, 0.1, 0.75],
        [0.0, 1.0, 0.0],
        [0.2, 0.4, 0.4],
    ]

    print(roc_auc_score(y_true, y_score, multi_class="ovr"))

    # for the "color" label key
    y_true = [3, 3, 2, 1, 0, 2]
    y_score = [
        [0.05, 0.2, 0.1, 0.65],
        [0.2, 0.5, 0.0, 0.3],
        [0.3, 0.1, 0.4, 0.2],
        [0.0, 0.0, 0.0, 1.0],
        [0.0, 0.2, 0.8, 0.0],
        [0.03, 0.01, 0.9, 0.06],
    ]
    ```

    outputs:

    ```
    0.8009259259259259
    0.43125
    ```
    """

    # groundtruth, prediction, score
    animals_and_colors = np.array(
        [
            # (animal, bird)
            [0, 0, 0, 0.6, 0],
            [0, 1, 0, 0.2, 0],
            [0, 0, 0, 0.15, 0],
            [0, 0, 0, 0.15, 0],
            [0, 1, 0, 0.0, 0],
            [0, 2, 0, 0.0, 0],
            # (animal, dog)
            [0, 0, 1, 0.75, 0],
            [0, 1, 1, 0.4, 0],
            [0, 0, 1, 0.2, 0],
            [0, 1, 1, 0.1, 0],
            [0, 0, 1, 0.05, 0],
            [0, 2, 1, 0.0, 0],
            # (animal, cat)
            [0, 2, 2, 1.0, 0],
            [0, 1, 2, 0.9, 0],
            [0, 0, 2, 0.8, 0],
            [0, 1, 2, 0.4, 0],
            [0, 0, 2, 0.2, 0],
            [0, 0, 2, 0.1, 0],
            # (color, white)
            [0, 5, 3, 1.0, 0],
            [0, 3, 3, 0.65, 0],
            [0, 3, 3, 0.3, 0],
            [0, 4, 3, 0.2, 0],
            [0, 4, 3, 0.06, 0],
            [0, 6, 3, 0.0, 0],
            # (color, red)
            [0, 4, 4, 0.9, 0],
            [0, 6, 4, 0.8, 0],
            [0, 3, 4, 0.1, 0],
            [0, 4, 4, 0.4, 0],
            [0, 3, 4, 0.0, 0],
            [0, 5, 4, 0.0, 0],
            # (color, blue)
            [0, 3, 5, 0.5, 0],
            [0, 3, 5, 0.2, 0],
            [0, 6, 5, 0.2, 0],
            [0, 4, 5, 0.1, 0],
            [0, 4, 5, 0.01, 0],
            [0, 5, 5, 0.0, 0],
            # (color, black)
            [0, 4, 6, 0.3, 0],
            [0, 3, 6, 0.2, 0],
            [0, 3, 6, 0.05, 0],
            [0, 4, 6, 0.03, 0],
            [0, 5, 6, 0.0, 0],
            [0, 6, 6, 0.0, 0],
        ],
        dtype=np.float64,
    )
    indices = np.argsort(-animals_and_colors[:, 3], axis=0)
    animals_and_colors = animals_and_colors[indices]

    # create args
    label_metadata = np.array(
        [
            [3, 6, 0],
            [2, 6, 0],
            [1, 6, 0],
            [2, 6, 1],
            [2, 6, 1],
            [1, 6, 1],
            [1, 6, 1],
        ],
        dtype=np.int32,
    )

    # compute ROCAUC and mROCAUC
    (_, _, _, _, _, rocauc, mean_rocauc) = compute_metrics(
        data=animals_and_colors,
        label_metadata=label_metadata,
        n_datums=6,
        score_thresholds=np.array([]),
        hardmax=False,
    )

    # test ROCAUC
    assert rocauc.shape == (label_metadata.shape[0],)
    assert rocauc[0] == 0.7777777777777778  # (animal, bird)
    assert rocauc[1] == 0.625  # (animal, dog)
    assert rocauc[2] == 1.0  # (animal, cat)
    assert rocauc[3] == 0.75  # (color, white)
    assert rocauc[4] == 0.875  # (color, red)
    assert rocauc[5] == 0.0  # (color, blue)
    assert rocauc[6] == 0.09999999999999998  # (color, black)

    # test mROCAUC
    assert mean_rocauc.shape == (np.unique(label_metadata[:, 2]).size,)
    assert (
        mean_rocauc == np.array([0.8009259259259259, 0.43125])  # animal, color
    ).all()


def test_rocauc_with_example(
    classifications_two_categories: list[Classification],
):

    loader = DataLoader()
    loader.add_data(classifications_two_categories)
    evaluator = loader.finalize()

    metrics = evaluator.evaluate(as_dict=True)

    # test ROCAUC
    actual_metrics = [m for m in metrics[MetricType.ROCAUC]]
    expected_metrics = [
        {
            "type": "ROCAUC",
            "value": 0.7777777777777778,
            "parameters": {
                "label": {"key": "animal", "value": "bird"},
            },
        },
        {
            "type": "ROCAUC",
            "value": 0.625,
            "parameters": {
                "label": {"key": "animal", "value": "dog"},
            },
        },
        {
            "type": "ROCAUC",
            "value": 1.0,
            "parameters": {
                "label": {"key": "animal", "value": "cat"},
            },
        },
        {
            "type": "ROCAUC",
            "value": 0.75,
            "parameters": {
                "label": {"key": "color", "value": "white"},
            },
        },
        {
            "type": "ROCAUC",
            "value": 0.875,
            "parameters": {
                "label": {"key": "color", "value": "red"},
            },
        },
        {
            "type": "ROCAUC",
            "value": 0.0,
            "parameters": {
                "label": {"key": "color", "value": "blue"},
            },
        },
        {
            "type": "ROCAUC",
            "value": 0.09999999999999998,
            "parameters": {
                "label": {"key": "color", "value": "black"},
            },
        },
    ]
    for m in actual_metrics:
        assert m in expected_metrics
    for m in expected_metrics:
        assert m in actual_metrics

    # test mROCAUC
    actual_metrics = [m for m in metrics[MetricType.mROCAUC]]
    expected_metrics = [
        {
            "type": "mROCAUC",
            "value": 0.8009259259259259,
            "parameters": {
                "label_key": "animal",
            },
        },
        {
            "type": "mROCAUC",
            "value": 0.43125,
            "parameters": {
                "label_key": "color",
            },
        },
    ]
    for m in actual_metrics:
        assert m in expected_metrics
    for m in expected_metrics:
        assert m in actual_metrics


def test_rocauc_with_image_example(
    classifications_image_example: list[Classification],
):
    loader = DataLoader()
    loader.add_data(classifications_image_example)
    loader.finalize()
    evaluator = loader.finalize()
    evaluator.evaluate()

    metrics = evaluator.evaluate(as_dict=True)

    actual_metrics = [m for m in metrics[MetricType.ROCAUC]]
    expected_metrics = [
        {
            "type": "ROCAUC",
            "value": 0.0,
            "parameters": {"label": {"key": "k4", "value": "v4"}},
        },
        {
            "type": "ROCAUC",
            "value": 0.0,
            "parameters": {"label": {"key": "k5", "value": "v5"}},
        },
        {
            "type": "ROCAUC",
            "value": 0.0,
            "parameters": {"label": {"key": "k3", "value": "v3"}},
        },
    ]
    for m in actual_metrics:
        assert m in expected_metrics
    for m in expected_metrics:
        assert m in actual_metrics

    actual_metrics = [m for m in metrics[MetricType.mROCAUC]]
    expected_metrics = [
        {"type": "mROCAUC", "value": 0.0, "parameters": {"label_key": "k3"}},
        {"type": "mROCAUC", "value": 0.0, "parameters": {"label_key": "k4"}},
        {"type": "mROCAUC", "value": 0.0, "parameters": {"label_key": "k5"}},
    ]
    for m in actual_metrics:
        assert m in expected_metrics
    for m in expected_metrics:
        assert m in actual_metrics


def test_rocauc_with_tabular_example(
    classifications_tabular_example: list[Classification],
):
    loader = DataLoader()
    loader.add_data(classifications_tabular_example)
    loader.finalize()
    evaluator = loader.finalize()
    evaluator.evaluate()

    metrics = evaluator.evaluate(as_dict=True)

    actual_metrics = [m for m in metrics[MetricType.ROCAUC]]
    expected_metrics = [
        {
            "type": "ROCAUC",
            "value": 0.75,
            "parameters": {"label": {"key": "class", "value": "1"}},
        },
        {
            "type": "ROCAUC",
            "value": 1.0,
            "parameters": {"label": {"key": "class", "value": "0"}},
        },
        {
            "type": "ROCAUC",
            "value": 0.5555555555555556,
            "parameters": {"label": {"key": "class", "value": "2"}},
        },
    ]
    for m in actual_metrics:
        assert m in expected_metrics
    for m in expected_metrics:
        assert m in actual_metrics

    actual_metrics = [m for m in metrics[MetricType.mROCAUC]]
    expected_metrics = [
        {
            "type": "mROCAUC",
            "value": 0.7685185185185185,
            "parameters": {"label_key": "class"},
        }
    ]
    for m in actual_metrics:
        assert m in expected_metrics
    for m in expected_metrics:
        assert m in actual_metrics
