import numpy as np
import pyarrow as pa
import pyarrow.compute as pc
import pytest

from valor_lite.classification import Classification, MetricType
from valor_lite.classification.loader import Loader


def test_roc_points_match_threshold_reference(loader, tmp_path, monkeypatch):
    labels = ["cat", "dog", "bird"]
    rng = np.random.default_rng(19)
    loader.add_data(
        [
            Classification(
                str(datum),
                labels[datum % 3],
                labels,
                (rng.integers(3, size=3) / 2).tolist(),
            )
            for datum in range(27)
        ]
    )

    def fail_scalar_write(*args, **kwargs):
        pytest.fail("ROC points should be emitted in columnar batches")

    monkeypatch.setattr(
        loader._roc_curve_writer, "write_rows", fail_scalar_write
    )
    source = loader.finalize(batch_size=2)
    filtered = source.filter(
        datums=pc.field("datum_uid").isin(["0", "3", "6", "9"]),
        predictions=pc.field("pd_label") != "bird",
        path=tmp_path / "filtered",
    )
    for evaluator in (source, filtered):
        table = pa.concat_tables(list(evaluator._reader.iterate_tables()))
        roc = pa.concat_tables(
            list(evaluator._roc_curve_reader.iterate_tables())
        )
        assert roc.schema.field("pd_label_id").type == pa.int64()
        assert roc.schema.field("cumulative_fp").type == pa.uint64()
        assert roc.schema.field("cumulative_tp").type == pa.uint64()
        for label_id in evaluator._index_to_label:
            rows = table.filter(
                pc.field("pd_label_id") == label_id
            ).to_pylist()
            expected = {(0, 0)}
            for threshold in {row["pd_score"] for row in rows}:
                retained = [
                    row for row in rows if row["pd_score"] >= threshold
                ]
                expected.add(
                    (
                        sum(not row["match"] for row in retained),
                        sum(row["match"] for row in retained),
                    )
                )
            actual = {(0, 0)} | {
                (row["cumulative_fp"], row["cumulative_tp"])
                for row in roc.filter(
                    pc.field("pd_label_id") == label_id
                ).to_pylist()
            }
            assert actual == expected


def test_rocauc_with_animal_example(
    loader: Loader,
    classifications_animal_example: list[Classification],
):
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
    ```
    outputs ==> 0.8009259259259259
    """

    loader.add_data(classifications_animal_example)
    evaluator = loader.finalize()

    metrics = evaluator.compute_rocauc()

    # test ROCAUC
    actual_metrics = [m.to_dict() for m in metrics[MetricType.ROCAUC]]
    expected_metrics = [
        {
            "type": "ROCAUC",
            "value": 0.7777777777777779,
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
    """
    Test ROC auc computation. This agrees with scikit-learn: the code (whose data
    comes from classification_test_data)

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
    print(roc_auc_score(y_true, y_score, multi_class="ovr"))
    ```

    outputs:

    ```
    0.43125
    ```
    """

    loader.add_data(classifications_color_example)
    evaluator = loader.finalize()
    metrics = evaluator.compute_rocauc()

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


def test_rocauc_single_classification(loader: Loader):
    data = [
        Classification(
            uid="uid",
            groundtruth="dog",
            predictions=["dog", "cat"],
            scores=[1.0, 0.0],
        )
    ]
    loader.add_data(data)
    evaluator = loader.finalize()

    metrics = evaluator.compute_rocauc()

    # test ROCAUC
    actual_metrics = [m.to_dict() for m in metrics[MetricType.ROCAUC]]
    expected_metrics = [
        {
            "type": "ROCAUC",
            "value": 0.0,
            "parameters": {
                "label": "dog",
            },
        },
        {
            "type": "ROCAUC",
            "value": 0.0,
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
        {"type": "mROCAUC", "value": 0.0, "parameters": {}},
    ]
    for m in actual_metrics:
        assert m in expected_metrics
    for m in expected_metrics:
        assert m in actual_metrics
