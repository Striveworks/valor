import pytest
from sqlalchemy.orm import Session

from valor_api import crud, enums, schemas
from valor_api.backend import models
from valor_api.backend.core import (
    create_or_get_evaluations,
    fetch_union_of_labels,
)
from valor_api.backend.metrics.classification import (
    _aggregate_data,
    _compute_accuracy_from_cm,
    _compute_clf_metrics,
    _compute_confusion_matrices,
    _compute_curves,
    _compute_roc_auc,
    compute_clf_metrics,
)


@pytest.fixture
def label_map():
    return [
        [["animal", "dog"], ["animal", "mammal"]],
        [["animal", "cat"], ["animal", "mammal"]],
    ]


@pytest.fixture
def classification_test_data(db: Session, dataset_name: str, model_name: str):
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

    imgs = [
        schemas.Datum(
            uid=f"uid{i}",
            metadata={
                "height": 128,
                "width": 256,
                "md1": f"md1-val{int(i == 4)}",
                "md2": f"md1-val{i % 3}",
            },
        )
        for i in range(6)
    ]

    gts = [
        schemas.GroundTruth(
            dataset_name=dataset_name,
            datum=imgs[i],
            annotations=[
                schemas.Annotation(
                    labels=[
                        schemas.Label(key="animal", value=animal_gts[i]),
                        schemas.Label(key="color", value=color_gts[i]),
                    ],
                )
            ],
        )
        for i in range(6)
    ]

    preds = [
        schemas.Prediction(
            dataset_name=dataset_name,
            model_name=model_name,
            datum=imgs[i],
            annotations=[
                schemas.Annotation(
                    labels=[
                        schemas.Label(key="animal", value=value, score=score)
                        for value, score in animal_preds[i].items()
                    ]
                    + [
                        schemas.Label(key="color", value=value, score=score)
                        for value, score in color_preds[i].items()
                    ],
                )
            ],
        )
        for i in range(6)
    ]

    crud.create_dataset(
        db=db,
        dataset=schemas.Dataset(
            name=dataset_name,
            metadata={"type": "image"},
        ),
    )

    crud.create_groundtruths(db=db, groundtruths=gts)
    crud.finalize(db=db, dataset_name=dataset_name)

    crud.create_model(
        db=db,
        model=schemas.Model(
            name=model_name,
            metadata={"type": "image"},
        ),
    )
    crud.create_predictions(db=db, predictions=preds)
    crud.finalize(db=db, dataset_name=dataset_name, model_name=model_name)

    assert len(db.query(models.Datum).all()) == 6
    assert len(db.query(models.Annotation).all()) == 12
    assert len(db.query(models.Label).all()) == 7
    assert len(db.query(models.GroundTruth).all()) == 6 * 2
    assert len(db.query(models.Prediction).all()) == 6 * 7


def test_compute_confusion_matrices(
    db: Session,
    dataset_name: str,
    model_name: str,
    classification_test_data,
):
    prediction_filter = schemas.Filter(
        predictions=schemas.LogicalFunction(
            args=[
                schemas.Condition(
                    lhs=schemas.Symbol(
                        name=schemas.SupportedSymbol.MODEL_NAME
                    ),
                    rhs=schemas.Value.infer(model_name),
                    op=schemas.FilterOperator.EQ,
                ),
                schemas.Condition(
                    lhs=schemas.Symbol(name=schemas.SupportedSymbol.TASK_TYPE),
                    rhs=schemas.Value.infer(enums.TaskType.CLASSIFICATION),
                    op=schemas.FilterOperator.CONTAINS,
                ),
            ],
            op=schemas.LogicalOperator.AND,
        )
    )
    groundtruth_filter = schemas.Filter(
        groundtruths=schemas.LogicalFunction(
            args=[
                schemas.Condition(
                    lhs=schemas.Symbol(
                        name=schemas.SupportedSymbol.DATASET_NAME
                    ),
                    rhs=schemas.Value.infer(dataset_name),
                    op=schemas.FilterOperator.EQ,
                ),
                schemas.Condition(
                    lhs=schemas.Symbol(name=schemas.SupportedSymbol.TASK_TYPE),
                    rhs=schemas.Value.infer(enums.TaskType.CLASSIFICATION),
                    op=schemas.FilterOperator.CONTAINS,
                ),
            ],
            op=schemas.LogicalOperator.AND,
        )
    )

    groundtruths, predictions, labels = _aggregate_data(
        db=db,
        groundtruth_filter=groundtruth_filter,
        prediction_filter=prediction_filter,
        label_map=None,
    )

    confusion_matrices = _compute_confusion_matrices(
        db=db,
        predictions=predictions,
        groundtruths=groundtruths,
        labels=labels,
    )

    cm = confusion_matrices["animal"]
    expected_entries = [
        schemas.ConfusionMatrixEntry(
            prediction="bird", groundtruth="bird", count=1
        ),
        schemas.ConfusionMatrixEntry(
            prediction="cat", groundtruth="dog", count=2
        ),
        schemas.ConfusionMatrixEntry(
            prediction="cat", groundtruth="cat", count=1
        ),
        schemas.ConfusionMatrixEntry(
            prediction="cat", groundtruth="bird", count=1
        ),
        schemas.ConfusionMatrixEntry(
            prediction="dog", groundtruth="bird", count=1
        ),
    ]
    assert cm
    assert len(cm.entries) == len(expected_entries)
    for entry in cm.entries:
        assert entry in expected_entries
    for entry in expected_entries:
        assert entry in cm.entries
    assert _compute_accuracy_from_cm(cm) == 2 / 6

    # test for color
    cm = confusion_matrices["color"]
    expected_entries = [
        schemas.ConfusionMatrixEntry(
            prediction="white", groundtruth="white", count=1
        ),
        schemas.ConfusionMatrixEntry(
            prediction="white", groundtruth="blue", count=1
        ),
        schemas.ConfusionMatrixEntry(
            prediction="blue", groundtruth="white", count=1
        ),
        schemas.ConfusionMatrixEntry(
            prediction="red", groundtruth="red", count=2
        ),
        schemas.ConfusionMatrixEntry(
            prediction="red", groundtruth="black", count=1
        ),
    ]
    assert cm
    assert len(cm.entries) == len(expected_entries)
    for entry in cm.entries:
        assert entry in expected_entries
    for entry in expected_entries:
        assert entry in cm.entries
    assert _compute_accuracy_from_cm(cm) == 3 / 6


def test_compute_confusion_matrices_and_filter(
    db: Session,
    dataset_name: str,
    model_name: str,
    classification_test_data,
):
    """
    Test filtering by metadata (md1: md1-val0).
    """
    prediction_filter = schemas.Filter(
        predictions=schemas.LogicalFunction(
            args=[
                schemas.Condition(
                    lhs=schemas.Symbol(
                        name=schemas.SupportedSymbol.DATASET_NAME
                    ),
                    rhs=schemas.Value.infer(dataset_name),
                    op=schemas.FilterOperator.EQ,
                ),
                schemas.Condition(
                    lhs=schemas.Symbol(
                        name=schemas.SupportedSymbol.MODEL_NAME
                    ),
                    rhs=schemas.Value.infer(model_name),
                    op=schemas.FilterOperator.EQ,
                ),
            ],
            op=schemas.LogicalOperator.AND,
        )
    )
    groundtruth_filter = schemas.Filter(
        groundtruths=schemas.LogicalFunction(
            args=[
                schemas.Condition(
                    lhs=schemas.Symbol(
                        name=schemas.SupportedSymbol.DATASET_NAME
                    ),
                    rhs=schemas.Value.infer(dataset_name),
                    op=schemas.FilterOperator.EQ,
                ),
                schemas.Condition(
                    lhs=schemas.Symbol(
                        name=schemas.SupportedSymbol.MODEL_NAME
                    ),
                    rhs=schemas.Value.infer(model_name),
                    op=schemas.FilterOperator.EQ,
                ),
                schemas.Condition(
                    lhs=schemas.Symbol(name=schemas.SupportedSymbol.TASK_TYPE),
                    rhs=schemas.Value.infer(enums.TaskType.CLASSIFICATION),
                    op=schemas.FilterOperator.CONTAINS,
                ),
                schemas.Condition(
                    lhs=schemas.Symbol(
                        name=schemas.SupportedSymbol.DATUM_META, key="md1"
                    ),
                    rhs=schemas.Value.infer("md1-val0"),
                    op=schemas.FilterOperator.EQ,
                ),
            ],
            op=schemas.LogicalOperator.AND,
        )
    )

    labels = fetch_union_of_labels(
        db=db,
        rhs=prediction_filter,
        lhs=groundtruth_filter,
    )

    groundtruths, predictions, labels = _aggregate_data(
        db=db,
        groundtruth_filter=groundtruth_filter,
        prediction_filter=prediction_filter,
        label_map=None,
    )

    confusion_matrices = _compute_confusion_matrices(
        db=db,
        predictions=predictions,
        groundtruths=groundtruths,
        labels=labels,
    )

    # for this metadatum and label id we have the gts
    # ["bird", "dog", "bird", "bird", "dog"] and the preds
    # ["bird", "cat", "cat", "dog", "cat"]
    cm = confusion_matrices["animal"]
    expected_entries = [
        schemas.ConfusionMatrixEntry(
            groundtruth="bird", prediction="bird", count=1
        ),
        schemas.ConfusionMatrixEntry(
            groundtruth="dog", prediction="cat", count=2
        ),
        schemas.ConfusionMatrixEntry(
            groundtruth="bird", prediction="cat", count=1
        ),
        schemas.ConfusionMatrixEntry(
            groundtruth="bird", prediction="dog", count=1
        ),
    ]
    assert cm
    assert len(cm.entries) == len(expected_entries)
    for e in expected_entries:
        assert e in cm.entries


def test_compute_confusion_matrices_using_label_map(
    db: Session,
    dataset_name: str,
    model_name: str,
    label_map,
    classification_test_data,
):
    """
    Test grouping using the label_map
    """
    prediction_filter = schemas.Filter(
        predictions=schemas.LogicalFunction(
            args=[
                schemas.Condition(
                    lhs=schemas.Symbol(
                        name=schemas.SupportedSymbol.DATASET_NAME
                    ),
                    rhs=schemas.Value.infer(dataset_name),
                    op=schemas.FilterOperator.EQ,
                ),
                schemas.Condition(
                    lhs=schemas.Symbol(
                        name=schemas.SupportedSymbol.MODEL_NAME
                    ),
                    rhs=schemas.Value.infer(model_name),
                    op=schemas.FilterOperator.EQ,
                ),
            ],
            op=schemas.LogicalOperator.AND,
        )
    )
    groundtruth_filter = schemas.Filter(
        groundtruths=schemas.LogicalFunction(
            args=[
                schemas.Condition(
                    lhs=schemas.Symbol(
                        name=schemas.SupportedSymbol.DATASET_NAME
                    ),
                    rhs=schemas.Value.infer(dataset_name),
                    op=schemas.FilterOperator.EQ,
                ),
                schemas.Condition(
                    lhs=schemas.Symbol(
                        name=schemas.SupportedSymbol.MODEL_NAME
                    ),
                    rhs=schemas.Value.infer(model_name),
                    op=schemas.FilterOperator.EQ,
                ),
                schemas.Condition(
                    lhs=schemas.Symbol(name=schemas.SupportedSymbol.TASK_TYPE),
                    rhs=schemas.Value.infer(enums.TaskType.CLASSIFICATION),
                    op=schemas.FilterOperator.CONTAINS,
                ),
                schemas.Condition(
                    lhs=schemas.Symbol(
                        name=schemas.SupportedSymbol.DATUM_META, key="md1"
                    ),
                    rhs=schemas.Value.infer("md1-val0"),
                    op=schemas.FilterOperator.EQ,
                ),
            ],
            op=schemas.LogicalOperator.AND,
        )
    )

    groundtruths, predictions, labels = _aggregate_data(
        db=db,
        groundtruth_filter=groundtruth_filter,
        prediction_filter=prediction_filter,
        label_map=label_map,
    )

    confusion_matrices = _compute_confusion_matrices(
        db=db,
        predictions=predictions,
        groundtruths=groundtruths,
        labels=labels,
    )

    cm = confusion_matrices["animal"]
    expected_entries = [
        schemas.ConfusionMatrixEntry(
            groundtruth="bird", prediction="bird", count=1
        ),
        schemas.ConfusionMatrixEntry(
            groundtruth="mammal", prediction="mammal", count=2
        ),
        schemas.ConfusionMatrixEntry(
            groundtruth="mammal", prediction="mammal", count=2
        ),
    ]

    assert cm
    assert len(cm.entries) == len(expected_entries)
    for e in expected_entries:
        assert e in cm.entries


def test_compute_roc_auc(
    db: Session,
    dataset_name: str,
    model_name: str,
    classification_test_data,
):
    """Test ROC auc computation. This agrees with scikit-learn: the code (whose data
    comes from classification_test_data)

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
    prediction_filter = schemas.Filter(
        predictions=schemas.LogicalFunction(
            args=[
                schemas.Condition(
                    lhs=schemas.Symbol(
                        name=schemas.SupportedSymbol.MODEL_NAME
                    ),
                    rhs=schemas.Value.infer(model_name),
                    op=schemas.FilterOperator.EQ,
                ),
                schemas.Condition(
                    lhs=schemas.Symbol(name=schemas.SupportedSymbol.TASK_TYPE),
                    rhs=schemas.Value.infer(enums.TaskType.CLASSIFICATION),
                    op=schemas.FilterOperator.CONTAINS,
                ),
            ],
            op=schemas.LogicalOperator.AND,
        )
    )
    groundtruth_filter = schemas.Filter(
        groundtruths=schemas.LogicalFunction(
            args=[
                schemas.Condition(
                    lhs=schemas.Symbol(
                        name=schemas.SupportedSymbol.DATASET_NAME
                    ),
                    rhs=schemas.Value.infer(dataset_name),
                    op=schemas.FilterOperator.EQ,
                ),
                schemas.Condition(
                    lhs=schemas.Symbol(name=schemas.SupportedSymbol.TASK_TYPE),
                    rhs=schemas.Value.infer(enums.TaskType.CLASSIFICATION),
                    op=schemas.FilterOperator.CONTAINS,
                ),
            ],
            op=schemas.LogicalOperator.AND,
        )
    )

    groundtruths, predictions, labels = _aggregate_data(
        db=db,
        groundtruth_filter=groundtruth_filter,
        prediction_filter=prediction_filter,
        label_map=None,
    )

    rocaucs = _compute_roc_auc(
        db=db,
        groundtruths=groundtruths,
        predictions=predictions,
        labels=set(labels.values()),
    )

    results = {rocauc.label_key: rocauc.value for rocauc in rocaucs}
    assert len(results) == 2
    assert results["animal"] == 0.8009259259259259
    assert results["color"] == 0.43125


def test_compute_roc_auc_groupby_metadata(
    db: Session, dataset_name: str, model_name: str, classification_test_data
):
    """Test computing ROC AUC for a given grouping. This agrees with:

    Scikit-learn won't do multiclass ROC AUC when there are only two predictive classes. So we
    compare this to doing the following in scikit-learn: first computing binary ROC for the "dog" class via:

    ```
    from sklearn.metrics import roc_auc_score

    y_true = [0, 1, 0, 0, 1]
    y_score = [0.2, 0.1, 0.05, 0.75, 0.4]

    roc_auc_score(y_true, y_score)
    ```

    which gives 0.5. Then we do it for the "bird" class via:

    ```
    from sklearn.metrics import roc_auc_score

    y_true = [1, 0, 1, 1, 0]
    y_score = [0.6, 0.0, 0.15, 0.15, 0.2]

    roc_auc_score(y_true, y_score)
    ```

    which gives 2/3. So we expect our implementation to give the average of 0.5 and 2/3
    """

    prediction_filter = schemas.Filter(
        predictions=schemas.Condition(
            lhs=schemas.Symbol(name=schemas.SupportedSymbol.MODEL_NAME),
            rhs=schemas.Value.infer(model_name),
            op=schemas.FilterOperator.EQ,
        ),
    )
    groundtruth_filter = schemas.Filter(
        groundtruths=schemas.LogicalFunction(
            args=[
                schemas.Condition(
                    lhs=schemas.Symbol(
                        name=schemas.SupportedSymbol.DATASET_NAME
                    ),
                    rhs=schemas.Value.infer(dataset_name),
                    op=schemas.FilterOperator.EQ,
                ),
                schemas.Condition(
                    lhs=schemas.Symbol(name=schemas.SupportedSymbol.TASK_TYPE),
                    rhs=schemas.Value.infer(enums.TaskType.CLASSIFICATION),
                    op=schemas.FilterOperator.CONTAINS,
                ),
                schemas.Condition(
                    lhs=schemas.Symbol(
                        name=schemas.SupportedSymbol.DATUM_META, key="md1"
                    ),
                    rhs=schemas.Value.infer("md1-val0"),
                    op=schemas.FilterOperator.EQ,
                ),
            ],
            op=schemas.LogicalOperator.AND,
        )
    )

    groundtruths, predictions, labels = _aggregate_data(
        db=db,
        groundtruth_filter=groundtruth_filter,
        prediction_filter=prediction_filter,
        label_map=None,
    )

    rocaucs = _compute_roc_auc(
        db=db,
        groundtruths=groundtruths,
        predictions=predictions,
        labels=set(labels.values()),
    )

    results = {rocauc.label_key: rocauc.value for rocauc in rocaucs}
    assert len(results) == 2
    assert results["animal"] == (0.5 + 2 / 3) / 2


def test_compute_roc_auc_with_label_map(
    db: Session,
    dataset_name: str,
    model_name: str,
    classification_test_data,
    label_map,
):
    """Test ROC auc computation using a label_map to group labels together. Matches the following output from sklearn:

    import numpy as np
    from sklearn.metrics import roc_auc_score

    # for the "animal" label key
    y_true = np.array([0, 1, 0, 0, 1, 1])
    y_score = np.array(
        [
            [0.6, 0.4],
            [0.0, 1],
            [0.15, 0.85],
            [0.15, 0.85],
            [0.0, 1.0],
            [0.2, 0.8],
        ]
    )

    score = roc_auc_score(y_true, y_score[:, 1], multi_class="ovr")
    assert score == 0.7777777777777778

    """
    prediction_filter = schemas.Filter(
        predictions=schemas.LogicalFunction(
            args=[
                schemas.Condition(
                    lhs=schemas.Symbol(
                        name=schemas.SupportedSymbol.MODEL_NAME
                    ),
                    rhs=schemas.Value.infer(model_name),
                    op=schemas.FilterOperator.EQ,
                ),
                schemas.Condition(
                    lhs=schemas.Symbol(name=schemas.SupportedSymbol.TASK_TYPE),
                    rhs=schemas.Value.infer(enums.TaskType.CLASSIFICATION),
                    op=schemas.FilterOperator.CONTAINS,
                ),
            ],
            op=schemas.LogicalOperator.AND,
        )
    )
    groundtruth_filter = schemas.Filter(
        groundtruths=schemas.LogicalFunction(
            args=[
                schemas.Condition(
                    lhs=schemas.Symbol(
                        name=schemas.SupportedSymbol.DATASET_NAME
                    ),
                    rhs=schemas.Value.infer(dataset_name),
                    op=schemas.FilterOperator.EQ,
                ),
                schemas.Condition(
                    lhs=schemas.Symbol(name=schemas.SupportedSymbol.TASK_TYPE),
                    rhs=schemas.Value.infer(enums.TaskType.CLASSIFICATION),
                    op=schemas.FilterOperator.CONTAINS,
                ),
            ],
            op=schemas.LogicalOperator.AND,
        )
    )

    groundtruths, predictions, labels = _aggregate_data(
        db=db,
        groundtruth_filter=groundtruth_filter,
        prediction_filter=prediction_filter,
        label_map=label_map,
    )

    rocaucs = _compute_roc_auc(
        db=db,
        groundtruths=groundtruths,
        predictions=predictions,
        labels=set(labels.values()),
    )

    results = {
        rocauc.label_key: rocauc.value for rocauc in rocaucs if rocauc.value
    }
    assert len(results) == 2
    assert (results["animal"] - 0.7777777777777779) < 1e-6


def test_compute_classification(
    db: Session,
    dataset_name: str,
    model_name: str,
    classification_test_data,
):
    """
    Tests the _compute_classification function.
    """

    prediction_filter = schemas.Filter(
        predictions=schemas.LogicalFunction(
            args=[
                schemas.Condition(
                    lhs=schemas.Symbol(
                        name=schemas.SupportedSymbol.DATASET_NAME
                    ),
                    rhs=schemas.Value.infer(dataset_name),
                    op=schemas.FilterOperator.EQ,
                ),
                schemas.Condition(
                    lhs=schemas.Symbol(
                        name=schemas.SupportedSymbol.MODEL_NAME
                    ),
                    rhs=schemas.Value.infer(model_name),
                    op=schemas.FilterOperator.EQ,
                ),
            ],
            op=schemas.LogicalOperator.AND,
        )
    )
    groundtruth_filter = schemas.Filter(
        groundtruths=schemas.LogicalFunction(
            args=[
                schemas.Condition(
                    lhs=schemas.Symbol(
                        name=schemas.SupportedSymbol.DATASET_NAME
                    ),
                    rhs=schemas.Value.infer(dataset_name),
                    op=schemas.FilterOperator.EQ,
                ),
                schemas.Condition(
                    lhs=schemas.Symbol(
                        name=schemas.SupportedSymbol.MODEL_NAME
                    ),
                    rhs=schemas.Value.infer(model_name),
                    op=schemas.FilterOperator.EQ,
                ),
                schemas.Condition(
                    lhs=schemas.Symbol(name=schemas.SupportedSymbol.TASK_TYPE),
                    rhs=schemas.Value.infer(enums.TaskType.CLASSIFICATION),
                    op=schemas.FilterOperator.CONTAINS,
                ),
            ],
            op=schemas.LogicalOperator.AND,
        )
    )

    gt, pd, labels = _aggregate_data(
        db=db,
        groundtruth_filter=groundtruth_filter,
        prediction_filter=prediction_filter,
        label_map=None,
    )

    metrics = _compute_clf_metrics(
        db,
        groundtruths=gt,
        predictions=pd,
        labels=labels,
        pr_curve_max_examples=0,
        metrics_to_return=[
            enums.MetricType.Precision,
            enums.MetricType.Recall,
            enums.MetricType.F1,
            enums.MetricType.Accuracy,
            enums.MetricType.ROCAUC,
            enums.MetricType.PrecisionRecallCurve,
        ],
    )

    confusion = [
        metric
        for metric in metrics
        if isinstance(metric, schemas.ConfusionMatrix)
    ]
    metrics = [
        metric
        for metric in metrics
        if not isinstance(metric, schemas.ConfusionMatrix)
    ]

    # Make matrices accessible by label_key
    confusion = {matrix.label_key: matrix for matrix in confusion}

    # Test confusion matrix w/ label_key "animal"
    expected_entries = [
        schemas.ConfusionMatrixEntry(
            groundtruth="bird", prediction="bird", count=1
        ),
        schemas.ConfusionMatrixEntry(
            groundtruth="dog", prediction="cat", count=2
        ),
        schemas.ConfusionMatrixEntry(
            groundtruth="bird", prediction="cat", count=1
        ),
        schemas.ConfusionMatrixEntry(
            groundtruth="bird", prediction="dog", count=1
        ),
        schemas.ConfusionMatrixEntry(
            groundtruth="cat", prediction="cat", count=1
        ),
    ]
    assert len(confusion["animal"].entries) == len(expected_entries)
    for e in expected_entries:
        assert e in confusion["animal"].entries

    # Test confusion matrix w/ label_key "color"
    expected_entries = [
        schemas.ConfusionMatrixEntry(
            groundtruth="white", prediction="white", count=1
        ),
        schemas.ConfusionMatrixEntry(
            groundtruth="white", prediction="blue", count=1
        ),
        schemas.ConfusionMatrixEntry(
            groundtruth="red", prediction="red", count=2
        ),
        schemas.ConfusionMatrixEntry(
            groundtruth="blue", prediction="white", count=1
        ),
        schemas.ConfusionMatrixEntry(
            groundtruth="black", prediction="red", count=1
        ),
    ]
    assert len(confusion["color"].entries) == len(expected_entries)
    for e in expected_entries:
        assert e in confusion["color"].entries

    # Test metrics (only ROCAUC)
    for metric in metrics:
        if isinstance(metric, schemas.ROCAUCMetric):
            if metric.label_key == "animal":
                assert metric.value == 0.8009259259259259
            elif metric.label_key == "color":
                assert metric.value == 0.43125


def test_classification(
    db: Session,
    dataset_name: str,
    model_name: str,
    classification_test_data,
):
    # default request
    job_request = schemas.EvaluationRequest(
        dataset_names=[dataset_name],
        model_names=[model_name],
        parameters=schemas.EvaluationParameters(
            task_type=enums.TaskType.CLASSIFICATION,
        ),
    )

    # creates evaluation job
    evaluations = create_or_get_evaluations(db=db, job_request=job_request)
    assert len(evaluations) == 1
    assert evaluations[0].status == enums.EvaluationStatus.PENDING

    # computation, normally run as background task
    _ = compute_clf_metrics(
        db=db,
        evaluation_id=evaluations[0].id,
    )

    # get evaluations
    evaluations = create_or_get_evaluations(db=db, job_request=job_request)
    assert len(evaluations) == 1
    assert evaluations[0].status in {
        enums.EvaluationStatus.RUNNING,
        enums.EvaluationStatus.DONE,
    }

    metrics = evaluations[0].metrics
    confusion = evaluations[0].confusion_matrices

    # Make matrices accessible by label_key
    assert confusion
    confusion = {matrix.label_key: matrix for matrix in confusion}

    # Test confusion matrix w/ label_key "animal"
    expected_entries = [
        schemas.ConfusionMatrixEntry(
            groundtruth="bird", prediction="bird", count=1
        ),
        schemas.ConfusionMatrixEntry(
            groundtruth="dog", prediction="cat", count=2
        ),
        schemas.ConfusionMatrixEntry(
            groundtruth="bird", prediction="cat", count=1
        ),
        schemas.ConfusionMatrixEntry(
            groundtruth="bird", prediction="dog", count=1
        ),
        schemas.ConfusionMatrixEntry(
            groundtruth="cat", prediction="cat", count=1
        ),
    ]
    assert len(confusion["animal"].entries) == len(expected_entries)
    for e in expected_entries:
        assert e in confusion["animal"].entries

    # Test confusion matrix w/ label_key "color"
    expected_entries = [
        schemas.ConfusionMatrixEntry(
            groundtruth="white", prediction="white", count=1
        ),
        schemas.ConfusionMatrixEntry(
            groundtruth="white", prediction="blue", count=1
        ),
        schemas.ConfusionMatrixEntry(
            groundtruth="red", prediction="red", count=2
        ),
        schemas.ConfusionMatrixEntry(
            groundtruth="blue", prediction="white", count=1
        ),
        schemas.ConfusionMatrixEntry(
            groundtruth="black", prediction="red", count=1
        ),
    ]
    assert len(confusion["color"].entries) == len(expected_entries)
    for e in expected_entries:
        assert e in confusion["color"].entries

    # Test metrics (only ROCAUC)
    assert metrics
    for metric in metrics:
        if isinstance(metric, schemas.ROCAUCMetric):
            if metric.label_key == "animal":
                assert metric.value == 0.8009259259259259
            elif metric.label_key == "color":
                assert metric.value == 0.43125


def test__compute_curves(
    db: Session,
    dataset_name: str,
    model_name: str,
    classification_test_data,
):
    """Test that _compute_curves correctly returns precision-recall curves for our animal ground truths."""

    prediction_filter = schemas.Filter(
        predictions=schemas.LogicalFunction(
            args=[
                schemas.Condition(
                    lhs=schemas.Symbol(
                        name=schemas.SupportedSymbol.MODEL_NAME
                    ),
                    rhs=schemas.Value.infer(model_name),
                    op=schemas.FilterOperator.EQ,
                ),
                schemas.Condition(
                    lhs=schemas.Symbol(name=schemas.SupportedSymbol.TASK_TYPE),
                    rhs=schemas.Value.infer(enums.TaskType.CLASSIFICATION),
                    op=schemas.FilterOperator.CONTAINS,
                ),
            ],
            op=schemas.LogicalOperator.AND,
        )
    )
    groundtruth_filter = schemas.Filter(
        groundtruths=schemas.LogicalFunction(
            args=[
                schemas.Condition(
                    lhs=schemas.Symbol(
                        name=schemas.SupportedSymbol.DATASET_NAME
                    ),
                    rhs=schemas.Value.infer(dataset_name),
                    op=schemas.FilterOperator.EQ,
                ),
                schemas.Condition(
                    lhs=schemas.Symbol(name=schemas.SupportedSymbol.TASK_TYPE),
                    rhs=schemas.Value.infer(enums.TaskType.CLASSIFICATION),
                    op=schemas.FilterOperator.CONTAINS,
                ),
            ],
            op=schemas.LogicalOperator.AND,
        )
    )

    groundtruths, predictions, labels = _aggregate_data(
        db=db,
        groundtruth_filter=groundtruth_filter,
        prediction_filter=prediction_filter,
        label_map=None,
    )

    # calculate the number of unique datums
    # used to determine the number of true negatives
    gt_datums = (
        db.query(
            groundtruths.c.datum_id,
            groundtruths.c.dataset_name,
            groundtruths.c.datum_uid,
        )
        .distinct()
        .all()
    )
    pd_datums = (
        db.query(
            predictions.c.datum_id,
            predictions.c.dataset_name,
            predictions.c.datum_uid,
        )
        .distinct()
        .all()
    )

    unique_datums = {
        datum_id: (dataset_name, datum_uid)
        for datum_id, dataset_name, datum_uid in gt_datums
    }
    unique_datums.update(
        {
            datum_id: (dataset_name, datum_uid)
            for datum_id, dataset_name, datum_uid in pd_datums
        }
    )

    curves = _compute_curves(
        db=db,
        predictions=predictions,
        groundtruths=groundtruths,
        labels=labels,
        unique_datums=unique_datums,
        pr_curve_max_examples=1,
        metrics_to_return=[
            enums.MetricType.PrecisionRecallCurve,
            enums.MetricType.DetailedPrecisionRecallCurve,
        ],
    )

    assert len(curves) == 4
    pr_curves = {
        curve.label_key: curve
        for curve in curves
        if isinstance(curve, schemas.PrecisionRecallCurve)
    }
    detailed_pr_curves = {
        curve.label_key: curve
        for curve in curves
        if isinstance(curve, schemas.DetailedPrecisionRecallCurve)
    }

    # check PrecisionRecallCurve
    pr_expected_answers = {
        # bird
        ("bird", 0.05, "tp"): 3,
        ("bird", 0.05, "fp"): 1,
        ("bird", 0.05, "tn"): 2,
        ("bird", 0.05, "fn"): 0,
        ("bird", 0.3, "tp"): 1,
        ("bird", 0.3, "fn"): 2,
        ("bird", 0.3, "fp"): 0,
        ("bird", 0.3, "tn"): 3,
        ("bird", 0.65, "fn"): 3,
        ("bird", 0.65, "tn"): 3,
        ("bird", 0.65, "tp"): 0,
        ("bird", 0.65, "fp"): 0,
        # dog
        ("dog", 0.05, "tp"): 2,
        ("dog", 0.05, "fp"): 3,
        ("dog", 0.05, "tn"): 1,
        ("dog", 0.05, "fn"): 0,
        ("dog", 0.45, "fn"): 2,
        ("dog", 0.45, "fp"): 1,
        ("dog", 0.45, "tn"): 3,
        ("dog", 0.45, "tp"): 0,
        ("dog", 0.8, "fn"): 2,
        ("dog", 0.8, "fp"): 0,
        ("dog", 0.8, "tn"): 4,
        ("dog", 0.8, "tp"): 0,
        # cat
        ("cat", 0.05, "tp"): 1,
        ("cat", 0.05, "tn"): 0,
        ("cat", 0.05, "fp"): 5,
        ("cat", 0.05, "fn"): 0,
        ("cat", 0.95, "tp"): 1,
        ("cat", 0.95, "fp"): 0,
        ("cat", 0.95, "tn"): 5,
        ("cat", 0.95, "fn"): 0,
    }

    for (
        value,
        threshold,
        metric,
    ), expected_length in pr_expected_answers.items():
        classification = pr_curves["animal"].value[value][threshold][metric]
        assert classification == expected_length

    # check DetailedPrecisionRecallCurve
    detailed_pr_expected_answers = {
        # bird
        ("bird", 0.05, "tp"): {"all": 3, "total": 3},
        ("bird", 0.05, "fp"): {
            "misclassifications": 1,
            "total": 1,
        },
        ("bird", 0.05, "tn"): {"all": 2, "total": 2},
        ("bird", 0.05, "fn"): {
            "no_predictions": 0,
            "misclassifications": 0,
            "total": 0,
        },
        # dog
        ("dog", 0.05, "tp"): {"all": 2, "total": 2},
        ("dog", 0.05, "fp"): {
            "misclassifications": 3,
            "total": 3,
        },
        ("dog", 0.05, "tn"): {"all": 1, "total": 1},
        ("dog", 0.8, "fn"): {
            "no_predictions": 1,
            "misclassifications": 1,
            "total": 2,
        },
        # cat
        ("cat", 0.05, "tp"): {"all": 1, "total": 1},
        ("cat", 0.05, "fp"): {
            "misclassifications": 5,
            "total": 5,
        },
        ("cat", 0.05, "tn"): {"all": 0, "total": 0},
        ("cat", 0.8, "fn"): {
            "no_predictions": 0,
            "misclassifications": 0,
            "total": 0,
        },
    }

    for (
        value,
        threshold,
        metric,
    ), expected_output in detailed_pr_expected_answers.items():
        model_output = detailed_pr_curves["animal"].value[value][threshold][
            metric
        ]
        assert isinstance(model_output, dict)
        assert model_output["total"] == expected_output["total"]
        assert all(
            [
                model_output["observations"][key]["count"]  # type: ignore - we know this element is a dict
                == expected_output[key]
                for key in [
                    key
                    for key in expected_output.keys()
                    if key not in ["total"]
                ]
            ]
        )

    # spot check number of examples
    assert (
        len(
            detailed_pr_curves["animal"].value["bird"][0.05]["tp"]["observations"]["all"][  # type: ignore - we know this element is a dict
                "examples"
            ]
        )
        == 1
    )
    assert (
        len(
            detailed_pr_curves["animal"].value["bird"][0.05]["tn"]["observations"]["all"][  # type: ignore - we know this element is a dict
                "examples"
            ]
        )
        == 1
    )

    # repeat the above, but with a higher pr_max_curves_example
    curves = _compute_curves(
        db=db,
        predictions=predictions,
        groundtruths=groundtruths,
        labels=labels,
        unique_datums=unique_datums,
        pr_curve_max_examples=3,
        metrics_to_return=[
            enums.MetricType.PrecisionRecallCurve,
            enums.MetricType.DetailedPrecisionRecallCurve,
        ],
    )

    assert len(curves) == 4
    pr_curves = {
        curve.label_key: curve
        for curve in curves
        if isinstance(curve, schemas.PrecisionRecallCurve)
    }
    detailed_pr_curves = {
        curve.label_key: curve
        for curve in curves
        if isinstance(curve, schemas.DetailedPrecisionRecallCurve)
    }

    # these outputs shouldn't have changed
    for (
        value,
        threshold,
        metric,
    ), expected_output in detailed_pr_expected_answers.items():
        model_output = detailed_pr_curves["animal"].value[value][threshold][
            metric
        ]
        assert isinstance(model_output, dict)
        assert model_output["total"] == expected_output["total"]
        assert all(
            [
                model_output["observations"][key]["count"]  # type: ignore - we know this element is a dict
                == expected_output[key]
                for key in [
                    key
                    for key in expected_output.keys()
                    if key not in ["total"]
                ]
            ]
        )

    assert (
        len(
            detailed_pr_curves["animal"].value["bird"][0.05]["tp"]["observations"]["all"][  # type: ignore - we know this element is a dict
                "examples"
            ]
        )
        == 3
    )
    assert (
        len(
            (
                detailed_pr_curves["animal"].value["bird"][0.05]["tn"]["observations"]["all"][  # type: ignore - we know this element is a dict
                    "examples"
                ]
            )
        )
        == 2  # only two examples exist
    )

    # test behavior if pr_curve_max_examples == 0
    curves = _compute_curves(
        db=db,
        predictions=predictions,
        groundtruths=groundtruths,
        labels=labels,
        unique_datums=unique_datums,
        pr_curve_max_examples=0,
        metrics_to_return=[
            enums.MetricType.PrecisionRecallCurve,
            enums.MetricType.DetailedPrecisionRecallCurve,
        ],
    )

    assert len(curves) == 4
    pr_curves = {
        curve.label_key: curve
        for curve in curves
        if isinstance(curve, schemas.PrecisionRecallCurve)
    }
    detailed_pr_curves = {
        curve.label_key: curve
        for curve in curves
        if isinstance(curve, schemas.DetailedPrecisionRecallCurve)
    }

    # these outputs shouldn't have changed
    for (
        value,
        threshold,
        metric,
    ), expected_output in detailed_pr_expected_answers.items():
        model_output = detailed_pr_curves["animal"].value[value][threshold][
            metric
        ]
        assert isinstance(model_output, dict)
        assert model_output["total"] == expected_output["total"]
        assert all(
            [
                model_output["observations"][key]["count"]  # type: ignore - we know this element is a dict
                == expected_output[key]
                for key in [
                    key
                    for key in expected_output.keys()
                    if key not in ["total"]
                ]
            ]
        )

    assert (
        len(
            detailed_pr_curves["animal"].value["bird"][0.05]["tp"]["observations"]["all"][  # type: ignore - we know this element is a dict
                "examples"
            ]
        )
        == 0
    )
    assert (
        len(
            (
                detailed_pr_curves["animal"].value["bird"][0.05]["tn"]["observations"]["all"][  # type: ignore - we know this element is a dict
                    "examples"
                ]
            )
        )
        == 0
    )
