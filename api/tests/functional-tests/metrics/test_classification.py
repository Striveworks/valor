import pytest
from sqlalchemy.orm import Session

from velour_api import crud, enums, schemas
from velour_api.backend.metrics.classification import (
    _compute_accuracy_from_cm,
    _compute_clf_metrics,
    _compute_confusion_matrix_at_label_key,
    _compute_roc_auc,
    create_clf_evaluation,
    create_clf_metrics,
)
from api.velour_api.backend.metrics.metric_utils import get_evaluations


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
            dataset=dataset_name,
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
            datum=imgs[i],
            annotations=[
                schemas.Annotation(
                    task_type=enums.TaskType.CLASSIFICATION,
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
            model=model_name,
            datum=imgs[i],
            annotations=[
                schemas.Annotation(
                    task_type=enums.TaskType.CLASSIFICATION,
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
            metadata={"type": enums.DataType.IMAGE.value},
        ),
    )
    for gt in gts:
        crud.create_groundtruth(db=db, groundtruth=gt)
    crud.finalize(db=db, dataset_name=dataset_name)

    crud.create_model(
        db=db,
        model=schemas.Model(
            name=model_name, metadata={"type": enums.DataType.IMAGE}
        ),
    )
    for pd in preds:
        crud.create_prediction(db=db, prediction=pd)


def test_compute_confusion_matrix_at_label_key(
    db: Session,
    dataset_name: str,
    model_name: str,
    classification_test_data,
):
    label_key = "animal"
    job_request = schemas.EvaluationJob(
        dataset=dataset_name,
        model=model_name,
        task_type=enums.TaskType.CLASSIFICATION,
        settings=schemas.EvaluationSettings(filters=schemas.Filter()),
    )

    cm = _compute_confusion_matrix_at_label_key(db, job_request, label_key)
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
    for entry in cm.entries:
        assert entry in expected_entries
    for entry in expected_entries:
        assert entry in cm.entries
    assert _compute_accuracy_from_cm(cm) == 2 / 6

    label_key = "color"
    cm = _compute_confusion_matrix_at_label_key(db, job_request, label_key)
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
    for entry in cm.entries:
        assert entry in expected_entries
    for entry in expected_entries:
        assert entry in cm.entries
    assert _compute_accuracy_from_cm(cm) == 3 / 6


def test_compute_confusion_matrix_at_label_key_and_filter(
    db: Session,
    dataset_name: str,
    model_name: str,
    classification_test_data,
):
    """
    Test filtering by metadata (md1: md1-val0).
    """
    job_request = schemas.EvaluationJob(
        dataset=dataset_name,
        model=model_name,
        task_type=enums.TaskType.CLASSIFICATION,
        settings=schemas.EvaluationSettings(
            filters=schemas.Filter(
                task_types=[enums.TaskType.CLASSIFICATION],
                datum_metadata={"md1": schemas.StringFilter(value="md1-val0")},
            )
        ),
    )

    cm = _compute_confusion_matrix_at_label_key(
        db,
        job_request=job_request,
        label_key="animal",
    )

    # for this metadatum and label id we have the gts
    # ["bird", "dog", "bird", "bird", "dog"] and the preds
    # ["bird", "cat", "cat", "dog", "cat"]
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
    job_request = schemas.EvaluationJob(
        dataset=dataset_name,
        model=model_name,
        task_type=enums.TaskType.CLASSIFICATION,
        settings=schemas.EvaluationSettings(
            filters=schemas.Filter(task_types=[enums.TaskType.CLASSIFICATION])
        ),
    )

    assert (
        _compute_roc_auc(db, job_request, label_key="animal")
        == 0.8009259259259259
    )
    assert _compute_roc_auc(db, job_request, label_key="color") == 0.43125

    with pytest.raises(RuntimeError) as exc_info:
        _compute_roc_auc(db, job_request, label_key="not a key")
    assert "is not a classification label" in str(exc_info)


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
    job_request = schemas.EvaluationJob(
        dataset=dataset_name,
        model=model_name,
        task_type=enums.TaskType.CLASSIFICATION,
        settings=schemas.EvaluationSettings(
            filters=schemas.Filter(
                task_types=[enums.TaskType.CLASSIFICATION],
                datum_metadata={"md1": schemas.StringFilter(value="md1-val0")},
            )
        ),
    )

    assert (
        _compute_roc_auc(
            db,
            job_request=job_request,
            label_key="animal",
        )
        == (0.5 + 2 / 3) / 2
    )


def test_compute_classification(
    db: Session,
    dataset_name: str,
    model_name: str,
    classification_test_data,
):
    """
    Tests the _compute_classification function.
    """
    job_request = schemas.EvaluationJob(
        dataset=dataset_name,
        model=model_name,
        task_type=enums.TaskType.CLASSIFICATION,
        settings=schemas.EvaluationSettings(
            filters=schemas.Filter(
                task_types=[enums.TaskType.CLASSIFICATION],
            )
        ),
    )

    confusion, metrics = _compute_clf_metrics(db, job_request)

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
    job_request = schemas.EvaluationJob(
        dataset=dataset_name,
        model=model_name,
        task_type=enums.TaskType.CLASSIFICATION,
        settings=schemas.EvaluationSettings(),
    )

    # creates evaluation job
    job_id = create_clf_evaluation(db, job_request)

    # computation, normally run as background task
    _ = create_clf_metrics(db, job_id)  # returns job_ud

    # get evaluations
    evaluations = get_evaluations(db, job_ids=[job_id])

    assert len(evaluations) == 1
    metrics = evaluations[0].metrics
    confusion = evaluations[0].confusion_matrices

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
