""" These integration tests should be run with a back end at http://localhost:8000
that is no auth
"""

from datetime import date, datetime, timedelta, timezone

import pytest

from valor import (
    Annotation,
    Client,
    Dataset,
    Datum,
    GroundTruth,
    Label,
    Model,
    Prediction,
)
from valor.enums import EvaluationStatus, TaskType
from valor.exceptions import ClientException


def test_evaluate_image_clf(
    client: Client,
    gt_clfs: list[GroundTruth],
    pred_clfs: list[Prediction],
    dataset_name: str,
    model_name: str,
):
    dataset = Dataset.create(dataset_name)
    for gt in gt_clfs:
        dataset.add_groundtruth(gt)
    dataset.finalize()

    model = Model.create(model_name)
    for pd in pred_clfs:
        model.add_prediction(dataset, pd)
    model.finalize_inferences(dataset)

    eval_job = model.evaluate_classification(dataset)

    assert eval_job.id

    assert eval_job.wait_for_completion(timeout=30) == EvaluationStatus.DONE

    metrics = eval_job.metrics

    expected_metrics = [
        {"type": "Accuracy", "parameters": {"label_key": "k4"}, "value": 0.5},
        {"type": "ROCAUC", "parameters": {"label_key": "k4"}, "value": 1.0},
        {
            "type": "Precision",
            "value": 1.0,  # no false predictions
            "label": {"key": "k4", "value": "v4"},
        },
        {
            "type": "Recall",
            "value": 0.5,  # img5 had the correct prediction, but not img6
            "label": {"key": "k4", "value": "v4"},
        },
        {
            "type": "F1",
            "value": 0.6666666666666666,
            "label": {"key": "k4", "value": "v4"},
        },
        {
            "type": "Precision",
            "value": 0.0,
            "label": {"key": "k4", "value": "v8"},
        },
        {
            "type": "Recall",
            "value": 0.0,
            "label": {"key": "k4", "value": "v8"},
        },
        {"type": "F1", "value": 0.0, "label": {"key": "k4", "value": "v8"}},
        {
            "type": "Precision",
            "value": -1.0,
            "label": {"key": "k4", "value": "v5"},
        },
        {
            "type": "Recall",
            "value": -1.0,
            "label": {"key": "k4", "value": "v5"},
        },
        {"type": "F1", "value": -1.0, "label": {"key": "k4", "value": "v5"}},
        {
            "type": "Precision",
            "value": -1.0,  # this value is -1 (not 0) because this label is never used anywhere; (k4, v8) has the higher score for img5, so it's chosen over (k4, v1)
            "label": {"key": "k4", "value": "v1"},
        },
        {
            "type": "Recall",
            "value": -1.0,
            "label": {"key": "k4", "value": "v1"},
        },
        {"type": "F1", "value": -1.0, "label": {"key": "k4", "value": "v1"}},
        {"type": "Accuracy", "parameters": {"label_key": "k5"}, "value": 0.0},
        {"type": "ROCAUC", "parameters": {"label_key": "k5"}, "value": 1.0},
        {
            "type": "Precision",
            "value": 0.0,
            "label": {"key": "k5", "value": "v1"},
        },
        {
            "type": "Recall",
            "value": 0.0,
            "label": {"key": "k5", "value": "v1"},
        },
        {"type": "F1", "value": 0.0, "label": {"key": "k5", "value": "v1"}},
        {
            "type": "Precision",
            "value": 0.0,
            "label": {"key": "k5", "value": "v5"},
        },
        {
            "type": "Recall",
            "value": 0.0,
            "label": {"key": "k5", "value": "v5"},
        },
        {"type": "F1", "value": 0.0, "label": {"key": "k5", "value": "v5"}},
        {"type": "Accuracy", "parameters": {"label_key": "k3"}, "value": 0.0},
        {"type": "ROCAUC", "parameters": {"label_key": "k3"}, "value": 1.0},
        {
            "type": "Precision",
            "value": 0.0,
            "label": {"key": "k3", "value": "v1"},
        },
        {
            "type": "Recall",
            "value": 0.0,
            "label": {"key": "k3", "value": "v1"},
        },
        {"type": "F1", "value": 0.0, "label": {"key": "k3", "value": "v1"}},
        {
            "type": "Precision",
            "value": 0.0,
            "label": {"key": "k3", "value": "v3"},
        },
        {
            "type": "Recall",
            "value": 0.0,
            "label": {"key": "k3", "value": "v3"},
        },
        {"type": "F1", "value": 0.0, "label": {"key": "k3", "value": "v3"}},
    ]

    expected_confusion_matrices = [
        {
            "label_key": "k5",
            "entries": [{"prediction": "v1", "groundtruth": "v5", "count": 1}],
        },
        {
            "label_key": "k4",
            "entries": [
                {"prediction": "v4", "groundtruth": "v4", "count": 1},
                {"prediction": "v8", "groundtruth": "v4", "count": 1},
            ],
        },
        {
            "label_key": "k3",
            "entries": [{"prediction": "v1", "groundtruth": "v3", "count": 1}],
        },
    ]

    for m in metrics:
        assert m in expected_metrics
    for m in expected_metrics:
        assert m in metrics

    confusion_matrices = eval_job.confusion_matrices
    for m in confusion_matrices:
        assert m in expected_confusion_matrices
    for m in expected_confusion_matrices:
        assert m in confusion_matrices


def test_evaluate_tabular_clf(
    client: Client,
    dataset_name: str,
    model_name: str,
    gt_clfs_tabular: list[int],
    pred_clfs_tabular: list[list[float]],
):
    assert len(gt_clfs_tabular) == len(pred_clfs_tabular)

    dataset = Dataset.create(name=dataset_name)
    gts = [
        GroundTruth(
            datum=Datum(uid=f"uid{i}"),
            annotations=[
                Annotation(
                    task_type=TaskType.CLASSIFICATION,
                    labels=[Label(key="class", value=str(t))],
                )
            ],
        )
        for i, t in enumerate(gt_clfs_tabular)
    ]
    for gt in gts:
        dataset.add_groundtruth(gt)

    # test dataset finalization
    model = Model.create(name=model_name)
    with pytest.raises(ClientException) as exc_info:
        model.evaluate_classification(dataset).wait_for_completion(timeout=30)
    assert "has not been finalized" in str(exc_info)

    dataset.finalize()

    pds = [
        Prediction(
            datum=Datum(uid=f"uid{i}"),
            annotations=[
                Annotation(
                    task_type=TaskType.CLASSIFICATION,
                    labels=[
                        Label(key="class", value=str(i), score=pred[i])
                        for i in range(len(pred))
                    ],
                )
            ],
        )
        for i, pred in enumerate(pred_clfs_tabular)
    ]
    for pd in pds[:-1]:
        model.add_prediction(dataset, pd)

    # test model finalization
    with pytest.raises(ClientException) as exc_info:
        model.evaluate_classification(dataset)
    assert "has not been finalized" in str(exc_info)

    # model is automatically finalized if all datums have a prediction
    model.add_prediction(dataset, pds[-1])

    # evaluate
    eval_job = model.evaluate_classification(dataset)

    assert eval_job.wait_for_completion(timeout=30) == EvaluationStatus.DONE

    metrics = eval_job.metrics

    expected_metrics = [
        {
            "type": "Accuracy",
            "parameters": {"label_key": "class"},
            "value": 0.5,
        },
        {
            "type": "ROCAUC",
            "parameters": {"label_key": "class"},
            "value": 0.7685185185185185,
        },
        {
            "type": "Precision",
            "value": 0.6666666666666666,
            "label": {"key": "class", "value": "1"},
        },
        {
            "type": "Recall",
            "value": 0.3333333333333333,
            "label": {"key": "class", "value": "1"},
        },
        {
            "type": "F1",
            "value": 0.4444444444444444,
            "label": {"key": "class", "value": "1"},
        },
        {
            "type": "Precision",
            "value": 0.0,
            "label": {"key": "class", "value": "2"},
        },
        {
            "type": "Recall",
            "value": 0.0,
            "label": {"key": "class", "value": "2"},
        },
        {"type": "F1", "value": 0.0, "label": {"key": "class", "value": "2"}},
        {
            "type": "Precision",
            "value": 0.5,
            "label": {"key": "class", "value": "0"},
        },
        {
            "type": "Recall",
            "value": 1.0,
            "label": {"key": "class", "value": "0"},
        },
        {
            "type": "F1",
            "value": 0.6666666666666666,
            "label": {"key": "class", "value": "0"},
        },
    ]
    for m in metrics:
        assert m in expected_metrics
    for m in expected_metrics:
        assert m in metrics

    confusion_matrices = eval_job.confusion_matrices

    expected_confusion_matrix = {
        "label_key": "class",
        "entries": [
            {"prediction": "0", "groundtruth": "0", "count": 3},
            {"prediction": "0", "groundtruth": "1", "count": 3},
            {"prediction": "1", "groundtruth": "1", "count": 2},
            {"prediction": "1", "groundtruth": "2", "count": 1},
            {"prediction": "2", "groundtruth": "1", "count": 1},
        ],
    }

    # validate that we can fetch the confusion matrices through get_evaluations()
    bulk_evals = client.get_evaluations(datasets=[dataset_name])

    assert len(bulk_evals) == 1
    for metric in bulk_evals[0].metrics:
        assert metric in expected_metrics
    assert len(bulk_evals[0].confusion_matrices[0]) == len(
        expected_confusion_matrix
    )

    # validate return schema
    assert len(confusion_matrices) == 1
    confusion_matrix = confusion_matrices[0]
    assert "label_key" in confusion_matrix
    assert "entries" in confusion_matrix

    # validate values
    assert (
        confusion_matrix["label_key"] == expected_confusion_matrix["label_key"]
    )
    for entry in confusion_matrix["entries"]:
        assert entry in expected_confusion_matrix["entries"]
    for entry in expected_confusion_matrix["entries"]:
        assert entry in confusion_matrix["entries"]

    # check model methods
    model.get_labels()

    assert model.name == model_name
    assert model.metadata is not None
    assert len(model.metadata) == 0

    # check evaluation
    results = model.get_evaluations()
    assert len(results) == 1
    assert results[0].datum_filter.dataset_names is not None
    assert len(results[0].datum_filter.dataset_names) == 1
    assert results[0].datum_filter.dataset_names[0] == dataset_name
    assert results[0].model_name == model_name
    assert isinstance(results[0].created_at, datetime)
    # check created at is within a minute of the current time
    assert (datetime.now(timezone.utc) - results[0].created_at) < timedelta(
        minutes=1
    )

    metrics_from_eval_settings_id = results[0].metrics
    assert len(metrics_from_eval_settings_id) == len(expected_metrics)
    for m in metrics_from_eval_settings_id:
        assert m in expected_metrics
    for m in expected_metrics:
        assert m in metrics_from_eval_settings_id

    # check confusion matrix
    confusion_matrices = results[0].confusion_matrices

    # validate return schema
    assert len(confusion_matrices) == 1
    confusion_matrix = confusion_matrices[0]
    assert "label_key" in confusion_matrix
    assert "entries" in confusion_matrix

    # validate values
    assert (
        confusion_matrix["label_key"] == expected_confusion_matrix["label_key"]
    )
    for entry in confusion_matrix["entries"]:
        assert entry in expected_confusion_matrix["entries"]
    for entry in expected_confusion_matrix["entries"]:
        assert entry in confusion_matrix["entries"]

    model.delete()
    assert len(client.get_models()) == 0


def test_stratify_clf_metrics(
    client: Client,
    gt_clfs_tabular: list[int],
    pred_clfs_tabular: list[list[float]],
    dataset_name: str,
    model_name: str,
):
    assert len(gt_clfs_tabular) == len(pred_clfs_tabular)

    # create data and two-different defining groups of cohorts
    dataset = Dataset.create(name=dataset_name)
    for i, label_value in enumerate(gt_clfs_tabular):
        gt = GroundTruth(
            datum=Datum(
                uid=f"uid{i}",
                metadata={
                    "md1": f"md1-val{i % 3}",
                    "md2": f"md2-val{i % 4}",
                    "md3": i % 3 == 2,
                },
            ),
            annotations=[
                Annotation(
                    task_type=TaskType.CLASSIFICATION,
                    labels=[Label(key="class", value=str(label_value))],
                )
            ],
        )
        dataset.add_groundtruth(gt)
    dataset.finalize()

    model = Model.create(name=model_name)
    for i, pred in enumerate(pred_clfs_tabular):
        pd = Prediction(
            datum=Datum(
                uid=f"uid{i}",
                metadata={
                    "md1": f"md1-val{i % 3}",
                    "md2": f"md2-val{i % 4}",
                    "md3": i % 3 == 2,
                },
            ),
            annotations=[
                Annotation(
                    task_type=TaskType.CLASSIFICATION,
                    labels=[
                        Label(key="class", value=str(pidx), score=pred[pidx])
                        for pidx in range(len(pred))
                    ],
                )
            ],
        )
        model.add_prediction(dataset, pd)
    model.finalize_inferences(dataset)

    eval_results_val2 = model.evaluate_classification(
        dataset,
        filter_by=[
            Datum.metadata["md1"] == "md1-val2",  # type: ignore - filter type error
        ],
    )
    assert (
        eval_results_val2.wait_for_completion(timeout=30)
        == EvaluationStatus.DONE
    )
    val2_metrics = eval_results_val2.metrics

    # should get the same thing if we use the boolean filter
    eval_results_bool = model.evaluate_classification(
        dataset,
        filter_by=[Datum.metadata["md3"] == True],  # type: ignore - filter type error # noqa 712
    )
    assert (
        eval_results_bool.wait_for_completion(timeout=30)
        == EvaluationStatus.DONE
    )
    val_bool_metrics = eval_results_bool.metrics

    # for value 2: the gts are [2, 0, 1] and preds are [[0.03, 0.88, 0.09], [1.0, 0.0, 0.0], [0.78, 0.21, 0.01]]
    # (hard preds [1, 0, 0])
    expected_metrics = [
        {
            "type": "Accuracy",
            "parameters": {"label_key": "class"},
            "value": 0.3333333333333333,
        },
        {
            "type": "ROCAUC",
            "parameters": {"label_key": "class"},
            "value": 0.8333333333333334,
        },
        {
            "type": "Precision",
            "value": 0.0,
            "label": {"key": "class", "value": "1"},
        },
        {
            "type": "Recall",
            "value": 0.0,
            "label": {"key": "class", "value": "1"},
        },
        {
            "type": "F1",
            "value": 0.0,
            "label": {"key": "class", "value": "1"},
        },
        {
            "type": "Precision",
            "value": 0.0,
            "label": {"key": "class", "value": "2"},
        },
        {
            "type": "Recall",
            "value": 0.0,
            "label": {"key": "class", "value": "2"},
        },
        {
            "type": "F1",
            "value": 0.0,
            "label": {"key": "class", "value": "2"},
        },
        {
            "type": "Precision",
            "value": 0.5,
            "label": {"key": "class", "value": "0"},
        },
        {
            "type": "Recall",
            "value": 1.0,
            "label": {"key": "class", "value": "0"},
        },
        {
            "type": "F1",
            "value": 0.6666666666666666,
            "label": {"key": "class", "value": "0"},
        },
    ]

    for metrics in [val2_metrics, val_bool_metrics]:
        assert len(metrics) == len(expected_metrics)
        for m in metrics:
            assert m in expected_metrics
        for m in expected_metrics:
            assert m in metrics


def test_stratify_clf_metrics_by_time(
    client: Client,
    gt_clfs_tabular: list[int],
    pred_clfs_tabular: list[list[float]],
    dataset_name: str,
    model_name: str,
):
    assert len(gt_clfs_tabular) == len(pred_clfs_tabular)

    # create data and two-different defining groups of cohorts
    dataset = Dataset.create(name=dataset_name)
    for i, label_value in enumerate(gt_clfs_tabular):
        gt = GroundTruth(
            datum=Datum(
                uid=f"uid{i}",
                metadata={
                    "md1": date.fromisoformat(f"{2000 + (i % 3)}-01-01"),
                    "md2": datetime.fromisoformat(f"{2000 + (i % 4)}-01-01"),
                },
            ),
            annotations=[
                Annotation(
                    task_type=TaskType.CLASSIFICATION,
                    labels=[Label(key="class", value=str(label_value))],
                )
            ],
        )
        dataset.add_groundtruth(gt)
    dataset.finalize()

    model = Model.create(name=model_name)
    for i, pred in enumerate(pred_clfs_tabular):
        pd = Prediction(
            datum=Datum(
                uid=f"uid{i}",
                metadata={
                    "md1": date.fromisoformat(f"{2000 + (i % 3)}-01-01"),
                    "md2": datetime.fromisoformat(f"{2000 + (i % 4)}-01-01"),
                },
            ),
            annotations=[
                Annotation(
                    task_type=TaskType.CLASSIFICATION,
                    labels=[
                        Label(key="class", value=str(pidx), score=pred[pidx])
                        for pidx in range(len(pred))
                    ],
                )
            ],
        )
        model.add_prediction(dataset, pd)
    model.finalize_inferences(dataset)

    eval_results_val2 = model.evaluate_classification(
        dataset,
        filter_by=[
            Datum.metadata["md1"] == date.fromisoformat("2002-01-01"),  # type: ignore - filter type error
        ],
    )
    assert (
        eval_results_val2.wait_for_completion(timeout=30)
        == EvaluationStatus.DONE
    )
    val2_metrics = eval_results_val2.metrics

    # for value 2: the gts are [2, 0, 1] and preds are [[0.03, 0.88, 0.09], [1.0, 0.0, 0.0], [0.78, 0.21, 0.01]]
    # (hard preds [1, 0, 0])
    expected_metrics = [
        {
            "type": "Accuracy",
            "parameters": {"label_key": "class"},
            "value": 0.3333333333333333,
        },
        {
            "type": "ROCAUC",
            "parameters": {"label_key": "class"},
            "value": 0.8333333333333334,
        },
        {
            "type": "Precision",
            "value": 0.0,
            "label": {"key": "class", "value": "1"},
        },
        {
            "type": "Recall",
            "value": 0.0,
            "label": {"key": "class", "value": "1"},
        },
        {
            "type": "F1",
            "value": 0.0,
            "label": {"key": "class", "value": "1"},
        },
        {
            "type": "Precision",
            "value": 0.0,
            "label": {"key": "class", "value": "2"},
        },
        {
            "type": "Recall",
            "value": 0.0,
            "label": {"key": "class", "value": "2"},
        },
        {
            "type": "F1",
            "value": 0.0,
            "label": {"key": "class", "value": "2"},
        },
        {
            "type": "Precision",
            "value": 0.5,
            "label": {"key": "class", "value": "0"},
        },
        {
            "type": "Recall",
            "value": 1.0,
            "label": {"key": "class", "value": "0"},
        },
        {
            "type": "F1",
            "value": 0.6666666666666666,
            "label": {"key": "class", "value": "0"},
        },
    ]

    assert len(val2_metrics) == len(expected_metrics)
    for m in val2_metrics:
        assert m in expected_metrics
    for m in expected_metrics:
        assert m in val2_metrics


@pytest.fixture
def gt_clfs_with_label_maps(
    img5: Datum,
    img6: Datum,
    img8: Datum,
) -> list[GroundTruth]:
    return [
        GroundTruth(
            datum=img5,
            annotations=[
                Annotation(
                    task_type=TaskType.CLASSIFICATION,
                    labels=[
                        Label(key="k4", value="v4"),
                        Label(key="k5", value="v5"),
                        Label(key="class", value="siamese cat"),
                    ],
                ),
            ],
        ),
        GroundTruth(
            datum=img6,
            annotations=[
                Annotation(
                    task_type=TaskType.CLASSIFICATION,
                    labels=[
                        Label(key="k4", value="v4"),
                        Label(key="class", value="british shorthair"),
                    ],
                )
            ],
        ),
        GroundTruth(
            datum=img8,
            annotations=[
                Annotation(
                    task_type=TaskType.CLASSIFICATION,
                    labels=[
                        Label(key="k3", value="v3"),
                        Label(key="class", value="tabby cat"),
                    ],
                )
            ],
        ),
    ]


@pytest.fixture
def pred_clfs_with_label_maps(
    model_name: str,
    img5: Datum,
    img6: Datum,
    img8: Datum,
) -> list[Prediction]:
    return [
        Prediction(
            datum=img5,
            annotations=[
                Annotation(
                    task_type=TaskType.CLASSIFICATION,
                    labels=[
                        Label(key="k4", value="v1", score=0.47),
                        Label(key="k4", value="v8", score=0.53),
                        Label(key="k5", value="v1", score=1.0),
                        Label(key="class", value="cat", score=1.0),
                    ],
                )
            ],
        ),
        Prediction(
            datum=img6,
            annotations=[
                Annotation(
                    task_type=TaskType.CLASSIFICATION,
                    labels=[
                        Label(key="k4", value="v4", score=0.71),
                        Label(key="k4", value="v5", score=0.29),
                        Label(key="class_name", value="cat", score=1.0),
                    ],
                )
            ],
        ),
        Prediction(
            datum=img8,
            annotations=[
                Annotation(
                    task_type=TaskType.CLASSIFICATION,
                    labels=[
                        Label(key="k3", value="v1", score=1.0),
                        Label(key="class", value="cat", score=1.0),
                    ],
                )
            ],
        ),
    ]


def test_evaluate_classification_with_label_maps(
    client: Client,
    gt_clfs_with_label_maps: list[GroundTruth],
    pred_clfs_with_label_maps: list[Prediction],
    dataset_name: str,
    model_name: str,
):
    dataset = Dataset.create(dataset_name)
    for gt in gt_clfs_with_label_maps:
        dataset.add_groundtruth(gt)
    dataset.finalize()

    model = Model.create(model_name)
    for pd in pred_clfs_with_label_maps:
        model.add_prediction(dataset, pd)
    model.finalize_inferences(dataset)

    # check baseline case, where we have mismatched ground truth and prediction label keys
    result = model.evaluate_classification(dataset).wait_for_completion(
        timeout=30
    )

    assert result.value == "failed"

    # now try using a label map to connect all the cats

    label_mapping = {
        # map the ground truths
        Label(key="class", value="tabby cat"): Label(
            key="special_class", value="cat_type1"
        ),
        Label(key="class", value="siamese cat"): Label(
            key="special_class", value="cat_type1"
        ),
        Label(key="class", value="british shorthair"): Label(
            key="special_class", value="cat_type1"
        ),
        # map the predictions
        Label(key="class", value="cat"): Label(
            key="special_class", value="cat_type1"
        ),
        Label(key="class_name", value="cat"): Label(
            key="special_class", value="cat_type1"
        ),
    }

    cat_expected_metrics = [
        {"type": "Accuracy", "parameters": {"label_key": "k3"}, "value": 0.0},
        {"type": "ROCAUC", "parameters": {"label_key": "k3"}, "value": 1.0},
        {
            "type": "PrecisionRecallCurve",
            "parameters": {"label_key": "k3", "pr_curve_iou_threshold": None},
            "value": {
                "v1": {
                    "0.1": {
                        "fn": 0,
                        "fp": 1,  # img8
                        "tp": 0,
                        "recall": -1,
                        "f1_score": -1,
                        "precision": 0.0,
                    },
                    "0.2": {
                        "fn": 0,
                        "fp": 1,
                        "tp": 0,
                        "recall": -1,
                        "f1_score": -1,
                        "precision": 0.0,
                    },
                    "0.3": {
                        "fn": 0,
                        "fp": 1,
                        "tp": 0,
                        "recall": -1,
                        "f1_score": -1,
                        "precision": 0.0,
                    },
                    "0.4": {
                        "fn": 0,
                        "fp": 1,
                        "tp": 0,
                        "recall": -1,
                        "f1_score": -1,
                        "precision": 0.0,
                    },
                    "0.5": {
                        "fn": 0,
                        "fp": 1,
                        "tp": 0,
                        "recall": -1,
                        "f1_score": -1,
                        "precision": 0.0,
                    },
                    "0.6": {
                        "fn": 0,
                        "fp": 1,
                        "tp": 0,
                        "recall": -1,
                        "f1_score": -1,
                        "precision": 0.0,
                    },
                    "0.7": {
                        "fn": 0,
                        "fp": 1,
                        "tp": 0,
                        "recall": -1,
                        "f1_score": -1,
                        "precision": 0.0,
                    },
                    "0.8": {
                        "fn": 0,
                        "fp": 1,
                        "tp": 0,
                        "recall": -1,
                        "f1_score": -1,
                        "precision": 0.0,
                    },
                    "0.9": {
                        "fn": 0,
                        "fp": 1,
                        "tp": 0,
                        "recall": -1,
                        "f1_score": -1,
                        "precision": 0.0,
                    },
                    "0.05": {
                        "fn": 0,
                        "fp": 1,
                        "tp": 0,
                        "recall": -1,
                        "f1_score": -1,
                        "precision": 0.0,
                    },
                    "0.15": {
                        "fn": 0,
                        "fp": 1,
                        "tp": 0,
                        "recall": -1,
                        "f1_score": -1,
                        "precision": 0.0,
                    },
                    "0.25": {
                        "fn": 0,
                        "fp": 1,
                        "tp": 0,
                        "recall": -1,
                        "f1_score": -1,
                        "precision": 0.0,
                    },
                    "0.35": {
                        "fn": 0,
                        "fp": 1,
                        "tp": 0,
                        "recall": -1,
                        "f1_score": -1,
                        "precision": 0.0,
                    },
                    "0.45": {
                        "fn": 0,
                        "fp": 1,
                        "tp": 0,
                        "recall": -1,
                        "f1_score": -1,
                        "precision": 0.0,
                    },
                    "0.55": {
                        "fn": 0,
                        "fp": 1,
                        "tp": 0,
                        "recall": -1,
                        "f1_score": -1,
                        "precision": 0.0,
                    },
                    "0.65": {
                        "fn": 0,
                        "fp": 1,
                        "tp": 0,
                        "recall": -1,
                        "f1_score": -1,
                        "precision": 0.0,
                    },
                    "0.75": {
                        "fn": 0,
                        "fp": 1,
                        "tp": 0,
                        "recall": -1,
                        "f1_score": -1,
                        "precision": 0.0,
                    },
                    "0.85": {
                        "fn": 0,
                        "fp": 1,
                        "tp": 0,
                        "recall": -1,
                        "f1_score": -1,
                        "precision": 0.0,
                    },
                    "0.95": {
                        "fn": 0,
                        "fp": 1,
                        "tp": 0,
                        "recall": -1,
                        "f1_score": -1,
                        "precision": 0.0,
                    },
                },
                "v3": {
                    "0.1": {
                        "fn": 1,  # img5
                        "fp": 0,
                        "tp": 0,
                        "recall": 0.0,
                        "f1_score": -1,
                        "precision": -1,
                    },
                    "0.2": {
                        "fn": 1,
                        "fp": 0,
                        "tp": 0,
                        "recall": 0.0,
                        "f1_score": -1,
                        "precision": -1,
                    },
                    "0.3": {
                        "fn": 1,
                        "fp": 0,
                        "tp": 0,
                        "recall": 0.0,
                        "f1_score": -1,
                        "precision": -1,
                    },
                    "0.4": {
                        "fn": 1,
                        "fp": 0,
                        "tp": 0,
                        "recall": 0.0,
                        "f1_score": -1,
                        "precision": -1,
                    },
                    "0.5": {
                        "fn": 1,
                        "fp": 0,
                        "tp": 0,
                        "recall": 0.0,
                        "f1_score": -1,
                        "precision": -1,
                    },
                    "0.6": {
                        "fn": 1,
                        "fp": 0,
                        "tp": 0,
                        "recall": 0.0,
                        "f1_score": -1,
                        "precision": -1,
                    },
                    "0.7": {
                        "fn": 1,
                        "fp": 0,
                        "tp": 0,
                        "recall": 0.0,
                        "f1_score": -1,
                        "precision": -1,
                    },
                    "0.8": {
                        "fn": 1,
                        "fp": 0,
                        "tp": 0,
                        "recall": 0.0,
                        "f1_score": -1,
                        "precision": -1,
                    },
                    "0.9": {
                        "fn": 1,
                        "fp": 0,
                        "tp": 0,
                        "recall": 0.0,
                        "f1_score": -1,
                        "precision": -1,
                    },
                    "0.05": {
                        "fn": 1,
                        "fp": 0,
                        "tp": 0,
                        "recall": 0.0,
                        "f1_score": -1,
                        "precision": -1,
                    },
                    "0.15": {
                        "fn": 1,
                        "fp": 0,
                        "tp": 0,
                        "recall": 0.0,
                        "f1_score": -1,
                        "precision": -1,
                    },
                    "0.25": {
                        "fn": 1,
                        "fp": 0,
                        "tp": 0,
                        "recall": 0.0,
                        "f1_score": -1,
                        "precision": -1,
                    },
                    "0.35": {
                        "fn": 1,
                        "fp": 0,
                        "tp": 0,
                        "recall": 0.0,
                        "f1_score": -1,
                        "precision": -1,
                    },
                    "0.45": {
                        "fn": 1,
                        "fp": 0,
                        "tp": 0,
                        "recall": 0.0,
                        "f1_score": -1,
                        "precision": -1,
                    },
                    "0.55": {
                        "fn": 1,
                        "fp": 0,
                        "tp": 0,
                        "recall": 0.0,
                        "f1_score": -1,
                        "precision": -1,
                    },
                    "0.65": {
                        "fn": 1,
                        "fp": 0,
                        "tp": 0,
                        "recall": 0.0,
                        "f1_score": -1,
                        "precision": -1,
                    },
                    "0.75": {
                        "fn": 1,
                        "fp": 0,
                        "tp": 0,
                        "recall": 0.0,
                        "f1_score": -1,
                        "precision": -1,
                    },
                    "0.85": {
                        "fn": 1,
                        "fp": 0,
                        "tp": 0,
                        "recall": 0.0,
                        "f1_score": -1,
                        "precision": -1,
                    },
                    "0.95": {
                        "fn": 1,
                        "fp": 0,
                        "tp": 0,
                        "recall": 0.0,
                        "f1_score": -1,
                        "precision": -1,
                    },
                },
            },
        },
        {
            "type": "Precision",
            "value": 0.0,
            "label": {"key": "k3", "value": "v1"},
        },
        {
            "type": "Recall",
            "value": 0.0,
            "label": {"key": "k3", "value": "v1"},
        },
        {"type": "F1", "value": 0.0, "label": {"key": "k3", "value": "v1"}},
        {
            "type": "Precision",
            "value": 0.0,
            "label": {"key": "k3", "value": "v3"},
        },
        {
            "type": "Recall",
            "value": 0.0,
            "label": {"key": "k3", "value": "v3"},
        },
        {"type": "F1", "value": 0.0, "label": {"key": "k3", "value": "v3"}},
        {"type": "Accuracy", "parameters": {"label_key": "k5"}, "value": 0.0},
        {"type": "ROCAUC", "parameters": {"label_key": "k5"}, "value": 1.0},
        {
            "type": "PrecisionRecallCurve",
            "parameters": {"label_key": "k5", "pr_curve_iou_threshold": None},
            "value": {
                "v1": {
                    "0.1": {
                        "fn": 0,
                        "fp": 1,
                        "tp": 0,
                        "recall": -1,
                        "f1_score": -1,
                        "precision": 0.0,
                    },
                    "0.2": {
                        "fn": 0,
                        "fp": 1,
                        "tp": 0,
                        "recall": -1,
                        "f1_score": -1,
                        "precision": 0.0,
                    },
                    "0.3": {
                        "fn": 0,
                        "fp": 1,
                        "tp": 0,
                        "recall": -1,
                        "f1_score": -1,
                        "precision": 0.0,
                    },
                    "0.4": {
                        "fn": 0,
                        "fp": 1,
                        "tp": 0,
                        "recall": -1,
                        "f1_score": -1,
                        "precision": 0.0,
                    },
                    "0.5": {
                        "fn": 0,
                        "fp": 1,
                        "tp": 0,
                        "recall": -1,
                        "f1_score": -1,
                        "precision": 0.0,
                    },
                    "0.6": {
                        "fn": 0,
                        "fp": 1,
                        "tp": 0,
                        "recall": -1,
                        "f1_score": -1,
                        "precision": 0.0,
                    },
                    "0.7": {
                        "fn": 0,
                        "fp": 1,
                        "tp": 0,
                        "recall": -1,
                        "f1_score": -1,
                        "precision": 0.0,
                    },
                    "0.8": {
                        "fn": 0,
                        "fp": 1,
                        "tp": 0,
                        "recall": -1,
                        "f1_score": -1,
                        "precision": 0.0,
                    },
                    "0.9": {
                        "fn": 0,
                        "fp": 1,
                        "tp": 0,
                        "recall": -1,
                        "f1_score": -1,
                        "precision": 0.0,
                    },
                    "0.05": {
                        "fn": 0,
                        "fp": 1,
                        "tp": 0,
                        "recall": -1,
                        "f1_score": -1,
                        "precision": 0.0,
                    },
                    "0.15": {
                        "fn": 0,
                        "fp": 1,
                        "tp": 0,
                        "recall": -1,
                        "f1_score": -1,
                        "precision": 0.0,
                    },
                    "0.25": {
                        "fn": 0,
                        "fp": 1,
                        "tp": 0,
                        "recall": -1,
                        "f1_score": -1,
                        "precision": 0.0,
                    },
                    "0.35": {
                        "fn": 0,
                        "fp": 1,
                        "tp": 0,
                        "recall": -1,
                        "f1_score": -1,
                        "precision": 0.0,
                    },
                    "0.45": {
                        "fn": 0,
                        "fp": 1,
                        "tp": 0,
                        "recall": -1,
                        "f1_score": -1,
                        "precision": 0.0,
                    },
                    "0.55": {
                        "fn": 0,
                        "fp": 1,
                        "tp": 0,
                        "recall": -1,
                        "f1_score": -1,
                        "precision": 0.0,
                    },
                    "0.65": {
                        "fn": 0,
                        "fp": 1,
                        "tp": 0,
                        "recall": -1,
                        "f1_score": -1,
                        "precision": 0.0,
                    },
                    "0.75": {
                        "fn": 0,
                        "fp": 1,
                        "tp": 0,
                        "recall": -1,
                        "f1_score": -1,
                        "precision": 0.0,
                    },
                    "0.85": {
                        "fn": 0,
                        "fp": 1,
                        "tp": 0,
                        "recall": -1,
                        "f1_score": -1,
                        "precision": 0.0,
                    },
                    "0.95": {
                        "fn": 0,
                        "fp": 1,
                        "tp": 0,
                        "recall": -1,
                        "f1_score": -1,
                        "precision": 0.0,
                    },
                },
                "v5": {
                    "0.1": {
                        "fn": 1,  # img5
                        "fp": 0,
                        "tp": 0,
                        "recall": 0.0,
                        "f1_score": -1,
                        "precision": -1,
                    },
                    "0.2": {
                        "fn": 1,
                        "fp": 0,
                        "tp": 0,
                        "recall": 0.0,
                        "f1_score": -1,
                        "precision": -1,
                    },
                    "0.3": {
                        "fn": 1,
                        "fp": 0,
                        "tp": 0,
                        "recall": 0.0,
                        "f1_score": -1,
                        "precision": -1,
                    },
                    "0.4": {
                        "fn": 1,
                        "fp": 0,
                        "tp": 0,
                        "recall": 0.0,
                        "f1_score": -1,
                        "precision": -1,
                    },
                    "0.5": {
                        "fn": 1,
                        "fp": 0,
                        "tp": 0,
                        "recall": 0.0,
                        "f1_score": -1,
                        "precision": -1,
                    },
                    "0.6": {
                        "fn": 1,
                        "fp": 0,
                        "tp": 0,
                        "recall": 0.0,
                        "f1_score": -1,
                        "precision": -1,
                    },
                    "0.7": {
                        "fn": 1,
                        "fp": 0,
                        "tp": 0,
                        "recall": 0.0,
                        "f1_score": -1,
                        "precision": -1,
                    },
                    "0.8": {
                        "fn": 1,
                        "fp": 0,
                        "tp": 0,
                        "recall": 0.0,
                        "f1_score": -1,
                        "precision": -1,
                    },
                    "0.9": {
                        "fn": 1,
                        "fp": 0,
                        "tp": 0,
                        "recall": 0.0,
                        "f1_score": -1,
                        "precision": -1,
                    },
                    "0.05": {
                        "fn": 1,
                        "fp": 0,
                        "tp": 0,
                        "recall": 0.0,
                        "f1_score": -1,
                        "precision": -1,
                    },
                    "0.15": {
                        "fn": 1,
                        "fp": 0,
                        "tp": 0,
                        "recall": 0.0,
                        "f1_score": -1,
                        "precision": -1,
                    },
                    "0.25": {
                        "fn": 1,
                        "fp": 0,
                        "tp": 0,
                        "recall": 0.0,
                        "f1_score": -1,
                        "precision": -1,
                    },
                    "0.35": {
                        "fn": 1,
                        "fp": 0,
                        "tp": 0,
                        "recall": 0.0,
                        "f1_score": -1,
                        "precision": -1,
                    },
                    "0.45": {
                        "fn": 1,
                        "fp": 0,
                        "tp": 0,
                        "recall": 0.0,
                        "f1_score": -1,
                        "precision": -1,
                    },
                    "0.55": {
                        "fn": 1,
                        "fp": 0,
                        "tp": 0,
                        "recall": 0.0,
                        "f1_score": -1,
                        "precision": -1,
                    },
                    "0.65": {
                        "fn": 1,
                        "fp": 0,
                        "tp": 0,
                        "recall": 0.0,
                        "f1_score": -1,
                        "precision": -1,
                    },
                    "0.75": {
                        "fn": 1,
                        "fp": 0,
                        "tp": 0,
                        "recall": 0.0,
                        "f1_score": -1,
                        "precision": -1,
                    },
                    "0.85": {
                        "fn": 1,
                        "fp": 0,
                        "tp": 0,
                        "recall": 0.0,
                        "f1_score": -1,
                        "precision": -1,
                    },
                    "0.95": {
                        "fn": 1,
                        "fp": 0,
                        "tp": 0,
                        "recall": 0.0,
                        "f1_score": -1,
                        "precision": -1,
                    },
                },
            },
        },
        {
            "type": "Precision",
            "value": 0.0,
            "label": {"key": "k5", "value": "v5"},
        },
        {
            "type": "Recall",
            "value": 0.0,
            "label": {"key": "k5", "value": "v5"},
        },
        {"type": "F1", "value": 0.0, "label": {"key": "k5", "value": "v5"}},
        {
            "type": "Precision",
            "value": 0.0,
            "label": {"key": "k5", "value": "v1"},
        },
        {
            "type": "Recall",
            "value": 0.0,
            "label": {"key": "k5", "value": "v1"},
        },
        {"type": "F1", "value": 0.0, "label": {"key": "k5", "value": "v1"}},
        {
            "type": "Accuracy",
            "parameters": {"label_key": "special_class"},
            "value": 1.0,
        },
        {
            "type": "ROCAUC",
            "parameters": {"label_key": "special_class"},
            "value": -1.0,
        },
        {
            "type": "PrecisionRecallCurve",
            "parameters": {
                "label_key": "special_class",
                "pr_curve_iou_threshold": None,
            },
            "value": {
                "cat_type1": {
                    "0.1": {
                        "fn": 0,
                        "fp": 0,
                        "tp": 3,
                        "recall": 1.0,
                        "f1_score": 1.0,
                        "precision": 1.0,
                    },
                    "0.2": {
                        "fn": 0,
                        "fp": 0,
                        "tp": 3,
                        "recall": 1.0,
                        "f1_score": 1.0,
                        "precision": 1.0,
                    },
                    "0.3": {
                        "fn": 0,
                        "fp": 0,
                        "tp": 3,
                        "recall": 1.0,
                        "f1_score": 1.0,
                        "precision": 1.0,
                    },
                    "0.4": {
                        "fn": 0,
                        "fp": 0,
                        "tp": 3,
                        "recall": 1.0,
                        "f1_score": 1.0,
                        "precision": 1.0,
                    },
                    "0.5": {
                        "fn": 0,
                        "fp": 0,
                        "tp": 3,
                        "recall": 1.0,
                        "f1_score": 1.0,
                        "precision": 1.0,
                    },
                    "0.6": {
                        "fn": 0,
                        "fp": 0,
                        "tp": 3,
                        "recall": 1.0,
                        "f1_score": 1.0,
                        "precision": 1.0,
                    },
                    "0.7": {
                        "fn": 0,
                        "fp": 0,
                        "tp": 3,
                        "recall": 1.0,
                        "f1_score": 1.0,
                        "precision": 1.0,
                    },
                    "0.8": {
                        "fn": 0,
                        "fp": 0,
                        "tp": 3,
                        "recall": 1.0,
                        "f1_score": 1.0,
                        "precision": 1.0,
                    },
                    "0.9": {
                        "fn": 0,
                        "fp": 0,
                        "tp": 3,
                        "recall": 1.0,
                        "f1_score": 1.0,
                        "precision": 1.0,
                    },
                    "0.05": {
                        "fn": 0,
                        "fp": 0,
                        "tp": 3,
                        "recall": 1.0,
                        "f1_score": 1.0,
                        "precision": 1.0,
                    },
                    "0.15": {
                        "fn": 0,
                        "fp": 0,
                        "tp": 3,
                        "recall": 1.0,
                        "f1_score": 1.0,
                        "precision": 1.0,
                    },
                    "0.25": {
                        "fn": 0,
                        "fp": 0,
                        "tp": 3,
                        "recall": 1.0,
                        "f1_score": 1.0,
                        "precision": 1.0,
                    },
                    "0.35": {
                        "fn": 0,
                        "fp": 0,
                        "tp": 3,
                        "recall": 1.0,
                        "f1_score": 1.0,
                        "precision": 1.0,
                    },
                    "0.45": {
                        "fn": 0,
                        "fp": 0,
                        "tp": 3,
                        "recall": 1.0,
                        "f1_score": 1.0,
                        "precision": 1.0,
                    },
                    "0.55": {
                        "fn": 0,
                        "fp": 0,
                        "tp": 3,
                        "recall": 1.0,
                        "f1_score": 1.0,
                        "precision": 1.0,
                    },
                    "0.65": {
                        "fn": 0,
                        "fp": 0,
                        "tp": 3,
                        "recall": 1.0,
                        "f1_score": 1.0,
                        "precision": 1.0,
                    },
                    "0.75": {
                        "fn": 0,
                        "fp": 0,
                        "tp": 3,
                        "recall": 1.0,
                        "f1_score": 1.0,
                        "precision": 1.0,
                    },
                    "0.85": {
                        "fn": 0,
                        "fp": 0,
                        "tp": 3,
                        "recall": 1.0,
                        "f1_score": 1.0,
                        "precision": 1.0,
                    },
                    "0.95": {
                        "fn": 0,
                        "fp": 0,
                        "tp": 3,
                        "recall": 1.0,
                        "f1_score": 1.0,
                        "precision": 1.0,
                    },
                }
            },
        },
        {
            "type": "Precision",
            "value": 1.0,
            "label": {"key": "special_class", "value": "cat_type1"},
        },
        {
            "type": "Recall",
            "value": 1.0,
            "label": {"key": "special_class", "value": "cat_type1"},
        },
        {
            "type": "F1",
            "value": 1.0,
            "label": {"key": "special_class", "value": "cat_type1"},
        },
        {"type": "Accuracy", "parameters": {"label_key": "k4"}, "value": 0.5},
        {
            "type": "ROCAUC",
            "parameters": {
                "label_key": "k4",
            },
            "value": 1.0,
        },
        {
            "type": "PrecisionRecallCurve",
            "parameters": {"label_key": "k4", "pr_curve_iou_threshold": None},
            "value": {
                "v1": {
                    "0.1": {
                        "fn": 0,
                        "fp": 1,
                        "tp": 0,
                        "recall": -1,
                        "f1_score": -1,
                        "precision": 0.0,
                    },
                    "0.2": {
                        "fn": 0,
                        "fp": 1,
                        "tp": 0,
                        "recall": -1,
                        "f1_score": -1,
                        "precision": 0.0,
                    },
                    "0.3": {
                        "fn": 0,
                        "fp": 1,
                        "tp": 0,
                        "recall": -1,
                        "f1_score": -1,
                        "precision": 0.0,
                    },
                    "0.4": {
                        "fn": 0,
                        "fp": 1,
                        "tp": 0,
                        "recall": -1,
                        "f1_score": -1,
                        "precision": 0.0,
                    },
                    "0.5": {
                        "fn": 0,
                        "fp": 0,  # img5 pred drops out with a score of .47
                        "tp": 0,
                        "recall": -1,
                        "f1_score": -1.0,
                        "precision": -1,
                    },
                    "0.6": {
                        "fn": 0,
                        "fp": 0,
                        "tp": 0,
                        "recall": -1,
                        "f1_score": -1.0,
                        "precision": -1,
                    },
                    "0.7": {
                        "fn": 0,
                        "fp": 0,
                        "tp": 0,
                        "recall": -1,
                        "f1_score": -1.0,
                        "precision": -1,
                    },
                    "0.8": {
                        "fn": 0,
                        "fp": 0,
                        "tp": 0,
                        "recall": -1,
                        "f1_score": -1.0,
                        "precision": -1,
                    },
                    "0.9": {
                        "fn": 0,
                        "fp": 0,
                        "tp": 0,
                        "recall": -1,
                        "f1_score": -1.0,
                        "precision": -1,
                    },
                    "0.05": {
                        "fn": 0,
                        "fp": 1,
                        "tp": 0,
                        "recall": -1,
                        "f1_score": -1,
                        "precision": 0.0,
                    },
                    "0.15": {
                        "fn": 0,
                        "fp": 1,
                        "tp": 0,
                        "recall": -1,
                        "f1_score": -1,
                        "precision": 0.0,
                    },
                    "0.25": {
                        "fn": 0,
                        "fp": 1,
                        "tp": 0,
                        "recall": -1,
                        "f1_score": -1,
                        "precision": 0.0,
                    },
                    "0.35": {
                        "fn": 0,
                        "fp": 1,
                        "tp": 0,
                        "recall": -1,
                        "f1_score": -1,
                        "precision": 0.0,
                    },
                    "0.45": {
                        "fn": 0,
                        "fp": 1,
                        "tp": 0,
                        "recall": -1,
                        "f1_score": -1,
                        "precision": 0.0,
                    },
                    "0.55": {
                        "fn": 0,
                        "fp": 0,
                        "tp": 0,
                        "recall": -1,
                        "f1_score": -1.0,
                        "precision": -1,
                    },
                    "0.65": {
                        "fn": 0,
                        "fp": 0,
                        "tp": 0,
                        "recall": -1,
                        "f1_score": -1.0,
                        "precision": -1,
                    },
                    "0.75": {
                        "fn": 0,
                        "fp": 0,
                        "tp": 0,
                        "recall": -1,
                        "f1_score": -1.0,
                        "precision": -1,
                    },
                    "0.85": {
                        "fn": 0,
                        "fp": 0,
                        "tp": 0,
                        "recall": -1,
                        "f1_score": -1.0,
                        "precision": -1,
                    },
                    "0.95": {
                        "fn": 0,
                        "fp": 0,
                        "tp": 0,
                        "recall": -1,
                        "f1_score": -1.0,
                        "precision": -1,
                    },
                },
                "v4": {
                    "0.1": {
                        "fn": 1,  # faily to classify img5
                        "fp": 0,
                        "tp": 1,  # correctly classify img6
                        "recall": 0.5,
                        "f1_score": 0.6666666666666666,
                        "precision": 1.0,
                    },
                    "0.2": {
                        "fn": 1,
                        "fp": 0,
                        "tp": 1,
                        "recall": 0.5,
                        "f1_score": 0.6666666666666666,
                        "precision": 1.0,
                    },
                    "0.3": {
                        "fn": 1,
                        "fp": 0,
                        "tp": 1,
                        "recall": 0.5,
                        "f1_score": 0.6666666666666666,
                        "precision": 1.0,
                    },
                    "0.4": {
                        "fn": 1,
                        "fp": 0,
                        "tp": 1,
                        "recall": 0.5,
                        "f1_score": 0.6666666666666666,
                        "precision": 1.0,
                    },
                    "0.5": {
                        "fn": 1,
                        "fp": 0,
                        "tp": 1,
                        "recall": 0.5,
                        "f1_score": 0.6666666666666666,
                        "precision": 1.0,
                    },
                    "0.6": {
                        "fn": 1,
                        "fp": 0,
                        "tp": 1,
                        "recall": 0.5,
                        "f1_score": 0.6666666666666666,
                        "precision": 1.0,
                    },
                    "0.7": {
                        "fn": 1,
                        "fp": 0,
                        "tp": 1,
                        "recall": 0.5,
                        "f1_score": 0.6666666666666666,
                        "precision": 1.0,
                    },
                    "0.8": {
                        "fn": 2,
                        "fp": 0,
                        "tp": 0,
                        "recall": 0.0,
                        "f1_score": -1,
                        "precision": -1,
                    },
                    "0.9": {
                        "fn": 2,
                        "fp": 0,
                        "tp": 0,
                        "recall": 0.0,
                        "f1_score": -1,
                        "precision": -1,
                    },
                    "0.05": {
                        "fn": 1,
                        "fp": 0,
                        "tp": 1,
                        "recall": 0.5,
                        "f1_score": 0.6666666666666666,
                        "precision": 1.0,
                    },
                    "0.15": {
                        "fn": 1,
                        "fp": 0,
                        "tp": 1,
                        "recall": 0.5,
                        "f1_score": 0.6666666666666666,
                        "precision": 1.0,
                    },
                    "0.25": {
                        "fn": 1,
                        "fp": 0,
                        "tp": 1,
                        "recall": 0.5,
                        "f1_score": 0.6666666666666666,
                        "precision": 1.0,
                    },
                    "0.35": {
                        "fn": 1,
                        "fp": 0,
                        "tp": 1,
                        "recall": 0.5,
                        "f1_score": 0.6666666666666666,
                        "precision": 1.0,
                    },
                    "0.45": {
                        "fn": 1,
                        "fp": 0,
                        "tp": 1,
                        "recall": 0.5,
                        "f1_score": 0.6666666666666666,
                        "precision": 1.0,
                    },
                    "0.55": {
                        "fn": 1,
                        "fp": 0,
                        "tp": 1,
                        "recall": 0.5,
                        "f1_score": 0.6666666666666666,
                        "precision": 1.0,
                    },
                    "0.65": {
                        "fn": 1,
                        "fp": 0,
                        "tp": 1,
                        "recall": 0.5,
                        "f1_score": 0.6666666666666666,
                        "precision": 1.0,
                    },
                    "0.75": {
                        "fn": 2,
                        "fp": 0,
                        "tp": 0,
                        "recall": 0.0,
                        "f1_score": -1,
                        "precision": -1,
                    },
                    "0.85": {
                        "fn": 2,
                        "fp": 0,
                        "tp": 0,
                        "recall": 0.0,
                        "f1_score": -1,
                        "precision": -1,
                    },
                    "0.95": {
                        "fn": 2,
                        "fp": 0,
                        "tp": 0,
                        "recall": 0.0,
                        "f1_score": -1,
                        "precision": -1,
                    },
                },
                "v5": {
                    "0.1": {
                        "fn": 0,
                        "fp": 1,
                        "tp": 0,
                        "recall": -1,
                        "f1_score": -1,
                        "precision": 0.0,
                    },
                    "0.2": {
                        "fn": 0,
                        "fp": 1,
                        "tp": 0,
                        "recall": -1,
                        "f1_score": -1,
                        "precision": 0.0,
                    },
                    "0.3": {
                        "fn": 0,
                        "fp": 0,
                        "tp": 0,
                        "recall": -1,
                        "f1_score": -1.0,
                        "precision": -1,
                    },
                    "0.4": {
                        "fn": 0,
                        "fp": 0,
                        "tp": 0,
                        "recall": -1,
                        "f1_score": -1.0,
                        "precision": -1,
                    },
                    "0.5": {
                        "fn": 0,
                        "fp": 0,
                        "tp": 0,
                        "recall": -1,
                        "f1_score": -1.0,
                        "precision": -1,
                    },
                    "0.6": {
                        "fn": 0,
                        "fp": 0,
                        "tp": 0,
                        "recall": -1,
                        "f1_score": -1.0,
                        "precision": -1,
                    },
                    "0.7": {
                        "fn": 0,
                        "fp": 0,
                        "tp": 0,
                        "recall": -1,
                        "f1_score": -1,
                        "precision": -1,
                    },
                    "0.8": {
                        "fn": 0,
                        "fp": 0,
                        "tp": 0,
                        "recall": -1,
                        "f1_score": -1,
                        "precision": -1,
                    },
                    "0.9": {
                        "fn": 0,
                        "fp": 0,
                        "tp": 0,
                        "recall": -1,
                        "f1_score": -1,
                        "precision": -1,
                    },
                    "0.05": {
                        "fn": 0,
                        "fp": 1,
                        "tp": 0,
                        "recall": -1,
                        "f1_score": -1,
                        "precision": 0.0,
                    },
                    "0.15": {
                        "fn": 0,
                        "fp": 1,
                        "tp": 0,
                        "recall": -1,
                        "f1_score": -1,
                        "precision": 0.0,
                    },
                    "0.25": {
                        "fn": 0,
                        "fp": 1,
                        "tp": 0,
                        "recall": -1,
                        "f1_score": -1,
                        "precision": 0.0,
                    },
                    "0.35": {
                        "fn": 0,
                        "fp": 0,
                        "tp": 0,
                        "recall": -1,
                        "f1_score": -1,
                        "precision": -1,
                    },
                    "0.45": {
                        "fn": 0,
                        "fp": 0,
                        "tp": 0,
                        "recall": -1,
                        "f1_score": -1,
                        "precision": -1,
                    },
                    "0.55": {
                        "fn": 0,
                        "fp": 0,
                        "tp": 0,
                        "recall": -1,
                        "f1_score": -1,
                        "precision": -1,
                    },
                    "0.65": {
                        "fn": 0,
                        "fp": 0,
                        "tp": 0,
                        "recall": -1,
                        "f1_score": -1.0,
                        "precision": -1,
                    },
                    "0.75": {
                        "fn": 0,
                        "fp": 0,
                        "tp": 0,
                        "recall": -1,
                        "f1_score": -1,
                        "precision": -1,
                    },
                    "0.85": {
                        "fn": 0,
                        "fp": 0,
                        "tp": 0,
                        "recall": -1,
                        "f1_score": -1,
                        "precision": -1,
                    },
                    "0.95": {
                        "fn": 0,
                        "fp": 0,
                        "tp": 0,
                        "recall": -1,
                        "f1_score": -1.0,
                        "precision": -1,
                    },
                },
                "v8": {
                    "0.1": {
                        "fn": 0,
                        "fp": 1,
                        "tp": 0,
                        "recall": -1,
                        "f1_score": -1,
                        "precision": 0.0,
                    },
                    "0.2": {
                        "fn": 0,
                        "fp": 1,
                        "tp": 0,
                        "recall": -1,
                        "f1_score": -1,
                        "precision": 0.0,
                    },
                    "0.3": {
                        "fn": 0,
                        "fp": 1,
                        "tp": 0,
                        "recall": -1,
                        "f1_score": -1,
                        "precision": 0.0,
                    },
                    "0.4": {
                        "fn": 0,
                        "fp": 1,
                        "tp": 0,
                        "recall": -1,
                        "f1_score": -1,
                        "precision": 0.0,
                    },
                    "0.5": {
                        "fn": 0,
                        "fp": 1,
                        "tp": 0,
                        "recall": -1,
                        "f1_score": -1,
                        "precision": 0.0,
                    },
                    "0.6": {
                        "fn": 0,
                        "fp": 0,
                        "tp": 0,
                        "recall": -1,
                        "f1_score": -1.0,
                        "precision": -1,
                    },
                    "0.7": {
                        "fn": 0,
                        "fp": 0,
                        "tp": 0,
                        "recall": -1,
                        "f1_score": -1.0,
                        "precision": -1,
                    },
                    "0.8": {
                        "fn": 0,
                        "fp": 0,
                        "tp": 0,
                        "recall": -1,
                        "f1_score": -1.0,
                        "precision": -1,
                    },
                    "0.9": {
                        "fn": 0,
                        "fp": 0,
                        "tp": 0,
                        "recall": -1,
                        "f1_score": -1.0,
                        "precision": -1,
                    },
                    "0.05": {
                        "fn": 0,
                        "fp": 1,
                        "tp": 0,
                        "recall": -1,
                        "f1_score": -1,
                        "precision": 0.0,
                    },
                    "0.15": {
                        "fn": 0,
                        "fp": 1,
                        "tp": 0,
                        "recall": -1,
                        "f1_score": -1,
                        "precision": 0.0,
                    },
                    "0.25": {
                        "fn": 0,
                        "fp": 1,
                        "tp": 0,
                        "recall": -1,
                        "f1_score": -1,
                        "precision": 0.0,
                    },
                    "0.35": {
                        "fn": 0,
                        "fp": 1,
                        "tp": 0,
                        "recall": -1,
                        "f1_score": -1,
                        "precision": 0.0,
                    },
                    "0.45": {
                        "fn": 0,
                        "fp": 1,
                        "tp": 0,
                        "recall": -1,
                        "f1_score": -1,
                        "precision": 0.0,
                    },
                    "0.55": {
                        "fn": 0,
                        "fp": 0,
                        "tp": 0,
                        "recall": -1,
                        "f1_score": -1.0,
                        "precision": -1,
                    },
                    "0.65": {
                        "fn": 0,
                        "fp": 0,
                        "tp": 0,
                        "recall": -1,
                        "f1_score": -1.0,
                        "precision": -1,
                    },
                    "0.75": {
                        "fn": 0,
                        "fp": 0,
                        "tp": 0,
                        "recall": -1,
                        "f1_score": -1.0,
                        "precision": -1,
                    },
                    "0.85": {
                        "fn": 0,
                        "fp": 0,
                        "tp": 0,
                        "recall": -1,
                        "f1_score": -1.0,
                        "precision": -1,
                    },
                    "0.95": {
                        "fn": 0,
                        "fp": 0,
                        "tp": 0,
                        "recall": -1,
                        "f1_score": -1.0,
                        "precision": -1,
                    },
                },
            },
        },
        {
            "type": "Precision",
            "value": -1.0,
            "label": {"key": "k4", "value": "v5"},
        },
        {
            "type": "Recall",
            "value": -1.0,
            "label": {"key": "k4", "value": "v5"},
        },
        {"type": "F1", "value": -1.0, "label": {"key": "k4", "value": "v5"}},
        {
            "type": "Precision",
            "value": -1.0,
            "label": {"key": "k4", "value": "v1"},
        },
        {
            "type": "Recall",
            "value": -1.0,
            "label": {"key": "k4", "value": "v1"},
        },
        {"type": "F1", "value": -1.0, "label": {"key": "k4", "value": "v1"}},
        {
            "type": "Precision",
            "value": 1.0,
            "label": {"key": "k4", "value": "v4"},
        },
        {
            "type": "Recall",
            "value": 0.5,
            "label": {"key": "k4", "value": "v4"},
        },
        {
            "type": "F1",
            "value": 0.6666666666666666,
            "label": {"key": "k4", "value": "v4"},
        },
        {
            "type": "Precision",
            "value": 0.0,
            "label": {"key": "k4", "value": "v8"},
        },
        {
            "type": "Recall",
            "value": 0.0,
            "label": {"key": "k4", "value": "v8"},
        },
        {"type": "F1", "value": 0.0, "label": {"key": "k4", "value": "v8"}},
    ]

    cat_expected_cm = [
        {
            "label_key": "special_class",
            "entries": [
                {
                    "prediction": "cat_type1",
                    "groundtruth": "cat_type1",
                    "count": 3,
                }
            ],
        }
        # other label keys not included for testing purposes
    ]
    eval_job = model.evaluate_classification(
        dataset, label_map=label_mapping, compute_pr_curves=True
    )

    assert eval_job.id
    assert eval_job.wait_for_completion(timeout=30) == EvaluationStatus.DONE

    metrics = eval_job.metrics
    for m in metrics:
        assert m in cat_expected_metrics
    for m in cat_expected_metrics:
        assert m in metrics

    confusion_matrix = eval_job.confusion_matrices

    for row in confusion_matrix:
        if row["label_key"] == "special_class":
            for entry in cat_expected_cm[0]["entries"]:
                assert entry in row["entries"]
            for entry in row["entries"]:
                assert entry in cat_expected_cm[0]["entries"]

    # finally, check invalid label_map
    with pytest.raises(TypeError):
        eval_job = model.evaluate_classification(
            dataset,
            label_map=[
                [
                    [
                        Label(key="class", value="tabby cat"),
                        Label(key="class", value="mammals"),
                    ]
                ]
            ],  # type: ignore - purposefully raising error
        )


@pytest.fixture
def gt_clfs_label_key_mismatch(
    img5: Datum,
    img6: Datum,
    img8: Datum,
) -> list[GroundTruth]:
    return [
        GroundTruth(
            datum=img5,
            annotations=[
                Annotation(
                    task_type=TaskType.CLASSIFICATION,
                    labels=[
                        Label(key="k4", value="v4"),
                        Label(key="k5", value="v5"),
                    ],
                ),
            ],
        ),
        GroundTruth(
            datum=img6,
            annotations=[
                Annotation(
                    task_type=TaskType.CLASSIFICATION,
                    labels=[Label(key="k4", value="v4")],
                )
            ],
        ),
        GroundTruth(
            datum=img8,
            annotations=[
                Annotation(
                    task_type=TaskType.CLASSIFICATION,
                    labels=[Label(key="k3", value="v3")],
                )
            ],
        ),
    ]


@pytest.fixture
def pred_clfs_label_key_mismatch(
    model_name: str, img5: Datum, img6: Datum
) -> list[Prediction]:
    return [
        Prediction(
            datum=img5,
            annotations=[
                Annotation(
                    task_type=TaskType.CLASSIFICATION,
                    labels=[
                        Label(key="k12", value="v12", score=0.47),
                        Label(key="k12", value="v16", score=0.53),
                        Label(key="k13", value="v13", score=1.0),
                    ],
                )
            ],
        ),
        Prediction(
            datum=img6,
            annotations=[
                Annotation(
                    task_type=TaskType.CLASSIFICATION,
                    labels=[
                        Label(key="k4", value="v4", score=0.71),
                        Label(key="k4", value="v5", score=0.29),
                    ],
                )
            ],
        ),
    ]


def test_evaluate_classification_mismatched_label_keys(
    client: Client,
    gt_clfs_label_key_mismatch: list[GroundTruth],
    pred_clfs_label_key_mismatch: list[Prediction],
    dataset_name: str,
    model_name: str,
):
    """Check that we get an error when trying to evaluate over ground truths and predictions with different sets of label keys."""

    dataset = Dataset.create(dataset_name)
    for gt in gt_clfs_label_key_mismatch:
        dataset.add_groundtruth(gt)
    dataset.finalize()

    model = Model.create(model_name)
    for pd in pred_clfs_label_key_mismatch:
        model.add_prediction(dataset, pd)

    model.finalize_inferences(dataset)

    result = model.evaluate_classification(dataset).wait_for_completion(
        timeout=30
    )

    assert result.value == "failed"
