""" These integration tests should be run with a backend at http://localhost:8000
that is no auth
"""
from dataclasses import asdict

import pandas
import pytest
from sqlalchemy.orm import Session

from velour import (
    Annotation,
    Dataset,
    Datum,
    GroundTruth,
    Label,
    Model,
    Prediction,
)
from velour.client import Client, ClientException
from velour.enums import JobStatus, TaskType


def test_evaluate_image_clf(
    client: Client,
    gt_clfs: list[GroundTruth],
    pred_clfs: list[Prediction],
    dataset_name: str,
    model_name: str,
):
    dataset = Dataset.create(client, dataset_name)
    for gt in gt_clfs:
        dataset.add_groundtruth(gt)
    dataset.finalize()

    model = Model.create(client, model_name)
    for pd in pred_clfs:
        model.add_prediction(pd)
    model.finalize_inferences(dataset)

    eval_job = model.evaluate_classification(dataset=dataset, timeout=30)

    assert eval_job.id
    assert eval_job.task_type == "classification"
    assert eval_job.status.value == "done"
    assert set(eval_job.ignored_pred_keys) == {"k12", "k13"}
    assert set(eval_job.missing_pred_keys) == {"k3", "k5"}

    assert eval_job.status == JobStatus.DONE

    metrics = eval_job.results.metrics

    expected_metrics = [
        {"type": "Accuracy", "parameters": {"label_key": "k4"}, "value": 1.0},
        {"type": "ROCAUC", "parameters": {"label_key": "k4"}, "value": 1.0},
        {
            "type": "Precision",
            "value": 1.0,
            "label": {"key": "k4", "value": "v4"},
        },
        {
            "type": "Recall",
            "value": 1.0,
            "label": {"key": "k4", "value": "v4"},
        },
        {"type": "F1", "value": 1.0, "label": {"key": "k4", "value": "v4"}},
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
    ]
    for m in metrics:
        assert m in expected_metrics
    for m in expected_metrics:
        assert m in metrics

    confusion_matrices = eval_job.results.confusion_matrices
    assert confusion_matrices == [
        {
            "label_key": "k4",
            "entries": [{"prediction": "v4", "groundtruth": "v4", "count": 1}],
        }
    ]


def test_evaluate_tabular_clf(
    client: Session,
    dataset_name: str,
    model_name: str,
    gt_clfs_tabular: list[int],
    pred_clfs_tabular: list[list[float]],
):
    assert len(gt_clfs_tabular) == len(pred_clfs_tabular)

    dataset = Dataset.create(client, name=dataset_name)
    gts = [
        GroundTruth(
            datum=Datum(dataset=dataset_name, uid=f"uid{i}"),
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

    # test
    model = Model.create(client, name=model_name)
    with pytest.raises(ClientException) as exc_info:
        model.evaluate_classification(dataset=dataset, timeout=30)
    assert "has not been finalized" in str(exc_info)

    dataset.finalize()

    pds = [
        Prediction(
            model=model_name,
            datum=Datum(dataset=dataset_name, uid=f"uid{i}"),
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
    for pd in pds:
        model.add_prediction(pd)

    # test
    with pytest.raises(ClientException) as exc_info:
        model.evaluate_classification(dataset=dataset, timeout=30)
    assert "has not been finalized" in str(exc_info)

    model.finalize_inferences(dataset)

    # evaluate
    eval_job = model.evaluate_classification(dataset=dataset, timeout=30)
    assert eval_job.ignored_pred_keys == []
    assert eval_job.missing_pred_keys == []

    assert eval_job.status == JobStatus.DONE

    metrics = eval_job.results.metrics

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

    confusion_matrices = eval_job.results.confusion_matrices

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

    # validate that we can fetch the confusion matrices through get_bulk_evaluations()
    bulk_evals = client.get_bulk_evaluations(datasets=dataset_name)

    assert len(bulk_evals) == 1
    assert all(
        [
            name
            in [
                "dataset",
                "model",
                "settings",
                "job_id",
                "status",
                "metrics",
                "confusion_matrices",
            ]
            for evaluation in bulk_evals
            for name in evaluation.keys()
        ]
    )

    for metric in bulk_evals[0]["metrics"]:
        assert metric in expected_metrics

    assert len(bulk_evals[0]["confusion_matrices"][0]) == len(
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
    labels = model.get_labels()
    df = model.get_metric_dataframes()

    assert isinstance(model.id, int)
    assert model.name == model_name
    assert len(model.metadata) == 0

    assert len(labels) == 3
    assert isinstance(df[0]["df"], pandas.DataFrame)

    # check evaluation
    eval_jobs = model.get_evaluations()
    assert len(eval_jobs) == 1
    eval_settings = asdict(eval_jobs[0].settings)
    eval_settings.pop("id")
    assert eval_settings == {
        "model": model_name,
        "dataset": "test_dataset",
        "task_type": "classification",
        "settings": {},
    }

    metrics_from_eval_settings_id = eval_jobs[0].results.metrics
    assert len(metrics_from_eval_settings_id) == len(expected_metrics)
    for m in metrics_from_eval_settings_id:
        assert m in expected_metrics
    for m in expected_metrics:
        assert m in metrics_from_eval_settings_id

    # check confusion matrix
    confusion_matrices = eval_jobs[0].results.confusion_matrices

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

    client.delete_model(model.name, timeout=30)

    assert len(client.get_models()) == 0


def test_stratify_clf_metrics(
    client: Session,
    gt_clfs_tabular: list[int],
    pred_clfs_tabular: list[list[float]],
    dataset_name: str,
    model_name: str,
):
    assert len(gt_clfs_tabular) == len(pred_clfs_tabular)

    # create data and two-different defining groups of cohorts
    dataset = Dataset.create(client, name=dataset_name)
    for i, label_value in enumerate(gt_clfs_tabular):
        gt = GroundTruth(
            datum=Datum(
                uid=f"uid{i}",
                dataset=dataset_name,
                metadata={
                    "md1": f"md1-val{i % 3}",
                    "md2": f"md2-val{i % 4}",
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

    model = Model.create(client, name=model_name)
    for i, pred in enumerate(pred_clfs_tabular):
        pd = Prediction(
            model=model_name,
            datum=Datum(
                uid=f"uid{i}",
                dataset=dataset_name,
                metadata={
                    "md1": f"md1-val{i % 3}",
                    "md2": f"md2-val{i % 4}",
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
        model.add_prediction(pd)
    model.finalize_inferences(dataset)

    eval_job_val2 = model.evaluate_classification(
        dataset=dataset,
        filters=[
            Datum.metadata["md1"] == "md1-val2",
        ],
        timeout=30,
    )
    val2_metrics = eval_job_val2.results.metrics

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
