import pandas as pd
from valor_core.classification import evaluate_classification
from valor_core import enums, schemas
import random


def test_evaluate_image_clf(
    evaluate_image_clf_groundtruths, evaluate_image_clf_predictions
):

    eval_job = evaluate_classification(
        groundtruths=evaluate_image_clf_groundtruths,
        predictions=evaluate_image_clf_predictions,
    )

    metrics = eval_job.metrics

    expected_metrics = [
        {"type": "Accuracy", "parameters": {"label_key": "k4"}, "value": 0.5},
        {
            "type": "ROCAUC",
            "parameters": {"label_key": "k4"},
            "value": 1.0,
        },
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
        {
            "type": "ROCAUC",
            "parameters": {"label_key": "k5"},
            "value": 1.0,
        },
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
        if m["type"] not in [
            "PrecisionRecallCurve",
            "DetailedPrecisionRecallCurve",
        ]:
            assert m in expected_metrics
    for m in expected_metrics:
        assert m in metrics

    confusion_matrices = eval_job.confusion_matrices
    for m in confusion_matrices:
        assert m in expected_confusion_matrices
    for m in expected_confusion_matrices:
        assert m in confusion_matrices

    # test evaluation metadata
    expected_metadata = {
        "datums": 3,
        "labels": 8,
        "annotations": 6,
    }

    for key, value in expected_metadata.items():
        assert eval_job.meta[key] == value  # type: ignore - issue #605

    # eval should definitely take less than 5 seconds, usually around .4
    assert eval_job.meta["duration"] <= 5  # type: ignore - issue #605

    # check that metrics arg works correctly
    selected_metrics = random.sample(
        [
            enums.MetricType.Accuracy,
            enums.MetricType.ROCAUC,
            enums.MetricType.Precision,
            enums.MetricType.F1,
            enums.MetricType.Recall,
            enums.MetricType.PrecisionRecallCurve,
        ],
        2,
    )

    eval_job = evaluate_classification(
        groundtruths=evaluate_image_clf_groundtruths,
        predictions=evaluate_image_clf_predictions,
        parameters=schemas.EvaluationParameters(
            metrics_to_return=selected_metrics,
        ),
    )

    assert set([metric["type"] for metric in eval_job.metrics]) == set(
        selected_metrics
    )

    # check that passing None to metrics returns the assumed list of default metrics
    default_metrics = [
        enums.MetricType.Precision,
        enums.MetricType.Recall,
        enums.MetricType.F1,
        enums.MetricType.Accuracy,
        enums.MetricType.ROCAUC,
        enums.MetricType.PrecisionRecallCurve,
    ]
    eval_job = evaluate_classification(
        groundtruths=evaluate_image_clf_groundtruths,
        predictions=evaluate_image_clf_predictions,
        parameters=schemas.EvaluationParameters(
            metrics_to_return=None,
        ),
    )
    assert set([metric["type"] for metric in eval_job.metrics]) == set(
        default_metrics
    )


def test_evaluate_tabular_clf():
    groundtruth_df = pd.DataFrame(
        [
            {
                "id": 9040,
                "annotation_id": 11373,
                "label_id": 8031,
                "created_at": 1722267392923,
                "dataset_name": "test_dataset",
                "label_key": "class",
                "label_value": "1",
                "datum_id": 822,
                "datum_uid": "uid0",
            },
            {
                "id": 9041,
                "annotation_id": 11374,
                "label_id": 8031,
                "created_at": 1722267392967,
                "dataset_name": "test_dataset",
                "label_key": "class",
                "label_value": "1",
                "datum_id": 823,
                "datum_uid": "uid1",
            },
            {
                "id": 9042,
                "annotation_id": 11375,
                "label_id": 8033,
                "created_at": 1722267393007,
                "dataset_name": "test_dataset",
                "label_key": "class",
                "label_value": "2",
                "datum_id": 824,
                "datum_uid": "uid2",
            },
            {
                "id": 9043,
                "annotation_id": 11376,
                "label_id": 8034,
                "created_at": 1722267393047,
                "dataset_name": "test_dataset",
                "label_key": "class",
                "label_value": "0",
                "datum_id": 825,
                "datum_uid": "uid3",
            },
            {
                "id": 9044,
                "annotation_id": 11377,
                "label_id": 8034,
                "created_at": 1722267393088,
                "dataset_name": "test_dataset",
                "label_key": "class",
                "label_value": "0",
                "datum_id": 826,
                "datum_uid": "uid4",
            },
            {
                "id": 9045,
                "annotation_id": 11378,
                "label_id": 8034,
                "created_at": 1722267393125,
                "dataset_name": "test_dataset",
                "label_key": "class",
                "label_value": "0",
                "datum_id": 827,
                "datum_uid": "uid5",
            },
            {
                "id": 9046,
                "annotation_id": 11379,
                "label_id": 8031,
                "created_at": 1722267393166,
                "dataset_name": "test_dataset",
                "label_key": "class",
                "label_value": "1",
                "datum_id": 828,
                "datum_uid": "uid6",
            },
            {
                "id": 9047,
                "annotation_id": 11380,
                "label_id": 8031,
                "created_at": 1722267393215,
                "dataset_name": "test_dataset",
                "label_key": "class",
                "label_value": "1",
                "datum_id": 829,
                "datum_uid": "uid7",
            },
            {
                "id": 9048,
                "annotation_id": 11381,
                "label_id": 8031,
                "created_at": 1722267393263,
                "dataset_name": "test_dataset",
                "label_key": "class",
                "label_value": "1",
                "datum_id": 830,
                "datum_uid": "uid8",
            },
            {
                "id": 9049,
                "annotation_id": 11382,
                "label_id": 8031,
                "created_at": 1722267393306,
                "dataset_name": "test_dataset",
                "label_key": "class",
                "label_value": "1",
                "datum_id": 831,
                "datum_uid": "uid9",
            },
        ]
    )
    prediction_df = pd.DataFrame(
        [
            {
                "id": 4600,
                "annotation_id": 11385,
                "label_id": 8033,
                "score": 0.09,
                "created_at": pd.Timestamp("2024-07-29 15:36:33.502504"),
                "dataset_name": "test_dataset",
                "label_key": "class",
                "label_value": "2",
                "datum_id": 824,
                "datum_uid": "uid2",
            },
            {
                "id": 4599,
                "annotation_id": 11385,
                "label_id": 8031,
                "score": 0.88,
                "created_at": pd.Timestamp("2024-07-29 15:36:33.502504"),
                "dataset_name": "test_dataset",
                "label_key": "class",
                "label_value": "1",
                "datum_id": 824,
                "datum_uid": "uid2",
            },
            {
                "id": 4598,
                "annotation_id": 11385,
                "label_id": 8034,
                "score": 0.03,
                "created_at": pd.Timestamp("2024-07-29 15:36:33.502504"),
                "dataset_name": "test_dataset",
                "label_key": "class",
                "label_value": "0",
                "datum_id": 824,
                "datum_uid": "uid2",
            },
            {
                "id": 4603,
                "annotation_id": 11386,
                "label_id": 8033,
                "score": 0.0,
                "created_at": pd.Timestamp("2024-07-29 15:36:33.546293"),
                "dataset_name": "test_dataset",
                "label_key": "class",
                "label_value": "2",
                "datum_id": 825,
                "datum_uid": "uid3",
            },
            {
                "id": 4602,
                "annotation_id": 11386,
                "label_id": 8031,
                "score": 0.03,
                "created_at": pd.Timestamp("2024-07-29 15:36:33.546293"),
                "dataset_name": "test_dataset",
                "label_key": "class",
                "label_value": "1",
                "datum_id": 825,
                "datum_uid": "uid3",
            },
            {
                "id": 4601,
                "annotation_id": 11386,
                "label_id": 8034,
                "score": 0.97,
                "created_at": pd.Timestamp("2024-07-29 15:36:33.546293"),
                "dataset_name": "test_dataset",
                "label_key": "class",
                "label_value": "0",
                "datum_id": 825,
                "datum_uid": "uid3",
            },
            {
                "id": 4606,
                "annotation_id": 11387,
                "label_id": 8033,
                "score": 0.0,
                "created_at": pd.Timestamp("2024-07-29 15:36:33.586264"),
                "dataset_name": "test_dataset",
                "label_key": "class",
                "label_value": "2",
                "datum_id": 826,
                "datum_uid": "uid4",
            },
            {
                "id": 4605,
                "annotation_id": 11387,
                "label_id": 8031,
                "score": 0.0,
                "created_at": pd.Timestamp("2024-07-29 15:36:33.586264"),
                "dataset_name": "test_dataset",
                "label_key": "class",
                "label_value": "1",
                "datum_id": 826,
                "datum_uid": "uid4",
            },
            {
                "id": 4604,
                "annotation_id": 11387,
                "label_id": 8034,
                "score": 1.0,
                "created_at": pd.Timestamp("2024-07-29 15:36:33.586264"),
                "dataset_name": "test_dataset",
                "label_key": "class",
                "label_value": "0",
                "datum_id": 826,
                "datum_uid": "uid4",
            },
            {
                "id": 4609,
                "annotation_id": 11388,
                "label_id": 8033,
                "score": 0.0,
                "created_at": pd.Timestamp("2024-07-29 15:36:33.631094"),
                "dataset_name": "test_dataset",
                "label_key": "class",
                "label_value": "2",
                "datum_id": 827,
                "datum_uid": "uid5",
            },
            {
                "id": 4608,
                "annotation_id": 11388,
                "label_id": 8031,
                "score": 0.0,
                "created_at": pd.Timestamp("2024-07-29 15:36:33.631094"),
                "dataset_name": "test_dataset",
                "label_key": "class",
                "label_value": "1",
                "datum_id": 827,
                "datum_uid": "uid5",
            },
            {
                "id": 4607,
                "annotation_id": 11388,
                "label_id": 8034,
                "score": 1.0,
                "created_at": pd.Timestamp("2024-07-29 15:36:33.631094"),
                "dataset_name": "test_dataset",
                "label_key": "class",
                "label_value": "0",
                "datum_id": 827,
                "datum_uid": "uid5",
            },
            {
                "id": 4612,
                "annotation_id": 11389,
                "label_id": 8033,
                "score": 0.03,
                "created_at": pd.Timestamp("2024-07-29 15:36:33.673800"),
                "dataset_name": "test_dataset",
                "label_key": "class",
                "label_value": "2",
                "datum_id": 828,
                "datum_uid": "uid6",
            },
            {
                "id": 4611,
                "annotation_id": 11389,
                "label_id": 8031,
                "score": 0.96,
                "created_at": pd.Timestamp("2024-07-29 15:36:33.673800"),
                "dataset_name": "test_dataset",
                "label_key": "class",
                "label_value": "1",
                "datum_id": 828,
                "datum_uid": "uid6",
            },
            {
                "id": 4610,
                "annotation_id": 11389,
                "label_id": 8034,
                "score": 0.01,
                "created_at": pd.Timestamp("2024-07-29 15:36:33.673800"),
                "dataset_name": "test_dataset",
                "label_key": "class",
                "label_value": "0",
                "datum_id": 828,
                "datum_uid": "uid6",
            },
            {
                "id": 4615,
                "annotation_id": 11390,
                "label_id": 8033,
                "score": 0.7,
                "created_at": pd.Timestamp("2024-07-29 15:36:33.709818"),
                "dataset_name": "test_dataset",
                "label_key": "class",
                "label_value": "2",
                "datum_id": 829,
                "datum_uid": "uid7",
            },
            {
                "id": 4614,
                "annotation_id": 11390,
                "label_id": 8031,
                "score": 0.02,
                "created_at": pd.Timestamp("2024-07-29 15:36:33.709818"),
                "dataset_name": "test_dataset",
                "label_key": "class",
                "label_value": "1",
                "datum_id": 829,
                "datum_uid": "uid7",
            },
            {
                "id": 4613,
                "annotation_id": 11390,
                "label_id": 8034,
                "score": 0.28,
                "created_at": pd.Timestamp("2024-07-29 15:36:33.709818"),
                "dataset_name": "test_dataset",
                "label_key": "class",
                "label_value": "0",
                "datum_id": 829,
                "datum_uid": "uid7",
            },
            {
                "id": 4618,
                "annotation_id": 11391,
                "label_id": 8033,
                "score": 0.01,
                "created_at": pd.Timestamp("2024-07-29 15:36:33.745536"),
                "dataset_name": "test_dataset",
                "label_key": "class",
                "label_value": "2",
                "datum_id": 830,
                "datum_uid": "uid8",
            },
            {
                "id": 4617,
                "annotation_id": 11391,
                "label_id": 8031,
                "score": 0.21,
                "created_at": pd.Timestamp("2024-07-29 15:36:33.745536"),
                "dataset_name": "test_dataset",
                "label_key": "class",
                "label_value": "1",
                "datum_id": 830,
                "datum_uid": "uid8",
            },
            {
                "id": 4616,
                "annotation_id": 11391,
                "label_id": 8034,
                "score": 0.78,
                "created_at": pd.Timestamp("2024-07-29 15:36:33.745536"),
                "dataset_name": "test_dataset",
                "label_key": "class",
                "label_value": "0",
                "datum_id": 830,
                "datum_uid": "uid8",
            },
            {
                "id": 4621,
                "annotation_id": 11392,
                "label_id": 8033,
                "score": 0.44,
                "created_at": pd.Timestamp("2024-07-29 15:36:33.797759"),
                "dataset_name": "test_dataset",
                "label_key": "class",
                "label_value": "2",
                "datum_id": 831,
                "datum_uid": "uid9",
            },
            {
                "id": 4620,
                "annotation_id": 11392,
                "label_id": 8031,
                "score": 0.11,
                "created_at": pd.Timestamp("2024-07-29 15:36:33.797759"),
                "dataset_name": "test_dataset",
                "label_key": "class",
                "label_value": "1",
                "datum_id": 831,
                "datum_uid": "uid9",
            },
            {
                "id": 4619,
                "annotation_id": 11392,
                "label_id": 8034,
                "score": 0.45,
                "created_at": pd.Timestamp("2024-07-29 15:36:33.797759"),
                "dataset_name": "test_dataset",
                "label_key": "class",
                "label_value": "0",
                "datum_id": 831,
                "datum_uid": "uid9",
            },
            {
                "id": 4594,
                "annotation_id": 11383,
                "label_id": 8033,
                "score": 0.28,
                "created_at": pd.Timestamp("2024-07-29 15:36:33.411278"),
                "dataset_name": "test_dataset",
                "label_key": "class",
                "label_value": "2",
                "datum_id": 822,
                "datum_uid": "uid0",
            },
            {
                "id": 4593,
                "annotation_id": 11383,
                "label_id": 8031,
                "score": 0.35,
                "created_at": pd.Timestamp("2024-07-29 15:36:33.411278"),
                "dataset_name": "test_dataset",
                "label_key": "class",
                "label_value": "1",
                "datum_id": 822,
                "datum_uid": "uid0",
            },
            {
                "id": 4592,
                "annotation_id": 11383,
                "label_id": 8034,
                "score": 0.37,
                "created_at": pd.Timestamp("2024-07-29 15:36:33.411278"),
                "dataset_name": "test_dataset",
                "label_key": "class",
                "label_value": "0",
                "datum_id": 822,
                "datum_uid": "uid0",
            },
            {
                "id": 4597,
                "annotation_id": 11384,
                "label_id": 8033,
                "score": 0.15,
                "created_at": pd.Timestamp("2024-07-29 15:36:33.465625"),
                "dataset_name": "test_dataset",
                "label_key": "class",
                "label_value": "2",
                "datum_id": 823,
                "datum_uid": "uid1",
            },
            {
                "id": 4596,
                "annotation_id": 11384,
                "label_id": 8031,
                "score": 0.61,
                "created_at": pd.Timestamp("2024-07-29 15:36:33.465625"),
                "dataset_name": "test_dataset",
                "label_key": "class",
                "label_value": "1",
                "datum_id": 823,
                "datum_uid": "uid1",
            },
            {
                "id": 4595,
                "annotation_id": 11384,
                "label_id": 8034,
                "score": 0.24,
                "created_at": pd.Timestamp("2024-07-29 15:36:33.465625"),
                "dataset_name": "test_dataset",
                "label_key": "class",
                "label_value": "0",
                "datum_id": 823,
                "datum_uid": "uid1",
            },
        ]
    )

    eval_job = evaluate_classification(
        groundtruths=groundtruth_df,
        predictions=prediction_df,
    )

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
        if m["type"] not in [
            "PrecisionRecallCurve",
            "DetailedPrecisionRecallCurve",
        ]:
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
