import random

import pytest
from valor_core import enums, schemas
from valor_core.classification import (
    _calculate_rocauc,
    evaluate_classification,
)


def test_evaluate_image_clf(
    evaluate_image_clf_groundtruths, evaluate_image_clf_predictions
):

    eval_job = evaluate_classification(
        groundtruths=evaluate_image_clf_groundtruths,
        predictions=evaluate_image_clf_predictions,
    )

    eval_job_metrics = eval_job.metrics

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

    for m in eval_job_metrics:
        if m["type"] not in [
            "PrecisionRecallCurve",
            "DetailedPrecisionRecallCurve",
        ]:
            assert m in expected_metrics
    for m in expected_metrics:
        assert m in eval_job_metrics

    confusion_matrices = eval_job.confusion_matrices
    assert confusion_matrices
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
        metrics_to_return=selected_metrics,
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
    ]
    eval_job = evaluate_classification(
        groundtruths=evaluate_image_clf_groundtruths,
        predictions=evaluate_image_clf_predictions,
        metrics_to_return=None,
    )
    assert set([metric["type"] for metric in eval_job.metrics]) == set(
        default_metrics
    )


def test_evaluate_tabular_clf(
    evaluate_tabular_clf_groundtruths, evaluate_tabular_clf_predictions
):
    eval_job = evaluate_classification(
        groundtruths=evaluate_tabular_clf_groundtruths,
        predictions=evaluate_tabular_clf_predictions,
    )

    eval_job_metrics = eval_job.metrics

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
    for m in eval_job_metrics:
        if m["type"] not in [
            "PrecisionRecallCurve",
            "DetailedPrecisionRecallCurve",
        ]:
            assert m in expected_metrics
    for m in expected_metrics:
        assert m in eval_job_metrics

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
    assert confusion_matrices
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


def test_stratify_clf_metrics(
    gt_clfs_tabular: list[int],
    pred_clfs_tabular: list[list[float]],
):
    assert len(gt_clfs_tabular) == len(pred_clfs_tabular)

    groundtruths = []
    predictions = []
    for i, label_value in enumerate(gt_clfs_tabular):
        if (
            i % 3 == 2
        ):  # core doesn't use filters, so this emulates the results of the original test
            groundtruths.append(
                schemas.GroundTruth(
                    datum=schemas.Datum(
                        uid=f"uid{i}",
                        metadata={
                            "md1": f"md1-val{i % 3}",
                            "md2": f"md2-val{i % 4}",
                            "md3": i % 3 == 2,
                        },
                    ),
                    annotations=[
                        schemas.Annotation(
                            labels=[
                                schemas.Label(
                                    key="class", value=str(label_value)
                                )
                            ],
                        )
                    ],
                )
            )

    for i, pred in enumerate(pred_clfs_tabular):
        if (
            i % 3 == 2
        ):  # core doesn't use filters, so this emulates the results of the original test
            predictions.append(
                schemas.Prediction(
                    datum=schemas.Datum(
                        uid=f"uid{i}",
                        metadata={
                            "md1": f"md1-val{i % 3}",
                            "md2": f"md2-val{i % 4}",
                            "md3": i % 3 == 2,
                        },
                    ),
                    annotations=[
                        schemas.Annotation(
                            labels=[
                                schemas.Label(
                                    key="class",
                                    value=str(pidx),
                                    score=pred[pidx],
                                )
                                for pidx in range(len(pred))
                            ],
                        )
                    ],
                )
            )

    result = evaluate_classification(
        groundtruths=groundtruths,
        predictions=predictions,
    )

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

    for m in result.metrics:
        if m["type"] not in [
            "PrecisionRecallCurve",
            "DetailedPrecisionRecallCurve",
        ]:
            assert m in expected_metrics
    for m in expected_metrics:
        assert m in result.metrics


def test_evaluate_classification_with_label_maps(
    gt_clfs_with_label_maps: list[schemas.GroundTruth],
    pred_clfs_with_label_maps: list[schemas.Prediction],
):
    # check baseline case, where we have mismatched ground truth and prediction label keys
    with pytest.raises(ValueError) as e:
        evaluate_classification(
            groundtruths=gt_clfs_with_label_maps,
            predictions=pred_clfs_with_label_maps,
        )
    assert "label keys must match" in str(e)

    # now try using a label map to connect all the cats
    label_mapping = {
        # map the ground truths
        schemas.Label(key="class", value="tabby cat"): schemas.Label(
            key="special_class", value="cat_type1"
        ),
        schemas.Label(key="class", value="siamese cat"): schemas.Label(
            key="special_class", value="cat_type1"
        ),
        schemas.Label(key="class", value="british shorthair"): schemas.Label(
            key="special_class", value="cat_type1"
        ),
        # map the predictions
        schemas.Label(key="class", value="cat"): schemas.Label(
            key="special_class", value="cat_type1"
        ),
        schemas.Label(key="class_name", value="cat"): schemas.Label(
            key="special_class", value="cat_type1"
        ),
    }

    cat_expected_metrics = [
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
        {"type": "Accuracy", "parameters": {"label_key": "k5"}, "value": 0.0},
        {"type": "ROCAUC", "parameters": {"label_key": "k5"}, "value": 1.0},
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
            "value": 1.0,
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

    eval_job = evaluate_classification(
        groundtruths=gt_clfs_with_label_maps,
        predictions=pred_clfs_with_label_maps,
        label_map=label_mapping,
        pr_curve_max_examples=3,
        metrics_to_return=[
            enums.MetricType.Precision,
            enums.MetricType.Recall,
            enums.MetricType.F1,
            enums.MetricType.Accuracy,
            enums.MetricType.ROCAUC,
            enums.MetricType.PrecisionRecallCurve,
            enums.MetricType.DetailedPrecisionRecallCurve,
        ],
    )

    pr_expected_values = {
        # k3
        (0, "k3", "v1", "0.1", "fp"): 1,
        (0, "k3", "v1", "0.1", "tn"): 2,
        (0, "k3", "v3", "0.1", "fn"): 1,
        (0, "k3", "v3", "0.1", "tn"): 2,
        (0, "k3", "v3", "0.1", "accuracy"): 2 / 3,
        (0, "k3", "v3", "0.1", "precision"): -1,
        (0, "k3", "v3", "0.1", "recall"): 0,
        (0, "k3", "v3", "0.1", "f1_score"): -1,
        # k4
        (1, "k4", "v1", "0.1", "fp"): 1,
        (1, "k4", "v1", "0.1", "tn"): 2,
        (1, "k4", "v4", "0.1", "fn"): 1,
        (1, "k4", "v4", "0.1", "tn"): 1,
        (1, "k4", "v4", "0.1", "tp"): 1,
        (1, "k4", "v4", "0.9", "tp"): 0,
        (1, "k4", "v4", "0.9", "tn"): 1,
        (1, "k4", "v4", "0.9", "fn"): 2,
        (1, "k4", "v5", "0.1", "fp"): 1,
        (1, "k4", "v5", "0.1", "tn"): 2,
        (1, "k4", "v5", "0.3", "fp"): 0,
        (1, "k4", "v5", "0.3", "tn"): 3,
        (1, "k4", "v8", "0.1", "tn"): 2,
        (1, "k4", "v8", "0.6", "fp"): 0,
        (1, "k4", "v8", "0.6", "tn"): 3,
        # k5
        (2, "k5", "v1", "0.1", "fp"): 1,
        (2, "k5", "v1", "0.1", "tn"): 2,
        (2, "k5", "v5", "0.1", "fn"): 1,
        (
            2,
            "k5",
            "v5",
            "0.1",
            "tn",
        ): 2,
        (2, "k5", "v1", "0.1", "accuracy"): 2 / 3,
        (2, "k5", "v1", "0.1", "precision"): 0,
        (2, "k5", "v1", "0.1", "recall"): -1,
        (2, "k5", "v1", "0.1", "f1_score"): -1,
        # special_class
        (3, "special_class", "cat_type1", "0.1", "tp"): 3,
        (3, "special_class", "cat_type1", "0.1", "tn"): 0,
        (3, "special_class", "cat_type1", "0.95", "tp"): 3,
    }

    pr_metrics = []
    detailed_pr_metrics = []
    for m in eval_job.metrics:
        if m["type"] == "PrecisionRecallCurve":
            pr_metrics.append(m)
        elif m["type"] == "DetailedPrecisionRecallCurve":
            detailed_pr_metrics.append(m)
        else:
            assert m in cat_expected_metrics

    for m in cat_expected_metrics:
        assert m in eval_job.metrics

    pr_metrics.sort(key=lambda x: x["parameters"]["label_key"])
    detailed_pr_metrics.sort(key=lambda x: x["parameters"]["label_key"])

    for (
        index,
        key,
        value,
        threshold,
        metric,
    ), expected_value in pr_expected_values.items():
        assert (
            pr_metrics[index]["value"][value][float(threshold)][metric]
            == expected_value
        )

    # check DetailedPrecisionRecallCurve
    detailed_pr_expected_answers = {
        # k3
        (0, "v1", "0.1", "tp"): {"all": 0, "total": 0},
        (0, "v1", "0.1", "fp"): {
            "misclassifications": 1,
            "total": 1,
        },
        (0, "v1", "0.1", "tn"): {"all": 2, "total": 2},
        (0, "v1", "0.1", "fn"): {
            "no_predictions": 0,
            "misclassifications": 0,
            "total": 0,
        },
        # k4
        (1, "v1", "0.1", "tp"): {"all": 0, "total": 0},
        (1, "v1", "0.1", "fp"): {
            "misclassifications": 1,
            "total": 1,
        },
        (1, "v1", "0.1", "tn"): {"all": 2, "total": 2},
        (1, "v1", "0.1", "fn"): {
            "no_predictions": 0,
            "misclassifications": 0,
            "total": 0,
        },
        (1, "v4", "0.1", "fn"): {
            "no_predictions": 0,
            "misclassifications": 1,
            "total": 1,
        },
        (1, "v8", "0.1", "tn"): {"all": 2, "total": 2},
    }

    for (
        index,
        value,
        threshold,
        metric,
    ), expected_output in detailed_pr_expected_answers.items():
        model_output = detailed_pr_metrics[index]["value"][value][
            float(threshold)
        ][metric]
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

    # check metadata
    assert eval_job and eval_job.meta
    assert eval_job.meta["datums"] == 3
    assert eval_job.meta["labels"] == 13
    assert eval_job.meta["annotations"] == 6
    assert eval_job.meta["duration"] <= 10  # usually 2

    # check confusion matrix
    confusion_matrix = eval_job.confusion_matrices

    assert confusion_matrix
    for row in confusion_matrix:
        if row["label_key"] == "special_class":
            for entry in cat_expected_cm[0]["entries"]:
                assert entry in row["entries"]
            for entry in row["entries"]:
                assert entry in cat_expected_cm[0]["entries"]

    # finally, check invalid label_map
    with pytest.raises(TypeError):
        _ = evaluate_classification(
            groundtruths=gt_clfs_with_label_maps,
            predictions=pred_clfs_with_label_maps,
            label_map=[
                [
                    [
                        schemas.Label(key="class", value="tabby cat"),
                        schemas.Label(key="class", value="mammals"),
                    ]
                ]
            ],  # type: ignore - purposefully raising error,
            pr_curve_max_examples=3,
            metrics_to_return=[
                enums.MetricType.Precision,
                enums.MetricType.Recall,
                enums.MetricType.F1,
                enums.MetricType.Accuracy,
                enums.MetricType.ROCAUC,
                enums.MetricType.PrecisionRecallCurve,
                enums.MetricType.DetailedPrecisionRecallCurve,
            ],
        )


def test_evaluate_classification_mismatched_label_keys(
    gt_clfs_label_key_mismatch: list[schemas.GroundTruth],
    pred_clfs_label_key_mismatch: list[schemas.Prediction],
):
    """Check that we get an error when trying to evaluate over ground truths and predictions with different sets of label keys."""

    with pytest.raises(ValueError) as e:
        evaluate_classification(
            groundtruths=gt_clfs_label_key_mismatch,
            predictions=pred_clfs_label_key_mismatch,
        )
    assert "label keys must match" in str(e)


def test_evaluate_classification_model_with_no_predictions(
    gt_clfs: list[schemas.GroundTruth],
):

    expected_metrics = [
        {"type": "Accuracy", "parameters": {"label_key": "k5"}, "value": 0.0},
        {"type": "ROCAUC", "parameters": {"label_key": "k5"}, "value": 0.0},
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
        {"type": "Accuracy", "parameters": {"label_key": "k4"}, "value": 0.0},
        {"type": "ROCAUC", "parameters": {"label_key": "k4"}, "value": 0.0},
        {
            "type": "Precision",
            "value": 0.0,
            "label": {"key": "k4", "value": "v4"},
        },
        {
            "type": "Recall",
            "value": 0.0,
            "label": {"key": "k4", "value": "v4"},
        },
        {"type": "F1", "value": 0.0, "label": {"key": "k4", "value": "v4"}},
        {"type": "Accuracy", "parameters": {"label_key": "k3"}, "value": 0.0},
        {"type": "ROCAUC", "parameters": {"label_key": "k3"}, "value": 0.0},
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

    evaluation = evaluate_classification(
        groundtruths=gt_clfs,
        predictions=[
            schemas.Prediction(datum=gt_clfs[0].datum, annotations=[])
        ],
    )

    computed_metrics = evaluation.metrics

    assert all([metric["value"] == 0 for metric in computed_metrics])
    assert all([metric in computed_metrics for metric in expected_metrics])
    assert all([metric in expected_metrics for metric in computed_metrics])


def test_compute_confusion_matrix_at_grouper_key_using_label_map(
    classification_functional_test_data,
):
    """
    Test grouping using the label_map
    """

    groundtruths, predictions = classification_functional_test_data

    eval_job = evaluate_classification(
        groundtruths=groundtruths,
        predictions=predictions,
        label_map={
            schemas.Label(key="animal", value="dog"): schemas.Label(
                key="class", value="mammal"
            ),
            schemas.Label(key="animal", value="cat"): schemas.Label(
                key="class", value="mammal"
            ),
            schemas.Label(key="animal", value="bird"): schemas.Label(
                key="class", value="avian"
            ),
        },
    )

    cm = eval_job.confusion_matrices

    expected_entries = [
        {
            "label_key": "class",
            "entries": [
                {"prediction": "avian", "groundtruth": "avian", "count": 1},
                {"prediction": "mammal", "groundtruth": "avian", "count": 2},
                {"prediction": "mammal", "groundtruth": "mammal", "count": 3},
            ],
        },
        {
            "label_key": "color",
            "entries": [
                {"prediction": "blue", "groundtruth": "white", "count": 1},
                {"prediction": "red", "groundtruth": "black", "count": 1},
                {"prediction": "red", "groundtruth": "red", "count": 2},
                {"prediction": "white", "groundtruth": "blue", "count": 1},
                {"prediction": "white", "groundtruth": "white", "count": 1},
            ],
        },
    ]

    assert cm
    assert len(cm) == len(expected_entries)
    for entry in cm:
        assert entry in expected_entries
    for entry in expected_entries:
        assert entry in cm


def test_rocauc_with_label_map(
    classification_functional_test_prediction_df,
    classification_functional_test_groundtruth_df,
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

    Note that the label map is already built into the pandas dataframes used in this test.

    """

    computed_metrics = [
        m.to_dict()
        for m in _calculate_rocauc(
            prediction_df=classification_functional_test_prediction_df,
            groundtruth_df=classification_functional_test_groundtruth_df,
        )
    ]

    expected_metrics = [
        {
            "parameters": {"label_key": "class"},
            "value": 0.5972222222222222,
            "type": "ROCAUC",
        },
        {
            "parameters": {"label_key": "color"},
            "value": 0.43125,
            "type": "ROCAUC",
        },
        {
            "parameters": {"label_key": "other_class"},
            "value": 0.7777777777777779,
            "type": "ROCAUC",
        },
    ]

    for entry in computed_metrics:
        assert entry in expected_metrics
    for entry in expected_metrics:
        assert entry in computed_metrics


def test_compute_classification(
    classification_functional_test_data,
):
    """
    Tests the _compute_classification function.
    """
    groundtruths, predictions = classification_functional_test_data

    eval_job = evaluate_classification(
        groundtruths=groundtruths,
        predictions=predictions,
        metrics_to_return=[
            enums.MetricType.Precision,
            enums.MetricType.Recall,
            enums.MetricType.F1,
            enums.MetricType.Accuracy,
            enums.MetricType.ROCAUC,
            enums.MetricType.PrecisionRecallCurve,
            enums.MetricType.DetailedPrecisionRecallCurve,
        ],
    )

    computed_metrics = [
        m
        for m in eval_job.metrics
        if m["type"]
        not in ["PrecisionRecallCurve", "DetailedPrecisionRecallCurve"]
    ]
    pr_curves = [
        m for m in eval_job.metrics if m["type"] == "PrecisionRecallCurve"
    ]
    detailed_pr_curves = [
        m
        for m in eval_job.metrics
        if m["type"] == "DetailedPrecisionRecallCurve"
    ]
    confusion_matrices = eval_job.confusion_matrices

    expected_metrics = [
        {
            "label": {"key": "animal", "value": "bird"},
            "value": 1.0,
            "type": "Precision",
        },
        {
            "label": {"key": "animal", "value": "bird"},
            "value": 0.3333333333333333,
            "type": "Recall",
        },
        {
            "label": {"key": "animal", "value": "bird"},
            "value": 0.5,
            "type": "F1",
        },
        {
            "label": {"key": "animal", "value": "cat"},
            "value": 0.25,
            "type": "Precision",
        },
        {
            "label": {"key": "animal", "value": "cat"},
            "value": 1.0,
            "type": "Recall",
        },
        {
            "label": {"key": "animal", "value": "cat"},
            "value": 0.4,
            "type": "F1",
        },
        {
            "label": {"key": "animal", "value": "dog"},
            "value": 0.0,
            "type": "Precision",
        },
        {
            "label": {"key": "animal", "value": "dog"},
            "value": 0.0,
            "type": "Recall",
        },
        {
            "label": {"key": "animal", "value": "dog"},
            "value": 0.0,
            "type": "F1",
        },
        {
            "label": {"key": "color", "value": "blue"},
            "value": 0.0,
            "type": "Precision",
        },
        {
            "label": {"key": "color", "value": "blue"},
            "value": 0.0,
            "type": "Recall",
        },
        {
            "label": {"key": "color", "value": "blue"},
            "value": 0.0,
            "type": "F1",
        },
        {
            "label": {"key": "color", "value": "red"},
            "value": 0.6666666666666666,
            "type": "Precision",
        },
        {
            "label": {"key": "color", "value": "red"},
            "value": 1.0,
            "type": "Recall",
        },
        {
            "label": {"key": "color", "value": "red"},
            "value": 0.8,
            "type": "F1",
        },
        {
            "label": {"key": "color", "value": "white"},
            "value": 0.5,
            "type": "Precision",
        },
        {
            "label": {"key": "color", "value": "white"},
            "value": 0.5,
            "type": "Recall",
        },
        {
            "label": {"key": "color", "value": "white"},
            "value": 0.5,
            "type": "F1",
        },
        {
            "label": {"key": "color", "value": "black"},
            "value": 0.0,
            "type": "Precision",
        },
        {
            "label": {"key": "color", "value": "black"},
            "value": 0.0,
            "type": "Recall",
        },
        {
            "label": {"key": "color", "value": "black"},
            "value": 0.0,
            "type": "F1",
        },
        {
            "parameters": {"label_key": "animal"},
            "value": 0.3333333333333333,
            "type": "Accuracy",
        },
        {
            "parameters": {"label_key": "color"},
            "value": 0.5,
            "type": "Accuracy",
        },
        {
            "parameters": {"label_key": "animal"},
            "value": 0.8009259259259259,
            "type": "ROCAUC",
        },
        {
            "parameters": {"label_key": "color"},
            "value": 0.43125,
            "type": "ROCAUC",
        },
    ]
    expected_pr_curves = {
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
    expected_detailed_pr_curves = {
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
    expected_cm = [
        {
            "label_key": "animal",
            "entries": [
                {"prediction": "bird", "groundtruth": "bird", "count": 1},
                {"prediction": "cat", "groundtruth": "bird", "count": 1},
                {"prediction": "cat", "groundtruth": "cat", "count": 1},
                {"prediction": "cat", "groundtruth": "dog", "count": 2},
                {"prediction": "dog", "groundtruth": "bird", "count": 1},
            ],
        },
        {
            "label_key": "color",
            "entries": [
                {"prediction": "blue", "groundtruth": "white", "count": 1},
                {"prediction": "red", "groundtruth": "black", "count": 1},
                {"prediction": "red", "groundtruth": "red", "count": 2},
                {"prediction": "white", "groundtruth": "blue", "count": 1},
                {"prediction": "white", "groundtruth": "white", "count": 1},
            ],
        },
    ]

    # assert base metrics
    for actual, expected in [
        (computed_metrics, expected_metrics),
        (confusion_matrices, expected_cm),
    ]:
        for entry in actual:
            assert entry in expected
        for entry in expected:
            assert entry in actual

    # assert pr curves
    for (
        value,
        threshold,
        metric,
    ), expected_length in expected_pr_curves.items():
        classification = pr_curves[0]["value"][value][threshold][metric]
        assert classification == expected_length

    # assert DetailedPRCurves
    for (
        value,
        threshold,
        metric,
    ), expected_output in expected_detailed_pr_curves.items():
        model_output = detailed_pr_curves[0]["value"][value][threshold][metric]
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
    # test that DetailedPRCurve gives more examples when we adjust pr_curve_max_examples
    eval_job = evaluate_classification(
        groundtruths=groundtruths,
        predictions=predictions,
        pr_curve_max_examples=3,
        metrics_to_return=[
            enums.MetricType.DetailedPrecisionRecallCurve,
        ],
    )

    assert (
        len(
            eval_job.metrics[0]["value"]["bird"][0.05]["tp"]["observations"][
                "all"
            ]["examples"]
        )
        == 3
    )
    assert (
        len(
            eval_job.metrics[0]["value"]["bird"][0.05]["tn"]["observations"][
                "all"
            ]["examples"]
        )
        == 2
    )  # only two examples exist

    # test behavior if pr_curve_max_examples == 0
    eval_job = evaluate_classification(
        groundtruths=groundtruths,
        predictions=predictions,
        pr_curve_max_examples=0,
        metrics_to_return=[
            enums.MetricType.DetailedPrecisionRecallCurve,
        ],
    )

    assert (
        len(
            eval_job.metrics[0]["value"]["bird"][0.05]["tp"]["observations"][
                "all"
            ]["examples"]
        )
        == 0
    )
    assert (
        len(
            eval_job.metrics[0]["value"]["bird"][0.05]["tn"]["observations"][
                "all"
            ]["examples"]
        )
        == 0
    )
