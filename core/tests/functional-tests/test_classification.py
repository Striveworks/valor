import random

import pandas as pd
import pytest
from valor_core import enums, schemas
from valor_core.classification import (
    _calculate_rocauc,
    _create_joint_df,
    evaluate_classification,
)


def test_evaluate_image_clf(
    evaluate_image_clf_groundtruths: list[schemas.GroundTruth],
    evaluate_image_clf_predictions: list[schemas.Prediction],
    evaluate_image_clf_expected: tuple,
):

    expected_metrics, expected_confusion_matrices = evaluate_image_clf_expected

    eval_job = evaluate_classification(
        groundtruths=evaluate_image_clf_groundtruths,
        predictions=evaluate_image_clf_predictions,
    )

    eval_job_metrics = eval_job.metrics

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
    evaluate_tabular_clf_groundtruths_df: pd.DataFrame,
    evaluate_tabular_clf_predictions_df: pd.DataFrame,
    evaluate_tabular_clf_expected: tuple,
):
    expected_metrics, expected_confusion_matrix = evaluate_tabular_clf_expected

    eval_job = evaluate_classification(
        groundtruths=evaluate_tabular_clf_groundtruths_df,
        predictions=evaluate_tabular_clf_predictions_df,
    )

    eval_job_metrics = eval_job.metrics

    for m in eval_job_metrics:
        if m["type"] not in [
            "PrecisionRecallCurve",
            "DetailedPrecisionRecallCurve",
        ]:
            assert m in expected_metrics
    for m in expected_metrics:
        assert m in eval_job_metrics

    confusion_matrices = eval_job.confusion_matrices

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


def test_evaluate_classification_with_label_maps(
    gt_clfs_with_label_maps: list[schemas.GroundTruth],
    pred_clfs_with_label_maps: list[schemas.Prediction],
    cat_label_map: dict,
    evaluate_classification_with_label_maps_expected: tuple,
):

    (
        cat_expected_metrics,
        cat_expected_cm,
        pr_expected_values,
        detailed_pr_expected_answers,
    ) = evaluate_classification_with_label_maps_expected

    # check baseline case, where we have mismatched ground truth and prediction label keys
    with pytest.raises(ValueError) as e:
        evaluate_classification(
            groundtruths=gt_clfs_with_label_maps,
            predictions=pred_clfs_with_label_maps,
        )
    assert "label keys must match" in str(e)

    eval_job = evaluate_classification(
        groundtruths=gt_clfs_with_label_maps,
        predictions=pred_clfs_with_label_maps,
        label_map=cat_label_map,
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
    assert eval_job.meta["labels"] == 9
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
    evaluate_classification_model_with_no_predictions_expected: list,
):

    # can't pass empty lists, but can pass predictions without annotations
    with pytest.raises(ValueError) as e:
        evaluation = evaluate_classification(
            groundtruths=gt_clfs,
            predictions=[],
        )
    assert (
        "it's neither a dataframe nor a list of Valor Prediction objects"
        in str(e)
    )

    evaluation = evaluate_classification(
        groundtruths=gt_clfs,
        predictions=[
            schemas.Prediction(datum=gt_clfs[0].datum, annotations=[])
        ],
    )

    computed_metrics = evaluation.metrics

    assert all([metric["value"] == 0 for metric in computed_metrics])
    assert all(
        [
            metric in computed_metrics
            for metric in evaluate_classification_model_with_no_predictions_expected
        ]
    )
    assert all(
        [
            metric
            in evaluate_classification_model_with_no_predictions_expected
            for metric in computed_metrics
        ]
    )


def test_compute_confusion_matrix_at_label_key_using_label_map(
    classification_functional_test_data: tuple,
    mammal_label_map: dict,
    compute_confusion_matrix_at_label_key_using_label_map_expected: list,
):
    """
    Test grouping using the label_map
    """

    groundtruths, predictions = classification_functional_test_data

    eval_job = evaluate_classification(
        groundtruths=groundtruths,
        predictions=predictions,
        label_map=mammal_label_map,
    )

    cm = eval_job.confusion_matrices

    assert cm
    assert len(cm) == len(
        compute_confusion_matrix_at_label_key_using_label_map_expected
    )
    for entry in cm:
        assert (
            entry
            in compute_confusion_matrix_at_label_key_using_label_map_expected
        )
    for (
        entry
    ) in compute_confusion_matrix_at_label_key_using_label_map_expected:
        assert entry in cm


def test_rocauc_with_label_map(
    classification_functional_test_prediction_df,
    classification_functional_test_groundtruth_df,
    rocauc_with_label_map_expected: list,
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

    joint_df = _create_joint_df(
        groundtruth_df=classification_functional_test_groundtruth_df,
        prediction_df=classification_functional_test_prediction_df,
    )

    computed_metrics = [
        m.to_dict() for m in _calculate_rocauc(joint_df=joint_df)
    ]

    for entry in computed_metrics:
        assert entry in rocauc_with_label_map_expected
    for entry in rocauc_with_label_map_expected:
        assert entry in computed_metrics


def test_compute_classification(
    classification_functional_test_data, compute_classification_expected: list
):
    """
    Tests the _compute_classification function.
    """

    (
        expected_metrics,
        expected_cm,
        expected_pr_curves,
        expected_detailed_pr_curves,
    ) = compute_classification_expected

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


def test_pr_curves_multiple_predictions_per_groundtruth(
    multiclass_pr_curve_groundtruths: list,
    multiclass_pr_curve_predictions: list,
    test_pr_curves_multiple_predictions_per_groundtruth_expected: dict,
):
    """Test that we get back the expected results when creating PR curves with multiple predictions per groundtruth."""

    eval_job = evaluate_classification(
        groundtruths=multiclass_pr_curve_groundtruths,
        predictions=multiclass_pr_curve_predictions,
        metrics_to_return=[enums.MetricType.PrecisionRecallCurve],
    )

    output = eval_job.metrics[0]["value"]

    # there are two cat, two bee, and one dog groundtruths
    # once we raise the score threshold above the maximum score, we expect the tps to become fns and the fps to become tns
    def _get_specific_keys_from_pr_output(output_dict):
        return {
            k: v
            for k, v in output_dict.items()
            if k in ["tp", "fp", "tn", "fn"]
        }

    for (
        animal,
        thresholds,
    ) in test_pr_curves_multiple_predictions_per_groundtruth_expected.items():
        for threshold in thresholds.keys():
            assert (
                _get_specific_keys_from_pr_output(output[animal][threshold])
                == test_pr_curves_multiple_predictions_per_groundtruth_expected[
                    animal
                ][
                    threshold
                ]
            )


def test_detailed_curve_examples(
    multiclass_pr_curve_groundtruths: list,
    multiclass_pr_curve_check_zero_count_examples_groundtruths: list,
    multiclass_pr_curve_check_true_negatives_groundtruths: list,
    multiclass_pr_curve_predictions: list,
    multiclass_pr_curve_check_zero_count_examples_predictions: list,
    multiclass_pr_curve_check_true_negatives_predictions: list,
    detailed_curve_examples_output: dict,
    detailed_curve_examples_check_zero_count_examples_output: dict,
    detailed_curve_examples_check_true_negatives_output: dict,
):
    """Test that we get back the right examples in DetailedPRCurves."""

    eval_job = evaluate_classification(
        groundtruths=multiclass_pr_curve_groundtruths,
        predictions=multiclass_pr_curve_predictions,
        metrics_to_return=[enums.MetricType.DetailedPrecisionRecallCurve],
        pr_curve_max_examples=5,
    )

    output = eval_job.metrics[0]["value"]

    for key, expected in detailed_curve_examples_output.items():
        assert (
            set(
                output[key[0]][key[1]][key[2]]["observations"][key[3]][
                    "examples"
                ]
            )
            == expected
        )

    # test additional cases to make sure that we aren't returning examples where count == 0
    eval_job = evaluate_classification(
        groundtruths=multiclass_pr_curve_check_zero_count_examples_groundtruths,
        predictions=multiclass_pr_curve_check_zero_count_examples_predictions,
        metrics_to_return=[enums.MetricType.DetailedPrecisionRecallCurve],
    )
    output = eval_job.metrics[0]["value"]

    for (
        key,
        expected,
    ) in detailed_curve_examples_check_zero_count_examples_output.items():
        assert (
            len(
                output[key[0]][key[1]][key[2]]["observations"][key[3]][
                    "examples"
                ]
            )
            == expected
        )
        assert (
            output[key[0]][key[1]][key[2]]["observations"][key[3]]["count"]
        ) == 0

    # test additional cases to make sure that we're getting back enough true negative examples
    eval_job = evaluate_classification(
        groundtruths=multiclass_pr_curve_check_true_negatives_groundtruths,
        predictions=multiclass_pr_curve_check_true_negatives_predictions,
        metrics_to_return=[enums.MetricType.DetailedPrecisionRecallCurve],
        pr_curve_max_examples=5,
    )
    output = eval_job.metrics[0]["value"]

    for (
        key,
        expected,
    ) in detailed_curve_examples_check_true_negatives_output.items():
        assert (
            set(
                output[key[0]][key[1]][key[2]]["observations"][key[3]][
                    "examples"
                ]
            )
            == expected
        )
