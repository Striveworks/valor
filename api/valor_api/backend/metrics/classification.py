import random
from collections import defaultdict
from typing import Any

import numpy as np
from sqlalchemy import CTE, ColumnElement, Integer, Subquery
from sqlalchemy.orm import Session
from sqlalchemy.sql import and_, func, select

from valor_api import enums, schemas
from valor_api.backend import core, models
from valor_api.backend.metrics.metric_utils import (
    commit_results,
    create_label_mapping,
    log_evaluation_duration,
    log_evaluation_item_counts,
    prepare_filter_for_evaluation,
    validate_computation,
)
from valor_api.backend.query import generate_select

LabelMapType = list[list[list[str]]]


def _compute_curves(
    db: Session,
    predictions: CTE,
    groundtruths: CTE,
    labels: set[tuple[str, str]],
    unique_datums: dict[str, tuple[str, str]],
    pr_curve_max_examples: int,
    metrics_to_return: list[enums.MetricType],
) -> list[schemas.PrecisionRecallCurve | schemas.DetailedPrecisionRecallCurve]:
    """
    Calculates precision-recall curves for each class.

    Parameters
    ----------
    db: Session
        The database Session to query against.
    predictions: CTE
        A CTE defining a set of predictions.
    groundtruths: CTE
        A CTE defining a set of ground truths.
    unique_datums: dict[str, tuple[str, str]]
        All of the unique datums associated with the ground truth and prediction filters.
    pr_curve_max_examples: int
        The maximum number of datum examples to store per true positive, false negative, etc.
    metrics_to_return: list[enums.MetricType]
        The list of metrics requested by the user.

    Returns
    -------
    list[schemas.PrecisionRecallCurve | schemas.DetailedPrecisionRecallCurve]
        The PrecisionRecallCurve and/or DetailedPrecisionRecallCurve metrics.
    """

    # create sets of all datums for which there is a prediction / groundtruth
    # used when separating misclassifications/no_predictions
    def _high_scores():
        return db.query(
            select(
                predictions.c.key,
                predictions.c.datum_id,
                func.max(predictions.c.score),
            )
            .select_from(predictions)
            .group_by(
                predictions.c.key,
                predictions.c.datum_id,
            )
            .subquery()
        ).all()

    high_scores = _high_scores()

    label_key_to_datum_high_score = defaultdict(dict)
    for label_key, datum_id, high_score in high_scores:
        label_key_to_datum_high_score[label_key][datum_id] = high_score

    high_score_subquery = (
        select(
            predictions.c.key.label("label_key"),
            predictions.c.datum_id,
            func.max(predictions.c.score).label("score"),
        )
        .select_from(predictions)
        .group_by(
            predictions.c.key,
            predictions.c.datum_id,
        )
        .subquery()
    )

    total_query = (
        select(
            func.coalesce(
                groundtruths.c.datum_id,
                predictions.c.datum_id,
            ).label("datum_id"),
            func.coalesce(
                groundtruths.c.key,
                predictions.c.key,
            ).label("label_key"),
            groundtruths.c.value.label("gt_value"),
            predictions.c.value.label("pd_value"),
            predictions.c.score,
        )
        .select_from(groundtruths)
        .join(
            predictions,
            and_(
                predictions.c.datum_id == groundtruths.c.datum_id,
                predictions.c.key == groundtruths.c.key,
            ),
            full=True,
        )
        .subquery()
    )

    thresholds_cte = select(
        func.generate_series(0.95, 0.05, -0.05).label("threshold")
    ).cte()

    next_query = (
        select(
            total_query.c.datum_id,
            total_query.c.label_key,
            total_query.c.gt_value,
            thresholds_cte.c.threshold,
            (
                and_(
                    (total_query.c.gt_value == total_query.c.pd_value),
                    (total_query.c.score >= thresholds_cte.c.threshold),
                )
            ).label("tp"),
            (
                and_(
                    (total_query.c.gt_value != total_query.c.pd_value),
                    (total_query.c.score >= thresholds_cte.c.threshold),
                )
            ).label("fp"),
            (
                and_(
                    (total_query.c.gt_value != total_query.c.pd_value),
                    (total_query.c.score < thresholds_cte.c.threshold),
                )
            ).label("tn"),
            (
                and_(
                    (total_query.c.gt_value != total_query.c.pd_value),
                    (total_query.c.score < thresholds_cte.c.threshold),
                )
            ).label("fn"),
            (high_score_subquery.c.score < thresholds_cte.c.threshold).label(
                "prediction_exists"
            ),
        )
        .select_from(total_query)
        .join(
            high_score_subquery,
            and_(
                high_score_subquery.c.datum_id == total_query.c.datum_id,
                high_score_subquery.c.label_key == total_query.c.label_key,
            ),
        )
        .cte()
    )

    def search_datums(condition: ColumnElement[bool]):
        return (
            select(
                next_query.c.label_key,
                next_query.c.gt_value,
                next_query.c.threshold,
                next_query.c.datum_id,
                func.row_number()
                .over(
                    partition_by=[
                        next_query.c.label_key,
                        next_query.c.gt_value,
                        next_query.c.threshold,
                    ],
                    order_by=func.random(),
                )
                .label("row_number"),
            )
            .where(condition)
            .subquery()
        )

    def limit_examples(subquery: Subquery[Any]):
        return (
            select(
                subquery.c.label_key,
                subquery.c.gt_value,
                subquery.c.threshold,
                subquery.c.datum_id,
            )
            .where(subquery.c.row_number <= pr_curve_max_examples)
            .cte()
        )

    tp_datums = search_datums(next_query.c.fp)
    fp_datums = search_datums(next_query.c.fp)
    tn_datums = search_datums(next_query.c.tn)
    fn_misclassification_datums = search_datums(
        and_(
            next_query.c.fn,
            next_query.c.prediction_exists,
        )
    )
    fn_missing_prediction_datums = search_datums(
        and_(
            next_query.c.fn,
            next_query.c.prediction_exists.isnot(True),
        )
    )

    tp_examples = limit_examples(tp_datums)
    fp_examples = limit_examples(fp_datums)
    tn_examples = limit_examples(tn_datums)
    fn_misclassification_examples = limit_examples(fn_misclassification_datums)
    fn_missing_prediction_examples = limit_examples(
        fn_missing_prediction_datums
    )

    pr_counts = (
        select(
            next_query.c.label_key,
            next_query.c.gt_value,
            next_query.c.threshold,
            func.sum(next_query.c.tp.cast(Integer)).label("tp"),
            func.sum(next_query.c.fp.cast(Integer)).label("fp"),
            func.sum(next_query.c.tn.cast(Integer)).label("tn"),
            func.sum(next_query.c.fn.cast(Integer)).label("fn"),
            func.array_agg(tp_examples.c.datum_id).label("tp_examples"),
            func.array_agg(fp_examples.c.datum_id).label("fp_examples"),
            func.array_agg(tn_examples.c.datum_id).label("tn_examples"),
            func.array_agg(fn_misclassification_examples.c.datum_id).label(
                "fn_misclf_examples"
            ),
            func.array_agg(fn_missing_prediction_examples.c.datum_id).label(
                "fn_misprd_examples"
            ),
        )
        .select_from(next_query)
        .join(
            tp_examples,
            and_(
                tp_examples.c.label_key == next_query.c.label_key,
                tp_examples.c.gt_value == next_query.c.gt_value,
                tp_examples.c.threshold == next_query.c.threshold,
            ),
        )
        .join(
            fp_examples,
            and_(
                fp_examples.c.label_key == next_query.c.label_key,
                fp_examples.c.gt_value == next_query.c.gt_value,
                fp_examples.c.threshold == next_query.c.threshold,
            ),
        )
        .join(
            tn_examples,
            and_(
                tn_examples.c.label_key == next_query.c.label_key,
                tn_examples.c.gt_value == next_query.c.gt_value,
                tn_examples.c.threshold == next_query.c.threshold,
            ),
        )
        .join(
            fn_misclassification_examples,
            and_(
                fn_misclassification_examples.c.label_key
                == next_query.c.label_key,
                fn_misclassification_examples.c.gt_value
                == next_query.c.gt_value,
                fn_misclassification_examples.c.threshold
                == next_query.c.threshold,
            ),
        )
        .join(
            fn_missing_prediction_examples,
            and_(
                fn_missing_prediction_examples.c.label_key
                == next_query.c.label_key,
                fn_missing_prediction_examples.c.gt_value
                == next_query.c.gt_value,
                fn_missing_prediction_examples.c.threshold
                == next_query.c.threshold,
            ),
        )
        .group_by(
            next_query.c.label_key,
            next_query.c.gt_value,
            next_query.c.threshold,
        )
        .subquery()
    )

    def _res():
        return db.query(pr_counts).all()

    res = _res()

    pr_output = defaultdict(lambda: defaultdict((lambda: defaultdict(dict))))
    detailed_pr_output = defaultdict(
        lambda: defaultdict((lambda: defaultdict(dict)))
    )
    visited_keys = set()
    for (
        label_key,
        label_value,
        threshold,
        tp_cnt,
        fp_cnt,
        tn_cnt,
        fn_cnt,
        tp,
        fp,
        tn,
        fn_misclf_examples,
        fn_misprd_examples,
    ) in res:

        visited_keys.add(label_key)

        precision = (
            (tp_cnt) / (tp_cnt + fp_cnt) if (tp_cnt + fp_cnt) > 0 else -1
        )
        recall = tp_cnt / (tp_cnt + fn_cnt) if (tp_cnt + fn_cnt) > 0 else -1
        accuracy = (
            (tp_cnt + tn_cnt) / len(unique_datums)
            if len(unique_datums) > 0
            else -1
        )
        f1_score = (
            (2 * precision * recall) / (precision + recall)
            if precision and recall
            else -1
        )

        pr_output[label_key][label_value][float(threshold)] = {
            "tp": tp_cnt,
            "fp": fp_cnt,
            "fn": fn_cnt,
            "tn": tn_cnt,
            "accuracy": accuracy,
            "precision": precision,
            "recall": recall,
            "f1_score": f1_score,
        }

        if enums.MetricType.DetailedPrecisionRecallCurve in metrics_to_return:
            tp = [unique_datums[datum_id] for datum_id in tp]
            fp = {
                "misclassifications": [
                    unique_datums[datum_id] for datum_id in fp
                ]
            }
            tn = [unique_datums[datum_id] for datum_id in tn]
            fn = {
                "misclassifications": [
                    unique_datums[datum_id] for datum_id in fn_misclf_examples
                ],
                "no_predictions": [
                    unique_datums[datum_id] for datum_id in fn_misprd_examples
                ],
            }

            detailed_pr_output[label_key][label_value][float(threshold)] = {
                "tp": {
                    "total": tp_cnt,
                    "observations": {
                        "all": {
                            "count": tp_cnt,
                            "examples": (
                                random.sample(tp, pr_curve_max_examples)
                                if len(tp) > pr_curve_max_examples
                                else tp
                            ),
                        }
                    },
                },
                "tn": {
                    "total": tn_cnt,
                    "observations": {
                        "all": {
                            "count": tn_cnt,
                            "examples": (
                                random.sample(tn, pr_curve_max_examples)
                                if len(tn) > pr_curve_max_examples
                                else tn
                            ),
                        }
                    },
                },
                "fn": {
                    "total": fn_cnt,
                    "observations": {
                        "misclassifications": {
                            "count": len(fn["misclassifications"]),
                            "examples": (
                                random.sample(
                                    fn["misclassifications"],
                                    pr_curve_max_examples,
                                )
                                if len(fn["misclassifications"])
                                > pr_curve_max_examples
                                else fn["misclassifications"]
                            ),
                        },
                        "no_predictions": {
                            "count": len(fn["no_predictions"]),
                            "examples": (
                                random.sample(
                                    fn["no_predictions"],
                                    pr_curve_max_examples,
                                )
                                if len(fn["no_predictions"])
                                > pr_curve_max_examples
                                else fn["no_predictions"]
                            ),
                        },
                    },
                },
                "fp": {
                    "total": fp_cnt,
                    "observations": {
                        "misclassifications": {
                            "count": len(fp["misclassifications"]),
                            "examples": (
                                random.sample(
                                    fp["misclassifications"],
                                    pr_curve_max_examples,
                                )
                                if len(fp["misclassifications"])
                                >= pr_curve_max_examples
                                else fp["misclassifications"]
                            ),
                        },
                    },
                },
            }

    label_keys = {key for key, _ in labels}
    missing_label_keys = label_keys - visited_keys

    pr_curves = [
        schemas.PrecisionRecallCurve(
            label_key=label_key,
            value=dict(value),
        )
        for label_key, value in pr_output.items()
    ] + [
        schemas.PrecisionRecallCurve(
            label_key=label_key,
            value={},
        )
        for label_key in missing_label_keys
    ]

    detailed_pr_curves = [
        schemas.DetailedPrecisionRecallCurve(
            label_key=label_key,
            value=dict(value),
        )
        for label_key, value in detailed_pr_output.items()
    ] + [
        schemas.DetailedPrecisionRecallCurve(
            label_key=label_key,
            value={},
        )
        for label_key in missing_label_keys
    ]

    return pr_curves + detailed_pr_curves


def _compute_roc_auc(
    db: Session,
    groundtruths: CTE,
    predictions: CTE,
    labels: set[tuple[str, str]],
) -> list[schemas.ROCAUCMetric]:
    """
    Computes the area under the ROC curve. Note that for the multi-class setting
    this does one-vs-rest AUC for each class and then averages those scores. This should give
    the same thing as `sklearn.metrics.roc_auc_score` with `multi_class="ovr"`.

    Parameters
    ----------
    db : Session
        The database Session to query against.
    groundtruths : CTE
        A cte returning ground truths.
    predictions : CTE
        A cte returning predictions.

    Returns
    -------
    list[schemas.ROCAUCMetric]
        The ROC AUC. Returns None if no labels exist for that label_key.
    """

    predictions_label_keys = {
        key for key in (db.scalars(select(predictions.c.key).distinct()).all())
    }

    groundtruths_per_label_kv_query = (
        select(
            groundtruths.c.key,
            groundtruths.c.value,
            func.count().label("gt_counts_per_label"),
        )
        .select_from(groundtruths)
        .group_by(
            groundtruths.c.key,
            groundtruths.c.value,
        )
        .cte("gt_counts")
    )

    label_key_to_count = defaultdict(int)
    label_to_count = dict()
    groundtruth_labels = set()
    for key, value, count in db.query(groundtruths_per_label_kv_query).all():
        label_key_to_count[key] += count
        label_to_count[(key, value)] = count
        groundtruth_labels.add((key, value))

    groundtruths_per_label_key_query = (
        select(
            groundtruths_per_label_kv_query.c.key,
            func.sum(groundtruths_per_label_kv_query.c.gt_counts_per_label)
            .cast(Integer)
            .label("gt_counts_per_key"),
        )
        .group_by(groundtruths_per_label_kv_query.c.key)
        .subquery()
    )

    basic_counts_query = (
        select(
            (groundtruths.c.value == predictions.c.value)
            .cast(Integer)
            .label("is_true_positive"),
            (groundtruths.c.value != predictions.c.value)
            .cast(Integer)
            .label("is_false_positive"),
            predictions.c.key.label("label_key"),
            predictions.c.value.label("prediction_label_value"),
            predictions.c.datum_id,
            predictions.c.score,
        )
        .select_from(predictions)
        .join(
            groundtruths,
            and_(
                groundtruths.c.datum_id == predictions.c.datum_id,
                groundtruths.c.key == predictions.c.key,
            ),
        )
        .subquery("basic_counts")
    )

    cumulative_tp = func.sum(basic_counts_query.c.is_true_positive).over(
        partition_by=[
            basic_counts_query.c.label_key,
            basic_counts_query.c.prediction_label_value,
        ],
        order_by=basic_counts_query.c.score.desc(),
    )

    cumulative_fp = func.sum(basic_counts_query.c.is_false_positive).over(
        partition_by=[
            basic_counts_query.c.label_key,
            basic_counts_query.c.prediction_label_value,
        ],
        order_by=basic_counts_query.c.score.desc(),
    )

    tpr_fpr_cumulative = select(
        cumulative_tp.label("cumulative_tp"),
        cumulative_fp.label("cumulative_fp"),
        basic_counts_query.c.label_key,
        basic_counts_query.c.prediction_label_value,
        basic_counts_query.c.score,
    ).subquery("tpr_fpr_cumulative")

    tpr_fpr_rates = (
        select(
            (
                tpr_fpr_cumulative.c.cumulative_tp
                / groundtruths_per_label_kv_query.c.gt_counts_per_label
            ).label("tpr"),
            (
                tpr_fpr_cumulative.c.cumulative_fp
                / (
                    groundtruths_per_label_key_query.c.gt_counts_per_key
                    - groundtruths_per_label_kv_query.c.gt_counts_per_label
                )
            ).label("fpr"),
            tpr_fpr_cumulative.c.label_key,
            tpr_fpr_cumulative.c.prediction_label_value,
            tpr_fpr_cumulative.c.score,
        )
        .join(
            groundtruths_per_label_key_query,
            groundtruths_per_label_key_query.c.key
            == tpr_fpr_cumulative.c.label_key,
        )
        .join(
            groundtruths_per_label_kv_query,
            and_(
                groundtruths_per_label_kv_query.c.key
                == tpr_fpr_cumulative.c.label_key,
                groundtruths_per_label_kv_query.c.value
                == tpr_fpr_cumulative.c.prediction_label_value,
                groundtruths_per_label_kv_query.c.gt_counts_per_label > 0,
                (
                    groundtruths_per_label_key_query.c.gt_counts_per_key
                    - groundtruths_per_label_kv_query.c.gt_counts_per_label
                )
                > 0,
            ),
        )
        .subquery("tpr_fpr_rates")
    )

    lagging_tpr = func.lag(tpr_fpr_rates.c.tpr).over(
        partition_by=[
            tpr_fpr_rates.c.label_key,
            tpr_fpr_rates.c.prediction_label_value,
        ],
        order_by=tpr_fpr_rates.c.score.desc(),
    )

    lagging_fpr = func.lag(tpr_fpr_rates.c.fpr).over(
        partition_by=[
            tpr_fpr_rates.c.label_key,
            tpr_fpr_rates.c.prediction_label_value,
        ],
        order_by=tpr_fpr_rates.c.score.desc(),
    )

    trap_areas = select(
        (
            0.5
            * (tpr_fpr_rates.c.tpr + lagging_tpr)
            * (tpr_fpr_rates.c.fpr - lagging_fpr)
        ).label("trap_area"),
        tpr_fpr_rates.c.label_key,
        tpr_fpr_rates.c.prediction_label_value,
    ).subquery()

    results = (
        db.query(
            trap_areas.c.label_key,
            trap_areas.c.prediction_label_value,
            func.sum(trap_areas.c.trap_area),
        )
        .group_by(
            trap_areas.c.label_key,
            trap_areas.c.prediction_label_value,
        )
        .all()
    )

    map_label_to_rocauc = {
        (key, value): rocauc
        for key, value, rocauc in results
        if rocauc is not None
    }

    label_key_to_rocauc = defaultdict(list)
    for key, value in labels:
        label = (key, value)
        if label not in groundtruth_labels:
            continue
        elif label_to_count[label] == 0:
            label_key_to_rocauc[key].append(0.0)
        elif label_key_to_count[key] - label_to_count[label] == 0:
            label_key_to_rocauc[key].append(1.0)
        else:
            rocauc = map_label_to_rocauc.get(label, np.nan)
            label_key_to_rocauc[key].append(float(rocauc))

    label_keys = {key for key, _ in labels}
    return [
        schemas.ROCAUCMetric(
            label_key=key,
            value=(
                float(np.mean(label_key_to_rocauc[key]))
                if len(label_key_to_rocauc[key]) >= 1
                else None
            ),
        )
        if (key in label_key_to_rocauc and key in predictions_label_keys)
        else schemas.ROCAUCMetric(
            label_key=key,
            value=0.0,
        )
        for key in label_keys
    ]


def _compute_confusion_matrices(
    db: Session,
    predictions: CTE,
    groundtruths: CTE,
    labels: set[tuple[str, str]],
) -> dict[str, schemas.ConfusionMatrix | None]:
    """
    Computes the confusion matrix at a label_key.

    Parameters
    ----------
    db : Session
        The database Session to query against.
    predictions: CTE
        A CTE defining a set of predictions.
    groundtruths: CTE
        A CTE defining a set of ground truths.

    Returns
    -------
    schemas.ConfusionMatrix | None
        Returns None in the case that there are no common images in the dataset
        that have both a ground truth and prediction with label key `label_key`. Otherwise
        returns the confusion matrix.
    """

    # 1. Get the max prediction scores by datum
    max_scores_by_datum_id = (
        select(
            predictions.c.datum_id,
            predictions.c.key,
            func.max(predictions.c.score).label("max_score"),
        )
        .group_by(
            predictions.c.key,
            predictions.c.datum_id,
        )
        .subquery()
    )

    # 2. Remove duplicate scores per datum
    # used for the edge case where the max confidence appears twice
    # the result of this query is all of the hard predictions
    min_id_query = (
        select(
            func.min(predictions.c.prediction_id).label("min_id"),
            predictions.c.datum_id,
            predictions.c.key,
        )
        .select_from(predictions)
        .join(
            max_scores_by_datum_id,
            and_(
                predictions.c.datum_id == max_scores_by_datum_id.c.datum_id,
                predictions.c.key == max_scores_by_datum_id.c.key,
                predictions.c.score == max_scores_by_datum_id.c.max_score,
            ),
        )
        .group_by(predictions.c.key, predictions.c.datum_id)
        .subquery()
    )

    # 3. Get labels for hard predictions, organize per datum
    hard_preds_query = (
        select(
            predictions.c.key,
            predictions.c.value,
            predictions.c.datum_id,
        )
        .select_from(predictions)
        .join(
            min_id_query,
            and_(
                min_id_query.c.min_id == predictions.c.prediction_id,
                min_id_query.c.key == predictions.c.key,
            ),
        )
        .subquery()
    )

    # 4. Generate confusion matrix
    total_query = (
        db.query(
            groundtruths.c.key,
            groundtruths.c.value,
            hard_preds_query.c.value,
            func.count(),
        )
        .select_from(hard_preds_query)
        .join(
            groundtruths,
            and_(
                groundtruths.c.datum_id == hard_preds_query.c.datum_id,
                groundtruths.c.key == hard_preds_query.c.key,
            ),
        )
        .group_by(
            groundtruths.c.key,
            groundtruths.c.value,
            hard_preds_query.c.value,
        )
        .all()
    )

    # 5. Unpack results.
    confusion_mapping = defaultdict(list)
    for label_key, gt_value, pd_value, count in total_query:
        confusion_mapping[label_key].append((gt_value, pd_value, count))

    return {
        key: (
            schemas.ConfusionMatrix(
                label_key=key,
                entries=[
                    schemas.ConfusionMatrixEntry(
                        prediction=pd, groundtruth=gt, count=count
                    )
                    for gt, pd, count in confusion_mapping[key]
                ],
            )
            if key in confusion_mapping
            else None
        )
        for key, _ in labels
    }


def _compute_accuracy_from_cm(cm: schemas.ConfusionMatrix) -> float:
    """
    Computes the accuracy score from a confusion matrix.

    Parameters
    ----------
    cm : schemas.ConfusionMatrix
        The confusion matrix to use.

    Returns
    ----------
    float
        The resultant accuracy score.
    """
    return cm.matrix.trace() / cm.matrix.sum()


def _compute_precision_and_recall_f1_from_confusion_matrix(
    cm: schemas.ConfusionMatrix,
    label_value: str,
) -> tuple[float, float, float]:
    """
    Computes the precision, recall, and f1 score at a class index

    Parameters
    ----------
    cm : schemas.ConfusionMatrix
        The confusion matrix to use.
    label_key : str
        The label key to compute scores for.

    Returns
    ----------
    Tuple[float, float, float]
        A tuple containing the precision, recall, and F1 score.
    """
    cm_matrix = cm.matrix
    if label_value not in cm.label_map:
        return np.nan, np.nan, np.nan
    class_index = cm.label_map[label_value]

    true_positives = cm_matrix[class_index, class_index]
    # number of times the class was predicted
    n_preds = cm_matrix[:, class_index].sum()
    n_gts = cm_matrix[class_index, :].sum()

    prec = true_positives / n_preds if n_preds else 0
    recall = true_positives / n_gts if n_gts else 0

    f1_denom = prec + recall
    if f1_denom == 0:
        f1 = 0
    else:
        f1 = 2 * prec * recall / f1_denom
    return prec, recall, f1


def _compute_confusion_matrices_and_metrics(
    db: Session,
    groundtruths: CTE,
    predictions: CTE,
    labels: set[tuple[str, str]],
    pr_curve_max_examples: int,
    metrics_to_return: list[enums.MetricType],
) -> (
    tuple[
        list[schemas.ConfusionMatrix],
        list[
            schemas.AccuracyMetric
            | schemas.ROCAUCMetric
            | schemas.PrecisionMetric
            | schemas.RecallMetric
            | schemas.F1Metric
            | schemas.PrecisionRecallCurve
            | schemas.DetailedPrecisionRecallCurve
        ],
    ]
):
    """
    Computes the confusion matrix and all metrics for a given label key.

    Parameters
    ----------
    db : Session
        The database Session to query against.
    predictions: CTE
        A CTE defining a set of predictions.
    groundtruths: CTE
        A CTE defining a set of ground truths.
    labels: set[tuple[str, str]]
        Labels referenced by groundtruths and predictions.
    pr_curve_max_examples: int
        The maximum number of datum examples to store per true positive, false negative, etc.
    metrics_to_return: list[MetricType]
        The list of metrics to compute, store, and return to the user.

    Returns
    -------
    tuple[schemas.ConfusionMatrix, list[schemas.AccuracyMetric | schemas.ROCAUCMetric | schemas.PrecisionMetric
                                        | schemas.RecallMetric | schemas.F1Metric]] | None
        Returns None if there are no predictions and groundtruths with the given label
        key for the same datum. Otherwise returns a tuple, with the first element the confusion
        matrix and the second a list of all metrics (accuracy, ROC AUC, precisions, recalls, and f1s).
    """

    metrics: list[
        schemas.AccuracyMetric
        | schemas.ROCAUCMetric
        | schemas.PrecisionMetric
        | schemas.RecallMetric
        | schemas.F1Metric
        | schemas.PrecisionRecallCurve
        | schemas.DetailedPrecisionRecallCurve
    ] = list()

    #
    confusion_matrices = _compute_confusion_matrices(
        db=db,
        groundtruths=groundtruths,
        predictions=predictions,
        labels=labels,
    )

    # aggregate metrics (over all label values)
    metrics += [
        schemas.AccuracyMetric(
            label_key=label_key,
            value=(
                _compute_accuracy_from_cm(confusion_matrix)
                if confusion_matrix
                else 0.0
            ),
        )
        for label_key, confusion_matrix in confusion_matrices.items()
    ]
    metrics += _compute_roc_auc(
        db=db,
        groundtruths=groundtruths,
        predictions=predictions,
        labels=labels,
    )

    if (
        enums.MetricType.PrecisionRecallCurve in metrics_to_return
        or enums.MetricType.DetailedPrecisionRecallCurve in metrics_to_return
    ):
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

        metrics += _compute_curves(
            db=db,
            groundtruths=groundtruths,
            predictions=predictions,
            labels=labels,
            unique_datums=unique_datums,
            pr_curve_max_examples=pr_curve_max_examples,
            metrics_to_return=metrics_to_return,
        )

    # metrics that are per label
    for key, value in labels:
        confusion_matrix = confusion_matrices.get(key, None)
        if confusion_matrix:
            (
                precision,
                recall,
                f1,
            ) = _compute_precision_and_recall_f1_from_confusion_matrix(
                confusion_matrix, value
            )
        else:
            precision = 0.0
            recall = 0.0
            f1 = 0.0

        pydantic_label = schemas.Label(key=key, value=value)

        metrics += [
            schemas.PrecisionMetric(
                label=pydantic_label,
                value=precision,
            ),
            schemas.RecallMetric(
                label=pydantic_label,
                value=recall,
            ),
            schemas.F1Metric(
                label=pydantic_label,
                value=f1,
            ),
        ]

    return [
        confusion_matrix
        for confusion_matrix in confusion_matrices.values()
        if confusion_matrix
    ], metrics


def _aggregate_data(
    db: Session,
    groundtruth_filter: schemas.Filter,
    prediction_filter: schemas.Filter,
    label_map: LabelMapType | None = None,
) -> tuple[CTE, CTE, set[tuple[str, str]]]:
    """
    Compute classification metrics.

    Parameters
    ----------
    db : Session
        The database Session to query against.
    groundtruth_filter : schemas.Filter
        The filter to be used to query groundtruths.
    prediction_filter : schemas.Filter
        The filter to be used to query predictions.
    label_map: LabelMapType, optional
        Optional mapping of individual labels to a grouper label. Useful when you need to evaluate performance using labels that differ across datasets and models.

    Returns
    ----------
    list[ConfusionMatrix, Metric]
        A list of confusion matrices and metrics.
    """
    labels = core.fetch_union_of_labels(
        db=db,
        lhs=groundtruth_filter,
        rhs=prediction_filter,
    )

    label_mapping = create_label_mapping(
        db=db,
        labels=labels,
        label_map=label_map,
    )

    groundtruths_subquery = generate_select(
        models.Datum.id.label("datum_id"),
        models.Datum.uid.label("datum_uid"),
        models.Dataset.name.label("dataset_name"),
        models.Label.id,
        label_mapping,
        filters=groundtruth_filter,
        label_source=models.GroundTruth,
    ).subquery()
    groundtruths_cte = (
        select(
            groundtruths_subquery.c.datum_id,
            groundtruths_subquery.c.datum_uid,
            groundtruths_subquery.c.dataset_name,
            models.Label.key,
            models.Label.value,
        )
        .select_from(groundtruths_subquery)
        .join(
            models.Label,
            models.Label.id == groundtruths_subquery.c.label_id,
        )
        .distinct()
        .cte()
    )

    predictions_subquery = generate_select(
        models.Datum.id.label("datum_id"),
        models.Datum.uid.label("datum_uid"),
        models.Dataset.name.label("dataset_name"),
        models.Model.name.label("model_name"),
        models.Prediction.id.label("prediction_id"),
        models.Prediction.score,
        models.Label.id,
        label_mapping,
        filters=prediction_filter,
        label_source=models.Prediction,
    ).subquery()
    predictions_cte = (
        select(
            predictions_subquery.c.dataset_name,
            predictions_subquery.c.model_name,
            predictions_subquery.c.datum_id,
            predictions_subquery.c.datum_uid,
            models.Label.key,
            models.Label.value,
            func.min(predictions_subquery.c.prediction_id).label(
                "prediction_id"
            ),
            func.max(predictions_subquery.c.score).label("score"),
        )
        .select_from(predictions_subquery)
        .join(
            models.Label,
            models.Label.id == predictions_subquery.c.label_id,
        )
        .group_by(
            predictions_subquery.c.dataset_name,
            predictions_subquery.c.model_name,
            predictions_subquery.c.datum_id,
            predictions_subquery.c.datum_uid,
            models.Label.key,
            models.Label.value,
        )
        .cte()
    )

    def groundtruth_label_query():
        return (
            db.query(groundtruths_cte.c.key, groundtruths_cte.c.value)
            .distinct()
            .all()
        )

    def prediction_label_query():
        return (
            db.query(predictions_cte.c.key, predictions_cte.c.value)
            .distinct()
            .all()
        )

    # get all labels
    groundtruth_labels = {
        (key, value) for key, value in groundtruth_label_query()
    }
    prediction_labels = {
        (key, value) for key, value in prediction_label_query()
    }
    labels = groundtruth_labels.union(prediction_labels)

    return (groundtruths_cte, predictions_cte, labels)


def _compute_clf_metrics(
    db: Session,
    groundtruth_filter: schemas.Filter,
    prediction_filter: schemas.Filter,
    pr_curve_max_examples: int,
    metrics_to_return: list[enums.MetricType],
    label_map: LabelMapType | None = None,
) -> list[
    schemas.ConfusionMatrix
    | schemas.AccuracyMetric
    | schemas.ROCAUCMetric
    | schemas.PrecisionMetric
    | schemas.RecallMetric
    | schemas.F1Metric
    | schemas.PrecisionRecallCurve
    | schemas.DetailedPrecisionRecallCurve
]:
    """
    Compute classification metrics.

    Parameters
    ----------
    db : Session
        The database Session to query against.
    groundtruth_filter : schemas.Filter
        The filter to be used to query groundtruths.
    prediction_filter : schemas.Filter
        The filter to be used to query predictions.
    metrics_to_return: list[MetricType]
        The list of metrics to compute, store, and return to the user.
    label_map: LabelMapType, optional
        Optional mapping of individual labels to a grouper label. Useful when you need to evaluate performance using labels that differ across datasets and models.
    pr_curve_max_examples: int
        The maximum number of datum examples to store per true positive, false negative, etc.

    Returns
    ----------
    list[ConfusionMatrix, Metric]
        A list of confusion matrices and metrics.
    """

    groundtruths, predictions, labels = _aggregate_data(
        db=db,
        groundtruth_filter=groundtruth_filter,
        prediction_filter=prediction_filter,
        label_map=label_map,
    )

    # compute metrics and confusion matrix for each grouper id
    confusion_matrices, metrics = _compute_confusion_matrices_and_metrics(
        db=db,
        groundtruths=groundtruths,
        predictions=predictions,
        labels=labels,
        pr_curve_max_examples=pr_curve_max_examples,
        metrics_to_return=metrics_to_return,
    )

    return confusion_matrices + metrics


@validate_computation
def compute_clf_metrics(
    *,
    db: Session,
    evaluation_id: int,
) -> int:
    """
    Create classification metrics. This function is intended to be run using FastAPI's `BackgroundTasks`.

    Parameters
    ----------
    db : Session
        The database Session to query against.
    evaluation_id : int
        The job ID to create metrics for.

    Returns
    ----------
    int
        The evaluation job id.
    """

    # fetch evaluation
    evaluation = core.fetch_evaluation_from_id(db, evaluation_id)

    # unpack filters and params
    parameters = schemas.EvaluationParameters(**evaluation.parameters)
    groundtruth_filter, prediction_filter = prepare_filter_for_evaluation(
        filters=schemas.Filter(**evaluation.filters),
        dataset_names=evaluation.dataset_names,
        model_name=evaluation.model_name,
        task_type=parameters.task_type,
    )

    log_evaluation_item_counts(
        db=db,
        evaluation=evaluation,
        prediction_filter=prediction_filter,
        groundtruth_filter=groundtruth_filter,
    )

    if parameters.metrics_to_return is None:
        raise RuntimeError("Metrics to return should always be defined here.")

    metrics = _compute_clf_metrics(
        db=db,
        prediction_filter=prediction_filter,
        groundtruth_filter=groundtruth_filter,
        label_map=parameters.label_map,
        pr_curve_max_examples=(
            parameters.pr_curve_max_examples
            if parameters.pr_curve_max_examples
            else 0
        ),
        metrics_to_return=parameters.metrics_to_return,
    )

    # add metrics to database
    commit_results(
        db=db,
        metrics=metrics,
        evaluation_id=evaluation.id,
    )

    log_evaluation_duration(
        evaluation=evaluation,
        db=db,
    )

    return evaluation_id
