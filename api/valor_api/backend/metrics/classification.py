import random
from collections import defaultdict
from typing import Sequence

import numpy as np
import pandas as pd
from sqlalchemy import CTE, Integer, Subquery, alias
from sqlalchemy.engine.base import Engine
from sqlalchemy.orm import Bundle, Session
from sqlalchemy.sql import and_, case, func, or_, select
from sqlalchemy.sql.selectable import NamedFromClause

from valor_api import enums, schemas
from valor_api.backend import core, models
from valor_api.backend.metrics.metric_utils import (
    commit_results,
    create_grouper_mappings,
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
    grouper_key: str,
    grouper_mappings: dict[str, dict[str, dict]],
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
    grouper_key: str
        The key of the grouper used to calculate the PR curves.
    grouper_mappings: dict[str, dict[str, dict]]
        A dictionary of mappings that connect groupers to their related labels.
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

    pr_output = defaultdict(lambda: defaultdict(dict))
    detailed_pr_output = defaultdict(lambda: defaultdict(dict))

    label_keys = grouper_mappings["grouper_key_to_label_keys_mapping"][
        grouper_key
    ]

    # create sets of all datums for which there is a prediction / groundtruth
    # used when separating misclassifications/no_predictions
    pd_datum_ids_to_high_score = {
        datum_id: high_score
        for datum_id, high_score in db.query(
            select(predictions.c.datum_id, func.max(predictions.c.score))
            .select_from(predictions)
            .join(models.Datum, models.Datum.id == predictions.c.datum_id)
            .join(
                models.Label,
                and_(
                    models.Label.id == predictions.c.label_id,
                    models.Label.key.in_(label_keys),
                ),
            )
            .group_by(predictions.c.datum_id)
            .subquery()
        ).all()
    }

    groundtruth_labels = alias(models.Label)
    prediction_labels = alias(models.Label)

    total_query = (
        select(
            case(
                grouper_mappings["label_value_to_grouper_value"],
                value=groundtruth_labels.c.value,
                else_=None,
            ).label("gt_label_value"),
            groundtruths.c.datum_id,
            case(
                grouper_mappings["label_value_to_grouper_value"],
                value=prediction_labels.c.value,
                else_=None,
            ).label("pd_label_value"),
            predictions.c.datum_id,
            groundtruths.c.dataset_name,
            models.Datum.uid.label("datum_uid"),
            predictions.c.score,
        )
        .select_from(groundtruths)
        .join(
            predictions,
            predictions.c.datum_id == groundtruths.c.datum_id,
            full=True,
        )
        .join(
            models.Datum,
            or_(
                models.Datum.id == groundtruths.c.datum_id,
                models.Datum.id == predictions.c.datum_id,
            ),
        )
        .join(
            groundtruth_labels,
            and_(
                groundtruth_labels.c.id == groundtruths.c.label_id,
                groundtruth_labels.c.key.in_(label_keys),
            ),
        )
        .join(
            prediction_labels,
            and_(
                prediction_labels.c.id == predictions.c.label_id,
                prediction_labels.c.key.in_(label_keys),
            ),
        )
        .subquery()
    )

    sorted_query = select(total_query).order_by(
        total_query.c.gt_label_value != total_query.c.pd_label_value,
        -total_query.c.score,
    )
    res = db.query(sorted_query.subquery()).all()

    for threshold in [x / 100 for x in range(5, 100, 5)]:

        for grouper_value in grouper_mappings["grouper_key_to_labels_mapping"][
            grouper_key
        ].keys():
            tp, tn, fp, fn = set(), set(), defaultdict(set), defaultdict(set)
            seen_datum_ids = set()

            for row in res:
                (
                    groundtruth_label,
                    gt_datum_id,
                    predicted_label,
                    pd_datum_id,
                    score,
                ) = (row[0], row[1], row[2], row[3], row[6])

                if (
                    groundtruth_label == grouper_value
                    and predicted_label == grouper_value
                    and score >= threshold
                ):
                    tp.add(pd_datum_id)
                    seen_datum_ids.add(pd_datum_id)
                elif predicted_label == grouper_value and score >= threshold:
                    # if there was a groundtruth for a given datum, then it was a misclassification
                    fp["misclassifications"].add(pd_datum_id)
                    seen_datum_ids.add(pd_datum_id)
                elif (
                    groundtruth_label == grouper_value
                    and gt_datum_id not in seen_datum_ids
                ):
                    # if there was a prediction for a given datum, then it was a misclassification
                    if (
                        gt_datum_id in pd_datum_ids_to_high_score
                        and pd_datum_ids_to_high_score[gt_datum_id]
                        >= threshold
                    ):
                        fn["misclassifications"].add(gt_datum_id)
                    else:
                        fn["no_predictions"].add(gt_datum_id)
                    seen_datum_ids.add(gt_datum_id)

            tn = set(unique_datums.keys()) - seen_datum_ids
            tp_cnt, fp_cnt, fn_cnt, tn_cnt = (
                len(tp),
                len(fp["misclassifications"]),
                len(fn["no_predictions"]) + len(fn["misclassifications"]),
                len(tn),
            )

            precision = (
                (tp_cnt) / (tp_cnt + fp_cnt) if (tp_cnt + fp_cnt) > 0 else -1
            )
            recall = (
                tp_cnt / (tp_cnt + fn_cnt) if (tp_cnt + fn_cnt) > 0 else -1
            )
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

            pr_output[grouper_value][threshold] = {
                "tp": tp_cnt,
                "fp": fp_cnt,
                "fn": fn_cnt,
                "tn": tn_cnt,
                "accuracy": accuracy,
                "precision": precision,
                "recall": recall,
                "f1_score": f1_score,
            }

            if (
                enums.MetricType.DetailedPrecisionRecallCurve
                in metrics_to_return
            ):
                tp = [unique_datums[datum_id] for datum_id in tp]
                fp = {
                    key: [unique_datums[datum_id] for datum_id in fp[key]]
                    for key in fp
                }
                tn = [unique_datums[datum_id] for datum_id in tn]
                fn = {
                    key: [unique_datums[datum_id] for datum_id in fn[key]]
                    for key in fn
                }

                detailed_pr_output[grouper_value][threshold] = {
                    "tp": {
                        "total": tp_cnt,
                        "observations": {
                            "all": {
                                "count": tp_cnt,
                                "examples": (
                                    random.sample(tp, pr_curve_max_examples)
                                    if len(tp) >= pr_curve_max_examples
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
                                    if len(tn) >= pr_curve_max_examples
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
                                    >= pr_curve_max_examples
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
                                    >= pr_curve_max_examples
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

    output = []

    output.append(
        schemas.PrecisionRecallCurve(
            label_key=grouper_key, value=dict(pr_output)
        ),
    )

    if enums.MetricType.DetailedPrecisionRecallCurve in metrics_to_return:
        output += [
            schemas.DetailedPrecisionRecallCurve(
                label_key=grouper_key, value=dict(detailed_pr_output)
            )
        ]

    return output


def _compute_roc_auc(
    db: Session,
    groundtruths: CTE,
    predictions: CTE,
    grouper_key: str,
    grouper_mappings: dict[str, dict[str, dict]],
) -> float | None:
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
    grouper_key : str
        The key of the grouper to calculate the ROCAUC for.
    grouper_mappings: dict[str, dict[str, dict]]
        A dictionary of mappings that connect groupers to their related labels.

    Returns
    -------
    float | None
        The ROC AUC. Returns None if no labels exist for that label_key.
    """

    groundtruths_per_label_kv_query = (
        select(
            models.Label.key,
            models.Label.value,
            func.count(groundtruths.c.id).label("gt_counts_per_label"),
        )
        .select_from(models.Label)
        .join(groundtruths, groundtruths.c.label_id == models.Label.id)
        .group_by(
            models.Label.key,
            models.Label.value,
        )
        .cte("gt_counts")
    )

    label_key_to_count = defaultdict(int)
    label_to_count = dict()
    filtered_label_set = set()
    for key, value, count in db.query(groundtruths_per_label_kv_query).all():
        label_key_to_count[key] += count
        label_to_count[schemas.Label(key=key, value=value)] = count
        filtered_label_set.add(schemas.Label(key=key, value=value))

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

    groundtruth_labels = alias(models.Label)
    prediction_labels = alias(models.Label)

    basic_counts_query = (
        select(
            predictions.c.score,
            (groundtruth_labels.c.value == prediction_labels.c.value)
            .cast(Integer)
            .label("is_true_positive"),
            (groundtruth_labels.c.value != prediction_labels.c.value)
            .cast(Integer)
            .label("is_false_positive"),
            groundtruth_labels.c.key.label("label_key"),
            prediction_labels.c.value.label("prediction_label_value"),
        )
        .select_from(predictions)
        .join(
            groundtruths,
            groundtruths.c.datum_id == predictions.c.datum_id,
        )
        .join(
            groundtruth_labels,
            groundtruth_labels.c.id == groundtruths.c.label_id,
        )
        .join(
            prediction_labels,
            and_(
                prediction_labels.c.id == predictions.c.label_id,
                prediction_labels.c.key == groundtruth_labels.c.key,
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
        basic_counts_query.c.score,
        cumulative_tp.label("cumulative_tp"),
        cumulative_fp.label("cumulative_fp"),
        basic_counts_query.c.label_key,
        basic_counts_query.c.prediction_label_value,
    ).subquery("tpr_fpr_cumulative")

    tpr_fpr_rates = (
        select(
            tpr_fpr_cumulative.c.score,
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
        select(
            trap_areas.c.label_key,
            trap_areas.c.prediction_label_value,
            func.sum(trap_areas.c.trap_area),
        )
        .group_by(
            trap_areas.c.label_key,
            trap_areas.c.prediction_label_value,
        )
        .subquery()
    )
    map_label_to_result = {
        schemas.Label(key=key, value=value): rocauc
        for key, value, rocauc in db.query(results).all()
    }

    # get all of the labels associated with the grouper
    value_to_labels_mapping = grouper_mappings[
        "grouper_key_to_labels_mapping"
    ][grouper_key]

    sum_roc_aucs = 0
    label_count = 0
    for _, label_rows in value_to_labels_mapping.items():
        for label_row in label_rows:

            label = schemas.Label(key=label_row.key, value=label_row.value)

            # some labels in the "labels" argument may be out-of-scope given our groundtruth_filter, so we fetch all labels that are within scope of the groundtruth_filter to make sure we don't calculate ROCAUC for inappropriate labels
            if label not in filtered_label_set:
                continue
            elif (
                label not in label_to_count
                or label.key not in label_key_to_count
            ):
                raise RuntimeError(
                    f"ROC AUC computation failed as the label `{label}` could not be found."
                )

            if label_to_count[label] == 0:
                ret = 0.0
            elif (
                label_key_to_count[label_row.key] - label_to_count[label] == 0
            ):
                ret = 1.0
            else:
                ret = map_label_to_result.get(label, np.nan)

            if ret is not None:
                sum_roc_aucs += float(ret)
                label_count += 1

    return sum_roc_aucs / label_count if label_count else None


def _compute_confusion_matrix_at_grouper_key(
    db: Session,
    predictions: Subquery | NamedFromClause,
    groundtruths: Subquery | NamedFromClause,
    grouper_key: str,
    grouper_mappings: dict[str, dict[str, dict]],
) -> schemas.ConfusionMatrix | None:
    """
    Computes the confusion matrix at a label_key.

    Parameters
    ----------
    db : Session
        The database Session to query against.
    prediction_filter : schemas.Filter
        The filter to be used to query predictions.
    groundtruth_filter : schemas.Filter
        The filter to be used to query groundtruths.
    grouper_key: str
        The key of the grouper used to calculate the confusion matrix.
    grouper_mappings: dict[str, dict[str, dict]]
        A dictionary of mappings that connect groupers to their related labels.

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
            func.max(predictions.c.score).label("max_score"),
            models.Annotation.datum_id.label("datum_id"),
        )
        .join(
            models.Annotation,
            models.Annotation.id == predictions.c.annotation_id,
        )
        .group_by(models.Annotation.datum_id)
        .alias()
    )

    # 2. Remove duplicate scores per datum
    # used for the edge case where the max confidence appears twice
    # the result of this query is all of the hard predictions
    min_id_query = (
        select(
            func.min(predictions.c.id).label("min_id"),
            models.Annotation.datum_id.label("datum_id"),
        )
        .select_from(predictions)
        .join(
            models.Annotation,
            models.Annotation.id == predictions.c.annotation_id,
        )
        .join(
            max_scores_by_datum_id,
            and_(
                models.Annotation.datum_id
                == max_scores_by_datum_id.c.datum_id,
                predictions.c.score == max_scores_by_datum_id.c.max_score,
            ),
        )
        .group_by(models.Annotation.datum_id)
        .alias()
    )

    # 3. Get labels for hard predictions, organize per datum
    hard_preds_query = (
        select(
            models.Label.value.label("pd_label_value"),
            min_id_query.c.datum_id.label("datum_id"),
        )
        .select_from(min_id_query)
        .join(
            models.Prediction,
            models.Prediction.id == min_id_query.c.min_id,
        )
        .join(
            models.Label,
            models.Label.id == models.Prediction.label_id,
        )
        .alias()
    )

    # 4. Link each label value to its corresponding grouper value
    b = Bundle(
        "cols",
        case(
            grouper_mappings["label_value_to_grouper_value"],
            value=hard_preds_query.c.pd_label_value,
        ),
        case(
            grouper_mappings["label_value_to_grouper_value"],
            value=models.Label.value,
        ),
    )

    # 5. Generate confusion matrix
    total_query = (
        select(b, func.count())
        .select_from(hard_preds_query)
        .join(
            groundtruths,
            groundtruths.c.datum_id == hard_preds_query.c.datum_id,
        )
        .join(
            models.Label,
            models.Label.id == groundtruths.c.label_id,
        )
        .group_by(b)  # type: ignore - SQLAlchemy Bundle typing issue
    )

    res = db.execute(total_query).all()
    if len(res) == 0:
        # this means there's no predictions and groundtruths with the label key
        # for the same image
        return None

    return schemas.ConfusionMatrix(
        label_key=grouper_key,
        entries=[
            schemas.ConfusionMatrixEntry(
                prediction=r[0][0], groundtruth=r[0][1], count=r[1]
            )
            for r in res
        ],
    )


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


def _add_columns_to_groundtruth_and_prediction_table(
    df: pd.DataFrame, grouper_mappings: dict
) -> None:
    """Add label, grouper_key, and grouper_value columns to a particular dataframe. Modifies the dataframe in place."""

    df["label"] = df.apply(
        lambda chain_df: (chain_df["label_key"], chain_df["label_value"]),
        axis=1,
    )
    df["grouper_key"] = df["label"].map(
        grouper_mappings["label_to_grouper_key_mapping"]
    )
    df["grouper_value"] = df["label_value"].map(
        grouper_mappings["label_value_to_grouper_value"]
    )


def _calculate_confusion_matrix_df(
    merged_groundtruths_and_predictions_df: pd.DataFrame,
) -> tuple:
    """Calculate our confusion matrix dataframe."""

    cm_counts_df = (
        merged_groundtruths_and_predictions_df[
            ["grouper_key", "pd_grouper_value", "gt_grouper_value"]
        ]
        .groupby(
            ["grouper_key", "pd_grouper_value", "gt_grouper_value"],
            as_index=False,
            dropna=False,
        )
        .size()
    )

    cm_counts_df["true_positive_flag"] = (
        cm_counts_df["pd_grouper_value"] == cm_counts_df["gt_grouper_value"]
    )

    # resolve pandas typing error
    assert isinstance(cm_counts_df, pd.DataFrame)

    # count of predictions per grouper key
    cm_counts_df = cm_counts_df.merge(
        cm_counts_df.groupby(
            ["grouper_key", "pd_grouper_value"],
            as_index=False,
            dropna=False,
        )
        .size()
        .rename({"size": "number_of_predictions"}, axis=1),
        on=["grouper_key", "pd_grouper_value"],
    )

    # count of groundtruths per grouper key
    cm_counts_df = cm_counts_df.merge(
        cm_counts_df.groupby(
            ["grouper_key", "gt_grouper_value"],
            as_index=False,
            dropna=False,
        )
        .size()
        .rename({"size": "number_of_groundtruths"}, axis=1),
    )

    cm_counts_df = cm_counts_df.merge(
        cm_counts_df[
            [
                "grouper_key",
                "pd_grouper_value",
                "true_positive_flag",
            ]
        ]
        .groupby(
            ["grouper_key", "pd_grouper_value"],
            as_index=False,
            dropna=False,
        )
        .sum()
        .rename(
            columns={
                "true_positive_flag": "true_positives_per_pd_grouper_value"
            }
        ),
        on=["grouper_key", "pd_grouper_value"],
    )

    cm_counts_df = cm_counts_df.merge(
        cm_counts_df[["grouper_key", "gt_grouper_value", "true_positive_flag"]]
        .groupby(
            ["grouper_key", "gt_grouper_value"],
            as_index=False,
            dropna=False,
        )
        .sum()
        .rename(
            columns={
                "true_positive_flag": "true_positives_per_gt_grouper_value"
            }
        ),
        on=["grouper_key", "gt_grouper_value"],
    )

    cm_counts_df = cm_counts_df.merge(
        cm_counts_df[["grouper_key", "true_positive_flag"]]
        .groupby("grouper_key", as_index=False, dropna=False)
        .sum()
        .rename(
            columns={"true_positive_flag": "true_positives_per_grouper_key"}
        ),
        on="grouper_key",
    )

    # create ConfusionMatrix objects
    confusion_matrices = []
    for grouper_key in cm_counts_df.loc[:, "grouper_key"].unique():
        revelant_rows = cm_counts_df.loc[
            (cm_counts_df["grouper_key"] == grouper_key)
            & cm_counts_df["gt_grouper_value"].notnull()
        ]
        relevant_confusion_matrices = schemas.ConfusionMatrix(
            label_key=grouper_key,
            entries=[
                schemas.ConfusionMatrixEntry(
                    prediction=row["pd_grouper_value"],
                    groundtruth=row["gt_grouper_value"],
                    count=row["size"],
                )
                for row in revelant_rows.to_dict(orient="records")
                if isinstance(row["pd_grouper_value"], str)
                and isinstance(row["gt_grouper_value"], str)
            ],
        )
        confusion_matrices.append(relevant_confusion_matrices)

    return cm_counts_df, confusion_matrices


def _calculate_metrics_at_grouper_value_level(
    cm_counts_df: pd.DataFrame,
) -> pd.DataFrame:
    """Calculate metrics using the confusion matix dataframe."""

    # create base dataframe that's unique at the (grouper key, grouper value level)
    unique_grouper_values_per_grouper_key_df = pd.DataFrame(
        np.concatenate(
            [
                cm_counts_df[["grouper_key", "pd_grouper_value"]].values,
                cm_counts_df.loc[
                    cm_counts_df["gt_grouper_value"].notnull(),
                    ["grouper_key", "gt_grouper_value"],
                ].values,
            ]
        ),
        columns=[
            "grouper_key",
            "grouper_value",
        ],
    ).drop_duplicates()

    # compute metrics using confusion matrices
    metrics_per_grouper_key_and_grouper_value_df = (
        unique_grouper_values_per_grouper_key_df.assign(
            number_true_positives=lambda df: df.apply(
                lambda chain_df: (
                    cm_counts_df[
                        (
                            cm_counts_df["gt_grouper_value"]
                            == chain_df["grouper_value"]
                        )
                        & (
                            cm_counts_df["grouper_key"]
                            == chain_df["grouper_key"]
                        )
                        & (cm_counts_df["true_positive_flag"])
                    ]["size"].sum()
                ),
                axis=1,
            )
        )
        .assign(
            number_of_groundtruths=unique_grouper_values_per_grouper_key_df.apply(
                lambda chain_df: (
                    cm_counts_df[
                        (
                            cm_counts_df["gt_grouper_value"]
                            == chain_df["grouper_value"]
                        )
                        & (
                            cm_counts_df["grouper_key"]
                            == chain_df["grouper_key"]
                        )
                    ]["size"].sum()
                ),
                axis=1,
            )
        )
        .assign(
            number_of_predictions=unique_grouper_values_per_grouper_key_df.apply(
                lambda chain_df: (
                    cm_counts_df[
                        (
                            cm_counts_df["pd_grouper_value"]
                            == chain_df["grouper_value"]
                        )
                        & (
                            cm_counts_df["grouper_key"]
                            == chain_df["grouper_key"]
                        )
                    ]["size"].sum()
                ),
                axis=1,
            )
        )
        .assign(
            precision=lambda chain_df: chain_df["number_true_positives"]
            / chain_df["number_of_predictions"]
        )
        .assign(
            recall=lambda chain_df: chain_df["number_true_positives"]
            / chain_df["number_of_groundtruths"]
        )
        .assign(
            f1=lambda chain_df: (
                2 * chain_df["precision"] * chain_df["recall"]
            )
            / (chain_df["precision"] + chain_df["recall"])
        )
    )

    # replace nulls and infinities
    metrics_per_grouper_key_and_grouper_value_df[
        ["precision", "recall", "f1"]
    ] = metrics_per_grouper_key_and_grouper_value_df.loc[
        :, ["precision", "recall", "f1"]
    ].replace(
        [np.inf, -np.inf, np.nan], 0
    )

    # replace values of labels that only exist in predictions (not groundtruths) with -1
    labels_to_replace = cm_counts_df.loc[
        cm_counts_df["gt_grouper_value"].isnull(),
        ["grouper_key", "pd_grouper_value"],
    ].values.tolist()

    for key, value in labels_to_replace:
        metrics_per_grouper_key_and_grouper_value_df.loc[
            (
                metrics_per_grouper_key_and_grouper_value_df["grouper_key"]
                == key
            )
            & (
                metrics_per_grouper_key_and_grouper_value_df["grouper_value"]
                == value
            ),
            ["precision", "recall", "f1"],
        ] = -1

    return metrics_per_grouper_key_and_grouper_value_df


def _calculate_precision_recall_f1_metrics(
    metrics_per_grouper_key_and_grouper_value_df: pd.DataFrame,
) -> list[schemas.PrecisionMetric | schemas.RecallMetric | schemas.F1Metric]:
    # create metric objects
    output = []

    for row in metrics_per_grouper_key_and_grouper_value_df.loc[
        ~metrics_per_grouper_key_and_grouper_value_df[
            "grouper_value"
        ].isnull(),
        ["grouper_key", "grouper_value", "precision", "recall", "f1"],
    ].to_dict(orient="records"):
        pydantic_label = schemas.Label(
            key=row["grouper_key"], value=row["grouper_value"]
        )

        output += [
            schemas.PrecisionMetric(
                label=pydantic_label,
                value=row["precision"],
            ),
            schemas.RecallMetric(
                label=pydantic_label,
                value=row["recall"],
            ),
            schemas.F1Metric(
                label=pydantic_label,
                value=row["f1"],
            ),
        ]
    return output


def _calculate_accuracy_metrics(
    cm_counts_df: pd.DataFrame,
) -> list[schemas.AccuracyMetric]:

    accuracy_calculations = (
        cm_counts_df.loc[
            (
                cm_counts_df["gt_grouper_value"].notnull()
                & cm_counts_df["true_positive_flag"]
            ),
            ["grouper_key", "size"],
        ]
        .groupby(["grouper_key"], as_index=False)
        .sum()
        .rename({"size": "true_positives_per_grouper_key"}, axis=1)
    ).merge(
        cm_counts_df.loc[
            (cm_counts_df["gt_grouper_value"].notnull()),
            ["grouper_key", "size"],
        ]
        .groupby(["grouper_key"], as_index=False)
        .sum()
        .rename({"size": "observations_per_grouper_key"}, axis=1),
        on="grouper_key",
        how="outer",
    )

    accuracy_calculations["accuracy"] = (
        accuracy_calculations["true_positives_per_grouper_key"]
        / accuracy_calculations["observations_per_grouper_key"]
    )

    # some elements may be np.nan if a given grouper key has no true positives
    # replace those accuracy scores with 0
    accuracy_calculations["accuracy"] = accuracy_calculations[
        "accuracy"
    ].fillna(value=0)

    return [
        schemas.AccuracyMetric(
            label_key=values["grouper_key"], value=values["accuracy"]
        )
        for _, values in accuracy_calculations.iterrows()
    ]


def _get_merged_dataframe(pd_df: pd.DataFrame, gt_df: pd.DataFrame):
    max_scores_by_grouper_key_and_datum_id = (
        pd_df[["grouper_key", "datum_id", "score"]]
        .groupby(
            [
                "grouper_key",
                "datum_id",
            ],
            as_index=False,
        )
        .max()
    )

    # catch pandas typing error
    if not isinstance(pd_df, pd.DataFrame) or not isinstance(
        max_scores_by_grouper_key_and_datum_id, pd.DataFrame
    ):
        raise ValueError

    best_prediction_id_per_grouper_key_and_datum_id = (
        pd.merge(
            pd_df,
            max_scores_by_grouper_key_and_datum_id,
            on=["grouper_key", "datum_id", "score"],
            how="inner",
        )[["grouper_key", "datum_id", "id", "score"]]
        .groupby(["grouper_key", "datum_id"], as_index=False)
        .min()
        .rename(columns={"score": "best_score"})
    )

    best_prediction_label_for_each_grouper_key_and_datum = pd.merge(
        pd_df[["grouper_key", "grouper_value", "datum_id", "id"]],
        best_prediction_id_per_grouper_key_and_datum_id,
        on=["grouper_key", "datum_id", "id"],
        how="inner",
    )[["grouper_key", "datum_id", "grouper_value", "best_score"]]

    # count the number of matches for each (pd_label_value, gt_label_value) for each grouper_key
    merged_groundtruths_and_predictions_df = pd.merge(
        gt_df[["datum_id", "grouper_key", "grouper_value"]].rename(
            columns={"grouper_value": "gt_grouper_value"}
        ),
        best_prediction_label_for_each_grouper_key_and_datum.rename(
            columns={"grouper_value": "pd_grouper_value"}
        ),
        on=["datum_id", "grouper_key"],
        how="left",
    )

    # add back any labels that appear in predictions but not groundtruths
    missing_grouper_labels_from_predictions = list(
        set(
            zip(
                [None] * len(pd_df),
                pd_df["grouper_key"],
                [None] * len(pd_df),
                pd_df["grouper_value"],
                [None] * len(pd_df),
            )
        ).difference(
            set(
                zip(
                    [None] * len(merged_groundtruths_and_predictions_df),
                    merged_groundtruths_and_predictions_df["grouper_key"],
                    [None] * len(merged_groundtruths_and_predictions_df),
                    merged_groundtruths_and_predictions_df["pd_grouper_value"],
                    [None] * len(pd_df),
                )
            ).union(
                set(
                    zip(
                        [None] * len(merged_groundtruths_and_predictions_df),
                        merged_groundtruths_and_predictions_df["grouper_key"],
                        [None] * len(merged_groundtruths_and_predictions_df),
                        merged_groundtruths_and_predictions_df[
                            "gt_grouper_value"
                        ],
                        [None] * len(pd_df),
                    )
                )
            )
        )
    )

    merged_groundtruths_and_predictions_df = pd.concat(
        [
            merged_groundtruths_and_predictions_df,
            pd.DataFrame(
                missing_grouper_labels_from_predictions,
                columns=merged_groundtruths_and_predictions_df.columns,
            ),
        ],
        ignore_index=True,
    )

    return merged_groundtruths_and_predictions_df


def _calculate_rocauc(
    pd_df: pd.DataFrame, gt_df: pd.DataFrame
) -> list[schemas.ROCAUCMetric]:

    # if there are no predictions, then ROCAUC should be 0 for all groundtruth grouper keys
    if pd_df.empty:
        return [
            schemas.ROCAUCMetric(label_key=grouper_key, value=0)
            for grouper_key in gt_df["grouper_key"].unique()
        ]

    merged_predictions_and_groundtruths = (
        pd_df[["datum_id", "grouper_key", "grouper_value", "score"]]
        .merge(
            gt_df[["datum_id", "grouper_key", "grouper_value"]].rename(
                columns={
                    "grouper_value": "gt_grouper_value",
                }
            ),
            on=["datum_id", "grouper_key"],
            how="left",
        )
        .assign(
            is_true_positive=lambda chain_df: chain_df["grouper_value"]
            == chain_df["gt_grouper_value"],
        )
        .assign(
            is_false_positive=lambda chain_df: chain_df["grouper_value"]
            != chain_df["gt_grouper_value"],
        )
    ).sort_values(
        by=["score", "grouper_key", "gt_grouper_value"],
        ascending=[False, False, True],
    )

    # count the number of observations (i.e., predictions) and true positives for each grouper key
    total_observations_per_grouper_key_and_grouper_value = (
        merged_predictions_and_groundtruths.groupby(
            ["grouper_key", "grouper_value"], as_index=False
        )["gt_grouper_value"]
        .size()
        .rename({"size": "n"}, axis=1)
    )

    total_true_positves_per_grouper_key_and_grouper_value = (
        merged_predictions_and_groundtruths.loc[
            merged_predictions_and_groundtruths["is_true_positive"], :
        ]
        .groupby(["grouper_key", "grouper_value"], as_index=False)[
            "gt_grouper_value"
        ]
        .size()
        .rename({"size": "n_true_positives"}, axis=1)
    )

    merged_counts = merged_predictions_and_groundtruths.merge(
        total_observations_per_grouper_key_and_grouper_value,
        on=["grouper_key", "grouper_value"],
        how="left",
    ).merge(
        total_true_positves_per_grouper_key_and_grouper_value,
        on=["grouper_key", "grouper_value"],
        how="left",
    )

    cumulative_sums = (
        merged_counts[
            [
                "grouper_key",
                "grouper_value",
                "is_true_positive",
                "is_false_positive",
            ]
        ]
        .groupby(["grouper_key", "grouper_value"], as_index=False)
        .cumsum()
    ).rename(
        columns={
            "is_true_positive": "cum_true_positive_cnt",
            "is_false_positive": "cum_false_positive_cnt",
        }
    )

    rates = pd.concat([merged_counts, cumulative_sums], axis=1)

    # correct cumulative sums to be the max value for a given datum_id / grouper_key / grouper_value (this logic brings pandas' cumsum logic in line with psql's sum().over())
    max_cum_sums = (
        rates.groupby(
            ["grouper_key", "grouper_value", "score"], as_index=False
        )[["cum_true_positive_cnt", "cum_false_positive_cnt"]]
        .max()
        .rename(
            columns={
                "cum_true_positive_cnt": "max_cum_true_positive_cnt",
                "cum_false_positive_cnt": "max_cum_false_positive_cnt",
            }
        )
    )
    rates = rates.merge(
        max_cum_sums, on=["grouper_key", "grouper_value", "score"]
    )
    rates["cum_true_positive_cnt"] = rates[
        ["cum_true_positive_cnt", "max_cum_true_positive_cnt"]
    ].max(axis=1)
    rates["cum_false_positive_cnt"] = rates[
        ["cum_false_positive_cnt", "max_cum_false_positive_cnt"]
    ].max(axis=1)

    # calculate tpr and fpr
    rates = rates.assign(
        tpr=lambda chain_df: chain_df["cum_true_positive_cnt"]
        / chain_df["n_true_positives"]
    ).assign(
        fpr=lambda chain_df: chain_df["cum_false_positive_cnt"]
        / (chain_df["n"] - chain_df["n_true_positives"])
    )

    # sum trapezoidal areas by grouper key and grouper value
    trap_areas_per_grouper_value = pd.concat(
        [
            rates[
                [
                    "grouper_key",
                    "grouper_value",
                    "n",
                    "n_true_positives",
                    "tpr",
                    "fpr",
                ]
            ],
            rates.groupby(["grouper_key", "grouper_value"], as_index=False)[
                ["tpr", "fpr"]
            ]
            .shift(1)
            .rename(columns={"tpr": "lagged_tpr", "fpr": "lagged_fpr"}),
        ],
        axis=1,
    ).assign(
        trap_area=lambda chain_df: 0.5
        * (
            (chain_df["tpr"] + chain_df["lagged_tpr"])
            * (chain_df["fpr"] - chain_df["lagged_fpr"])
        )
    )

    summed_trap_areas_per_grouper_value = trap_areas_per_grouper_value.groupby(
        ["grouper_key", "grouper_value"], as_index=False
    )[["n", "n_true_positives", "trap_area"]].sum(min_count=1)

    # replace values if specific conditions are met
    summed_trap_areas_per_grouper_value = (
        summed_trap_areas_per_grouper_value.assign(
            trap_area=lambda chain_df: np.select(
                [
                    chain_df["n_true_positives"].isnull(),
                    ((chain_df["n"] - chain_df["n_true_positives"]) == 0),
                ],
                [1, 1],
                default=chain_df["trap_area"],
            )
        )
    )

    # take the average across grouper keys
    average_across_grouper_keys = summed_trap_areas_per_grouper_value.groupby(
        "grouper_key", as_index=False
    )["trap_area"].mean()

    return [
        schemas.ROCAUCMetric(
            label_key=values["grouper_key"], value=values["trap_area"]
        )
        for _, values in average_across_grouper_keys.iterrows()
    ]


def _get_datum_samples(
    pr_calc_df: pd.DataFrame,
    pr_curve_max_examples: int,
    flag_column: str,
    grouper_key: str,
    grouper_value: str,
    confidence_threshold: float,
) -> list:
    """Sample a dataframe to get geojsons based on input criteria."""
    samples = pr_calc_df[
        (pr_calc_df["grouper_key"] == grouper_key)
        & (
            (pr_calc_df["grouper_value_gt"] == grouper_value)
            | (pr_calc_df["grouper_value_pd"] == grouper_value)
        )
        & (pr_calc_df["confidence_threshold"] == confidence_threshold)
        & (pr_calc_df[flag_column])
    ]

    if samples.empty:
        samples = []
    else:
        samples = list(
            samples.sample(min(pr_curve_max_examples, len(samples)))[
                [
                    "dataset_name",
                    "datum_uid",
                ]
            ].itertuples(index=False, name=None)
        )
    return samples


def _calculate_pr_curves(
    pd_df: pd.DataFrame,
    gt_df: pd.DataFrame,
    metrics_to_return: list,
    pr_curve_max_examples: int,
):

    if not (
        enums.MetricType.PrecisionRecallCurve in metrics_to_return
        or enums.MetricType.DetailedPrecisionRecallCurve in metrics_to_return
    ):
        return []

    joint_df = (
        pd.merge(
            gt_df,
            pd_df,
            on=["dataset_name", "datum_id", "datum_uid", "grouper_key"],
            how="outer",
            suffixes=("_gt", "_pd"),
        )
        .assign(
            is_label_match=lambda chain_df: (
                (chain_df["grouper_value_pd"] == chain_df["grouper_value_gt"])
            )
        )
        .drop(
            columns=[
                "annotation_id_gt",
                "created_at_gt",
                "label_key_gt",
                "label_value_gt",
                "label_gt",
                "label_id_gt",
                "annotation_id_pd",
                "created_at_pd",
                "label_key_pd",
                "label_value_pd",
                "label_pd",
                "label_id_pd",
            ]
        )
    )

    # add confidence_threshold to the dataframe and sort
    pr_calc_df = pd.concat(
        [
            joint_df.assign(confidence_threshold=threshold)
            for threshold in [x / 100 for x in range(5, 100, 5)]
        ],
        ignore_index=True,
    ).sort_values(
        by=[
            "grouper_key",
            "grouper_value_pd",
            "confidence_threshold",
            "score",
        ],
        ascending=False,
    )

    # create flags where the predictions meet criteria
    pr_calc_df["true_positive_flag"] = (
        pr_calc_df["score"] >= pr_calc_df["confidence_threshold"]
    ) & pr_calc_df["is_label_match"]

    # for all the false positives, we consider them to be a misclassification if they share a key but not a value with a gt
    pr_calc_df["misclassification_false_positive_flag"] = (
        pr_calc_df["score"] >= pr_calc_df["confidence_threshold"]
    ) & ~pr_calc_df["is_label_match"]

    # any prediction IDs that aren't true positives or misclassification false positives are hallucination false positives
    pr_calc_df["hallucination_false_positive_flag"] = (
        pr_calc_df["score"] >= pr_calc_df["confidence_threshold"]
    )

    predictions_associated_with_tps_or_misclassification_fps = (
        pr_calc_df.groupby(["confidence_threshold", "id_pd"], as_index=False)
        .filter(
            lambda x: x["true_positive_flag"].any()
            or x["misclassification_false_positive_flag"].any()
        )
        .groupby(["confidence_threshold"], as_index=False)["id_pd"]
        .unique()
    )
    if not predictions_associated_with_tps_or_misclassification_fps.empty:
        predictions_associated_with_tps_or_misclassification_fps.columns = [
            "confidence_threshold",
            "misclassification_fp_pd_ids",
        ]
        pr_calc_df = pr_calc_df.merge(
            predictions_associated_with_tps_or_misclassification_fps,
            on=["confidence_threshold"],
            how="left",
        )
        pr_calc_df["misclassification_fp_pd_ids"] = pr_calc_df[
            "misclassification_fp_pd_ids"
        ].map(lambda x: x if isinstance(x, np.ndarray) else [])

        id_pd_in_set = pr_calc_df.apply(
            lambda row: (row["id_pd"] in row["misclassification_fp_pd_ids"]),
            axis=1,
        )
        pr_calc_df.loc[
            (id_pd_in_set) & (pr_calc_df["hallucination_false_positive_flag"]),
            "hallucination_false_positive_flag",
        ] = False

    # next, we flag false negatives by declaring any groundtruth that isn't associated with a true positive to be a false negative
    fn_gt_ids = (
        pr_calc_df.groupby(["confidence_threshold", "id_gt"], as_index=False)
        .filter(lambda x: ~x["true_positive_flag"].any())
        .groupby(["confidence_threshold"], as_index=False)["id_gt"]
        .unique()
    )
    fn_gt_ids.columns = ["confidence_threshold", "false_negative_ids"]

    pr_calc_df = pr_calc_df.merge(
        fn_gt_ids, on=["confidence_threshold"], how="left"
    )

    pr_calc_df["false_negative_flag"] = pr_calc_df.apply(
        lambda row: row["id_gt"] in row["false_negative_ids"], axis=1
    )

    # it's a misclassification if there is a corresponding misclassification false positive
    pr_calc_df["misclassification_false_negative_flag"] = (
        pr_calc_df["misclassification_false_positive_flag"]
        & pr_calc_df["false_negative_flag"]
    )

    # assign all id_gts that aren't misclassifications but are false negatives to be no_predictions
    no_predictions_fn_gt_ids = (
        pr_calc_df.groupby(["confidence_threshold", "id_gt"], as_index=False)
        .filter(lambda x: ~x["misclassification_false_negative_flag"].any())
        .groupby(["confidence_threshold"], as_index=False)["id_gt"]
        .unique()
    )
    no_predictions_fn_gt_ids.columns = [
        "confidence_threshold",
        "no_predictions_fn_gt_ids",
    ]

    pr_calc_df = pr_calc_df.merge(
        no_predictions_fn_gt_ids, on=["confidence_threshold"], how="left"
    )

    pr_calc_df["no_predictions_false_negative_flag"] = pr_calc_df.apply(
        lambda row: (row["false_negative_flag"])
        & (row["id_gt"] in row["no_predictions_fn_gt_ids"]),
        axis=1,
    )

    # true negatives are any rows which don't have another flag
    pr_calc_df["true_negative_flag"] = (
        ~pr_calc_df["true_positive_flag"]
        & ~pr_calc_df["false_negative_flag"]
        & ~pr_calc_df["misclassification_false_positive_flag"]
        & ~pr_calc_df["hallucination_false_positive_flag"]
    )

    # next, we sum up the occurences of each classification and merge them together into one dataframe
    true_positives = (
        pr_calc_df[pr_calc_df["true_positive_flag"]]
        .groupby(["grouper_key", "grouper_value_pd", "confidence_threshold"])[
            "id_pd"
        ]
        .nunique()
    )
    true_positives.name = "true_positives"

    hallucination_false_positives = (
        pr_calc_df[pr_calc_df["hallucination_false_positive_flag"]]
        .groupby(["grouper_key", "grouper_value_pd", "confidence_threshold"])[
            "id_pd"
        ]
        .nunique()
    )
    hallucination_false_positives.name = "hallucinations_false_positives"

    misclassification_false_positives = (
        pr_calc_df[pr_calc_df["misclassification_false_positive_flag"]]
        .groupby(["grouper_key", "grouper_value_pd", "confidence_threshold"])[
            "id_pd"
        ]
        .nunique()
    )
    misclassification_false_positives.name = (
        "misclassification_false_positives"
    )

    misclassification_false_negatives = (
        pr_calc_df[pr_calc_df["misclassification_false_negative_flag"]]
        .groupby(["grouper_key", "grouper_value_gt", "confidence_threshold"])[
            "id_gt"
        ]
        .nunique()
    )
    misclassification_false_negatives.name = (
        "misclassification_false_negatives"
    )

    no_predictions_false_negatives = (
        pr_calc_df[pr_calc_df["no_predictions_false_negative_flag"]]
        .groupby(["grouper_key", "grouper_value_gt", "confidence_threshold"])[
            "id_gt"
        ]
        .nunique()
    )
    no_predictions_false_negatives.name = "no_predictions_false_negatives"

    # combine these outputs
    pr_curve_counts_df = (
        pd.concat(
            [
                pr_calc_df.loc[
                    ~pr_calc_df["grouper_value_pd"].isnull(),
                    [
                        "grouper_key",
                        "grouper_value_pd",
                        "confidence_threshold",
                    ],
                ].rename(columns={"grouper_value_pd": "grouper_value"}),
                pr_calc_df.loc[
                    ~pr_calc_df["grouper_value_gt"].isnull(),
                    [
                        "grouper_key",
                        "grouper_value_gt",
                        "confidence_threshold",
                    ],
                ].rename(columns={"grouper_value_gt": "grouper_value"}),
            ],
            axis=0,
        )
        .drop_duplicates()
        .merge(
            true_positives,
            left_on=[
                "grouper_key",
                "grouper_value",
                "confidence_threshold",
            ],
            right_index=True,
            how="outer",
        )
        .merge(
            hallucination_false_positives,
            left_on=[
                "grouper_key",
                "grouper_value",
                "confidence_threshold",
            ],
            right_index=True,
            how="outer",
        )
        .merge(
            misclassification_false_positives,
            left_on=[
                "grouper_key",
                "grouper_value",
                "confidence_threshold",
            ],
            right_index=True,
            how="outer",
        )
        .merge(
            misclassification_false_negatives,
            left_on=[
                "grouper_key",
                "grouper_value",
                "confidence_threshold",
            ],
            right_index=True,
            how="outer",
        )
        .merge(
            no_predictions_false_negatives,
            left_on=[
                "grouper_key",
                "grouper_value",
                "confidence_threshold",
            ],
            right_index=True,
            how="outer",
        )
    )

    # we're doing an outer join, so any nulls should be zeroes
    pr_curve_counts_df.fillna(0, inplace=True)

    # find all unique datums for use when identifying true negatives
    unique_datum_ids = set(pr_calc_df["datum_id"].unique())

    # calculate additional metrics
    pr_curve_counts_df["false_positives"] = (
        pr_curve_counts_df["misclassification_false_positives"]
        + pr_curve_counts_df["hallucinations_false_positives"]
    )
    pr_curve_counts_df["false_negatives"] = (
        pr_curve_counts_df["misclassification_false_negatives"]
        + pr_curve_counts_df["no_predictions_false_negatives"]
    )
    pr_curve_counts_df["true_negatives"] = len(unique_datum_ids) - (
        pr_curve_counts_df["true_positives"]
        + pr_curve_counts_df["false_positives"]
        + pr_curve_counts_df["false_negatives"]
    )
    pr_curve_counts_df["precision"] = pr_curve_counts_df["true_positives"] / (
        pr_curve_counts_df["true_positives"]
        + pr_curve_counts_df["false_positives"]
    )
    pr_curve_counts_df["recall"] = pr_curve_counts_df["true_positives"] / (
        pr_curve_counts_df["true_positives"]
        + pr_curve_counts_df["false_negatives"]
    )
    pr_curve_counts_df["accuracy"] = (
        pr_curve_counts_df["true_positives"]
        + pr_curve_counts_df["true_negatives"]
    ) / len(unique_datum_ids)
    pr_curve_counts_df["f1_score"] = (
        2 * pr_curve_counts_df["precision"] * pr_curve_counts_df["recall"]
    ) / (pr_curve_counts_df["precision"] + pr_curve_counts_df["recall"])

    # any NaNs that are left are from division by zero errors
    pr_curve_counts_df.fillna(-1, inplace=True)

    pr_output = defaultdict(lambda: defaultdict(dict))
    detailed_pr_output = defaultdict(lambda: defaultdict(dict))

    for _, row in pr_curve_counts_df.iterrows():
        pr_output[row["grouper_key"]][row["grouper_value"]][
            row["confidence_threshold"]
        ] = {
            "tp": row["true_positives"],
            "fp": row["false_positives"],
            "fn": row["false_negatives"],
            "tn": row["true_negatives"],
            "accuracy": row["accuracy"],
            "precision": row["precision"],
            "recall": row["recall"],
            "f1_score": row["f1_score"],
        }

        detailed_pr_output[row["grouper_key"]][row["grouper_value"]][
            row["confidence_threshold"]
        ] = {
            "tp": {
                "total": row["true_positives"],
                "observations": {
                    "all": {
                        "count": row["true_positives"],
                        "examples": _get_datum_samples(
                            pr_calc_df=pr_calc_df,
                            pr_curve_max_examples=pr_curve_max_examples,
                            flag_column="true_positive_flag",
                            grouper_key=row["grouper_key"],
                            grouper_value=row["grouper_value"],
                            confidence_threshold=row["confidence_threshold"],
                        ),
                    }
                },
            },
            "tn": {
                "total": row["true_negatives"],
                "observations": {
                    "all": {
                        "count": row["true_negatives"],
                        "examples": _get_datum_samples(
                            pr_calc_df=pr_calc_df,
                            pr_curve_max_examples=pr_curve_max_examples,
                            flag_column="true_negative_flag",
                            grouper_key=row["grouper_key"],
                            grouper_value=row["grouper_value"],
                            confidence_threshold=row["confidence_threshold"],
                        ),
                    }
                },
            },
            "fn": {
                "total": row["false_negatives"],
                "observations": {
                    "misclassifications": {
                        "count": row["misclassification_false_negatives"],
                        "examples": _get_datum_samples(
                            pr_calc_df=pr_calc_df,
                            pr_curve_max_examples=pr_curve_max_examples,
                            flag_column="misclassification_false_negative_flag",
                            grouper_key=row["grouper_key"],
                            grouper_value=row["grouper_value"],
                            confidence_threshold=row["confidence_threshold"],
                        ),
                    },
                    "no_predictions": {
                        "count": row["no_predictions_false_negatives"],
                        "examples": _get_datum_samples(
                            pr_calc_df=pr_calc_df,
                            pr_curve_max_examples=pr_curve_max_examples,
                            flag_column="no_predictions_false_negative_flag",
                            grouper_key=row["grouper_key"],
                            grouper_value=row["grouper_value"],
                            confidence_threshold=row["confidence_threshold"],
                        ),
                    },
                },
            },
            "fp": {
                "total": row["false_positives"],
                "observations": {
                    "misclassifications": {
                        "count": row["misclassification_false_positives"],
                        "examples": _get_datum_samples(
                            pr_calc_df=pr_calc_df,
                            pr_curve_max_examples=pr_curve_max_examples,
                            flag_column="misclassification_false_positive_flag",
                            grouper_key=row["grouper_key"],
                            grouper_value=row["grouper_value"],
                            confidence_threshold=row["confidence_threshold"],
                        ),
                    },
                },
            },
        }

    output = []

    if enums.MetricType.PrecisionRecallCurve in metrics_to_return:
        output += [
            schemas.PrecisionRecallCurve(label_key=key, value=dict(value))
            for key, value in pr_output.items()
        ]

    if enums.MetricType.DetailedPrecisionRecallCurve in metrics_to_return:
        output += [
            schemas.DetailedPrecisionRecallCurve(
                label_key=key, value=dict(value)
            )
            for key, value in detailed_pr_output.items()
        ]

    return output


def _compute_clf_metrics(
    db: Session,
    prediction_filter: schemas.Filter,
    groundtruth_filter: schemas.Filter,
    pr_curve_max_examples: int,
    metrics_to_return: list[enums.MetricType],
    label_map: LabelMapType | None = None,
) -> tuple[
    list[schemas.ConfusionMatrix],
    Sequence[
        schemas.ConfusionMatrix
        | schemas.AccuracyMetric
        | schemas.ROCAUCMetric
        | schemas.PrecisionMetric
        | schemas.RecallMetric
        | schemas.F1Metric
    ],
]:
    """
    Compute classification metrics.

    Parameters
    ----------
    db : Session
        The database Session to query against.
    prediction_filter : schemas.Filter
        The filter to be used to query predictions.
    groundtruth_filter : schemas.Filter
        The filter to be used to query groundtruths.
    metrics_to_return: list[MetricType]
        The list of metrics to compute, store, and return to the user.
    label_map: LabelMapType, optional
        Optional mapping of individual labels to a grouper label. Useful when you need to evaluate performance using labels that differ across datasets and models.
    pr_curve_max_examples: int
        The maximum number of datum examples to store per true positive, false negative, etc.


    Returns
    ----------
    Tuple[List[schemas.ConfusionMatrix], List[schemas.ConfusionMatrix | schemas.AccuracyMetric | schemas.ROCAUCMetric| schemas.PrecisionMetric | schemas.RecallMetric | schemas.F1Metric]]
        A tuple of confusion matrices and metrics.
    """

    labels = core.fetch_union_of_labels(
        db=db,
        lhs=groundtruth_filter,
        rhs=prediction_filter,
    )

    grouper_mappings = create_grouper_mappings(
        db=db,
        labels=labels,
        label_map=label_map,
        evaluation_type=enums.TaskType.CLASSIFICATION,
    )

    confusion_matrices, metrics_to_output = [], []

    groundtruths = generate_select(
        models.GroundTruth,
        models.Dataset.name.label("dataset_name"),
        models.Label.key.label("label_key"),
        models.Label.value.label("label_value"),
        models.Annotation.datum_id,
        models.Datum.uid.label("datum_uid"),
        filters=groundtruth_filter,
        label_source=models.GroundTruth,
    )

    predictions = generate_select(
        models.Prediction,
        models.Dataset.name.label("dataset_name"),
        models.Label.key.label("label_key"),
        models.Label.value.label("label_value"),
        models.Annotation.datum_id,
        models.Datum.uid.label("datum_uid"),
        filters=prediction_filter,
        label_source=models.Prediction,
    )
    assert isinstance(db.bind, Engine)
    gt_df = pd.read_sql(groundtruths, db.bind)
    pd_df = pd.read_sql(predictions, db.bind)

    _add_columns_to_groundtruth_and_prediction_table(
        df=gt_df, grouper_mappings=grouper_mappings
    )
    _add_columns_to_groundtruth_and_prediction_table(
        df=pd_df, grouper_mappings=grouper_mappings
    )

    merged_groundtruths_and_predictions_df = _get_merged_dataframe(
        pd_df=pd_df, gt_df=gt_df
    )

    cm_counts_df, confusion_matrices = _calculate_confusion_matrix_df(
        merged_groundtruths_and_predictions_df=merged_groundtruths_and_predictions_df
    )

    metrics_per_grouper_key_and_grouper_value_df = (
        _calculate_metrics_at_grouper_value_level(cm_counts_df=cm_counts_df)
    )

    metrics_to_output += _calculate_precision_recall_f1_metrics(
        metrics_per_grouper_key_and_grouper_value_df=metrics_per_grouper_key_and_grouper_value_df
    )

    metrics_to_output += _calculate_accuracy_metrics(cm_counts_df=cm_counts_df)

    metrics_to_output += _calculate_rocauc(pd_df=pd_df, gt_df=gt_df)

    metrics_to_output += _calculate_pr_curves(
        pd_df=pd_df,
        gt_df=gt_df,
        metrics_to_return=metrics_to_return,
        pr_curve_max_examples=pr_curve_max_examples,
    )

    return confusion_matrices, metrics_to_output


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

    confusion_matrices, metrics = _compute_clf_metrics(
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

    # add confusion matrices to database
    commit_results(
        db=db,
        metrics=confusion_matrices,
        evaluation_id=evaluation.id,
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
