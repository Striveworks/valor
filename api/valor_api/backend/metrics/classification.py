import random
from collections import defaultdict
from typing import Sequence

import numpy as np
import pandas as pd
from sqlalchemy import CTE, Integer, Subquery, alias
from sqlalchemy.orm import Bundle, Session
from sqlalchemy.sql import and_, case, func, or_, select
from sqlalchemy.sql.selectable import NamedFromClause

from valor_api import enums, schemas
from valor_api.backend import core, models
from valor_api.backend.metrics.metric_utils import (
    create_grouper_mappings,
    create_metric_mappings,
    get_or_create_row,
    log_evaluation_duration,
    log_evaluation_item_counts,
    prepare_filter_for_evaluation,
    validate_computation,
)
from valor_api.backend.query import generate_query, generate_select

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


def _compute_binary_roc_auc(
    db: Session,
    prediction_filter: schemas.Filter,
    groundtruth_filter: schemas.Filter,
    label: schemas.Label,
) -> float:
    """
    Computes the binary ROC AUC score of a dataset and label.

    Parameters
    ----------
    db : Session
        The database Session to query against.
    prediction_filter : schemas.Filter
        The filter to be used to query predictions.
    groundtruth_filter : schemas.Filter
        The filter to be used to query groundtruths.
    label : schemas.Label
        The label to compute the score for.

    Returns
    -------
    float
        The binary ROC AUC score.
    """
    # query to get the datum_ids and label values of groundtruths that have the given label key
    gts_filter = groundtruth_filter.model_copy()
    gts_filter.labels = schemas.LogicalFunction.and_(
        gts_filter.labels,
        schemas.Condition(
            lhs=schemas.Symbol(name=schemas.SupportedSymbol.LABEL_KEY),
            rhs=schemas.Value.infer(label.key),
            op=schemas.FilterOperator.EQ,
        ),
    )
    gts_query = generate_select(
        models.Annotation.datum_id.label("datum_id"),
        models.Label.value.label("label_value"),
        filters=gts_filter,
        label_source=models.GroundTruth,
    ).subquery("groundtruth_subquery")

    # get the prediction scores for the given label (key and value)
    preds_filter = prediction_filter.model_copy()
    preds_filter.labels = schemas.LogicalFunction.and_(
        preds_filter.labels,
        schemas.Condition(
            lhs=schemas.Symbol(name=schemas.SupportedSymbol.LABEL_KEY),
            rhs=schemas.Value.infer(label.key),
            op=schemas.FilterOperator.EQ,
        ),
        schemas.Condition(
            lhs=schemas.Symbol(name=schemas.SupportedSymbol.LABEL_VALUE),
            rhs=schemas.Value.infer(label.value),
            op=schemas.FilterOperator.EQ,
        ),
    )

    preds_query = generate_select(
        models.Annotation.datum_id.label("datum_id"),
        models.Prediction.score.label("score"),
        models.Label.value.label("label_value"),
        filters=preds_filter,
        label_source=models.Prediction,
    ).subquery("prediction_subquery")

    # number of ground truth labels that match the given label value
    n_pos = db.scalar(
        select(func.count(gts_query.c.label_value)).where(
            gts_query.c.label_value == label.value
        )
    )
    # total number of groundtruths
    n = db.scalar(select(func.count(gts_query.c.label_value)))

    if n is None or n_pos is None:
        raise RuntimeError(
            "ROCAUC computation failed; db.scalar returned None for mathematical variables."
        )

    if n_pos == 0:
        return 0

    if n - n_pos == 0:
        return 1.0

    basic_counts_query = (
        select(
            preds_query.c.datum_id,
            preds_query.c.score,
            (gts_query.c.label_value == label.value)
            .cast(Integer)
            .label("is_true_positive"),
            (gts_query.c.label_value != label.value)
            .cast(Integer)
            .label("is_false_positive"),
        )
        .select_from(
            preds_query.join(
                gts_query, preds_query.c.datum_id == gts_query.c.datum_id
            )
        )
        .alias("basic_counts")
    )

    tpr_fpr_cumulative = select(
        basic_counts_query.c.score,
        func.sum(basic_counts_query.c.is_true_positive)
        .over(order_by=basic_counts_query.c.score.desc())
        .label("cumulative_tp"),
        func.sum(basic_counts_query.c.is_false_positive)
        .over(order_by=basic_counts_query.c.score.desc())
        .label("cumulative_fp"),
    ).alias("tpr_fpr_cumulative")

    tpr_fpr_rates = select(
        tpr_fpr_cumulative.c.score,
        (tpr_fpr_cumulative.c.cumulative_tp / n_pos).label("tpr"),
        (tpr_fpr_cumulative.c.cumulative_fp / (n - n_pos)).label("fpr"),
    ).alias("tpr_fpr_rates")

    trap_areas = select(
        (
            0.5
            * (
                tpr_fpr_rates.c.tpr
                + func.lag(tpr_fpr_rates.c.tpr).over(
                    order_by=tpr_fpr_rates.c.score.desc()
                )
            )
            * (
                tpr_fpr_rates.c.fpr
                - func.lag(tpr_fpr_rates.c.fpr).over(
                    order_by=tpr_fpr_rates.c.score.desc()
                )
            )
        ).label("trap_area")
    ).subquery()

    ret = db.scalar(func.sum(trap_areas.c.trap_area))

    if ret is None:
        return np.nan

    return float(ret)


# TODO copy function notes to include helpful details
def _compute_roc_auc(
    db: Session,
    prediction_filter: schemas.Filter,
    groundtruth_filter: schemas.Filter,
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
    prediction_filter : schemas.Filter
        The filter to be used to query predictions.
    groundtruth_filter : schemas.Filter
        The filter to be used to query groundtruths.
    grouper_key : str
        The key of the grouper to calculate the ROCAUC for.
    grouper_mappings: dict[str, dict[str, dict]]
        A dictionary of mappings that connect groupers to their related labels.

    Returns
    -------
    float | None
        The ROC AUC. Returns None if no labels exist for that label_key.
    """

    # get all of the labels associated with the grouper
    value_to_labels_mapping = grouper_mappings[
        "grouper_key_to_labels_mapping"
    ][grouper_key]

    sum_roc_aucs = 0
    label_count = 0

    for grouper_value, labels in value_to_labels_mapping.items():
        label_filter = groundtruth_filter.model_copy()
        label_filter.labels = schemas.LogicalFunction.and_(
            label_filter.labels,
            schemas.LogicalFunction.or_(
                *[
                    schemas.Condition(
                        lhs=schemas.Symbol(
                            name=schemas.SupportedSymbol.LABEL_ID
                        ),
                        rhs=schemas.Value.infer(label.id),
                        op=schemas.FilterOperator.EQ,
                    )
                    for label in labels
                ]
            ),
        )

        # some labels in the "labels" argument may be out-of-scope given our groundtruth_filter, so we fetch all labels that are within scope of the groundtruth_filter to make sure we don't calculate ROCAUC for inappropriate labels
        in_scope_labels = [
            label
            for label in labels
            if schemas.Label(key=label.key, value=label.value)
            in core.get_labels(
                db=db, filters=label_filter, ignore_predictions=True
            )
        ]

        if not in_scope_labels:
            continue

        for label in labels:
            bin_roc = _compute_binary_roc_auc(
                db=db,
                prediction_filter=prediction_filter,
                groundtruth_filter=groundtruth_filter,
                label=schemas.Label(key=label.key, value=label.value),
            )

            if bin_roc is not None:
                sum_roc_aucs += bin_roc
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

    def _add_columns_to_groundtruth_and_prediction_table(df: pd.DataFrame):
        """Derive label, grouper_key, and grouper_value columns for a particular dataframe. Modifies the dataframe in place."""

        df["label"] = df.apply(
            lambda chain_df: (chain_df["label_key"], chain_df["label_value"]),
            axis=1,
        )
        df["grouper_key"] = df["label"].map(
            grouper_mappings[
                "label_to_grouper_key_mapping"
            ]  # pyright: ignore - pandas typing error. the dictionary types are unknown here, though we know that they are actually dict[str, str]
        )
        df["grouper_value"] = df["label_value"].map(
            grouper_mappings[
                "label_value_to_grouper_value"
            ]  # pyright: ignore - pandas typing error. the dictionary types are unknown here, though we know that they are actually dict[str, str]
        )

    def _calculate_confusion_matrix_df(
        merged_groundtruths_and_predictions: pd.DataFrame,
    ):
        """Calculate our confusion matrix dataframe."""

        # TODO probably don't use assign here
        cm_counts = (
            merged_groundtruths_and_predictions[
                ["grouper_key", "pd_grouper_value", "gt_grouper_value"]
            ]
            .groupby(
                ["grouper_key", "pd_grouper_value", "gt_grouper_value"],
                as_index=False,
                dropna=False,
            )
            .size()
        ).assign(
            true_positive_flag=lambda chain_df: chain_df["pd_grouper_value"]
            == chain_df["gt_grouper_value"]
        )  # type: ignore - pandas typing error. the interpreter seems confused since .size is both a dataframe attribute and a function.

        # count of predictions per grouper key
        cm_counts = cm_counts.merge(
            cm_counts.groupby(
                ["grouper_key", "pd_grouper_value"],
                as_index=False,
                dropna=False,
            )
            .size()
            .rename(columns={"size": "number_of_predictions"}),  # type: ignore - pandas typing error. .rename is used correctly here, but the interpreter is confused since we're chaining operations; see docs: https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.rename.html
            on=["grouper_key", "pd_grouper_value"],
        )

        # count of groundtruths per grouper key
        cm_counts = cm_counts.merge(
            cm_counts.groupby(
                ["grouper_key", "gt_grouper_value"],
                as_index=False,
                dropna=False,
            )
            .size()
            .rename(columns={"size": "number_of_groundtruths"}),  # type: ignore - pandas typing error. .rename is used correctly here, but the interpreter is confused since we're chaining operations; see docs: https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.rename.html
            on=["grouper_key", "gt_grouper_value"],
        )

        cm_counts = cm_counts.merge(
            cm_counts[
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
            ),  # pyright: ignore - pandas typing error. .rename is used correctly here; see docs: https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.rename.html
            on=["grouper_key", "pd_grouper_value"],
        )

        cm_counts = cm_counts.merge(
            cm_counts[
                ["grouper_key", "gt_grouper_value", "true_positive_flag"]
            ]
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
            ),  # pyright: ignore - pandas typing error. .rename is used correctly here; see docs: https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.rename.html
            on=["grouper_key", "gt_grouper_value"],
        )

        cm_counts = cm_counts.merge(
            cm_counts[["grouper_key", "true_positive_flag"]]
            .groupby("grouper_key", as_index=False, dropna=False)
            .sum()
            .rename(
                columns={
                    "true_positive_flag": "true_positives_per_grouper_key"
                }
            ),  # pyright: ignore - pandas typing error. .rename is used correctly here; see docs: https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.rename.html
            on="grouper_key",
        )

        # create ConfusionMatrix objects
        confusion_matrices = []
        for grouper_key in cm_counts.loc[:, "grouper_key"].unique():
            revelant_rows = cm_counts.loc[
                (cm_counts["grouper_key"] == grouper_key)
                & cm_counts["gt_grouper_value"].notnull()
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

        return cm_counts, confusion_matrices

    def _calculate_metrics_at_grouper_value_level(cm_counts):
        # create base dataframe that's unique at the (grouper key, grouper value level)
        unique_grouper_values_per_grouper_key = pd.DataFrame(
            np.concatenate(
                [
                    cm_counts[["grouper_key", "pd_grouper_value"]].values,
                    cm_counts.loc[
                        cm_counts["gt_grouper_value"].notnull(),
                        ["grouper_key", "gt_grouper_value"],
                    ].values,
                ]
            ),
            columns=pd.Series(
                ["grouper_key", "grouper_value"]
            ),  # using Series here to resolve a typing error
        ).drop_duplicates()

        # compute metrics using confusion matrices
        metrics_per_grouper_key_and_grouper_value = (
            unique_grouper_values_per_grouper_key.assign(
                number_true_positives=unique_grouper_values_per_grouper_key.apply(
                    lambda chain_df: sum(
                        cm_counts.loc[
                            (
                                cm_counts["gt_grouper_value"]
                                == chain_df["grouper_value"]
                            )
                            & (
                                cm_counts["grouper_key"]
                                == chain_df["grouper_key"]
                            )
                            & (cm_counts["true_positive_flag"]),
                            "size",
                        ]
                    ),
                    axis=1,
                )
            )
            # ['grouper_key', 'pd_grouper_value', 'gt_grouper_value', 'size', 'true_positive_flag', 'number_of_predictions', 'number_of_groundtruths', 'true_positives_per_pd_grouper_value', 'true_positives_per_gt_grouper_value', 'true_positives_per_grouper_key']
            .assign(
                number_of_groundtruths=unique_grouper_values_per_grouper_key.apply(
                    lambda chain_df: sum(
                        cm_counts.loc[
                            (
                                cm_counts["gt_grouper_value"]
                                == chain_df["grouper_value"]
                            )
                            & (
                                cm_counts["grouper_key"]
                                == chain_df["grouper_key"]
                            ),
                            "size",
                        ]
                    ),
                    axis=1,
                )
            )
            .assign(
                number_of_predictions=unique_grouper_values_per_grouper_key.apply(
                    lambda chain_df: sum(
                        cm_counts.loc[
                            (
                                cm_counts["pd_grouper_value"]
                                == chain_df["grouper_value"]
                            )
                            & (
                                cm_counts["grouper_key"]
                                == chain_df["grouper_key"]
                            ),
                            "size",
                        ]
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
        metrics_per_grouper_key_and_grouper_value[
            ["precision", "recall", "f1"]
        ] = metrics_per_grouper_key_and_grouper_value.loc[
            :, ["precision", "recall", "f1"]
        ].replace(
            [np.inf, -np.inf, np.nan], 0
        )

        # replace values of labels that only exist in predictions (not groundtruths) with -1
        labels_to_replace = cm_counts.loc[
            cm_counts["gt_grouper_value"].isnull(),
            ["grouper_key", "pd_grouper_value"],
        ].values.tolist()

        for key, value in labels_to_replace:
            metrics_per_grouper_key_and_grouper_value.loc[
                (
                    metrics_per_grouper_key_and_grouper_value["grouper_key"]
                    == key
                )
                & (
                    metrics_per_grouper_key_and_grouper_value["grouper_value"]
                    == value
                ),
                ["precision", "recall", "f1"],
            ] = -1

        return metrics_per_grouper_key_and_grouper_value

    def _calculate_precision_recall_f1_metrics(
        metrics_per_grouper_key_and_grouper_value,
    ):
        # TODO better arg names and docstrings

        # create metric objects
        output = []

        for row in metrics_per_grouper_key_and_grouper_value.loc[
            ~metrics_per_grouper_key_and_grouper_value[
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

    def _calculate_accuracy_metrics(cm_counts):
        # TODO consider whether we use assign or not
        # TODO Check that all columns in the cm_counts df are actually used

        accuracy_calculations = (
            cm_counts.loc[
                (
                    cm_counts["gt_grouper_value"].notnull()
                    & cm_counts["true_positive_flag"]
                ),
                ["grouper_key", "size"],
            ]
            .groupby(["grouper_key"], as_index=False)
            .sum()
            .rename(columns={"size": "true_positives_per_grouper_key"})
        ).merge(
            cm_counts.loc[
                (cm_counts["gt_grouper_value"].notnull()),
                ["grouper_key", "size"],
            ]
            .groupby(["grouper_key"], as_index=False)
            .sum()
            .rename(columns={"size": "observations_per_grouper_key"}),
            on="grouper_key",
            how="outer",
        )

        accuracy_calculations["accuracy"] = (
            accuracy_calculations["true_positives_per_grouper_key"]
            / accuracy_calculations["observations_per_grouper_key"]
        )

        # some elements may be np.nan if a given grouper key has no true positives
        # replace those accuracy scores with 0

        accuracy_calculations["accuracy"].fillna(value=0, inplace=True)

        return [
            schemas.AccuracyMetric(
                label_key=grouper_key,
                value=accuracy_calculations.loc[
                    accuracy_calculations["grouper_key"] == grouper_key,
                    "accuracy",
                ],
            )
            for grouper_key in accuracy_calculations.loc[:, "grouper_key"]
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

        # TODO should we repeat this step?
        # get the id of the prediction with the maximum score for each grouper_key
        # we use min(id) in case there are multiple predictions that have the same max(score)
        # predictions_ids_with_max_score = predictions.where(score)

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
            .rename(
                columns={"score": "best_score"}
            )  # pyright: ignore - pandas typing error. .rename is used correctly here; see docs: https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.rename.html
        )

        # TODO redo names, should they all end in _df?
        best_prediction_label_for_each_grouper_key_and_datum = pd.merge(
            pd_df[["grouper_key", "grouper_value", "datum_id", "id"]],
            best_prediction_id_per_grouper_key_and_datum_id,
            on=["grouper_key", "datum_id", "id"],
            how="inner",
        )[["grouper_key", "datum_id", "grouper_value", "best_score"]]

        # count the number of matches for each (pd_label_value, gt_label_value) for each grouper_key
        merged_groundtruths_and_predictions = pd.merge(
            gt_df[["datum_id", "grouper_key", "grouper_value"]].rename(
                columns={"grouper_value": "gt_grouper_value"}
            ),  # pyright: ignore - pandas typing error. .rename is used correctly here; see docs: https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.rename.html
            best_prediction_label_for_each_grouper_key_and_datum.rename(
                columns={"grouper_value": "pd_grouper_value"}
            ),  # pyright: ignore - pandas typing error. .rename is used correctly here; see docs: https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.rename.html
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
                        [None] * len(merged_groundtruths_and_predictions),
                        merged_groundtruths_and_predictions["grouper_key"],
                        [None] * len(merged_groundtruths_and_predictions),
                        merged_groundtruths_and_predictions[
                            "pd_grouper_value"
                        ],
                        [None] * len(pd_df),
                    )
                ).union(
                    set(
                        zip(
                            [None] * len(merged_groundtruths_and_predictions),
                            merged_groundtruths_and_predictions["grouper_key"],
                            [None] * len(merged_groundtruths_and_predictions),
                            merged_groundtruths_and_predictions[
                                "gt_grouper_value"
                            ],
                            [None] * len(pd_df),
                        )
                    )
                )
            )
        )

        merged_groundtruths_and_predictions = pd.concat(
            [
                merged_groundtruths_and_predictions,
                pd.DataFrame(
                    missing_grouper_labels_from_predictions,
                    columns=merged_groundtruths_and_predictions.columns,
                ),
            ],
            ignore_index=True,
        )

        return merged_groundtruths_and_predictions

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
                ),  # pyright: ignore - pandas typing error. .rename is used correctly here; see docs: https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.rename.html
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
            .rename(columns={"size": "n"})  # type: ignore - pandas typing error. .rename is used correctly here, but the interpreter is confused since we're chaining operations; see docs: https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.rename.html
        )

        total_true_positves_per_grouper_key_and_grouper_value = (
            merged_predictions_and_groundtruths.loc[
                merged_predictions_and_groundtruths["is_true_positive"], :
            ]
            .groupby(["grouper_key", "grouper_value"], as_index=False)[
                "gt_grouper_value"
            ]
            .size()
            .rename(columns={"size": "n_true_positives"})  # type: ignore - pandas typing error. .rename is used correctly here, but the interpreter is confused since we're chaining operations; see docs: https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.rename.html
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
        )  # pyright: ignore - pandas typing error. .rename is used correctly here; see docs: https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.rename.html

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
            )  # pyright: ignore - pandas typing error. .rename is used correctly here; see docs: https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.rename.html
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
                rates.groupby(
                    ["grouper_key", "grouper_value"], as_index=False
                )[["tpr", "fpr"]]
                .shift(1)
                .rename(
                    columns={"tpr": "lagged_tpr", "fpr": "lagged_fpr"}
                ),  # pyright: ignore - pandas typing error. .rename is used correctly here; see docs: https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.rename.html
            ],
            axis=1,
        ).assign(
            trap_area=lambda chain_df: 0.5
            * (
                (chain_df["tpr"] + chain_df["lagged_tpr"])
                * (chain_df["fpr"] - chain_df["lagged_fpr"])
            )
        )

        summed_trap_areas_per_grouper_value = (
            trap_areas_per_grouper_value.groupby(
                ["grouper_key", "grouper_value"], as_index=False
            )[["n", "n_true_positives", "trap_area"]].sum(min_count=1)
        )

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
        average_across_grouper_keys = (
            summed_trap_areas_per_grouper_value.groupby(
                "grouper_key", as_index=False
            )["trap_area"].mean()
        )

        return [
            schemas.ROCAUCMetric(
                label_key=row["grouper_key"], value=row["trap_area"]
            )
            for row in average_across_grouper_keys.to_dict(orient="records")  # type: ignore - pandas typing error. orient="records" is used in their docs: https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.to_dict.html
        ]

    labels = core.fetch_union_of_labels(
        db=db,
        lhs=groundtruth_filter,
        rhs=prediction_filter,
    )

    grouper_mappings = create_grouper_mappings(
        labels=labels,
        label_map=label_map,
        evaluation_type=enums.TaskType.CLASSIFICATION,
    )

    # TODO delete old functions which are no longer being used at the end
    # TODO check that we output these
    confusion_matrices, metrics_to_output = [], []

    groundtruths = generate_select(
        models.GroundTruth,
        models.Dataset.name.label("dataset_name"),
        models.Label.key.label("label_key"),
        models.Label.value.label("label_value"),
        models.Annotation.datum_id,
        filters=groundtruth_filter,
        label_source=models.GroundTruth,
    )

    predictions = generate_select(
        models.Prediction,
        models.Dataset.name.label("dataset_name"),
        models.Label.key.label("label_key"),
        models.Label.value.label("label_value"),
        models.Annotation.datum_id,
        filters=prediction_filter,
        label_source=models.Prediction,
    )

    gt_df = pd.read_sql(groundtruths, db.bind)  # type: ignore - pandas typing error. passing Engine type works fine here.
    pd_df = pd.read_sql(predictions, db.bind)  # type: ignore - pandas typing error. passing Engine type works fine here.

    _add_columns_to_groundtruth_and_prediction_table(df=gt_df)
    _add_columns_to_groundtruth_and_prediction_table(df=pd_df)

    merged_groundtruths_and_predictions = _get_merged_dataframe(
        pd_df=pd_df, gt_df=gt_df
    )

    cm_counts, confusion_matrices = _calculate_confusion_matrix_df(
        merged_groundtruths_and_predictions=merged_groundtruths_and_predictions
    )

    metrics_per_grouper_key_and_grouper_value = (
        _calculate_metrics_at_grouper_value_level(cm_counts=cm_counts)
    )

    metrics_to_output += _calculate_precision_recall_f1_metrics(
        metrics_per_grouper_key_and_grouper_value=metrics_per_grouper_key_and_grouper_value
    )

    metrics_to_output += _calculate_accuracy_metrics(cm_counts=cm_counts)

    metrics_to_output += _calculate_rocauc(pd_df=pd_df, gt_df=gt_df)

    for grouper_key in grouper_mappings[
        "grouper_key_to_labels_mapping"
    ].keys():

        if (
            enums.MetricType.PrecisionRecallCurve in metrics_to_return
            or enums.MetricType.DetailedPrecisionRecallCurve
            in metrics_to_return
        ):
            # calculate the number of unique datums
            # used to determine the number of true negatives
            gt_datums = generate_query(
                models.Datum.id,
                models.Dataset.name,
                models.Datum.uid,
                db=db,
                filters=groundtruth_filter,
                label_source=models.GroundTruth,
            ).all()
            pd_datums = generate_query(
                models.Datum.id,
                models.Dataset.name,
                models.Datum.uid,
                db=db,
                filters=prediction_filter,
                label_source=models.Prediction,
            ).all()
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

            pr_curves = _compute_curves(
                db=db,
                groundtruths=groundtruths.cte(),
                predictions=predictions.cte(),
                grouper_key=grouper_key,
                grouper_mappings=grouper_mappings,
                unique_datums=unique_datums,
                pr_curve_max_examples=pr_curve_max_examples,
                metrics_to_return=metrics_to_return,
            )
            metrics_to_output += pr_curves

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
        db=db,
        filters=schemas.Filter(**evaluation.filters),
        dataset_names=evaluation.dataset_names,
        model_name=evaluation.model_name,
        task_type=parameters.task_type,
        label_map=parameters.label_map,
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

    confusion_matrices_mappings = create_metric_mappings(
        db=db,
        metrics=confusion_matrices,
        evaluation_id=evaluation.id,
    )

    for mapping in confusion_matrices_mappings:
        get_or_create_row(
            db,
            models.ConfusionMatrix,
            mapping,
        )

    metric_mappings = create_metric_mappings(
        db=db,
        metrics=metrics,
        evaluation_id=evaluation.id,
    )

    for mapping in metric_mappings:
        # ignore value since the other columns are unique identifiers
        # and have empirically noticed value can slightly change due to floating
        # point errors
        get_or_create_row(
            db,
            models.Metric,
            mapping,
            columns_to_ignore=["value"],
        )

    log_evaluation_duration(
        evaluation=evaluation,
        db=db,
    )

    return evaluation_id
