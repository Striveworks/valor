# from collections import defaultdict
# import pdb  # TODO remove
from typing import Sequence

# import numpy as np
# from sqlalchemy import Subquery  # Integer
from sqlalchemy.orm import Session  # Bundle
from sqlalchemy.sql import and_, select  # case, func, join,

from valor_api import enums, schemas
from valor_api.backend import core, models
from valor_api.backend.metrics.llm_call import OpenAIClient
from valor_api.backend.metrics.metric_utils import (
    create_grouper_mappings,
    create_metric_mappings,
    get_or_create_row,
    log_evaluation_duration,
    log_evaluation_item_counts,
    validate_computation,
)
from valor_api.backend.query import Query

# from sqlalchemy.sql.selectable import NamedFromClause


LabelMapType = list[list[list[str]]]


def _compute_llm_evaluation_metrics_at_grouper_key(
    db: Session,
    prediction_filter: schemas.Filter,
    groundtruth_filter: schemas.Filter,
    grouper_key: str,
    grouper_mappings: dict[str, dict[str, dict]],
    metric_list: list[str] = [],
) -> (
    Sequence[
        schemas.AnswerCorrectnessMetric
        | schemas.AnswerRelevanceMetric
        | schemas.BiasMetric
        | schemas.CoherenceMetric
        | schemas.ContextPrecisionMetric
        | schemas.ContextRecallMetric
        | schemas.ContextRelevanceMetric
        | schemas.FaithfulnessMetric
        | schemas.GrammaticalityMetric
        | schemas.HallucinationMetric
        | schemas.QAGMetric
        | schemas.ToxicityMetric
    ]
    | None
):
    """
    Computes the confusion matrix and all metrics for a given label key.

    Parameters
    ----------
    db : Session
        The database Session to query against.
    prediction_filter : schemas.Filter
        The filter to be used to query predictions.
    groundtruth_filter : schemas.Filter
        The filter to be used to query groundtruths.
    grouper_mappings: dict[str, dict[str, dict]]
        A dictionary of mappings that connect groupers to their related labels.
    metric_list: list[str]
        TODO

    Returns
    -------
    Sequence[schemas.AnswerCorrectnessMetric | schemas.AnswerRelevanceMetric | schemas.BiasMetric | schemas.CoherenceMetric
                | schemas.ContextPrecisionMetric | schemas.ContextRecallMetric | schemas.ContextRelevanceMetric
                | schemas.FaithfulnessMetric | schemas.GrammaticalityMetric | schemas.HallucinationMetric | schemas.QAGMetric
                | schemas.ToxicityMetric] | None
        Returns None if there are no predictions and groundtruths with the given label
        key for the same datum. Otherwise returns the desired metrics.
    """

    label_key_filter = list(
        grouper_mappings["grouper_key_to_label_keys_mapping"][grouper_key]
    )

    # get groundtruths and predictions that conform to filters
    gFilter = groundtruth_filter.model_copy()
    gFilter.label_keys = label_key_filter

    pFilter = prediction_filter.model_copy()
    pFilter.label_keys = label_key_filter

    groundtruths = (
        Query(
            models.GroundTruth,
            models.Annotation.datum_id.label("datum_id"),
            models.Label.key.label("label_key"),
            models.Label.value.label("groundtruth"),
            models.Dataset.name.label("dataset_name"),
        )
        .filter(gFilter)
        .groundtruths(as_subquery=False)
        .alias()
    )

    predictions = (
        Query(
            models.Prediction,
            models.Annotation.datum_id.label("datum_id"),
            models.Annotation.meta.label("prediction_annotation_metadata"),
            models.Annotation.task_type.label("task_type"),
            models.Label.key.label("label_key"),
            models.Label.value.label("prediction"),
            models.Dataset.name.label("dataset_name"),
        )
        .filter(pFilter)
        .predictions(as_subquery=False)
        .alias()
    )

    total_query = (
        select(
            groundtruths.c.datum_id.label("datum_id"),
            groundtruths.c.label_key.label("label_key"),
            predictions.c.task_type.label("task_type"),
            groundtruths.c.groundtruth.label("groundtruth"),
            predictions.c.prediction.label("prediction"),
            predictions.c.prediction_annotation_metadata.label(
                "prediction_annotation_metadata"
            ),
        )
        .select_from(groundtruths)
        .join(
            predictions,
            and_(
                groundtruths.c.datum_id == predictions.c.datum_id,
                groundtruths.c.label_key == predictions.c.label_key,
            ),
        )
    )

    # TODO rename this output
    res = db.execute(total_query).all()

    if len(res) == 0:
        # this means there's no predictions and groundtruths with the same datum id and label key
        return None

    client = OpenAIClient()
    metrics = []

    for metric_name in metric_list:
        if metric_name == "AnswerCorrectnessMetric":
            raise NotImplementedError
        elif metric_name == "AnswerRelevanceMetric":
            raise NotImplementedError
        elif metric_name == "BiasMetric":
            raise NotImplementedError
        elif metric_name == "CoherenceMetric":
            # TODO how does this work with grouping?
            for row in res:
                label_key = row[1]
                generated_text = row[4]
                metrics.append(
                    client.coherence(text=generated_text, label_key=label_key)
                )
        elif metric_name == "ContextPrecisionMetric":
            raise NotImplementedError
        elif metric_name == "ContextRecallMetric":
            raise NotImplementedError
        elif metric_name == "ContextRelevanceMetric":
            raise NotImplementedError
        elif metric_name == "FaithfulnessMetric":
            raise NotImplementedError
        elif metric_name == "GrammaticalityMetric":
            raise NotImplementedError
        elif metric_name == "HallucinationMetric":
            raise NotImplementedError
        elif metric_name == "QAGMetric":
            raise NotImplementedError
        elif metric_name == "ToxicityMetric":
            raise NotImplementedError

    # metrics that are per label
    for grouper_value in grouper_mappings["grouper_key_to_labels_mapping"][
        grouper_key
    ].keys():
        pass  # TODO is there anything we want to use grouper mappings for?

    return metrics


def _compute_llm_evaluation_metrics(
    db: Session,
    prediction_filter: schemas.Filter,
    groundtruth_filter: schemas.Filter,
    label_map: LabelMapType | None = None,
    metric_list: list[str] = [],
) -> Sequence[
    schemas.AnswerCorrectnessMetric
    | schemas.AnswerRelevanceMetric
    | schemas.BiasMetric
    | schemas.CoherenceMetric
    | schemas.ContextPrecisionMetric
    | schemas.ContextRecallMetric
    | schemas.ContextRelevanceMetric
    | schemas.FaithfulnessMetric
    | schemas.GrammaticalityMetric
    | schemas.HallucinationMetric
    | schemas.QAGMetric
    | schemas.ToxicityMetric
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
    label_map: LabelMapType, optional
        Optional mapping of individual labels to a grouper label. Useful when you need to evaluate performance using labels that differ across datasets and models.
    metric_list: list[str]
        TODO

    Returns
    ----------
    Sequence[schemas.AnswerCorrectnessMetric | schemas.AnswerRelevanceMetric | schemas.BiasMetric | schemas.CoherenceMetric | schemas.ContextPrecisionMetric | schemas.ContextRecallMetric | schemas.ContextRelevanceMetric | schemas.FaithfulnessMetric | schemas.GrammaticalityMetric | schemas.HallucinationMetric | schemas.QAGMetric | schemas.ToxicityMetric]
        A list of metrics.
    """
    labels = core.fetch_union_of_labels(
        db=db,
        rhs=prediction_filter,
        lhs=groundtruth_filter,
    )

    grouper_mappings = create_grouper_mappings(
        labels=labels,
        label_map=label_map,
        evaluation_type=enums.TaskType.LLM_EVALUATION,
    )

    # compute metrics for each grouper id
    metrics = []
    for grouper_key in grouper_mappings[
        "grouper_key_to_labels_mapping"
    ].keys():
        print(grouper_key)
        llm_evaluation_metrics = (
            _compute_llm_evaluation_metrics_at_grouper_key(
                db=db,
                prediction_filter=prediction_filter,
                groundtruth_filter=groundtruth_filter,
                grouper_key=grouper_key,
                grouper_mappings=grouper_mappings,
                metric_list=metric_list,
            )
        )
        if llm_evaluation_metrics is not None:
            metrics.extend(llm_evaluation_metrics)

    return metrics


@validate_computation
def compute_llm_evaluation_metrics(
    *,
    db: Session,
    evaluation_id: int,
) -> int:
    """
    Create llm guided evaluation metrics. This function is intended to be run using FastAPI's `BackgroundTasks`.

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
    groundtruth_filter = schemas.Filter(**evaluation.datum_filter)
    prediction_filter = groundtruth_filter.model_copy()
    prediction_filter.model_names = [evaluation.model_name]
    parameters = schemas.EvaluationParameters(**evaluation.parameters)

    # load task type into filters
    groundtruth_filter.task_types = [parameters.task_type]
    prediction_filter.task_types = [parameters.task_type]

    # get the metrics to compute
    metric_list = parameters.metrics_to_return

    assert metric_list, "No metrics to compute."

    log_evaluation_item_counts(
        db=db,
        evaluation=evaluation,
        prediction_filter=prediction_filter,
        groundtruth_filter=groundtruth_filter,
    )

    metrics = _compute_llm_evaluation_metrics(
        db=db,
        prediction_filter=prediction_filter,
        groundtruth_filter=groundtruth_filter,
        label_map=parameters.label_map,
        metric_list=metric_list,
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
