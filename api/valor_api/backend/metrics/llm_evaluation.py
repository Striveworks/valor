# from collections import defaultdict
# import pdb  # TODO remove
from typing import Sequence

# import numpy as np
from sqlalchemy import Subquery  # Integer
from sqlalchemy.orm import Session  # Bundle
from sqlalchemy.sql import and_, select  # case, func, join, Select,
from sqlalchemy.sql.selectable import NamedFromClause

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

LabelMapType = list[list[list[str]]]


def _generate_groundtruth_query(
    groundtruth_filter: schemas.Filter,
) -> NamedFromClause | Subquery:
    """Generate a sqlalchemy query to fetch a ground truth."""
    return (
        Query(
            models.GroundTruth,
            models.Annotation.datum_id.label("datum_id"),
            models.Label.key.label("label_key"),
            models.Label.value.label("groundtruth"),
            models.Dataset.name.label("dataset_name"),
        )
        .filter(groundtruth_filter)
        .groundtruths(
            as_subquery=False
        )  # TODO should I change this? as_subquery=False
        .alias()
    )


def _generate_prediction_query(
    prediction_filter: schemas.Filter,
) -> NamedFromClause | Subquery:
    """Generate a sqlalchemy query to fetch a prediction."""

    return (
        Query(
            models.Prediction,
            models.Annotation.datum_id.label("datum_id"),
            models.Annotation.meta.label("prediction_annotation_metadata"),
            models.Annotation.task_type.label("task_type"),
            models.Label.key.label("label_key"),
            models.Label.value.label("prediction"),
            models.Dataset.name.label("dataset_name"),
        )
        .filter(prediction_filter)
        .predictions(
            as_subquery=False
        )  # TODO should I change this? as_subquery=False
        .alias()
    )


def _compute_llm_evaluation_metrics_at_grouper_id(
    db: Session,
    prediction_filter: schemas.Filter,
    groundtruth_filter: schemas.Filter,
    grouper_label: models.Label,
    grouper_id,  # TODO what type?
    grouper_mappings: dict[str, dict[str, dict]],
    metrics_to_return: list[str] = [],
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
    grouper_label: models.Label
        TODO
    grouper_id: TODO
        TODO
    grouper_mappings: dict[str, dict[str, dict]]
        A dictionary of mappings that connect groupers to their related labels.
    metrics_to_return: list[str]
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
    groundtruth_subquery = _generate_groundtruth_query(groundtruth_filter)
    prediction_subquery = _generate_prediction_query(prediction_filter)

    # groundtruth_query = select(
    #     groundtruth_subquery.c.datum_id.label("datum_id"),
    #     groundtruth_subquery.c.label_key.label("label_key"),
    #     groundtruth_subquery.c.groundtruth.label("groundtruth"),
    #     groundtruth_subquery.c.dataset_name.label("dataset_name"),
    # ).select_from(groundtruth_subquery)

    # prediction_query = select(
    #     prediction_subquery.c.datum_id.label("datum_id"),
    #     prediction_subquery.c.label_key.label("label_key"),
    #     prediction_subquery.c.task_type.label("task_type"),
    #     prediction_subquery.c.prediction.label("prediction"),
    #     prediction_subquery.c.prediction_annotation_metadata.label(
    #         "prediction_annotation_metadata"
    #     ),
    #     prediction_subquery.c.dataset_name.label("dataset_name"),
    # ).select_from(prediction_subquery)

    # res_gt = db.execute(groundtruth_query).all()
    # res_pred = db.execute(prediction_query).all()

    total_query = (
        select(
            groundtruth_subquery.c.datum_id.label("datum_id"),
            groundtruth_subquery.c.label_key.label("label_key"),
            prediction_subquery.c.task_type.label("task_type"),
            groundtruth_subquery.c.groundtruth.label("groundtruth"),
            prediction_subquery.c.prediction.label("prediction"),
            prediction_subquery.c.prediction_annotation_metadata.label(
                "prediction_annotation_metadata"
            ),
        )
        .select_from(groundtruth_subquery)
        .join(
            prediction_subquery,
            and_(
                # groundtruth_subquery.c.datum_id == prediction_subquery.c.datum_id,
                groundtruth_subquery.c.label_key
                == prediction_subquery.c.label_key,
            ),
        )
    )

    # TODO rename this output
    res = db.execute(total_query).all()

    # pdb.set_trace()

    if len(res) == 0:
        # this means there's no predictions and groundtruths with the same datum id and label key
        return None

    client = OpenAIClient()
    client.connect()
    metrics = []

    # TODO use grouper_id and grouper_label
    for metric_name in metrics_to_return:
        if metric_name == "AnswerCorrectness":
            raise NotImplementedError
        elif metric_name == "AnswerRelevance":
            raise NotImplementedError
        elif metric_name == "Bias":
            raise NotImplementedError
        elif metric_name == "Coherence":
            # TODO how does this work with grouping?
            for row in res:
                label_key = row[1]
                generated_text = row[4]
                metrics.append(
                    client.coherence(text=generated_text, label_key=label_key)
                )
        elif metric_name == "ContextPrecision":
            raise NotImplementedError
        elif metric_name == "ContextRecall":
            raise NotImplementedError
        elif metric_name == "ContextRelevance":
            raise NotImplementedError
        elif metric_name == "Faithfulness":
            raise NotImplementedError
        elif metric_name == "Grammaticality":
            raise NotImplementedError
        elif metric_name == "Hallucination":
            raise NotImplementedError
        elif metric_name == "QAG":
            raise NotImplementedError
        elif metric_name == "Toxicity":
            raise NotImplementedError

    # pdb.set_trace()

    return metrics


def _compute_llm_evaluation_metrics(
    db: Session,
    prediction_filter: schemas.Filter,
    groundtruth_filter: schemas.Filter,
    label_map: LabelMapType | None = None,
    metrics_to_return: list[str] = [],
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
    metrics_to_return: list[str]
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
    ret = []
    for grouper_id, label_ids in grouper_mappings[
        "grouper_id_to_label_ids_mapping"
    ].items():
        # set filter
        groundtruth_filter.label_ids = [label_id for label_id in label_ids]
        prediction_filter.label_ids = [label_id for label_id in label_ids]

        grouper_label = grouper_mappings[
            "grouper_id_to_grouper_label_mapping"
        ][grouper_id]

        llm_evaluation_metrics = _compute_llm_evaluation_metrics_at_grouper_id(
            db=db,
            prediction_filter=prediction_filter,
            groundtruth_filter=groundtruth_filter,
            grouper_label=grouper_label,
            grouper_id=grouper_id,
            grouper_mappings=grouper_mappings,
            metrics_to_return=metrics_to_return,
        )

        if llm_evaluation_metrics is not None:
            ret.extend(llm_evaluation_metrics)

    # TODO remove
    # metrics = []
    # for grouper_key in grouper_mappings[
    #     "grouper_key_to_labels_mapping"
    # ].keys():
    #     print(grouper_key)
    #     llm_evaluation_metrics = (
    #         _compute_llm_evaluation_metrics_at_grouper_key(
    #             db=db,
    #             prediction_filter=prediction_filter,
    #             groundtruth_filter=groundtruth_filter,
    #             grouper_key=grouper_key,
    #             grouper_mappings=grouper_mappings,
    #             metrics_to_return=metrics_to_return,
    #         )
    #     )
    #     if llm_evaluation_metrics is not None:
    #         metrics.extend(llm_evaluation_metrics)

    return ret


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
    metrics_to_return = parameters.metrics_to_return

    assert metrics_to_return, "No metrics to compute."

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
        metrics_to_return=metrics_to_return,
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
