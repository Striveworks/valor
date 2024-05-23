from typing import Sequence

from sqlalchemy import Subquery
from sqlalchemy.orm import Session
from sqlalchemy.sql import select
from sqlalchemy.sql.selectable import NamedFromClause

from valor_api import schemas
from valor_api.backend import core, models
from valor_api.backend.metrics.llm_call import OpenAIValorClient
from valor_api.backend.metrics.metric_utils import (  # log_evaluation_item_counts,
    create_metric_mappings,
    get_or_create_row,
    log_evaluation_duration,
    validate_computation,
)
from valor_api.backend.query import Query

LabelMapType = list[list[list[str]]]


# TODO update this for text comparison metrics
def _generate_datum_query(
    datum_filter: schemas.Filter,
) -> NamedFromClause | Subquery:
    """Generate a sqlalchemy query to fetch datums."""
    return (
        Query(
            models.Datum,
            models.Datum.id.label("datum_id"),
            models.Datum.uid.label("datum_uid"),
            # models.Datum.dataset_id.label("dataset_id"),
            models.Dataset.name.label(
                "dataset_name"
            ),  # TODO Will this only get dataset_name that matches the dataset_id?
            models.Datum.text.label("datum_text"),
            models.Datum.meta.label("datum_meta"),
        )
        .filter(datum_filter)
        .datums(
            as_subquery=False
        )  # TODO should I change this? as_subquery=False
        .alias()
    )


# def _generate_groundtruth_query(
#     datum_filter: schemas.Filter,
# ) -> NamedFromClause | Subquery:
#     """Generate a sqlalchemy query to fetch a ground truth."""
#     return (
#         Query(
#             models.GroundTruth,
#             models.Annotation.datum_id.label("datum_id"),
#             models.GroundTruth.text.label("groundtruth_text"),
#             models.Dataset.name.label("dataset_name"),
#         )
#         .filter(datum_filter)
#         .groundtruths(
#             as_subquery=False
#         )  # TODO should I change this? as_subquery=False
#         .alias()
#     )


def _generate_prediction_query(
    prediction_filter: schemas.Filter,
) -> NamedFromClause | Subquery:
    """Generate a sqlalchemy query to fetch a prediction."""

    return (
        Query(
            models.Prediction,
            # models.Annotation.id.label("annotation_id"),
            models.Annotation.datum_id.label("datum_id"),
            models.Annotation.model_id.label("model_id"),
            models.Model.name.label(
                "model_name"
            ),  # TODO Will this only get the model_name that matches the model_id?
            models.Annotation.task_type.label("task_type"),
            models.Annotation.text.label("prediction_text"),
            models.Annotation.context.label("prediction_context"),
            models.Annotation.meta.label("prediction_annotation_meta"),
            models.Prediction.id.label("prediction_id"),
        )
        .filter(prediction_filter)
        .predictions(
            as_subquery=False
        )  # TODO should I change this? as_subquery=False
        .alias()
    )


def _compute_text_generation_metrics(
    db: Session,
    prediction_filter: schemas.Filter,
    datum_filter: schemas.Filter,
    metrics: list[str] = [],
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
    | schemas.SummarizationMetric
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
    datum_filter : schemas.Filter
        The filter to be used to query groundtruths.
    metrics: list[str]
        TODO

    Returns
    ----------
    Sequence[schemas.AnswerCorrectnessMetric | schemas.AnswerRelevanceMetric | schemas.BiasMetric | schemas.CoherenceMetric | schemas.ContextPrecisionMetric | schemas.ContextRecallMetric | schemas.ContextRelevanceMetric | schemas.FaithfulnessMetric | schemas.GrammaticalityMetric | schemas.HallucinationMetric | schemas.SummarizationMetric | schemas.ToxicityMetric]
        A list of metrics.
    """
    datum_subquery = _generate_datum_query(datum_filter)
    prediction_subquery = _generate_prediction_query(prediction_filter)

    # TODO Remove unnecessary columns in the select.
    total_query = (
        select(
            datum_subquery.c.datum_uid.label("datum_uid"),
            datum_subquery.c.dataset_name.label("dataset_name"),
            datum_subquery.c.datum_text.label("datum_text"),
            prediction_subquery.c.prediction_text.label("prediction_text"),
            prediction_subquery.c.prediction_context.label(
                "prediction_context"
            ),
            # datum_subquery.c.datum_id.label("datum_id"),
            # datum_subquery.c.dataset_id.label("dataset_id"),
            # datum_subquery.c.meta.label("datum_meta"),
            # prediction_subquery.c.datum_id.label("datum_id"),
            # prediction_subquery.c.model_id.label("model_id"),
            # prediction_subquery.c.model_name.label("model_name"),
            # prediction_subquery.c.task_type.label("task_type"),
            # prediction_subquery.c.prediction_annotation_meta.label("prediction_annotation_meta"),
            # prediction_subquery.c.prediction_id.label("prediction_id"),
        )
        .select_from(datum_subquery)
        .join(
            prediction_subquery,
            datum_subquery.c.datum_id == prediction_subquery.c.datum_id,
        )
    )

    res = db.execute(total_query).all()

    client = OpenAIValorClient()
    client.connect()
    ret = []

    # TODO Implement the rest of the metrics.
    for row in res:
        for metric_name in metrics:
            if metric_name == "AnswerCorrectness":
                raise NotImplementedError
            elif metric_name == "AnswerRelevance":
                raise NotImplementedError
            elif metric_name == "Bias":
                raise NotImplementedError
            elif metric_name == "Coherence":
                generated_text = row[3]
                response = client.coherence(text=generated_text)
                ret.append(
                    schemas.CoherenceMetric(
                        value=response,
                        parameters={
                            "dataset": row[1],
                            "datum_uid": row[0],
                            "prediction": row[3],
                        },
                    )
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
            elif metric_name == "Summarization":
                raise NotImplementedError
            elif metric_name == "Toxicity":
                raise NotImplementedError

    return ret


@validate_computation
def compute_text_generation_metrics(
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
    datum_filter = schemas.Filter(**evaluation.datum_filter)
    groundtruth_filter = datum_filter.model_copy()
    prediction_filter = datum_filter.model_copy()

    prediction_filter.model_names = [evaluation.model_name]
    parameters = schemas.EvaluationParameters(**evaluation.parameters)

    # load task type into filters
    datum_filter.task_types = [parameters.task_type]
    groundtruth_filter.task_types = [parameters.task_type]
    prediction_filter.task_types = [parameters.task_type]

    # get the metrics to compute
    metrics = parameters.metrics

    assert metrics, "No metrics to compute."

    # TODO What should I do here?
    # log_evaluation_item_counts(
    #     db=db,
    #     evaluation=evaluation,
    #     prediction_filter=prediction_filter,
    #     groundtruth_filter=groundtruth_filter,
    # )

    metrics = _compute_text_generation_metrics(
        db=db,
        prediction_filter=prediction_filter,
        datum_filter=datum_filter,
        metrics=metrics,
    )

    # TODO Add this for the text comparison metrics, which compare predictions to groundtruths.
    # text_comparison_metrics = _compute_text_comparison_metrics(
    #     db=db,
    #     prediction_filter=prediction_filter,
    #     datum_filter=datum_filter,
    #     metrics=metrics,
    # )

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
