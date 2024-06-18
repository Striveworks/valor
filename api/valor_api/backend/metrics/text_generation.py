from typing import Sequence

from sqlalchemy import Subquery
from sqlalchemy.orm import Session
from sqlalchemy.sql import select
from sqlalchemy.sql.selectable import NamedFromClause

from valor_api import schemas
from valor_api.backend import core, models
from valor_api.backend.metrics.llm_call import (
    LLMClient,
    MistralValorClient,
    OpenAIValorClient,
)
from valor_api.backend.metrics.metric_utils import (  # log_evaluation_item_counts,
    create_metric_mappings,
    get_or_create_row,
    log_evaluation_duration,
    validate_computation,
)
from valor_api.backend.query import generate_query
from valor_api.enums import MetricType

LabelMapType = list[list[list[str]]]

LLM_GUIDED_METRICS = {
    "AnswerRelevance",
    "AnswerCorrectness",
    "Bias",
    "Coherence",
    "ContextPrecision",
    "ContextRecall",
    "ContextRelevance",
    "Faithfulness",
    "Grammaticality",
    "Hallucination",
    "Summarization",
    "Toxicity",
}


# TODO update this for text comparison metrics
def _generate_datum_query(
    db: Session,
    datum_filter: schemas.Filter,
) -> NamedFromClause | Subquery:
    """Generate a sqlalchemy query to fetch datums."""
    return generate_query(
        models.Datum,
        models.Datum.id.label("datum_id"),
        models.Datum.uid.label("datum_uid"),
        # models.Datum.dataset_id.label("dataset_id"),
        models.Dataset.name.label(
            "dataset_name"
        ),  # TODO Will this only get dataset_name that matches the dataset_id?
        models.Datum.text.label("datum_text"),
        models.Datum.meta.label("datum_meta"),
        db=db,
        filters=datum_filter,
        label_source=models.Annotation,
    ).subquery()


# TODO remove this
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
    db: Session,
    prediction_filter: schemas.Filter,
) -> NamedFromClause | Subquery:
    """Generate a sqlalchemy query to fetch a prediction."""

    return generate_query(
        models.Prediction,
        # models.Annotation.id.label("annotation_id"),
        models.Annotation.datum_id.label("datum_id"),
        models.Annotation.model_id.label("model_id"),
        models.Model.name.label(
            "model_name"
        ),  # TODO Will this only get the model_name that matches the model_id?
        models.Annotation.text.label("prediction_text"),
        models.Annotation.context.label("prediction_context"),
        models.Annotation.meta.label("prediction_annotation_meta"),
        models.Prediction.id.label("prediction_id"),
        db=db,
        label_source=models.Prediction,
        filters=prediction_filter,
    ).subquery()


def setup_llm_client(
    llm_api_params: dict[str, str | dict],
) -> LLMClient:
    """
    Setup an LLM client for LLM guided evaluation.

    Parameters
    ----------
    llm_api_params : dict[str, str | dict], optional
        The parameters to setup the client with.

    Returns
    ----------
    LLMClient
        TODO
    """
    assert (
        "client" in llm_api_params or "api_url" in llm_api_params
    ), "Need to specify the client or api_url."
    assert not (
        "client" in llm_api_params and "api_url" in llm_api_params
    ), "Cannot specify both client and api_url."

    if llm_api_params.get("client") is not None:
        if llm_api_params["client"] == "openai":
            client_cls = OpenAIValorClient
        elif llm_api_params["client"] == "mistral":
            client_cls = MistralValorClient
        else:
            raise ValueError(
                f"Client {llm_api_params['client']} is not supported."
            )
    else:
        raise NotImplementedError(
            "Support has not been implemented for api_url."
        )

    client_kwargs = {}
    if "data" in llm_api_params:
        assert isinstance(llm_api_params["data"], dict)
        if "model" in llm_api_params["data"]:
            client_kwargs["model_name"] = llm_api_params["data"]["model"]
        if "seed" in llm_api_params["data"]:
            client_kwargs["seed"] = llm_api_params["data"]["seed"]

    client = client_cls(**client_kwargs)
    client.connect()
    return client


def _compute_text_generation_metrics(
    db: Session,
    datum_filter: schemas.Filter,
    prediction_filter: schemas.Filter,
    metrics_to_return: list[MetricType] = [],
    llm_api_params: dict[str, str | dict] | None = None,
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
    datum_filter : schemas.Filter
        The filter to be used to query datums.
    prediction_filter : schemas.Filter
        The filter to be used to query predictions.
    metrics_to_return: list[MetricType]
        The list of metrics to compute, store, and return to the user.
    llm_api_params: dict[str, str | dict], optional
        A dictionary of parameters for the LLM API.

    Returns
    ----------
    Sequence[schemas.AnswerCorrectnessMetric | schemas.AnswerRelevanceMetric | schemas.BiasMetric | schemas.CoherenceMetric | schemas.ContextPrecisionMetric | schemas.ContextRecallMetric | schemas.ContextRelevanceMetric | schemas.FaithfulnessMetric | schemas.GrammaticalityMetric | schemas.HallucinationMetric | schemas.SummarizationMetric | schemas.ToxicityMetric]
        A list of metrics.
    """
    datum_subquery = _generate_datum_query(db=db, datum_filter=datum_filter)
    prediction_subquery = _generate_prediction_query(
        db=db, prediction_filter=prediction_filter
    )

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

    client = None
    if any(
        metric_name in LLM_GUIDED_METRICS for metric_name in metrics_to_return
    ):
        assert (
            llm_api_params is not None
        ), f"llm_api_params must be provided for the following metrics: {[metric for metric in metrics_to_return if metric in LLM_GUIDED_METRICS]}."
        client = setup_llm_client(llm_api_params)

    # TODO Implement the rest of the metrics.
    ret = []
    for row in res:
        for metric_type in metrics_to_return:
            if metric_type == MetricType.AnswerCorrectness:
                raise NotImplementedError
            elif metric_type == MetricType.AnswerRelevance:
                raise NotImplementedError
            elif metric_type == MetricType.Bias:
                raise NotImplementedError
            elif metric_type == MetricType.Coherence:
                assert client
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
            elif metric_type == MetricType.ContextPrecision:
                raise NotImplementedError
            elif metric_type == MetricType.ContextRecall:
                raise NotImplementedError
            elif metric_type == MetricType.ContextRelevance:
                raise NotImplementedError
            elif metric_type == MetricType.Faithfulness:
                raise NotImplementedError
            elif metric_type == MetricType.Grammaticality:
                raise NotImplementedError
            elif metric_type == MetricType.Hallucination:
                raise NotImplementedError
            elif metric_type == MetricType.Summarization:
                raise NotImplementedError
            elif metric_type == MetricType.Toxicity:
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
    datum_filter = schemas.Filter(**evaluation.filters)
    # groundtruth_filter = datum_filter.model_copy()
    prediction_filter = datum_filter.model_copy()

    # prediction_filter.model_names = [evaluation.model_name]
    parameters = schemas.EvaluationParameters(**evaluation.parameters)

    # load task type into filters
    # datum_filter.task_types = [parameters.task_type]
    # groundtruth_filter.task_types = [parameters.task_type]
    # prediction_filter.task_types = [parameters.task_type]

    # get llm api params
    llm_api_params = parameters.llm_api_params

    # TODO What should I do here?
    # log_evaluation_item_counts(
    #     db=db,
    #     evaluation=evaluation,
    #     prediction_filter=prediction_filter,
    #     groundtruth_filter=groundtruth_filter,
    # )

    assert parameters.metrics_to_return

    metrics = _compute_text_generation_metrics(
        db=db,
        datum_filter=datum_filter,
        prediction_filter=prediction_filter,
        metrics_to_return=parameters.metrics_to_return,
        llm_api_params=llm_api_params,
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
