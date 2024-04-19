from geoalchemy2.functions import ST_Count, ST_MapAlgebra
from sqlalchemy.orm import Session, aliased
from sqlalchemy.sql import Select, and_, func, select

from valor_api import enums, schemas
from valor_api.backend import core, models
from valor_api.backend.metrics.metric_utils import (
    create_grouper_mappings,
    create_metric_mappings,
    get_or_create_row,
    log_evaluation_duration,
    log_evaluation_item_counts,
    validate_computation,
)
from valor_api.backend.query import Query


def _compute_embedding_distance(
    db: Session,
    groundtruth_filter: schemas.Filter,
    prediction_filter: schemas.Filter,
):
    pd_embeddings = (
        Query(
            models.Annotation.embedding_id, models.Label.id.label("label_id")
        )
        .filter(prediction_filter)
        .predictions(as_subquery=False)
    ).subquery("preds")

    pds1 = aliased(pd_embeddings)
    pds2 = aliased(pd_embeddings)

    emb1 = aliased(models.Embedding)
    emb2 = aliased(models.Embedding)

    result = db.query(
        select(
            emb1.id,
            emb2.id,
            emb1.value.l2_distance(emb2.value).label("l2"),
            emb1.value.cosine_distance(emb2.value).label("cosine_distance"),
        )
        .select_from(pds1)
        .join(
            pds2,
            and_(
                pds1.c.label_id == pds2.c.label_id,
                pds1.c.embedding_id > pds2.c.embedding_id,
            ),
        )
        .join(emb1, emb1.id == pds1.c.embedding_id)
        .join(emb2, emb2.id == pds2.c.embedding_id)
        .subquery()
    ).all()

    average_l2 = sum([item[2] for item in result]) / len(result)
    average_cosine = sum([item[3] for item in result]) / len(result)

    return average_l2, average_cosine


@validate_computation
def compute_embedding_distance_metrics(
    *,
    db: Session,
    evaluation_id: int,
) -> int:
    """
    Create semantic segmentation metrics. This function is intended to be run using FastAPI's `BackgroundTasks`.

    Parameters
    ----------
    db : Session
        The database Session to query against.
    evaluation_id : int
        The job ID to create metrics for.
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

    log_evaluation_item_counts(
        db=db,
        evaluation=evaluation,
        prediction_filter=prediction_filter,
        groundtruth_filter=groundtruth_filter,
    )

    metrics = _compute_embedding_distance(
        db=db,
        prediction_filter=prediction_filter,
        groundtruth_filter=groundtruth_filter,
    )

    metric_mappings = create_metric_mappings(db, metrics, evaluation_id)
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
