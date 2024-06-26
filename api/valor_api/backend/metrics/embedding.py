from collections import defaultdict

from scipy.stats import cramervonmises_2samp, ks_2samp
from sqlalchemy import and_, func, select
from sqlalchemy.orm import Session

from valor_api import schemas
from valor_api.backend import core, models
from valor_api.backend.metrics.metric_utils import (
    create_metric_mappings,
    get_or_create_row,
    log_evaluation_duration,
    log_evaluation_item_counts,
    prepare_filter_for_evaluation,
    validate_computation,
)
from valor_api.schemas import CramerVonMisesMetric, KolmgorovSmirnovMetric


def _get_embeddings(
    db: Session,
    dataset: models.Dataset,
    model: models.Model,
    label: models.Label,
    limit: int,
):
    datums = (
        select(models.Datum.id.label("id"))
        .join(
            models.Annotation,
            models.Annotation.datum_id == models.Datum.id,
        )
        .join(
            models.GroundTruth,
            models.GroundTruth.annotation_id == models.Annotation.id,
        )
        .join(
            models.Label,
            and_(
                models.Label.id == models.GroundTruth.label_id,
                models.Label.key == label.key,
                models.Label.value == label.value,
            ),
        )
        .where(models.Datum.dataset_id == dataset.id)
        .distinct()
        .subquery()
    )

    n_datums = db.scalar(func.count(datums.c.id))

    if limit > 0 and n_datums > limit:
        query = (
            select(models.Embedding)
            .join(
                models.Annotation,
                and_(
                    models.Annotation.embedding_id == models.Embedding.id,
                    models.Annotation.model_id == model.id,
                ),
            )
            .join(datums, datums.c.id == models.Annotation.datum_id)
            .order_by(func.random())
            .limit(limit=limit)
        )
    else:
        query = (
            select(models.Embedding)
            .join(
                models.Annotation,
                and_(
                    models.Annotation.embedding_id == models.Embedding.id,
                    models.Annotation.model_id == model.id,
                ),
            )
            .join(datums, datums.c.id == models.Annotation.datum_id)
            .order_by(func.random())
        )

    return (query.cte(), n_datums)


def _compute_self_distances(
    db: Session,
    dataset: models.Dataset,
    model: models.Model,
    label: models.Label,
    limit: int,
):
    embeddings, n_items = _get_embeddings(
        db=db,
        dataset=dataset,
        model=model,
        label=label,
        limit=limit * 2,
    )

    split_idx = n_items // 2
    split1 = select(embeddings).limit(split_idx).subquery()
    split2 = select(embeddings).limit(split_idx).offset(split_idx).subquery()

    return db.scalars(
        select(split1.c.value.l2_distance(split2.c.value))
        .select_from(split1)
        .join(split2, split2.c.id != split1.c.id)
    ).all()


def _compute_distances(
    db: Session,
    model: models.Model,
    query_dataset: models.Dataset,
    ref_label: models.Label,
    reference_dataset: models.Dataset,
    query_label: models.Label,
    limit: int,
):
    reference_embeddings, _ = _get_embeddings(
        db=db,
        dataset=reference_dataset,
        model=model,
        label=ref_label,
        limit=limit,
    )

    query_embeddings, _ = _get_embeddings(
        db=db,
        dataset=query_dataset,
        model=model,
        label=query_label,
        limit=limit,
    )

    return db.scalars(
        select(
            reference_embeddings.c.value.l2_distance(query_embeddings.c.value)
        )
        .select_from(reference_embeddings)
        .join(
            query_embeddings,
            query_embeddings.c.id != reference_embeddings.c.id,
        )
    ).all()


def _compute_metrics(
    db: Session,
    query_dataset: models.Dataset,
    reference_dataset: models.Dataset | None,
    model: models.Model,
    labels: list[models.Label],
    limit: int,
):
    cvm_statistics = defaultdict(dict)
    cvm_pvalues = defaultdict(dict)
    ks_statistics = defaultdict(dict)
    ks_pvalues = defaultdict(dict)

    no_references = reference_dataset is None
    reference_dataset = query_dataset

    for ref_label in labels:
        reference_distances = _compute_self_distances(
            db=db,
            dataset=reference_dataset,
            model=model,
            label=ref_label,
            limit=limit,
        )
        for query_label in labels:
            if ref_label == query_label and no_references:
                self_distances = _compute_self_distances(
                    db=db,
                    dataset=reference_dataset,
                    model=model,
                    label=ref_label,
                    limit=2 * limit,
                )
                split_idx = len(self_distances) // 2
                cvm = cramervonmises_2samp(
                    self_distances[:split_idx], self_distances[split_idx:]
                )
                ks = ks_2samp(
                    self_distances[:split_idx], self_distances[split_idx:]
                )
            else:
                query_distances = _compute_distances(
                    db=db,
                    model=model,
                    query_dataset=reference_dataset,
                    query_label=query_label,
                    reference_dataset=reference_dataset,
                    ref_label=ref_label,
                    limit=limit,
                )
                cvm = cramervonmises_2samp(
                    reference_distances,
                    query_distances,
                )
                ks = ks_2samp(
                    reference_distances,
                    query_distances,
                )

            cvm_statistics[ref_label.value][query_label.value] = cvm.statistic
            cvm_pvalues[ref_label.value][query_label.value] = cvm.pvalue

            ks_statistics[ref_label.value][query_label.value] = ks.statistic
            ks_pvalues[ref_label.value][query_label.value] = ks.pvalue

    cvm_metric = CramerVonMisesMetric(
        statistics=cvm_statistics,
        pvalues=cvm_pvalues,
    )
    ks_metric = KolmgorovSmirnovMetric(
        statistics=ks_statistics,
        pvalues=ks_pvalues,
    )
    return (cvm_metric, ks_metric)


@validate_computation
def compute_embedding_metrics(
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

    # TODO - implement filtering
    dataset = core.fetch_dataset(db=db, name=evaluation.dataset_names[-1])
    model = core.fetch_model(db=db, name=evaluation.model_name)
    labels = core.fetch_labels(
        db=db,
        filters=schemas.Filter(
            datasets=schemas.Condition(
                lhs=schemas.Symbol(name=schemas.SupportedSymbol.DATASET_NAME),
                rhs=schemas.Value.infer(dataset.name),
                op=schemas.FilterOperator.EQ,
            )
        ),
    )

    cvm, ks = _compute_metrics(
        db=db,
        query_dataset=dataset,
        reference_dataset=None,
        model=model,
        labels=list(labels),
        limit=parameters.limit,
        # prediction_filter=prediction_filter,
        # groundtruth_filter=groundtruth_filter,
        # label_map=parameters.label_map,
        # metrics_to_return=parameters.metrics_to_return,
    )

    metric_mappings = create_metric_mappings(
        db=db,
        metrics=[cvm, ks],
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
