from sqlalchemy import and_, or_, select
from sqlalchemy.orm import Session

from velour_api import enums, schemas
from velour_api.backend import core, models, ops, query


def get_labels(
    db: Session,
    dataset_name: str = None,
    model_name: str = None,
    datum_uid: str = None,
    keys: list[str] = [],
    task_types: list[enums.TaskType] = [],
    filter_by_annotation_type: list[enums.AnnotationType] = [],
    filter_by_metadata: list[schemas.MetaDatum] = [],
) -> list[schemas.Label]:
    """Returns a list of unique labels from a union of sources (dataset, model, datum, annotation) optionally filtered by (label key, task_type)."""

    dataset = core.get_dataset(db, dataset_name) if dataset_name else None
    model = core.get_model(db, model_name) if model_name else None
    datum = core.get_datum(db, uid=datum_uid) if datum_uid else None

    qf = ops.QueryFilter()
    qf.filter_by_id(
        dataset,
        model,
        datum,
    )
    qf.filter_by_str(models.Label.key, keys)
    qf.filter_by_task_types(task_types)
    qf.filter_by_annotation_types()

    labels = core.get_labels(
        db,
        dataset=dataset,
        model=model,
        datum=datum,
        annotation=annotation,
        filter_by_key=filter_by_key,
        filter_by_task_type=filter_by_task_type,
        filter_by_annotation_type=filter_by_annotation_type,
        filter_by_metadata=filter_by_metadata,
    )

    return [
        schemas.Label(
            key=label.key,
            value=label.value,
        )
        for label in labels
    ]


def get_scored_labels(
    db: Session,
    annotation: models.Annotation,
) -> list[schemas.ScoredLabel]:
    scored_labels = (
        db.query(models.Prediction.score, models.Label.key, models.Label.value)
        .select_from(models.Prediction)
        .join(models.Label, models.Label.id == models.Prediction.label_id)
        .where(models.Prediction.annotation_id == annotation.id)
        .all()
    )

    return [
        schemas.ScoredLabel(
            label=schemas.Label(
                key=label[1],
                value=label[2],
            ),
            score=label[0],
        )
        for label in scored_labels
    ]


def get_disjoint_labels(
    db: Session,
    dataset_name: str,
    model_name: str,
) -> dict[str, list[schemas.Label]]:
    """Returns tuple(gt_labels, pd_labels)"""

    dataset = core.get_dataset(db, dataset_name)
    model = core.get_model(db, model_name)

    ds_labels = set(query.get_labels(db, dataset=dataset))
    md_labels = set(query.get_labels(db, dataset=dataset, model=model))

    ds_unique = list(ds_labels - md_labels)
    md_unique = list(md_labels - ds_labels)

    return {
        "dataset": ds_unique,
        "model": md_unique,
    }


def get_label_distribution(
    db: Session,
) -> list[schemas.LabelDistribution]:
    return []
