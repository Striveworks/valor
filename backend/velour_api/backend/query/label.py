from sqlalchemy import and_, or_, select
from sqlalchemy.orm import Session

from velour_api import enums, schemas
from velour_api.backend import core, models, query, ops


def get_labels(
    db: Session,
    request_info: schemas.Filter | None,
) -> list[schemas.Label]:
    """Returns a list of unique labels from a union of sources (dataset, model, datum, annotation) optionally filtered by (label key, task_type)."""

    if request_info is None:
        labels = db.query(models.Label).all()
    else:
        labels = (
            ops.BackendQuery.label()
            .filter(request_info)
            .all(db)
        )

    return [
        schemas.Label(
            key=label.key,
            value=label.value,
        )
        for label in labels
    ]


def _get_dataset_labels(db: Session, dataset_name: str) -> set[schemas.Label]:
    return {
        schemas.Label(key=label[0], value=label[1])
        for label in (
            db.query(models.Label.key, models.Label.value)
            .join(models.GroundTruth, models.GroundTruth.label_id == models.Label.id)
            .join(models.Annotation, models.Annotation.id == models.GroundTruth.annotation_id)
            .join(models.Datum, models.Datum.id == models.Annotation.datum_id)
            .join(models.Dataset, models.Dataset.id == models.Dataset.id)
            .where(
                and_(
                    models.Dataset.name == dataset_name,
                    models.Annotation.model_id.is_(None),
                )
            )
            .all()
        )
    }


def _get_model_labels(db: Session, dataset_name: str, model_name: str) -> set[schemas.Label]:
    return {
        schemas.Label(key=label[0], value=label[1])
        for label in (
            db.query(models.Label.key, models.Label.value)
            .join(models.Prediction, models.Prediction.label_id == models.Label.id)
            .join(models.Annotation, models.Annotation.id == models.Prediction.annotation_id)
            .join(models.Datum, models.Datum.id == models.Annotation.datum_id)
            .join(models.Dataset, models.Dataset.id == models.Dataset.id)
            .join(models.Model, models.Model.id == models.Annotation.model_id)
            .where(
                and_(
                    models.Dataset.name == dataset_name,
                    models.Model.name == model_name,
                )
            )
            .all()
        )
    }


def _get_dataset_label_keys(db: Session, dataset_name: str) -> set[str]:
    return {
        label[0]
        for label in (
            db.query(models.Label.key)
            .join(models.GroundTruth, models.GroundTruth.label_id == models.Label.id)
            .join(models.Annotation, models.Annotation.id == models.GroundTruth.annotation_id)
            .join(models.Datum, models.Datum.id == models.Annotation.datum_id)
            .join(models.Dataset, models.Dataset.id == models.Dataset.id)
            .where(
                and_(
                    models.Dataset.name == dataset_name,
                    models.Annotation.model_id.is_(None),
                )
            )
            .all()
        )
    }


def _get_model_label_keys(db: Session, dataset_name: str, model_name: str) -> set[str]:
    return {
        label[0]
        for label in (
            db.query(models.Label.key)
            .join(models.Prediction, models.Prediction.label_id == models.Label.id)
            .join(models.Annotation, models.Annotation.id == models.Prediction.annotation_id)
            .join(models.Datum, models.Datum.id == models.Annotation.datum_id)
            .join(models.Dataset, models.Dataset.id == models.Dataset.id)
            .join(models.Model, models.Model.id == models.Annotation.model_id)
            .where(
                and_(
                    models.Dataset.name == dataset_name,
                    models.Model.name == model_name,
                )
            )
            .all()
        )
    }
    

def get_joint_labels(
    db: Session,
    dataset_name: str,
    model_name: str,
) -> list[schemas.Label]:
    return list(_get_dataset_labels(db, dataset_name).intersection(_get_model_labels(db, dataset_name, model_name)))


def get_joint_keys(
    db: Session,
    dataset_name: str,
    model_name: str,
) -> list[schemas.Label]:
    return list(_get_dataset_label_keys(db, dataset_name).intersection(_get_model_label_keys(db, dataset_name, model_name)))

def get_disjoint_labels(
    db: Session,
    dataset_name: str,
    model_name: str,
) -> dict[str, list[schemas.Label]]:
    """Returns tuple with elements (dataset, model) which contain lists of Labels. """

    # get labels
    ds_labels = _get_dataset_labels(db, dataset_name)
    md_labels = _get_model_labels(db, dataset_name, model_name)
    
    # set operation to get disjoint sets wrt the lhs operand
    ds_unique = list(ds_labels - md_labels)
    md_unique = list(md_labels - ds_labels)

    # returns tuple of label lists 
    return (ds_unique, md_unique)


def get_disjoint_keys(
    db: Session,
    dataset_name: str,
    model_name: str,
) -> dict[str, list[schemas.Label]]:
    """Returns tuple with elements (dataset, model) which contain lists of Labels. """

    ds_labels = _get_dataset_label_keys(db, dataset_name)
    md_labels = _get_model_label_keys(db, dataset_name, model_name)

    # set operation to get disjoint sets wrt the lhs operand
    ds_unique = list(ds_labels - md_labels)
    md_unique = list(md_labels - ds_labels)

    # returns tuple of label lists 
    return (ds_unique, md_unique)


def get_label_distribution(
    db: Session,
) -> list[schemas.LabelDistribution]:
    return []
