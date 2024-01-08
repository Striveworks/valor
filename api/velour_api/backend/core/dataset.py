import json

from geoalchemy2.functions import ST_AsGeoJSON
from sqlalchemy import and_, select
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session

from velour_api import exceptions, schemas
from velour_api.backend import models
from velour_api.backend.core.label import get_groundtruth_labels_in_dataset


def create_dataset(
    db: Session,
    dataset: schemas.Dataset,
):
    """
    Creates a dataset.

    Parameters
    ----------
    db : Session
        The database Session to query against.
    dataset : schemas.Dataset
        The dataset to create.
    """
    try:
        row = models.Dataset(
            name=dataset.name,
            meta=dataset.metadata,
            geo=dataset.geospatial.wkt() if dataset.geospatial else None,
        )
        db.add(row)
        db.commit()
        return row
    except IntegrityError:
        db.rollback()
        raise exceptions.DatasetAlreadyExistsError(dataset.name)


def fetch_dataset(
    db: Session,
    name: str,
) -> models.Dataset:
    """
    Fetch a dataset from the database.

    Parameters
    ----------
    db : Session
        The database Session you want to query against.
    name : str
        The name of the dataset.

    Returns
    ----------
    models.Dataset
        The requested dataset.

    """

    dataset = (
        db.query(models.Dataset)
        .where(models.Dataset.name == name)
        .one_or_none()
    )
    if dataset is None:
        raise exceptions.DatasetDoesNotExistError(name)
    return dataset


def get_n_datums_in_dataset(db: Session, name: str) -> int:
    """Returns the number of datums in a dataset."""
    return (
        db.query(models.Datum)
        .join(models.Dataset)
        .where(models.Dataset.name == name)
        .count()
    )


def get_n_groundtruth_annotations(db: Session, name: str) -> int:
    """Returns the number of groundtruth annotations in a dataset."""
    return (
        db.query(models.Annotation)
        .join(models.GroundTruth)
        .join(models.Datum)
        .join(models.Dataset)
        .where(models.Dataset.name == name)
        .count()
    )


def get_n_groundtruth_bounding_boxes_in_dataset(db: Session, name: str) -> int:
    return (
        db.query(models.Annotation.id)
        .join(models.GroundTruth)
        .join(models.Datum)
        .join(models.Dataset)
        .where(
            and_(
                models.Dataset.name == name, models.Annotation.box.isnot(None)
            )
        )
        .distinct()
        .count()
    )


def get_n_groundtruth_polygons_in_dataset(db: Session, name: str) -> int:
    return (
        db.query(models.Annotation.id)
        .join(models.GroundTruth)
        .join(models.Datum)
        .join(models.Dataset)
        .where(
            and_(
                models.Dataset.name == name,
                models.Annotation.polygon.isnot(None),
            )
        )
        .distinct()
        .count()
    )


def get_n_groundtruth_multipolygons_in_dataset(db: Session, name: str) -> int:
    return (
        db.query(models.Annotation.id)
        .join(models.GroundTruth)
        .join(models.Datum)
        .join(models.Dataset)
        .where(
            and_(
                models.Dataset.name == name,
                models.Annotation.multipolygon.isnot(None),
            )
        )
        .distinct()
        .count()
    )


def get_n_groundtruth_rasters_in_dataset(db: Session, name: str) -> int:
    return (
        db.query(models.Annotation.id)
        .join(models.GroundTruth)
        .join(models.Datum)
        .join(models.Dataset)
        .where(
            and_(
                models.Dataset.name == name,
                models.Annotation.raster.isnot(None),
            )
        )
        .distinct()
        .count()
    )


def get_unique_task_types_in_dataset(db: Session, name: str) -> list[str]:
    return db.scalars(
        select(models.Annotation.task_type)
        .join(models.GroundTruth)
        .join(models.Datum)
        .join(models.Dataset)
        .where(models.Dataset.name == name)
        .distinct()
    ).all()


def get_unique_datum_metadata_in_dataset(db: Session, name: str) -> list[str]:
    md = db.scalars(
        select(models.Datum.meta)
        .join(models.Dataset)
        .where(models.Dataset.name == name)
        .distinct()
    ).all()

    # remove trivial metadata
    md = [m for m in md if m != {}]
    return md


def get_unique_groundtruth_annotation_metadata_in_dataset(
    db: Session, name: str
) -> list[str]:
    md = db.scalars(
        select(models.Annotation.meta)
        .join(models.GroundTruth)
        .join(models.Datum)
        .join(models.Dataset)
        .where(models.Dataset.name == name)
        .distinct()
    ).all()

    # remove trivial metadata
    md = [m for m in md if m != {}]
    return md


def get_dataset_summary(db: Session, name: str) -> schemas.DatasetSummary:
    return schemas.DatasetSummary(
        name=name,
        num_datums=get_n_datums_in_dataset(db, name),
        num_annotations=get_n_groundtruth_annotations(db, name),
        num_bounding_boxes=get_n_groundtruth_bounding_boxes_in_dataset(
            db, name
        ),
        num_polygons=get_n_groundtruth_polygons_in_dataset(db, name),
        num_groundtruth_multipolygons=get_n_groundtruth_multipolygons_in_dataset(
            db, name
        ),
        num_rasters=get_n_groundtruth_rasters_in_dataset(db, name),
        task_types=get_unique_task_types_in_dataset(db, name),
        labels=get_groundtruth_labels_in_dataset(db, name),
        datum_metadata=get_unique_datum_metadata_in_dataset(db, name),
        annotation_metadata=get_unique_groundtruth_annotation_metadata_in_dataset(
            db, name
        ),
    )


def get_dataset(
    db: Session,
    name: str,
) -> schemas.Dataset:
    """
    Fetch a dataset.

    Parameters
    ----------
    db : Session
        The database Session to query against.
    name : str
        The name of the dataset.

    Returns
    ----------
    schemas.Dataset
        The requested dataset.
    """
    dataset = fetch_dataset(db, name=name)
    geo_dict = (
        schemas.geojson.from_dict(
            json.loads(db.scalar(ST_AsGeoJSON(dataset.geo)))
        )
        if dataset.geo
        else None
    )
    return schemas.Dataset(
        id=dataset.id,
        name=dataset.name,
        metadata=dataset.meta,
        geospatial=geo_dict,
    )


def get_datasets(
    db: Session,
) -> list[schemas.Dataset]:
    """
    Fetch all datasets.

    Parameters
    ----------
    db : Session
        The database Session to query against.

    Returns
    ----------
    List[schemas.Dataset]
        A list of all datasets.
    """
    return [
        get_dataset(db, name)
        for name in db.scalars(select(models.Dataset.name)).all()
    ]


def delete_dataset(
    db: Session,
    name: str,
):
    """
    Delete a dataset.

    Parameters
    ----------
    db : Session
        The database Session to query against.
    name : str
        The name of the dataset.
    """
    dataset = fetch_dataset(db, name=name)
    try:
        db.delete(dataset)
        db.commit()
    except IntegrityError as e:
        db.rollback()
        raise e
