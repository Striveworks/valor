from sqlalchemy import and_, select
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session

from valor_api import enums, exceptions, schemas
from valor_api.backend import models
from valor_api.backend.core.annotation import delete_dataset_annotations
from valor_api.backend.core.evaluation import (
    count_active_evaluations,
    delete_evaluations,
)
from valor_api.backend.core.groundtruth import delete_groundtruths
from valor_api.backend.core.label import get_labels
from valor_api.backend.core.prediction import delete_dataset_predictions
from valor_api.backend.query import Query
from valor_api.schemas.core import MetadataType


def _load_dataset_schema(
    db: Session,
    dataset: models.Dataset,
) -> schemas.Dataset:
    """Convert database row to schema."""
    return schemas.Dataset(
        id=dataset.id,
        name=dataset.name,
        metadata=dataset.meta,
    )


def create_dataset(
    db: Session,
    dataset: schemas.Dataset,
) -> models.Dataset:
    """
    Creates a dataset.

    Parameters
    ----------
    db : Session
        The database Session to query against.
    dataset : schemas.Dataset
        The dataset to create.

    Returns
    -------
    models.Dataset
        The created dataset row.

    Raises
    ------
    exceptions.DatasetAlreadyExistsError
        If a dataset with the provided name already exists.
    """
    try:
        row = models.Dataset(
            name=dataset.name,
            meta=dataset.metadata,
            status=enums.TableStatus.CREATING,
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

    Raises
    ------
    exceptions.DatasetDoesNotExistError
        If a dataset with the provided name does not exist.
    """
    dataset = (
        db.query(models.Dataset)
        .where(
            and_(
                models.Dataset.name == name  # type: ignore https://github.com/microsoft/pyright/issues/5062
                and models.Dataset.status != enums.TableStatus.DELETING  # type: ignore nhttps://github.com/microsoft/pyright/issues/5062
            )
        )
        .one_or_none()
    )
    if dataset is None:
        raise exceptions.DatasetDoesNotExistError(name)
    return dataset


def get_dataset(
    db: Session,
    name: str,
) -> schemas.Dataset:
    """
    Gets a dataset by name.

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
    dataset = fetch_dataset(db=db, name=name)
    return _load_dataset_schema(db=db, dataset=dataset)


def get_datasets(
    db: Session,
    filters: schemas.Filter | None = None,
) -> list[schemas.Dataset]:
    """
    Get datasets with optional filter constraint.

    Parameters
    ----------
    db : Session
        The database Session to query against.
    filters : schemas.Filter, optional
        Optional filter to constrain against.

    Returns
    ----------
    list[schemas.Dataset]
        A list of all datasets.
    """
    datasets_subquery = (
        Query(models.Dataset.id.label("id")).filter(filters).any()
    )

    if datasets_subquery is None:
        raise RuntimeError(
            "psql unexpectedly returned None instead of a Subquery."
        )

    datasets = (
        db.query(models.Dataset)
        .where(models.Dataset.id == datasets_subquery.c.id)
        .all()
    )
    return [
        _load_dataset_schema(db=db, dataset=dataset) for dataset in datasets
    ]


def get_dataset_status(
    db: Session,
    name: str,
) -> enums.TableStatus:
    """
    Get the status of a dataset.

    Parameters
    ----------
    db : Session
        The database session.
    name : str
        The name of the dataset.

    Returns
    -------
    enums.TableStatus
        The status of the dataset.
    """
    dataset = fetch_dataset(db, name)
    return enums.TableStatus(dataset.status)


def set_dataset_status(
    db: Session,
    name: str,
    status: enums.TableStatus,
):
    """
    Sets the status of a dataset.

    Parameters
    ----------
    db : Session
        The database session.
    name : str
        The name of the dataset.
    status : enums.TableStatus
        The desired dataset state.

    Raises
    ------
    exceptions.DatasetStateError
        If an illegal transition is requested.
    exceptions.EvaluationRunningError
        If the requested state is DELETING while an evaluation is running.
    """
    dataset = fetch_dataset(db, name)
    active_status = enums.TableStatus(dataset.status)

    if status == active_status:
        return

    if status not in active_status.next():
        raise exceptions.DatasetStateError(name, active_status, status)

    if status == enums.TableStatus.DELETING:
        if count_active_evaluations(
            db=db,
            dataset_names=[name],
        ):
            raise exceptions.EvaluationRunningError(dataset_name=name)

    try:
        dataset.status = status
        db.commit()
    except Exception as e:
        db.rollback()
        raise e


def get_n_datums_in_dataset(db: Session, name: str) -> int:
    """Returns the number of datums in a dataset."""
    return (
        db.query(models.Datum)
        .join(models.Dataset)
        .where(models.Dataset.name == name)
        .count()
    )


def get_n_groundtruth_annotations(db: Session, name: str) -> int:
    """Returns the number of ground truth annotations in a dataset."""
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


def get_unique_task_types_in_dataset(
    db: Session, name: str
) -> list[enums.TaskType]:
    return db.scalars(
        select(models.Annotation.task_type)
        .join(models.GroundTruth)
        .join(models.Datum)
        .join(models.Dataset)
        .where(models.Dataset.name == name)
        .distinct()
    ).all()  # type: ignore - sqlalchemy typing issue


def get_unique_datum_metadata_in_dataset(
    db: Session, name: str
) -> list[MetadataType]:
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
) -> list[MetadataType]:
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
    gt_labels = get_labels(
        db,
        schemas.Filter(dataset_names=[name]),
        ignore_predictions=True,
    )
    return schemas.DatasetSummary(
        name=name,
        num_datums=get_n_datums_in_dataset(db, name),
        num_annotations=get_n_groundtruth_annotations(db, name),
        num_bounding_boxes=get_n_groundtruth_bounding_boxes_in_dataset(
            db, name
        ),
        num_polygons=get_n_groundtruth_polygons_in_dataset(db, name),
        num_rasters=get_n_groundtruth_rasters_in_dataset(db, name),
        task_types=get_unique_task_types_in_dataset(db, name),
        labels=list(gt_labels),
        datum_metadata=get_unique_datum_metadata_in_dataset(db, name),
        annotation_metadata=get_unique_groundtruth_annotation_metadata_in_dataset(
            db, name
        ),
    )


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
    set_dataset_status(db, name, enums.TableStatus.DELETING)
    dataset = fetch_dataset(db, name=name)

    delete_evaluations(db=db, dataset_names=[name])
    delete_dataset_predictions(db, dataset)
    delete_groundtruths(db, dataset)
    delete_dataset_annotations(db, dataset)

    try:
        db.delete(dataset)
        db.commit()
    except IntegrityError as e:
        db.rollback()
        raise e
