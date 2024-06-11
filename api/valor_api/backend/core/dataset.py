from sqlalchemy import and_, desc, func, select
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session

from valor_api import api_utils, enums, exceptions, schemas
from valor_api.backend import core, models
from valor_api.backend.query import generate_select
from valor_api.schemas.types import MetadataType


def _load_dataset_schema(
    db: Session,
    dataset: models.Dataset,
) -> schemas.Dataset:
    """Convert database row to schema."""
    return schemas.Dataset(name=dataset.name, metadata=dataset.meta)


def _validate_dataset_contains_datums(db: Session, name: str):
    """
    Validates whether a dataset contains at least one datum.

    Raises
    ------
    DatasetEmptyError
        If the dataset contains no datums.
    """
    datum_count = (
        db.query(func.count(models.Datum.id))
        .join(models.Dataset, models.Dataset.id == models.Datum.dataset_id)
        .where(models.Dataset.name == name)
        .scalar()
    )
    if datum_count == 0:
        raise exceptions.DatasetEmptyError(name)


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
                models.Dataset.name == name,
                models.Dataset.status != enums.TableStatus.DELETING,
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


def get_paginated_datasets(
    db: Session,
    filters: schemas.Filter | None = None,
    offset: int = 0,
    limit: int = -1,
) -> tuple[list[schemas.Dataset], dict[str, str]]:
    """
    Get datasets with optional filter constraint.

    Parameters
    ----------
    db : Session
        The database Session to query against.
    filters : schemas.Filter, optional
        Optional filter to constrain against.
    offset : int, optional
        The start index of the items to return.
    limit : int, optional
        The number of items to return. Returns all items when set to -1.

    Returns
    ----------
    tuple[list[schemas.Dataset], dict[str, str]]
        A tuple containing the datasets and response headers to return to the user.
    """
    if offset < 0 or limit < -1:
        raise ValueError(
            "Offset should be an int greater than or equal to zero. Limit should be an int greater than or equal to -1."
        )

    datasets_subquery = generate_select(
        models.Dataset.id.label("id"),
        filters=filters,
        label_source=models.GroundTruth,
    ).subquery()

    if datasets_subquery is None:
        raise RuntimeError(
            "psql unexpectedly returned None instead of a Subquery."
        )

    count = (
        db.query(func.count(models.Dataset.id))
        .where(models.Dataset.id == datasets_subquery.c.id)
        .scalar()
    )

    if offset > count:
        raise ValueError(
            "Offset is greater than the total number of items returned in the query."
        )

    # return all rows when limit is -1
    if limit == -1:
        limit = count

    datasets = (
        db.query(models.Dataset)
        .where(
            and_(
                models.Dataset.id == datasets_subquery.c.id,
                models.Dataset.status != enums.TableStatus.DELETING,
            )
        )
        .order_by(desc(models.Dataset.created_at))
        .offset(offset)
        .limit(limit)
        .all()
    )

    content = [
        _load_dataset_schema(db=db, dataset=dataset) for dataset in datasets
    ]

    headers = api_utils._get_pagination_header(
        offset=offset,
        number_of_returned_items=len(datasets),
        total_number_of_items=count,
    )

    return (content, headers)


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
    dataset = (
        db.query(models.Dataset)
        .where(models.Dataset.name == name)
        .one_or_none()
    )
    if dataset is None:
        raise exceptions.DatasetDoesNotExistError(name)
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
        if core.count_active_evaluations(
            db=db,
            dataset_names=[name],
        ):
            raise exceptions.EvaluationRunningError(dataset_name=name)
    elif status == enums.TableStatus.FINALIZED:
        _validate_dataset_contains_datums(db=db, name=name)

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
        .where(
            and_(
                models.Dataset.name == name,
                models.Dataset.status != enums.TableStatus.DELETING,
            )
        )
        .count()
    )


def get_n_groundtruth_annotations(db: Session, name: str) -> int:
    """Returns the number of ground truth annotations in a dataset."""
    return (
        db.query(models.Annotation)
        .join(models.GroundTruth)
        .join(models.Datum)
        .join(models.Dataset)
        .where(
            and_(
                models.Dataset.name == name,
                models.Dataset.status != enums.TableStatus.DELETING,
            )
        )
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
                models.Dataset.name == name,
                models.Dataset.status != enums.TableStatus.DELETING,
                models.Annotation.box.isnot(None),
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
                models.Dataset.status != enums.TableStatus.DELETING,
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
                models.Dataset.status != enums.TableStatus.DELETING,
                models.Annotation.raster.isnot(None),
            )
        )
        .distinct()
        .count()
    )


def get_unique_task_types_in_dataset(
    db: Session, name: str
) -> list[enums.TaskType]:
    """
    Fetch the unique implied task types associated with the annotation in a dataset.

    Parameters
    -------
    db : Session
        The database Session you want to query against.
    name : str
        The name of the dataset to query for.
    """
    task_types = (
        db.query(
            func.jsonb_array_elements_text(
                models.Annotation.implied_task_types
            )
        )
        .select_from(models.Annotation)
        .join(models.Datum, models.Datum.id == models.Annotation.datum_id)
        .join(models.Dataset, models.Dataset.id == models.Datum.dataset_id)
        .where(
            and_(
                models.Dataset.name == name,
                models.Dataset.status != enums.TableStatus.DELETING,
            )
        )
        .distinct()
        .all()
    )
    return [
        enums.TaskType(task_type_tuple[0]) for task_type_tuple in task_types
    ]


def get_unique_datum_metadata_in_dataset(
    db: Session, name: str
) -> list[MetadataType]:
    md = db.scalars(
        select(models.Datum.meta)
        .join(models.Dataset)
        .where(
            and_(
                models.Dataset.name == name,
                models.Dataset.status != enums.TableStatus.DELETING,
            )
        )
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
        .where(
            and_(
                models.Dataset.name == name,
                models.Dataset.status != enums.TableStatus.DELETING,
            )
        )
        .distinct()
    ).all()

    # remove trivial metadata
    md = [m for m in md if m != {}]
    return md


def get_dataset_summary(db: Session, name: str) -> schemas.DatasetSummary:
    gt_labels = core.get_labels(
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
    dataset = fetch_dataset(db, name=name)
    set_dataset_status(db, name, enums.TableStatus.DELETING)

    core.delete_evaluations(db=db, dataset_names=[name])
    core.delete_dataset_predictions(db, dataset)
    core.delete_groundtruths(db, dataset)
    core.delete_dataset_annotations(db, dataset)

    try:
        db.delete(dataset)
        db.commit()
    except IntegrityError as e:
        db.rollback()
        raise e
