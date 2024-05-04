import time
from typing import Any

from geoalchemy2.functions import ST_AsGeoJSON
from sqlalchemy import and_, delete, insert, select
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session

from valor_api import schemas
from valor_api.backend import models
from valor_api.backend.core.geometry import _raster_to_png_b64
from valor_api.backend.query import Query
from valor_api.enums import ModelStatus, TableStatus, TaskType


def _create_embedding(
    db: Session,
    value: list[float],
) -> int:
    """
    Creates a row in the embedding table.

    Parameters
    ----------
    db : Session
        The current database session.
    value : list[float]
        The embedding, represented as a list of type float.

    Returns
    -------
    int
        The row id of the embedding.
    """
    try:
        row = models.Embedding(value=value)
        db.add(row)
        db.commit()
    except IntegrityError as e:
        db.rollback()
        raise e
    return row.id


def _create_annotation(
    db: Session,
    annotation: schemas.Annotation,
    datum: models.Datum,
    model: models.Model | None = None,
) -> dict[str, Any]:
    """
    Convert an individual annotation's attributes into a dictionary for upload to psql.

    Parameters
    ----------
    annotation : schemas.Annotation
        The annotation tom ap.
    datum : models.Datum
        The datum associated with the annotation.
    model : models.Model, optional
        The model associated with the annotation

    Returns
    ----------
    dict[str, Any]
        A populated models.Annotation object.
    """
    box = None
    polygon = None
    raster = None
    embedding_id = None

    # task-based conversion
    match annotation.task_type:
        case TaskType.OBJECT_DETECTION:
            if annotation.bounding_box:
                box = annotation.bounding_box.to_wkt()
            if annotation.polygon:
                polygon = annotation.polygon.to_wkt()
            if annotation.raster:
                raster = annotation.raster.to_psql()
        case TaskType.SEMANTIC_SEGMENTATION:
            if annotation.raster:
                raster = annotation.raster.to_psql()
        case TaskType.EMBEDDING:
            if annotation.embedding:
                embedding_id = _create_embedding(
                    db=db, value=annotation.embedding
                )
        case _:
            pass

    mapping = {
        "datum_id": datum.id,
        "model_id": model.id if model else None,
        "task_type": annotation.task_type,
        "meta": annotation.metadata,
        "box": box,
        "polygon": polygon,
        "raster": raster,
        "embedding_id": embedding_id,
    }
    return mapping


def create_annotations(
    db: Session,
    annotations: list[list[schemas.Annotation]],
    datums: list[models.Datum],
    models: list[models.Model] | None = None,
) -> list[list[models.Annotation]]:
    """
    Create a list of annotations and associated labels in psql.

    Parameters
    ----------
    db
        The database Session you want to query against.
    annotations
        The list of annotations to create.
    datum
        The datum associated with the annotation.
    model
        The model associated with the annotation.

    Returns
    ----------
    list[list[models.annotation]]
        The model associated with the annotation.

    Raises
    ------
    exceptions.AnnotationAlreadyExistsError
        If the provided datum already has existing annotations for that dataset or model.
    """
    models = models or [None] * len(datums)

    assert len(models) == len(datums) == len(annotations)

    # create annotations
    start = time.time()
    annotation_mappings = [
        _create_annotation(
            db=db, annotation=annotation, datum=datum, model=model
        )
        for annotations_per_datum, datum, model in zip(
            annotations, datums, models
        )
        for annotation in annotations_per_datum
    ]
    print(f"create_annotation: {time.time() - start:.4f}")

    start = time.time()
    try:
        insert_stmt = (
            insert(models.Annotation)
            .values(annotation_mappings)
            .returning(models.Annotation.id)
        )
        annotation_id_list = db.execute(insert_stmt).scalars().all()
    except IntegrityError as e:
        db.rollback()
        raise e
    print(f"add_all: {time.time() - start:.4f}")

    annotation_ids = []
    idx = 0
    for annotations_per_datum in annotations:
        annotation_ids.append(
            annotation_id_list[idx : idx + len(annotations_per_datum)]
        )
        idx += len(annotations_per_datum)

    return annotation_ids


def create_skipped_annotations(
    db: Session,
    datums: list[models.Datum],
    model: models.Model,
):
    """
    Create a list of skipped annotations and associated labels in psql.

    Parameters
    ----------
    db : Session
        The database Session you want to query against.
    datums : List[schemas.Datum]
        The list of datums to create skipped annotations for.
    model : models.Model
        The model associated with the annotation.
    """
    annotation_list = [
        _create_annotation(
            db=db,
            annotation=schemas.Annotation(task_type=TaskType.SKIP),
            datum=datum,
            model=model,
        )
        for datum in datums
    ]
    try:
        db.add_all(annotation_list)
        db.commit()
    except IntegrityError as e:
        db.rollback()
        raise e


def get_annotation(
    db: Session,
    annotation: models.Annotation,
) -> schemas.Annotation:
    """
    Fetch an annotation from the database.

    Parameters
    ----------
    db : Session
        The database Session you want to query against.
    annotation : models.Annotation
        The annotation you want to fetch.

    Returns
    -------
    schemas.Annotation
        The requested annotation.
    """
    # retrieve all labels associated with annotation
    if annotation.model_id:
        q = Query(
            models.Label.key,
            models.Label.value,
            models.Prediction.score,
        ).predictions(as_subquery=False)
        q = q.where(models.Prediction.annotation_id == annotation.id)  # type: ignore - SQLAlchemy type issue
        labels = [
            schemas.Label(
                key=scored_label[0],
                value=scored_label[1],
                score=scored_label[2],
            )
            for scored_label in db.query(q.subquery()).all()
        ]
    else:
        q = Query(
            models.Label.key,
            models.Label.value,
        ).groundtruths(as_subquery=False)
        q = q.where(models.GroundTruth.annotation_id == annotation.id)  # type: ignore - SQLAlchemy type issue
        labels = [
            schemas.Label(key=label[0], value=label[1])
            for label in db.query(q.subquery()).all()
        ]

    # initialize
    box = None
    polygon = None
    raster = None
    embedding = None

    # bounding box
    if annotation.box is not None:
        box = schemas.Box.from_json(db.scalar(ST_AsGeoJSON(annotation.box)))

    # polygon
    if annotation.polygon is not None:
        polygon = schemas.Polygon.from_json(
            db.scalar(ST_AsGeoJSON(annotation.polygon))
        )

    # raster
    if annotation.raster is not None:
        datum = db.scalar(
            select(models.Datum).where(models.Datum.id == annotation.datum_id)
        )

        if datum is None:
            raise RuntimeError(
                "psql unexpectedly returned None instead of a Datum."
            )
        raster = schemas.Raster(
            mask=_raster_to_png_b64(db=db, raster=annotation.raster),
        )

    # embedding
    if annotation.embedding_id:
        embedding = db.scalar(
            select(models.Embedding.value).where(
                models.Embedding.id == annotation.embedding_id
            )
        )

    return schemas.Annotation(
        task_type=annotation.task_type,  # type: ignore - models.Annotation.task_type should be a string in psql
        labels=labels,
        metadata=annotation.meta,
        bounding_box=box,
        polygon=polygon,  # type: ignore - guaranteed to be a polygon in this case
        raster=raster,
        embedding=embedding,
    )


def get_annotations(
    db: Session,
    datum: models.Datum,
    model: models.Model | None = None,
) -> list[schemas.Annotation]:
    """
    Query psql to get all annotations for a particular datum.

    Parameters
    -------
    db : Session
        The database session to query against.
    datum : models.Datum
        The datum you want to fetch annotations for.
    model : models.Model
        The model you want to query against (optional).

    Returns
    ----------
    List[schemas.Annotation]
        A list of annotations.
    """
    model_expr = (
        models.Annotation.model_id.is_(None)
        if model is None
        else models.Annotation.model_id == model.id
    )
    return [
        get_annotation(db, annotation=annotation)
        for annotation in (
            db.query(models.Annotation)
            .where(
                and_(
                    model_expr,
                    models.Annotation.datum_id == datum.id,
                )
            )
            .all()
        )
    ]


def delete_dataset_annotations(
    db: Session,
    dataset: models.Dataset,
):
    """
    Delete all annotations from a dataset.

    Parameters
    ----------
    db : Session
        The database session.
    dataset : models.Dataset
        The dataset row that is being deleted.

    Raises
    ------
    RuntimeError
        If dataset is not in deletion state.
    """

    if dataset.status != TableStatus.DELETING:
        raise RuntimeError(
            f"Attempted to delete annotations from dataset `{dataset.name}` which has status `{dataset.status}`"
        )

    subquery = (
        select(models.Annotation.id.label("id"))
        .join(models.Datum, models.Datum.id == models.Annotation.datum_id)
        .where(models.Datum.dataset_id == dataset.id)
        .subquery()
    )
    delete_stmt = delete(models.Annotation).where(
        models.Annotation.id == subquery.c.id
    )

    try:
        db.execute(delete_stmt)
        db.commit()
    except IntegrityError as e:
        db.rollback()
        raise e


def delete_model_annotations(
    db: Session,
    model: models.Model,
):
    """
    Delete all annotations from a model.

    Parameters
    ----------
    db : Session
        The database session.
    model : models.Model
        The model row that is being deleted.

    Raises
    ------
    RuntimeError
        If dataset is not in deletion state.
    """

    if model.status != ModelStatus.DELETING:
        raise RuntimeError(
            f"Attempted to delete annotations from dataset `{model.name}` which is not being deleted."
        )

    subquery = (
        select(models.Annotation.id.label("id"))
        .where(models.Annotation.model_id == model.id)
        .subquery()
    )
    delete_stmt = delete(models.Annotation).where(
        models.Annotation.id == subquery.c.id
    )

    try:
        db.execute(delete_stmt)
        db.commit()
    except IntegrityError as e:
        db.rollback()
        raise e
