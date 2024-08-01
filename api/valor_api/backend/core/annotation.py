from geoalchemy2.functions import ST_AsGeoJSON
from sqlalchemy import ScalarSelect, and_, delete, insert, select
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session

from valor_api import schemas
from valor_api.backend import models
from valor_api.backend.core.geometry import _raster_to_png_b64
from valor_api.backend.query import generate_query
from valor_api.enums import ModelStatus, TableStatus, TaskType


def _format_box(box: schemas.Box | None) -> str | None:
    return box.to_wkt() if box else None


def _format_polygon(polygon: schemas.Polygon | None) -> str | None:
    return polygon.to_wkt() if polygon else None


def _format_raster(
    raster: schemas.Raster | None,
) -> ScalarSelect | bytes | None:
    return raster.to_psql() if raster else None


def _create_embedding(
    db: Session,
    value: list[float] | None,
) -> int | None:
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
    if not value:
        return None
    try:
        row = models.Embedding(value=value)
        db.add(row)
        db.commit()
    except IntegrityError as e:
        db.rollback()
        raise e
    return row.id


def _format_context(
    context: str | list[str] | None,
) -> list[str] | None:
    if isinstance(context, str):
        context = [context]
    return context


def create_annotations(
    db: Session,
    annotations: list[list[schemas.Annotation]],
    datum_ids: list[int],
    models_: list[models.Model] | list[None] | None = None,
) -> list[list[models.Annotation]]:
    """
    Create a list of annotations and associated labels in psql.

    Parameters
    ----------
    db : Session
        The database Session you want to query against.
    annotations : list[list[schemas.Annotation]]
        The list of annotations to create.
    datums : dict[tuple[int, str], int]
        A mapping of (dataset_id, datum_uid) to a datum's row id.
    models_: list[models.Model], optional
        The model(s) associated with the annotations.

    Returns
    ----------
    list[list[models.annotation]]
        The model associated with the annotation.

    Raises
    ------
    exceptions.AnnotationAlreadyExistsError
        If the provided datum already has existing annotations for that dataset or model.
    """

    # cache model ids
    models_ = models_ or [None] * len(datum_ids)
    model_ids = [
        model.id if isinstance(model, models.Model) else model
        for model in models_
    ]

    if not (len(model_ids) == len(datum_ids) == len(annotations)):
        raise ValueError("Length mismatch between annotation elements.")

    values = [
        {
            "datum_id": datum_id,
            "model_id": model_id,
            "meta": annotation.metadata,
            "box": _format_box(annotation.bounding_box),
            "polygon": _format_polygon(annotation.polygon),
            "raster": _format_raster(annotation.raster),
            "embedding_id": _create_embedding(
                db=db, value=annotation.embedding
            ),
            "text": None,
            "context": None,
            "is_instance": annotation.is_instance,
            "implied_task_types": annotation.implied_task_types,
        }
        for annotations_per_datum, datum_id, model_id in zip(
            annotations, datum_ids, model_ids
        )
        for annotation in annotations_per_datum
    ]

    try:
        insert_stmt = (
            insert(models.Annotation)
            .values(values)
            .returning(models.Annotation.id)
        )
        annotation_ids = list(db.execute(insert_stmt).scalars().all())
        db.commit()
    except IntegrityError as e:
        db.rollback()
        raise e

    grouped_annotation_row_ids = []
    idx = 0
    for annotations_per_datum in annotations:
        grouped_annotation_row_ids.append(
            annotation_ids[idx : idx + len(annotations_per_datum)]
        )
        idx += len(annotations_per_datum)

    return grouped_annotation_row_ids


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
        models.Annotation(
            datum_id=datum.id,
            model_id=model.id if model else None,
            meta=dict(),
            box=None,
            polygon=None,
            raster=None,
            embedding_id=None,
            text=None,
            context=None,
            is_instance=False,
            implied_task_types=[TaskType.EMPTY],
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
        query = generate_query(
            models.Label.key,
            models.Label.value,
            models.Prediction.score,
            db=db,
            label_source=models.Prediction,
        ).where(models.Prediction.annotation_id == annotation.id)
        labels = [
            schemas.Label(
                key=scored_label[0],
                value=scored_label[1],
                score=scored_label[2],
            )
            for scored_label in query.all()
        ]
    else:
        query = generate_query(
            models.Label.key,
            models.Label.value,
            db=db,
            label_source=models.GroundTruth,
        ).where(models.GroundTruth.annotation_id == annotation.id)
        labels = [
            schemas.Label(key=label[0], value=label[1])
            for label in query.all()
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
        labels=labels,
        metadata=annotation.meta,
        bounding_box=box,
        polygon=polygon,
        raster=raster,
        embedding=embedding,
        is_instance=annotation.is_instance,
        implied_task_types=annotation.implied_task_types,
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

    try:
        # delete annotations
        annotations_to_delete = (
            select(models.Annotation)
            .join(models.Datum, models.Datum.id == models.Annotation.datum_id)
            .where(models.Datum.dataset_id == dataset.id)
            .subquery()
        )
        db.execute(
            delete(models.Annotation).where(
                models.Annotation.id == annotations_to_delete.c.id
            )
        )
        db.commit()

        # delete embeddings (if they exist)
        existing_ids = select(models.Annotation.embedding_id).where(
            models.Annotation.embedding_id.isnot(None)
        )
        db.execute(
            delete(models.Embedding).where(
                models.Embedding.id.not_in(existing_ids)
            )
        )
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

    try:
        # delete annotations
        annotations_to_delete = (
            select(models.Annotation)
            .where(models.Annotation.model_id == model.id)
            .subquery()
        )
        db.execute(
            delete(models.Annotation).where(
                models.Annotation.id == annotations_to_delete.c.id
            )
        )
        db.commit()

        # delete embeddings (if they exist)
        existing_ids = select(models.Annotation.embedding_id).where(
            models.Annotation.embedding_id.isnot(None)
        )
        db.execute(
            delete(models.Embedding).where(
                models.Embedding.id.not_in(existing_ids)
            )
        )
        db.commit()
    except IntegrityError as e:
        db.rollback()
        raise e
