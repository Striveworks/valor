import io
import json
from base64 import b64encode

from geoalchemy2.functions import ST_AsGeoJSON, ST_AsPNG, ST_Envelope
from PIL import Image
from sqlalchemy import and_, delete, func, select
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session

from valor_api import enums, exceptions, schemas
from valor_api.backend import models
from valor_api.backend.query import Query


def _get_bounding_box_of_raster(
    db: Session, raster: Image.Image
) -> tuple[int, int, int, int]:
    """Get the enveloping bounding box of a raster"""
    env = json.loads(db.scalar(ST_AsGeoJSON(ST_Envelope(raster))))
    assert len(env["coordinates"]) == 1
    xs = [pt[0] for pt in env["coordinates"][0]]
    ys = [pt[1] for pt in env["coordinates"][0]]

    return min(xs), min(ys), max(xs), max(ys)


def _raster_to_png_b64(
    db: Session, raster: Image.Image, height: float, width: float
) -> str:
    """Convert a raster to a png"""
    enveloping_box = _get_bounding_box_of_raster(db, raster)
    raster = Image.open(io.BytesIO(db.scalar(ST_AsPNG((raster))).tobytes()))

    assert raster.mode == "L"

    ret = Image.new(size=(int(width), int(height)), mode=raster.mode)

    ret.paste(raster, box=enveloping_box)

    # mask is greyscale with values 0 and 1. to convert to binary
    # we first need to map 1 to 255
    ret = ret.point(lambda x: 255 if x == 1 else 0).convert("1")

    f = io.BytesIO()
    ret.save(f, format="PNG")
    f.seek(0)
    mask_bytes = f.read()
    return b64encode(mask_bytes).decode()


def _wkt_multipolygon_to_raster(wkt: str):
    """
    Convert a multipolygon to a raster using psql.

    Parameters
    ----------
    wkt : str
        A psql multipolygon object in well-known text format.


    Returns
    ----------
    Query
        A scalar subquery from psql.
    """
    return select(
        func.ST_AsRaster(
            func.ST_GeomFromText(wkt),
            1.0,
            1.0,
        ),
    ).scalar_subquery()


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
) -> models.Annotation:
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
    models.Annotation
        A populated models.Annotation object.
    """
    box = None
    polygon = None
    raster = None
    embedding_id = None

    if annotation.bounding_box:
        box = annotation.bounding_box.wkt()
    if annotation.polygon:
        polygon = annotation.polygon.wkt()
    if annotation.multipolygon:
        raster = _wkt_multipolygon_to_raster(annotation.multipolygon.wkt())
    if annotation.raster:
        raster = annotation.raster.mask_bytes
    if annotation.embedding is not None:
        embedding_id = _create_embedding(db=db, value=annotation.embedding)

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
    return models.Annotation(**mapping)


def create_annotations(
    db: Session,
    annotations: list[schemas.Annotation],
    datum: models.Datum,
    model: models.Model | None = None,
) -> list[models.Annotation]:
    """
    Create a list of annotations and associated labels in psql.

    Parameters
    ----------
    db : Session
        The database Session you want to query against.
    annotations : List[schemas.Annotation]
        The list of annotations to create.
    datum : models.Datum
        The datum associated with the annotation.
    model : models.Model
        The model associated with the annotation.

    Returns
    ----------
    List[models.annotation]
        The model associated with the annotation.

    Raises
    ------
    exceptions.AnnotationAlreadyExistsError
        If the provided datum already has existing annotations for that dataset or model.
    """
    # validate that there are no existing annotations for this datum.
    if db.query(
        select(models.Annotation.id)
        .where(
            and_(
                models.Annotation.datum_id == datum.id,
                (
                    models.Annotation.model_id == model.id
                    if model
                    else models.Annotation.model_id.is_(None)
                ),
            )
        )
        .subquery()
    ).all():
        raise exceptions.AnnotationAlreadyExistsError(datum.uid)

    # create annotations
    annotation_list = [
        _create_annotation(
            db=db, annotation=annotation, datum=datum, model=model
        )
        for annotation in annotations
    ]

    try:
        db.add_all(annotation_list)
        db.commit()
    except IntegrityError as e:
        db.rollback()
        raise e
    return annotation_list


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
            annotation=schemas.Annotation(task_type=enums.TaskType.SKIP),
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
    bounding_box = None
    polygon = None
    raster = None
    embedding = None

    # bounding box
    if annotation.box is not None:
        geojson = json.loads(db.scalar(ST_AsGeoJSON(annotation.box)))
        bounding_box = schemas.BoundingBox(
            polygon=schemas.metadata.geojson_from_dict(data=geojson)
            .geometry()
            .boundary,  # type: ignore - this is guaranteed to be a polygon
        )

    # polygon
    if annotation.polygon is not None:
        geojson = json.loads(db.scalar(ST_AsGeoJSON(annotation.polygon)))
        polygon = schemas.metadata.geojson_from_dict(
            data=geojson
        ).geometry()  # type: ignore - guaranteed to be a polygon in this case

    # raster
    if annotation.raster is not None:
        datum = db.scalar(
            select(models.Datum).where(models.Datum.id == annotation.datum_id)
        )

        if datum is None:
            raise RuntimeError(
                "psql unexpectedly returned None instead of a Datum."
            )

        if "height" not in datum.meta or "width" not in datum.meta:
            raise ValueError("missing height or width")
        height = datum.meta["height"]
        width = datum.meta["width"]
        raster = schemas.Raster(
            mask=_raster_to_png_b64(
                db, raster=annotation.raster, height=height, width=width
            ),
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
        bounding_box=bounding_box,
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

    if dataset.status != enums.TableStatus.DELETING:
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

    if model.status != enums.ModelStatus.DELETING:
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
