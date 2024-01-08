import io
import json
from base64 import b64encode

from geoalchemy2 import RasterElement
from geoalchemy2.functions import ST_AsGeoJSON, ST_AsPNG, ST_Envelope
from PIL import Image
from sqlalchemy import and_, distinct, select, text
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session

from velour_api import enums, exceptions, schemas
from velour_api.backend import models
from velour_api.backend.ops import Query


def _get_bounding_box_of_raster(
    db: Session, raster: RasterElement
) -> tuple[int, int, int, int]:
    """Get the enveloping bounding box of a raster"""
    env = json.loads(db.scalar(ST_AsGeoJSON(ST_Envelope(raster))))
    assert len(env["coordinates"]) == 1
    xs = [pt[0] for pt in env["coordinates"][0]]
    ys = [pt[1] for pt in env["coordinates"][0]]

    return min(xs), min(ys), max(xs), max(ys)


def _raster_to_png_b64(
    db: Session, raster: RasterElement, height: float, width: float
) -> str:
    """COnvert a raster to a png"""
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
        text(f"ST_AsRaster(ST_GeomFromText('{wkt}'), {1.0}, {1.0})")
    ).scalar_subquery()


def _create_annotation(
    annotation: schemas.Annotation,
    datum: models.Datum,
    model: models.Model | None = None,
) -> list[models.Label]:
    """
    Convert an individual annotation's attributes into a dictionary for upload to psql.

    Parameters
    ----------
    annotation : schemas.Annotation
        The annotation tom ap.
    datum : models.Datum
        The datum associated with the annotation.
    model : models.Model
        The model associated with the annotation

    Returns
    ----------
    models.Annotation
        A populated models.Annotation object.

    """
    box = None
    polygon = None
    raster = None

    if isinstance(annotation.bounding_box, schemas.BoundingBox):
        box = annotation.bounding_box.wkt()
    if isinstance(annotation.polygon, schemas.Polygon):
        polygon = annotation.polygon.wkt()
    if isinstance(annotation.multipolygon, schemas.MultiPolygon):
        raster = _wkt_multipolygon_to_raster(annotation.multipolygon.wkt())
    if isinstance(annotation.raster, schemas.Raster):
        raster = annotation.raster.mask_bytes

    mapping = {
        "datum_id": datum.id,
        "model_id": model.id if model else None,
        "task_type": annotation.task_type,
        "meta": annotation.metadata,
        "box": box,
        "polygon": polygon,
        "raster": raster,
    }
    return models.Annotation(**mapping)


def _create_empty_annotation(
    datum: models.Datum,
    model: models.Model | None = None,
) -> models.Annotation:
    mapping = {
        "datum_id": datum.id,
        "model_id": model.id if model else None,
        "task_type": enums.TaskType.EMPTY,
        "meta": {},
        "box": None,
        "polygon": None,
        "raster": None,
    }
    return models.Annotation(**mapping)


def _create_skipped_annotation(
    datum: models.Datum,
    model: models.Model | None = None,
) -> models.Annotation:
    mapping = {
        "datum_id": datum.id,
        "model_id": model.id if model else None,
        "task_type": enums.TaskType.SKIP,
        "meta": {},
        "box": None,
        "polygon": None,
        "raster": None,
    }
    return models.Annotation(**mapping)


def create_annotations(
    db: Session,
    annotations: list[schemas.Annotation],
    datum: models.Datum,
    model: models.Model = None,
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
        raise exceptions.AnnotationAlreadyExistsError(datum.id)

    # create annotations
    annotation_list = (
        [
            _create_annotation(annotation=annotation, datum=datum, model=model)
            for annotation in annotations
        ]
        if annotations
        else [_create_empty_annotation(datum=datum, model=model)]
    )

    try:
        db.add_all(annotation_list)
        db.commit()
    except IntegrityError:
        db.rollback()
        raise exceptions.AnnotationAlreadyExistsError(datum.id)
    return annotation_list


def create_skipped_annotations(
    db: Session,
    datums: list[models.Datum],
    model: models.Model | None = None,
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
        _create_skipped_annotation(datum, model) for datum in datums
    ]
    try:
        db.add_all(annotation_list)
        db.commit()
    except IntegrityError:
        db.rollback()
        raise exceptions.AnnotationAlreadyExistsError("")


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
        q = q.where(models.Prediction.annotation_id == annotation.id)
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
        q = q.where(models.GroundTruth.annotation_id == annotation.id)
        labels = [
            schemas.Label(key=label[0], value=label[1])
            for label in db.query(q.subquery()).all()
        ]

    # Initialize
    retval = schemas.Annotation(
        task_type=annotation.task_type,
        labels=labels,
        metadata=annotation.meta,
        bounding_box=None,
        polygon=None,
        multipolygon=None,
        raster=None,
    )

    # Bounding Box
    if annotation.box is not None:
        geojson = json.loads(db.scalar(ST_AsGeoJSON(annotation.box)))
        retval.bounding_box = schemas.BoundingBox(
            polygon=schemas.geojson.from_dict(data=geojson)
            .geometry()
            .boundary,
            box=None,
        )

    # Polygon
    if annotation.polygon is not None:
        geojson = json.loads(db.scalar(ST_AsGeoJSON(annotation.polygon)))
        retval.polygon = schemas.geojson.from_dict(data=geojson).geometry()

    # Raster
    if annotation.raster is not None:
        datum = db.scalar(
            select(models.Datum).where(models.Datum.id == annotation.datum_id)
        )
        if "height" not in datum.meta or "width" not in datum.meta:
            raise ValueError("missing height or width")
        height = datum.meta["height"]
        width = datum.meta["width"]
        retval.raster = schemas.Raster(
            mask=_raster_to_png_b64(
                db, raster=annotation.raster, height=height, width=width
            ),
            height=height,
            width=width,
        )

    return retval


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


# @TODO: This only services detection annotations
def get_annotation_type(
    db: Session,
    dataset: models.Dataset,
    model: models.Model | None = None,
) -> enums.AnnotationType:
    """
    Fetch an annotation type from psql.

    Parameters
    ----------
    db : Session
        The database Session you want to query against.
    dataset : models.Dataset
        The dataset associated with the annotation.
    model : models.Model
        The model associated with the annotation.

    Returns
    ----------
    enums.AnnotationType
        The type of the annotation.
    """
    model_expr = (
        models.Annotation.model_id == model.id
        if model
        else models.Annotation.model_id.is_(None)
    )
    hierarchy = [
        (enums.AnnotationType.RASTER, models.Annotation.raster),
        (enums.AnnotationType.MULTIPOLYGON, models.Annotation.multipolygon),
        (enums.AnnotationType.POLYGON, models.Annotation.polygon),
        (enums.AnnotationType.BOX, models.Annotation.box),
    ]
    for atype, col in hierarchy:
        search = (
            db.query(distinct(models.Dataset.id))
            .select_from(models.Annotation)
            .join(models.Datum, models.Datum.id == models.Annotation.datum_id)
            .join(models.Dataset, models.Dataset.id == models.Datum.dataset_id)
            .where(
                models.Datum.dataset_id == dataset.id,
                models.Annotation.task_type == enums.TaskType.DETECTION.value,
                model_expr,
                col.isnot(None),
            )
            .one_or_none()
        )
        if search is not None:
            return atype
    return enums.AnnotationType.NONE
