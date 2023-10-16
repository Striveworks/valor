import io
import json
from base64 import b64encode

from geoalchemy2 import RasterElement
from geoalchemy2.functions import ST_AsGeoJSON, ST_AsPNG, ST_Envelope
from PIL import Image
from sqlalchemy import and_, distinct, or_, select, text
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session

from velour_api import enums, exceptions, schemas
from velour_api.backend import models
from velour_api.backend.core.metadata import create_metadata, get_metadata
from velour_api.enums import AnnotationType


# @TODO: Might introduce multipolygon type to annotations, convert to raster at evaluation time.
def _wkt_multipolygon_to_raster(wkt: str):
    return select(
        text(f"ST_AsRaster(ST_GeomFromText('{wkt}'), {1.0}, {1.0})")
    ).scalar_subquery()


def create_annotation(
    db: Session,
    annotation: schemas.Annotation,
    datum: models.Datum,
    model: models.Model = None,
) -> models.Annotation:
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
    # @TODO: Add more annotation types

    mapping = {
        "datum_id": datum.id,
        "model_id": model.id if model else None,
        "task_type": annotation.task_type,
        "box": box,
        "polygon": polygon,
        "raster": raster,
    }

    try:
        row = models.Annotation(**mapping)
        db.add(row)
        db.commit()
    except IntegrityError:
        db.rollback()
        raise exceptions.AnnotationAlreadyExistsError

    create_metadata(db, annotation.metadata, annotation=row)
    return row


def create_annotations(
    db: Session,
    annotations: list[schemas.Annotation],
    datum: models.Datum,
    model: models.Model = None,
) -> list[models.Annotation]:
    return [
        create_annotation(
            db,
            annotation=annotation,
            datum=datum,
            model=model,
        )
        for annotation in annotations
    ]


# @TODO: Clean up??
def _get_bounding_box_of_raster(
    db: Session, raster: RasterElement
) -> tuple[int, int, int, int]:
    env = json.loads(db.scalar(ST_AsGeoJSON(ST_Envelope(raster))))
    assert len(env["coordinates"]) == 1
    xs = [pt[0] for pt in env["coordinates"][0]]
    ys = [pt[1] for pt in env["coordinates"][0]]

    return min(xs), min(ys), max(xs), max(ys)


# @TODO: Clean up??
def _raster_to_png_b64(
    db: Session, raster: RasterElement, height: float, width: float
) -> str:
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


def get_annotation(
    db: Session, annotation: models.Annotation, datum: models.Datum = None
) -> schemas.Annotation:
    # Retrieve all labels associated with annotation
    labels = [
        schemas.Label(key=label[0], value=label[1])
        for label in (
            db.query(models.Label.key, models.Label.value)
            .join(
                models.GroundTruth,
                models.GroundTruth.label_id == models.Label.id,
            )
            .where(models.GroundTruth.annotation_id == annotation.id)
            .all()
        )
    ]

    # Initialize
    retval = schemas.Annotation(
        task_type=annotation.task_type,
        labels=labels,
        metadata=get_metadata(db, annotation=annotation),
        bounding_box=None,
        polygon=None,
        multipolygon=None,
        raster=None,
    )

    # Bounding Box
    if annotation.box is not None:
        geojson = db.scalar(ST_AsGeoJSON(annotation.box))
        retval.bounding_box = schemas.BoundingBox(
            polygon=schemas.GeoJSON.from_json(geojson=geojson)
            .shape()
            .boundary,
            box=None,
        )

    # Polygon
    if annotation.polygon is not None:
        geojson = (
            db.scalar(ST_AsGeoJSON(annotation.polygon))
            if annotation.polygon is not None
            else None
        )
        retval.polygon = schemas.GeoJSON.from_json(geojson=geojson).shape()

    # Raster
    if annotation.raster is not None:
        height = get_metadata(db, datum=datum, key="height", union_type="and")[
            0
        ].value
        width = get_metadata(db, datum=datum, key="width", union_type="and")[
            0
        ].value
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
) -> list[schemas.Annotation]:
    return [
        get_annotation(db, annotation=annotation, datum=datum)
        for annotation in (
            db.query(models.Annotation)
            .where(
                and_(
                    models.Annotation.model_id.is_(None),
                    models.Annotation.datum_id == datum.id,
                )
            )
            .all()
        )
    ]


def get_scored_annotation(
    db: Session,
    annotation: models.Annotation,
) -> schemas.Annotation:
    # Retrieve all labels associated with annotation
    scored_labels = [
        schemas.Label(
            key=scored_label[0],
            value=scored_label[1],
            score=scored_label[2],
        )
        for scored_label in (
            db.query(
                models.Label.key, models.Label.value, models.Prediction.score
            )
            .join(
                models.Prediction,
                models.Prediction.label_id == models.Label.id,
            )
            .where(models.Prediction.annotation_id == annotation.id)
            .all()
        )
    ]

    # Initialize
    retval = schemas.Annotation(
        task_type=annotation.task_type,
        labels=scored_labels,
        metadata=get_metadata(db, annotation=annotation),
        bounding_box=None,
        polygon=None,
        multipolygon=None,
        raster=None,
    )

    # Bounding Box
    if annotation.box is not None:
        geojson = db.scalar(ST_AsGeoJSON(annotation.box))
        retval.bounding_box = schemas.BoundingBox(
            polygon=schemas.GeoJSON.from_json(geojson=geojson)
            .shape()
            .boundary,
            box=None,
        )

    # Polygon
    if annotation.polygon is not None:
        geojson = (
            db.scalar(ST_AsGeoJSON(annotation.polygon))
            if annotation.polygon is not None
            else None
        )
        retval.polygon = schemas.GeoJSON.from_json(geojson=geojson).shape()

    # Raster
    if annotation.raster is not None:
        height = get_metadata(db, annotation=annotation, key="height")[0].value
        width = get_metadata(db, annotation=annotation, key="width")[0].value
        retval.raster = schemas.Raster(
            mask=_raster_to_png_b64(
                db, raster=annotation.raster, height=height, width=width
            ),
            height=height,
            width=width,
        )

    return retval


def get_scored_annotations(
    db: Session,
    model: models.Model,
    datum: models.Datum,
) -> list[schemas.Annotation]:
    return [
        get_scored_annotation(db, annotation=annotation)
        for annotation in (
            db.query(models.Annotation)
            .where(
                and_(
                    models.Annotation.model_id == model.id,
                    models.Annotation.datum_id == datum.id,
                ),
            )
            .all()
        )
    ]


# @FIXME: This only services detection annotations
def get_annotation_type(
    db: Session,
    dataset: models.Dataset,
    model: models.Model | None = None,
) -> AnnotationType:
    model_expr = (
        models.Annotation.model_id == model.id
        if model
        else models.Annotation.model_id.is_(None)
    )
    hierarchy = [
        (AnnotationType.RASTER, models.Annotation.raster),
        (AnnotationType.MULTIPOLYGON, models.Annotation.multipolygon),
        (AnnotationType.POLYGON, models.Annotation.polygon),
        (AnnotationType.BOX, models.Annotation.box),
    ]
    for atype, col in hierarchy:
        search = (
            db.query(distinct(models.Dataset.id))
            .select_from(models.Annotation)
            .join(models.Datum, models.Datum.id == models.Annotation.datum_id)
            .join(models.Dataset, models.Dataset.id == models.Datum.dataset_id)
            .where(
                models.Datum.dataset_id == dataset.id,
                model_expr,
                col.isnot(None),
                or_(
                    models.Annotation.task_type
                    == enums.TaskType.DETECTION.value,
                    models.Annotation.task_type
                    == enums.TaskType.INSTANCE_SEGMENTATION.value,
                ),
            )
            .one_or_none()
        )
        if search is not None:
            return atype
    return AnnotationType.NONE
