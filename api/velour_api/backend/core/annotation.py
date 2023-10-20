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
from velour_api.backend.core.label import create_labels
from velour_api.backend.core.metadata import deserialize_meta, serialize_meta
from velour_api.enums import AnnotationType


# @TODO: Might introduce multipolygon type to annotations, convert to raster at evaluation time.
def _wkt_multipolygon_to_raster(wkt: str):
    return select(
        text(f"ST_AsRaster(ST_GeomFromText('{wkt}'), {1.0}, {1.0})")
    ).scalar_subquery()


def _get_annotation_mapping(
    annotation: schemas.Annotation,
    datum: models.Datum,
    model: models.Model = None,
) -> models.Annotation:
    box = None
    polygon = None
    raster = None
    jsonb = None

    if isinstance(annotation.bounding_box, schemas.BoundingBox):
        box = annotation.bounding_box.wkt()
    if isinstance(annotation.polygon, schemas.Polygon):
        polygon = annotation.polygon.wkt()
    if isinstance(annotation.multipolygon, schemas.MultiPolygon):
        raster = _wkt_multipolygon_to_raster(annotation.multipolygon.wkt())
    if isinstance(annotation.raster, schemas.Raster):
        raster = annotation.raster.mask_bytes
    if isinstance(annotation.jsonb, dict):
        jsonb = annotation.jsonb
    # @TODO: Add more annotation types

    metadata = deserialize_meta(annotation.metadata)

    mapping = {
        "datum_id": datum.id,
        "model_id": model.id if model else None,
        "task_type": annotation.task_type,
        "meta": metadata,
        "box": box,
        "polygon": polygon,
        "raster": raster,
        "json": jsonb,
    }

    return mapping


def create_annotations_and_labels(
    db: Session,
    annotations: list[schemas.Annotation],
    datum: models.Datum,
    model: models.Model = None,
) -> list[models.Annotation]:
    """
    Create a list of annotations and associated labels in postgis

    Parameters
    ----------
    db
        The database Session you want to query against.
    annotations
        The list of annotations you want to create.
    datum
        The datum you want to create the annotations for.
    model
        The model you want to query against (optional).
    """
    annotation_list = []
    label_list = []

    for annotation in annotations:
        mapping = _get_annotation_mapping(
            annotation=annotation, datum=datum, model=model
        )
        annotation_list.append(models.Annotation(**mapping))
        label_list.append(create_labels(db=db, labels=annotation.labels))

    try:
        db.add_all(annotation_list)
        db.commit()
    except IntegrityError:
        db.rollback()
        raise exceptions.AnnotationAlreadyExistsError

    # return the label_list, too, since these are needed for GroundTruth
    return (annotation_list, label_list)


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
    groundtruth_labels = [
        schemas.Label(key=label[0], value=label[1])
        for label in (
            db.query(models.Label.key, models.Label.value)
            .join(
                models.GroundTruth,
                models.GroundTruth.label_id == models.Label.id,
            )
            .where(
                models.GroundTruth.annotation_id == annotation.id,
            )
            .all()
        )
    ]
    prediction_labels = [
        schemas.Label(key=label[0], value=label[1], score=label[2])
        for label in (
            db.query(
                models.Label.key, models.Label.value, models.Prediction.score
            )
            .join(
                models.Prediction,
                models.Prediction.label_id == models.Label.id,
            )
            .where(
                models.Prediction.annotation_id == annotation.id,
            )
            .all()
        )
    ]
    labels = groundtruth_labels if groundtruth_labels else prediction_labels

    # Initialize
    retval = schemas.Annotation(
        task_type=annotation.task_type,
        labels=labels,
        metadata=serialize_meta(annotation.meta),
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
    Query postgis to get all annotations for a particular datum
    Parameters
    -------
    db
        The database session to query against.
    datum
        The datum you want to fetch annotations for.
    model
        The model you want to query against (optional).
    """
    model_expr = (
        models.Annotation.model_id.is_(None)
        if model is None
        else models.Annotation.model_id == model.id
    )
    return [
        get_annotation(db, annotation=annotation, datum=datum)
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
                models.Annotation.task_type == enums.TaskType.DETECTION.value,
                model_expr,
                col.isnot(None),
            )
            .one_or_none()
        )
        if search is not None:
            return atype
    return AnnotationType.NONE
