import io
import json
from base64 import b64encode
from typing import List, Optional

from geoalchemy2 import RasterElement
from geoalchemy2.functions import ST_AsGeoJSON, ST_AsPNG, ST_Envelope
from PIL import Image
from sqlalchemy import select, text
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session

from velour_api import exceptions, schemas
from velour_api.backend import models
from velour_api.backend.core.metadata import (
    create_metadata,
    get_metadata,
    get_metadatum_schema,
)


# @TODO: Might introduce multipolygon type to annotations, convert to raster at evaluation time.
def _wkt_multipolygon_to_raster(wkt: str):
    return select(text(f"ST_AsRaster(ST_GeomFromText('{wkt}'), {1.0}, {1.0})"))


def create_annotation(
    db: Session,
    annotation: schemas.Annotation,
    datum: models.Datum,
    model: models.Model = None,
    commit: bool = True,
) -> models.Annotation:

    box = None
    polygon = None
    raster = None

    if isinstance(annotation.bounding_box, schemas.BoundingBox):
        box = annotation.bounding_box.wkt()
    elif isinstance(annotation.polygon, schemas.Polygon):
        polygon = annotation.polygon.wkt()
        print(polygon)
    elif isinstance(annotation.multipolygon, schemas.MultiPolygon):
        raster = _wkt_multipolygon_to_raster(annotation.multipolygon.wkt())
    elif isinstance(annotation.raster, schemas.Raster):
        raster = annotation.raster.mask_bytes
    # @TODO: Add more annotation types

    mapping = {
        "datum": datum,
        "model": model,
        "task_type": annotation.task_type,
        "box": box,
        "polygon": polygon,
        "raster": raster,
    }
    row = models.Annotation(**mapping)
    create_metadata(db, annotation.metadata, annotation=row)
    if commit:
        try:
            db.add(row)
            db.commit()
        except IntegrityError:
            db.rollback()
            raise exceptions.AnnotationAlreadyExistsError
    return row


def create_annotations(
    db: Session,
    annotations: list[schemas.Annotation],
    datum: models.Datum,
    model: models.Model = None,
) -> list[models.Annotation]:
    rows = [
        create_annotation(
            db,
            annotation=annotation,
            datum=datum,
            model=model,
            commit=False,
        )
        for annotation in annotations
    ]
    try:
        db.add_all(rows)
        db.commit()
    except IntegrityError:
        db.rollback()
        raise exceptions.AnnotationAlreadyExistsError
    return rows


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
    db: Session,
    datum: models.Datum,
    annotation: models.Annotation,
) -> schemas.Annotation:

    # Initialize
    retval = schemas.Annotation(
        task_type=annotation.task_type,
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
            .polygon()
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
        retval.polygon = schemas.GeoJSON.from_json(geojson=geojson).polygon()

    # Raster
    if annotation.raster is not None:
        height = get_metadatum_schema(db, datum=datum, name="height").value
        width = get_metadatum_schema(db, datum=datum, name="width").value
        retval.raster = schemas.Raster(
            mask=_raster_to_png_b64(
                db, raster=annotation.raster, height=height, width=width
            ),
            height=height,
            width=width,
        )

    return retval
