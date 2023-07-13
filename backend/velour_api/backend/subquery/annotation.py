import io
import json
from base64 import b64encode
from typing import List, Optional
from PIL import Image

from sqlalchemy.orm import Session
from geoalchemy2 import RasterElement
from geoalchemy2.functions import ST_AsGeoJSON, ST_AsPNG, ST_Envelope

from velour_api import schemas
from velour_api.backend import models
from velour_api.backend.subquery.metadata import get_metadata, get_metadatum

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
            polygon=schemas.GeoJSON(geojson=geojson).polygon().boundary,
            box=None,
        )

    # Polygon
    if annotation.polygon is not None:
        geojson = db.scalar(ST_AsGeoJSON(annotation.polygon)) if annotation.polygon is not None else None
        retval.polygon = schemas.GeoJSON(geojson=geojson).polygon()

    # Raster
    if annotation.raster is not None:
        height = get_metadatum(db, datum=datum, name="height").value
        width = get_metadatum(db, datum=datum, name="width").value
        retval.raster = schemas.Raster(
            mask=_raster_to_png_b64(db, raster=annotation.raster, height=height, width=width),
            height=height,
            width=width,
        )

    return retval
