import io
import struct
from base64 import b64encode

import numpy as np
from geoalchemy2 import Geometry, RasterElement
from geoalchemy2.types import CompositeType
from PIL import Image
from sqlalchemy import (
    BinaryExpression,
    Float,
    Update,
    distinct,
    func,
    select,
    type_coerce,
    update,
)
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session

from valor_api.backend import models
from valor_api.enums import AnnotationType, TaskType


class GeometricValueType(CompositeType):
    """
    SQLAlchemy typing override.

    Not to be confused with typing aliases used for PyRight.

    This prevents sqlalchemy from automatically converting geometries to WKB.
    """

    typemap = {"geom": Geometry("MULTIPOLYGON"), "val": Float}


class RawGeometry(Geometry):
    """Modified SQLAlchemy geometry type."""

    cache_ok = True

    def column_expression(self, col):
        return col


def get_annotation_type(
    db: Session,
    task_type: TaskType,
    dataset: models.Dataset,
    model: models.Model | None = None,
) -> AnnotationType:
    """
    Fetch annotation type from psql.

    Parameters
    ----------
    db : Session
        The database Session you want to query against.
    task_type: TaskType
        The implied task type to filter on.
    dataset : models.Dataset
        The dataset associated with the annotation.
    model : models.Model
        The model associated with the annotation.

    Returns
    ----------
    AnnotationType
        The type of the annotation.
    """
    model_expr = (
        models.Annotation.model_id == model.id
        if model
        else models.Annotation.model_id.is_(None)
    )
    hierarchy = [
        (AnnotationType.RASTER, models.Annotation.raster),
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
                models.Annotation.implied_task_types.op("?")(task_type.value),
                model_expr,
                col.isnot(None),
            )
            .one_or_none()
        )
        if search is not None:
            return atype
    return AnnotationType.NONE


def _convert_polygon_to_box(
    where_conditions: list[BinaryExpression],
) -> Update:
    """
    Converts annotation column 'polygon' into column 'box'.

    Parameters
    ----------
    where_conditions: list[BinaryExpression]
        A list of conditions that specify the desired source via model, dataset and task type.

    Returns
    ----------
    sqlalchemy.Update
        A SQL update to complete the conversion.
    """

    subquery = (
        select(models.Annotation.id)
        .join(models.Datum, models.Datum.id == models.Annotation.datum_id)
        .where(
            models.Annotation.box.is_(None),
            models.Annotation.polygon.isnot(None),
            *where_conditions,
        )
        .alias("subquery")
    )
    return (
        update(models.Annotation)
        .where(models.Annotation.id == subquery.c.id)
        .values(box=func.ST_Envelope(models.Annotation.polygon))
    )


def _convert_raster_to_box(where_conditions: list[BinaryExpression]) -> Update:
    """
    Converts annotation column 'raster' into column 'box'.

    Parameters
    ----------
    where_conditions: list[BinaryExpression]
        A list of conditions that specify the desired source via model, dataset and task type.

    Returns
    ----------
    sqlalchemy.Update
        A SQL update to complete the conversion.
    """
    subquery = (
        select(
            models.Annotation.id.label("id"),
            func.ST_Envelope(
                func.ST_MinConvexHull(models.Annotation.raster)
            ).label("box"),
        )
        .select_from(models.Annotation)
        .join(models.Datum, models.Datum.id == models.Annotation.datum_id)
        .where(
            models.Annotation.box.is_(None),
            models.Annotation.raster.isnot(None),
            *where_conditions,
        )
        .group_by(models.Annotation.id)
        .subquery()
    )
    return (
        update(models.Annotation)
        .where(models.Annotation.id == subquery.c.id)
        .values(box=subquery.c.box)
    )


def _convert_raster_to_polygon(
    where_conditions: list[BinaryExpression],
) -> Update:
    """
    Converts annotation column 'raster' into column 'polygon'.

    Parameters
    ----------
    where_conditions: list[BinaryExpression]
        A list of conditions that specify the desired source via model, dataset and task type.

    Returns
    ----------
    sqlalchemy.Update
        A SQL update to complete the conversion.
    """

    pixels_subquery = select(
        models.Annotation.id.label("id"),
        type_coerce(
            func.ST_PixelAsPoints(models.Annotation.raster, 1),
            type_=GeometricValueType,
        ).geom.label("geom"),
    ).lateral("pixels")
    subquery = (
        select(
            models.Annotation.id.label("id"),
            func.ST_ConvexHull(func.ST_Collect(pixels_subquery.c.geom)).label(
                "raster_polygon"
            ),
        )
        .select_from(models.Annotation)
        .join(models.Datum, models.Datum.id == models.Annotation.datum_id)
        .join(pixels_subquery, pixels_subquery.c.id == models.Annotation.id)
        .where(
            models.Annotation.polygon.is_(None),
            models.Annotation.raster.isnot(None),
            *where_conditions,
        )
        .group_by(models.Annotation.id)
        .subquery()
    )
    return (
        update(models.Annotation)
        .where(models.Annotation.id == subquery.c.id)
        .values(polygon=subquery.c.raster_polygon)
    )


def convert_geometry(
    db: Session,
    source_type: AnnotationType,
    target_type: AnnotationType,
    dataset: models.Dataset,
    model: models.Model | None = None,
    task_type: TaskType | None = None,
):
    """
    Converts geometry into some target type

    Parameters
    ----------
    db : Session
        The database Session you want to query against.
    source_type: AnnotationType
        The annotation type we have.
    target_type: AnnotationType
        The annotation type we wish to convert to.
    dataset : models.Dataset
        The dataset of the geometry.
    model : models.Model, optional
        The model of the geometry.
    task_type: TaskType, optional
        Optional task type to search by.
    """
    # Check typing
    valid_geometric_types = [
        AnnotationType.BOX,
        AnnotationType.POLYGON,
        AnnotationType.RASTER,
    ]
    if source_type not in valid_geometric_types:
        raise ValueError(
            f"Annotation source with type `{source_type}` not supported."
        )
    if target_type not in valid_geometric_types:
        raise ValueError(
            f"Annotation target with type `{target_type}` not supported."
        )

    # Check if source type can serve the target type
    if source_type == target_type:
        return
    elif source_type < target_type:
        raise ValueError(
            f"Source type `{source_type}` is not capable of being converted to target type `{target_type}`."
        )

    # define conversion function mapping
    source_to_target_conversion = {
        AnnotationType.RASTER: {
            AnnotationType.BOX: _convert_raster_to_box,
            AnnotationType.POLYGON: _convert_raster_to_polygon,
        },
        AnnotationType.POLYGON: {
            AnnotationType.BOX: _convert_polygon_to_box,
        },
    }

    # define model expression
    model_expr = (
        models.Annotation.model_id == model.id
        if model
        else models.Annotation.model_id.is_(None)
    )

    # define task type expression
    task_type_expr = (
        models.Annotation.implied_task_types.op("?")(task_type.value)
        if task_type
        else models.Annotation.implied_task_types.isnot(None)
    )

    # define where expression
    where_conditions = [
        task_type_expr,
        models.Datum.dataset_id == dataset.id,
        model_expr,
    ]

    # get update
    update_stmt = source_to_target_conversion[source_type][target_type](
        where_conditions
    )

    try:
        db.execute(update_stmt)
        db.commit()
    except IntegrityError:
        db.rollback()


def _raster_to_numpy(
    db: Session,
    raster: RasterElement,
) -> np.ndarray:
    # Ensure raster_wkb is a bytes-like object
    raster_wkb = bytes.fromhex(raster.data)

    # Unpack header to get width and height
    # reference: https://postgis.net/docs/manual-dev/RT_reference.html
    header_format = "<BHHddddddiHH"
    header_size = struct.calcsize(header_format)
    (
        ndr,
        version,
        num_bands,
        scale_x,
        scale_y,
        ip_x,
        ip_y,
        skew_x,
        skew_y,
        srid,
        width,
        height,
    ) = struct.unpack(header_format, raster_wkb[:header_size])

    # Check if the raster has a single band
    if num_bands != 1:
        raise ValueError("This function only supports single-band rasters.")

    # Calculate the number of bytes needed for the pixel data
    # Each byte represents 1 pixel
    num_pixels = width * height
    num_bytes = num_pixels

    # Convert the byte data to a binary array
    pixel_format = "B"
    pixel_data = struct.unpack(
        f"{width * height}{pixel_format}",
        raster_wkb[header_size + 2 : header_size + 2 + num_bytes],
    )

    # Convert pixel data to numpy array
    raster_numpy = np.array(pixel_data, dtype=bool)
    return raster_numpy.reshape((height, width))


def _raster_to_png_b64(
    db: Session,
    raster: RasterElement,
) -> str:
    """
    Convert a raster to a png.

    Parameters
    ----------
    db : Session
        The database session.
    raster : Image.Image
        The raster in bytes.

    Returns
    -------
    str
        The encoded raster.
    """
    # Ensure raster_wkb is a bytes-like object
    raster_wkb = bytes.fromhex(raster.data)

    # Unpack header to get width and height
    # reference: https://postgis.net/docs/manual-dev/RT_reference.html
    header_format = "<BHHddddddiHH"
    header_size = struct.calcsize(header_format)
    (
        ndr,
        version,
        num_bands,
        scale_x,
        scale_y,
        ip_x,
        ip_y,
        skew_x,
        skew_y,
        srid,
        width,
        height,
    ) = struct.unpack(header_format, raster_wkb[:header_size])

    # Check if the raster has a single band
    if num_bands != 1:
        raise ValueError("This function only supports single-band rasters.")

    # Calculate the number of bytes needed for the pixel data
    # Each byte represents 1 pixel
    num_pixels = width * height
    num_bytes = num_pixels

    # Convert the byte data to a binary array
    pixel_format = "B"
    pixel_data = struct.unpack(
        f"{width * height}{pixel_format}",
        raster_wkb[header_size + 2 : header_size + 2 + num_bytes],
    )

    # Convert pixel data to numpy array
    raster_numpy = np.array(pixel_data, dtype=bool)
    raster_numpy = raster_numpy.reshape((height, width))

    # Convert to Pillow Image
    raster_image = Image.fromarray(raster_numpy)

    # b64 encode PNG to mask str
    f = io.BytesIO()
    raster_image.save(f, format="PNG")
    f.seek(0)
    mask_bytes = f.read()
    return b64encode(mask_bytes).decode()
