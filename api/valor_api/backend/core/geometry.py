import io
from base64 import b64encode

from geoalchemy2 import Geometry
from geoalchemy2.functions import ST_AsPNG
from geoalchemy2.types import CompositeType
from PIL import Image
from sqlalchemy import (
    BinaryExpression,
    Float,
    Subquery,
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
        (AnnotationType.BOX, models.Annotation.bounding_box),
    ]
    for atype, col in hierarchy:
        search = (
            db.query(distinct(models.Dataset.id))
            .select_from(models.Annotation)
            .join(models.Datum, models.Datum.id == models.Annotation.datum_id)
            .join(models.Dataset, models.Dataset.id == models.Datum.dataset_id)
            .where(
                models.Datum.dataset_id == dataset.id,
                models.Annotation.task_type == task_type.value,
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
    dataset_id : int
        A dataset id.
    model_id : int, optional
        A model id.

    Returns
    ----------
    sqlalchemy.Update
        A SQL update to complete the conversion.
    """

    subquery = (
        select(models.Annotation.id)
        .join(models.Datum, models.Datum.id == models.Annotation.datum_id)
        .where(
            models.Annotation.bounding_box.is_(None),
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


def _convert_from_raster(where_conditions: list[BinaryExpression]) -> Subquery:
    """Returns a subquery to convert a raster to a polygon."""
    pixels_subquery = select(
        models.Annotation.id.label("id"),
        type_coerce(
            func.ST_PixelAsPoints(models.Annotation.raster, 1),
            type_=GeometricValueType,
        ).geom.label("geom"),
    ).lateral("pixels")
    return (
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


def _convert_raster_to_box(where_conditions: list[BinaryExpression]) -> Update:
    """
    Converts annotation column 'raster' into column 'box'.

    Parameters
    ----------
    dataset_id : int
        A dataset id.
    model_id : int, optional
        A model id.

    Returns
    ----------
    sqlalchemy.Update
        A SQL update to complete the conversion.
    """

    subquery = _convert_from_raster(where_conditions)
    return (
        update(models.Annotation)
        .where(models.Annotation.id == subquery.c.id)
        .values(box=func.ST_Envelope(subquery.c.raster_polygon))
    )


def _convert_raster_to_polygon(
    where_conditions: list[BinaryExpression],
) -> Update:
    """
    Converts annotation column 'raster' into column 'polygon'.

    Parameters
    ----------
    dataset_id : int
        A dataset id.
    model_id : int, optional
        A model id.

    Returns
    ----------
    sqlalchemy.Update
        A SQL update to complete the conversion.
    """

    subquery = _convert_from_raster(where_conditions)
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
    task_type : enums.TaskType, optional
        A task type to stratify the conversion by.
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
        models.Annotation.task_type == task_type.value
        if task_type
        else models.Annotation.task_type.isnot(None)
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


def _raster_to_png_b64(
    db: Session,
    raster: Image.Image,
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
    raster = Image.open(io.BytesIO(db.scalar(ST_AsPNG((raster))).tobytes()))
    if raster.mode != "L":
        raise RuntimeError

    # mask is greyscale with values 0 and 1. to convert to binary
    # we first need to map 1 to 255
    raster = raster.point(lambda x: 255 if x == 1 else 0).convert("1")

    f = io.BytesIO()
    raster.save(f, format="PNG")
    f.seek(0)
    mask_bytes = f.read()
    return b64encode(mask_bytes).decode()
