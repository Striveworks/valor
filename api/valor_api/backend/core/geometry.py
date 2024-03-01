from geoalchemy2 import Geometry
from geoalchemy2.types import CompositeType
from sqlalchemy import (
    Float,
    Update,
    distinct,
    func,
    literal_column,
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
    dataset_id: int, model_id: int | None = None
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

    model_expr = (
        models.Annotation.model_id == model_id
        if model_id
        else models.Annotation.model_id.is_(None)
    )
    subquery = (
        select(models.Annotation.id)
        .join(models.Datum, models.Datum.id == models.Annotation.datum_id)
        .where(
            models.Annotation.box.is_(None),
            models.Annotation.polygon.isnot(None),
            models.Datum.dataset_id == dataset_id,
            model_expr,
        )
        .alias("subquery")
    )
    return (
        update(models.Annotation)
        .where(models.Annotation.id == subquery.c.id)
        .values(box=func.ST_Envelope(models.Annotation.polygon))
    )


def _convert_multipolygon_to_box(
    dataset_id: int, model_id: int | None = None
) -> Update:
    raise NotImplementedError(
        "Conversion from multipolygon to box is currently unsupported. See Issue #470."
    )


def _convert_multipolygon_to_polygon(
    dataset_id: int, model_id: int | None = None
) -> Update:
    raise NotImplementedError(
        "Conversion from multipolygon to polygon is currently unsupported. See Issue #470."
    )


def _convert_raster_to_box(
    dataset_id: int, model_id: int | None = None
) -> Update:
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

    model_expr = (
        models.Annotation.model_id == model_id
        if model_id
        else models.Annotation.model_id.is_(None)
    )
    subquery = (
        select(
            models.Annotation.id.label("id"),
            func.ST_Envelope(
                func.ST_Union(models.Annotation.raster),
                type_=RawGeometry,
            ).label("raster_envelope"),
        )
        .join(models.Datum, models.Datum.id == models.Annotation.datum_id)
        .where(
            models.Annotation.box.is_(None),
            models.Annotation.raster.isnot(None),
            models.Datum.dataset_id == dataset_id,
            model_expr,
        )
        .group_by(models.Annotation.id)
        .alias("subquery")
    )
    return (
        update(models.Annotation)
        .where(models.Annotation.id == subquery.c.id)
        .values(box=subquery.c.raster_envelope)
    )


def _convert_raster_to_polygon(
    dataset_id: int,
    model_id: int | None = None,
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

    model_expr = (
        models.Annotation.model_id == model_id
        if model_id
        else models.Annotation.model_id.is_(None)
    )

    pixels_subquery = select(
        type_coerce(
            func.ST_PixelAsPoints(models.Annotation.raster, 1),
            type_=GeometricValueType,
        ).geom.label("geom")
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
        .join(
            pixels_subquery,
            literal_column(
                "true"
            ),  # Joining the lateral subquery doesn't require a condition
        )
        .where(
            models.Annotation.polygon.is_(None),
            models.Annotation.raster.isnot(None),
            models.Datum.dataset_id == dataset_id,
            model_expr,
        )
        .group_by(models.Annotation.id)
        .subquery()
    )

    return (
        update(models.Annotation)
        .where(models.Annotation.id == subquery.c.id)
        .values(polygon=subquery.c.raster_polygon)
    )


def _convert_raster_to_multipolygon(
    dataset_id: int,
    model_id: int | None = None,
) -> Update:
    raise NotImplementedError(
        "Conversion from raster to multipolygon is currently unsupported. See Issue #470."
    )


def convert_geometry(
    db: Session,
    source_type: AnnotationType,
    target_type: AnnotationType,
    dataset: models.Dataset,
    model: models.Model | None = None,
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
    model : models.Model
        The model of the geometry.
    """
    # Check typing
    valid_geometric_types = [
        AnnotationType.BOX,
        AnnotationType.POLYGON,
        AnnotationType.MULTIPOLYGON,
        AnnotationType.RASTER,
    ]
    if source_type not in valid_geometric_types:
        raise ValueError(f"Source type `{source_type}` not a geometric type.")
    if target_type not in valid_geometric_types:
        raise ValueError(f"Target type `{target_type}` not a geometric type.")

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
            AnnotationType.MULTIPOLYGON: _convert_raster_to_multipolygon,
        },
        AnnotationType.MULTIPOLYGON: {
            AnnotationType.BOX: _convert_multipolygon_to_box,
            AnnotationType.POLYGON: _convert_multipolygon_to_polygon,
        },
        AnnotationType.POLYGON: {
            AnnotationType.BOX: _convert_polygon_to_box,
        },
    }

    # get update
    update_stmt = source_to_target_conversion[source_type][target_type](
        dataset_id=dataset.id,
        model_id=model.id if model else None,
    )

    try:
        db.execute(update_stmt)
        db.commit()
    except IntegrityError:
        db.rollback()
