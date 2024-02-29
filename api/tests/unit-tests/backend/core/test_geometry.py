from sqlalchemy import func, select, type_coerce, update

from valor_api.backend import models
from valor_api.backend.core.geometry import (
    GeometricValueType,
    RawGeometry,
    _convert_polygon_to_box,
    _convert_raster_to_box,
    _convert_raster_to_multipolygon,
    _convert_raster_to_polygon,
)


def test__convert_polygon_to_box():
    # test input (0, None)
    subquery = (
        select(models.Annotation.id)
        .join(models.Datum, models.Datum.id == models.Annotation.datum_id)
        .where(
            models.Annotation.box.is_(None),
            models.Annotation.polygon.isnot(None),
            models.Datum.dataset_id == 0,
            models.Annotation.model_id.is_(None),
        )
        .alias("subquery")
    )
    stmt = (
        update(models.Annotation)
        .where(models.Annotation.id == subquery.c.id)
        .values(box=func.ST_Envelope(models.Annotation.polygon))
    )
    assert str(_convert_polygon_to_box(0, None)) == str(stmt)

    # test input (0, 1)
    subquery = (
        select(models.Annotation.id)
        .join(models.Datum, models.Datum.id == models.Annotation.datum_id)
        .where(
            models.Annotation.box.is_(None),
            models.Annotation.polygon.isnot(None),
            models.Datum.dataset_id == 0,
            models.Annotation.model_id == 1,
        )
        .alias("subquery")
    )
    stmt = (
        update(models.Annotation)
        .where(models.Annotation.id == subquery.c.id)
        .values(box=func.ST_Envelope(models.Annotation.polygon))
    )
    assert str(_convert_polygon_to_box(0, 1)) == str(stmt)


def test__convert_raster_to_box():
    # test input (0, None)
    subquery1 = (
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
            models.Datum.dataset_id == 0,
            models.Annotation.model_id.is_(None),
        )
        .group_by(models.Annotation.id)
        .alias("subquery1")
    )
    stmt = (
        update(models.Annotation)
        .where(models.Annotation.id == subquery1.c.id)
        .values(box=subquery1.c.raster_envelope)
    )
    assert str(_convert_raster_to_box(0, None)) == str(stmt)

    # test input (0, 1)
    subquery1 = (
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
            models.Datum.dataset_id == 0,
            models.Annotation.model_id == 1,
        )
        .group_by(models.Annotation.id)
        .alias("subquery1")
    )
    stmt = (
        update(models.Annotation)
        .where(models.Annotation.id == subquery1.c.id)
        .values(box=subquery1.c.raster_envelope)
    )
    assert str(_convert_raster_to_box(0, 1)) == str(stmt)


def test__convert_raster_to_polygon():
    # test input (0, None)
    subquery1 = (
        select(
            models.Annotation.id.label("id"),
            func.ST_MakeValid(
                type_coerce(
                    func.ST_DumpAsPolygons(models.Annotation.raster),
                    GeometricValueType(),
                ).geom,
                type_=RawGeometry,
            ).label("geom"),
        )
        .join(models.Datum, models.Datum.id == models.Annotation.datum_id)
        .where(
            models.Annotation.polygon.is_(None),
            models.Annotation.raster.isnot(None),
            models.Datum.dataset_id == 0,
            models.Annotation.model_id.is_(None),
        )
        .alias("subquery1")
    )
    subquery2 = select(
        subquery1.c.id.label("id"),
        func.ST_ConvexHull(
            func.ST_Collect(subquery1.c.geom),
            type_=RawGeometry,
        ).label("raster_polygon"),
    ).alias("subquery2")
    stmt = (
        update(models.Annotation)
        .where(models.Annotation.id == subquery2.c.id)
        .values(polygon=subquery2.c.raster_polygon)
    )
    assert str(_convert_raster_to_polygon(0, None)) == str(stmt)

    # test input (0, 1)
    subquery1 = (
        select(
            models.Annotation.id.label("id"),
            func.ST_MakeValid(
                type_coerce(
                    func.ST_DumpAsPolygons(models.Annotation.raster),
                    GeometricValueType(),
                ).geom,
                type_=RawGeometry,
            ).label("geom"),
        )
        .join(models.Datum, models.Datum.id == models.Annotation.datum_id)
        .where(
            models.Annotation.polygon.is_(None),
            models.Annotation.raster.isnot(None),
            models.Datum.dataset_id == 0,
            models.Annotation.model_id == 1,
        )
        .alias("subquery1")
    )
    subquery2 = select(
        subquery1.c.id.label("id"),
        func.ST_ConvexHull(
            func.ST_Collect(subquery1.c.geom),
            type_=RawGeometry,
        ).label("raster_polygon"),
    ).alias("subquery2")
    stmt = (
        update(models.Annotation)
        .where(models.Annotation.id == subquery2.c.id)
        .values(polygon=subquery2.c.raster_polygon)
    )
    assert str(_convert_raster_to_polygon(0, 1)) == str(stmt)


def test__convert_raster_to_multipolygon():
    # test input (0, None)
    subquery1 = (
        select(
            models.Annotation.id.label("id"),
            func.ST_MakeValid(
                type_coerce(
                    func.ST_DumpAsPolygons(models.Annotation.raster),
                    GeometricValueType(),
                ).geom,
                type_=RawGeometry,
            ).label("geom"),
        )
        .join(models.Datum, models.Datum.id == models.Annotation.datum_id)
        .where(
            models.Annotation.polygon.is_(None),
            models.Annotation.raster.isnot(None),
            models.Datum.dataset_id == 0,
            models.Annotation.model_id.is_(None),
        )
        .alias("subquery1")
    )
    subquery2 = select(
        subquery1.c.id.label("id"),
        func.ST_Union(
            subquery1.c.geom,
            type_=RawGeometry,
        ).label("raster_multipolygon"),
    ).alias("subquery2")
    stmt = (
        update(models.Annotation)
        .where(models.Annotation.id == subquery2.c.id)
        .values(multipolygon=subquery2.c.raster_multipolygon)
    )
    assert str(_convert_raster_to_multipolygon(0, None)) == str(stmt)

    # test input (0, 1)
    subquery1 = (
        select(
            models.Annotation.id.label("id"),
            func.ST_MakeValid(
                type_coerce(
                    func.ST_DumpAsPolygons(models.Annotation.raster),
                    GeometricValueType(),
                ).geom,
                type_=RawGeometry,
            ).label("geom"),
        )
        .join(models.Datum, models.Datum.id == models.Annotation.datum_id)
        .where(
            models.Annotation.polygon.is_(None),
            models.Annotation.raster.isnot(None),
            models.Datum.dataset_id == 0,
            models.Annotation.model_id == 1,
        )
        .alias("subquery1")
    )
    subquery2 = select(
        subquery1.c.id.label("id"),
        func.ST_Union(
            subquery1.c.geom,
            type_=RawGeometry,
        ).label("raster_multipolygon"),
    ).alias("subquery2")
    stmt = (
        update(models.Annotation)
        .where(models.Annotation.id == subquery2.c.id)
        .values(multipolygon=subquery2.c.raster_multipolygon)
    )
    assert str(_convert_raster_to_multipolygon(0, 1)) == str(stmt)
