from sqlalchemy import text
from sqlalchemy.orm import Session

from velour_api.backend import models
from velour_api.enums import AnnotationType


def convert_polygon_to_box(dataset_id: int, model_id: int | None = None):
    """Converts annotation column 'polygon' into column 'box'."""

    model_id = f" = {model_id}" if model_id else " IS NULL"

    return f"""
    UPDATE annotation
    SET box = ST_Envelope(annotation.polygon)
    FROM annotation ann
    JOIN datum ON datum.id = ann.datum_id
    WHERE
    annotation.id = ann.id
    AND ann.box IS NULL
    AND ann.polygon IS NOT NULL
    AND datum.dataset_id = {dataset_id}
    AND ann.model_id {model_id}
    """


def convert_raster_to_box(dataset_id: int, model_id: int | None = None):
    """Converts annotation column 'raster' into column 'box'."""

    model_id = f" = {model_id}" if model_id else " IS NULL"

    return f"""
    UPDATE annotation
    SET box = subquery.raster_envelope
    FROM (
        SELECT id, ST_Envelope(ST_Union(geom)) as raster_envelope
        FROM (
            SELECT ann.id as id, ST_MakeValid((ST_DumpAsPolygons(raster)).geom) as geom
            FROM annotation AS ann
            JOIN datum ON datum.id = ann.datum_id
            WHERE
            ann.box IS NULL
            AND ann.raster IS NOT NULL
            AND datum.dataset_id = {dataset_id}
            AND ann.model_id {model_id}
        ) AS conversion
        GROUP BY id
    ) as subquery
    WHERE annotation.id = subquery.id
    """


def convert_raster_to_polygon(dataset_id: int, model_id: int | None = None):
    """Converts annotation column 'raster' into column 'polygon'."""

    # @TODO: should this be purely an boundary around the raster,
    # multipolygon handles holes and odd regions better.

    raise NotImplementedError


def convert_raster_to_multipolygon(
    dataset_id: int, model_id: int | None = None
):
    """Converts annotation column 'raster' into column 'multipolygon'."""

    model_id = f" = {model_id}" if model_id else " IS NULL"

    return f"""
    UPDATE annotation
    SET multipolygon = subquery.raster_multipolygon
    FROM (
        SELECT id, ST_Union(geom) as raster_multipolygon
        FROM (
            SELECT ann.id as id, ST_MakeValid((ST_DumpAsPolygons(raster)).geom) as geom
            FROM annotation ann
            JOIN datum ON datum.id = ann.datum_id
            WHERE
            ann.id = annotation.id
            AND ann.multipolygon IS NULL
            AND ann.raster IS NOT NULL
            AND datum.dataset_id = {dataset_id}
            AND ann.model_id {model_id}
        ) AS conversion
        GROUP BY id
    ) as subquery
    WHERE annotation.id = subquery.id
    """


def convert_geometry(
    db: Session,
    dataset: models.Dataset,
    model: models.Model,
    dataset_source_type: AnnotationType,
    model_source_type: AnnotationType,
    evaluation_target_type: AnnotationType,
):
    # Check typing
    valid_types = [
        AnnotationType.BOX,
        AnnotationType.POLYGON,
        AnnotationType.RASTER,
    ]
    if evaluation_target_type not in valid_types:
        raise RuntimeError(
            f"Evaluation type `{evaluation_target_type}` not in valid set `{valid_types}`"
        )
    if dataset_source_type not in valid_types:
        raise RuntimeError(
            f"Groundtruth type `{evaluation_target_type}` not in valid set `{valid_types}`"
        )
    if model_source_type not in valid_types:
        raise RuntimeError(
            f"Prediction type `{evaluation_target_type}` not in valid set `{valid_types}`"
        )

    # Check if source type can serve the target type
    assert dataset_source_type >= evaluation_target_type
    assert model_source_type >= evaluation_target_type

    # Dataset type conversion
    if (
        dataset_source_type == AnnotationType.POLYGON
        and evaluation_target_type == AnnotationType.BOX
    ):
        db.execute(text(convert_polygon_to_box(dataset_id=dataset.id)))
    elif (
        dataset_source_type == AnnotationType.RASTER
        and evaluation_target_type == AnnotationType.BOX
    ):
        db.execute(text(convert_raster_to_box(dataset_id=dataset.id)))
    elif (
        dataset_source_type == AnnotationType.RASTER
        and evaluation_target_type == AnnotationType.POLYGON
    ):
        db.execute(text(convert_raster_to_polygon(dataset_id=dataset.id)))

    # Model type conversion
    if (
        model_source_type == AnnotationType.POLYGON
        and evaluation_target_type == AnnotationType.BOX
    ):
        db.execute(
            text(
                convert_polygon_to_box(
                    dataset_id=dataset.id, model_id=model.id
                )
            )
        )
    elif (
        model_source_type == AnnotationType.RASTER
        and evaluation_target_type == AnnotationType.BOX
    ):
        db.execute(
            text(
                convert_raster_to_box(dataset_id=dataset.id, model_id=model.id)
            )
        )
    elif (
        model_source_type == AnnotationType.RASTER
        and evaluation_target_type == AnnotationType.POLYGON
    ):
        db.execute(
            text(
                convert_raster_to_polygon(
                    dataset_id=dataset.id, model_id=model.id
                )
            )
        )

    db.commit()
