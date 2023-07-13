from sqlalchemy import select, text
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session

from velour_api import exceptions, schemas
from velour_api.backend import models
from velour_api.backend.core.metadata import create_metadata


# @TODO: Might introduce multipolygon type to annotations, convert to raster at evaluation time.
def _wkt_multipolygon_to_raster(data: schemas.MultiPolygon):
    return select(
        text(f"ST_AsRaster(ST_GeomFromText('{data.wkt}'), {1.0}, {1.0})")
    )


def create_annotation(
    db: Session,
    annotation: schemas.Annotation, 
    commit: bool = True
) -> models.Annotation:

    box = None
    polygon = None
    raster = None

    if isinstance(annotation.bounding_box, schemas.BoundingBox):
        box = annotation.bounding_box.wkt
    elif isinstance(annotation.polygon, schemas.Polygon):
        polygon = annotation.polygon.wkt
    elif isinstance(annotation.multipolygon, schemas.MultiPolygon):
        raster = _wkt_multipolygon_to_raster(annotation.multipolygon.wkt)
    elif isinstance(annotation.raster, schemas.Raster):
        raster = annotation.raster.mask_bytes
    # @TODO: Add more annotation types

    mapping = {
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
) -> list[models.Annotation]:
    rows = [
        create_annotation(db, annotation, commit=False)
        for annotation in annotations
    ]
    try:
        db.add_all(rows)
        db.commit()
    except IntegrityError:
        db.rollback()
        raise exceptions.AnnotationAlreadyExistsError
    return rows


def convert_polygon_to_box(
    dataset_id: int = None,
    model_id: int = None,
    datum_id: int = None,
    annotation_id: int = None,
):
    """Converts annotation column 'polygon' into column 'bbox'. Filter by input args."""
    pass
    # return select(
    #     text(f"ST_Envelope(ST_Union(ST_GeomFromText(polygon))")
    # )


def convert_raster_to_box(
    dataset_id: int = None,
    model_id: int = None,
    datum_id: int = None,
    annotation_id: int = None,
):
    """Converts annotation column 'raster' into column 'bbox'."""
    # SELECT id, ST_Envelope(ST_Union(geom)) as bbox
    # FROM (
    #     SELECT id, ST_MakeValid((ST_DumpAsPolygons(shape)).geom) as geom
    #     FROM {tablename}
    #     {f"WHERE ({criteria_id})" if criteria_id != '' else ''}
    # ) AS conversion
    # GROUP BY id
    pass


def convert_raster_to_polygon(
    dataset_id: int = None,
    model_id: int = None,
    datum_id: int = None,
    annotation_id: int = None,
):
    """Converts annotation column 'raster' into column 'polygon'."""
    # SELECT subquery.id as id, is_instance, boundary, datum_id
    # FROM(
    #     SELECT id, ST_Union(geom) as boundary
    #     FROM (
    #         SELECT id, ST_MakeValid((ST_DumpAsPolygons(shape)).geom) as geom
    #         FROM {tablename}
    #         {f"WHERE ({criteria_id})" if criteria_id != '' else ''}
    #     ) AS conversion
    #     GROUP BY id
    # )
    pass
