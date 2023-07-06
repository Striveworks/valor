from sqlalchemy import insert, select, text
from sqlalchemy.orm import Session

from velour_api import schemas
from velour_api.backend import models
from velour_api.backend.core.metadata import create_metadata


def wkt_multipolygon_to_raster(data: schemas.MultiPolygon):
    return select(
        text(f"ST_AsRaster(ST_GeomFromText('{data.wkt}'), {1.0}, {1.0})")
    )


def create_geometric_annotation(
    db: Session, annotation: schemas.Annotation, commit: bool = True
) -> models.GeometricAnnotation:

    # Check if annotation contains geometry
    if not annotation.geometry:
        return None

    box = None
    polygon = None
    raster = None

    if isinstance(annotation.geometry, schemas.BoundingBox):
        box = annotation.geometry.wkt
    elif isinstance(annotation.geometry, schemas.Polygon):
        polygon = annotation.geometry.wkt
    elif isinstance(annotation.geometry, schemas.MultiPolygon):
        raster = wkt_multipolygon_to_raster(annotation.geometry.wkt)
    elif isinstance(annotation.geometry, schemas.Raster):
        raster = annotation.geometry.mask_bytes
    else:
        raise ValueError(
            f"Unknown geometry with type '{type(annotation.geometry)}'."
        )

    mapping = {
        "box": box,
        "polygon": polygon,
        "raster": raster,
    }
    row = models.GeometricAnnotation(**mapping)
    create_metadata(db, annotation.metadata, geometry=row)
    if commit:
        db.add(row)
        db.commit()
    return row


def create_geometric_annotations(
    db: Session,
    annotations: list[schemas.Annotation],
) -> list[models.GeometricAnnotation]:
    rows = [
        create_geometric_annotation(db, annotation, commit=False)
        for annotation in annotations
    ]
    db.add_all(rows)
    db.commit()
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
