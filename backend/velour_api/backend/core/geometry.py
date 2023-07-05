from sqlalchemy.orm import Session
from sqlalchemy import (
    select, insert, text
)

from velour_api import schemas
from velour_api.backend import models


def wkt_multipolygon_to_raster(
    data: schemas.MultiPolygon
):
    return select(
        text(f"ST_AsRaster(ST_GeomFromText('{data.wkt}'), {1.0}, {1.0})")
    )


def create_geometry_mapping(
    data: schemas.BoundingBox | schemas.Polygon | schemas.MultiPolygon | schemas.Raster,
) -> dict:
    
    box = None
    polygon = None
    raster = None

    if isinstance(data, schemas.BoundingBox):
        box = data.wkt
    elif isinstance(data, schemas.Polygon):
        polygon = data.wkt
    elif isinstance(data, schemas.MultiPolygon):
        raster = wkt_multipolygon_to_raster(data.wkt)
    elif isinstance(data, schemas.Raster):
        raster = data.mask_bytes
    else:
        raise ValueError("Unknown geometric type.")
    
    return {
        "box": box,
        "polygon": polygon,
        "raster": raster,
    }
    

def create_geometry(
    db: Session,
    data: list[schemas.GeometricAnnotation]
) -> list[int]:
    
    geometries = [
        create_geometry_mapping(datum.geometry)
        for datum in data
    ]
    
    added_ids = db.scalars(
        insert(models.GeometricAnnotation)
        .values(geometries)
        .returning(models.GeometricAnnotation.id)
    )
    db.commit()
    return added_ids.all()


def convert_polygon_to_box(
    dataset_id: int = None,
    model_id: int = None,
    datum_id: int = None,
    annotation_id: int = None,
):
    """Converts annotation column 'polygon' into column 'bbox'. Filter by input args."""
    pass


def convert_raster_to_box(
    dataset_id: int = None,
    model_id: int = None,
    datum_id: int = None,
    annotation_id: int = None,
):
    """Converts annotation column 'raster' into column 'bbox'."""
    pass


def convert_raster_to_polygon(
    dataset_id: int = None,
    model_id: int = None,
    datum_id: int = None,
    annotation_id: int = None,
):
    """Converts annotation column 'raster' into column 'polygon'."""
    pass