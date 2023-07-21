from sqlalchemy import select, text
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session

from velour_api import exceptions, schemas
from velour_api.backend import models
from velour_api.backend.core.metadata import create_metadata


# @TODO: Might introduce multipolygon type to annotations, convert to raster at evaluation time.
def _wkt_multipolygon_to_raster(wkt: str):
    return select(
        text(f"ST_AsRaster(ST_GeomFromText('{wkt}'), {1.0}, {1.0})")
    )


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