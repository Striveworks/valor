from pydantic import BaseModel

from velour_api.schemas.geometry import (
    BoundingBox,
    MultiPolygon,
    Polygon,
    Raster,
)


class GeometricAnnotation(BaseModel):
    geometry: BoundingBox | Polygon | MultiPolygon | Raster
