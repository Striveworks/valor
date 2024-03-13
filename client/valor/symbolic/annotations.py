from typing import Union

from valor.symbolic.modifiers import Spatial, Nullable
from valor.symbolic.attributes import Area
from valor.symbolic.geojson import Polygon, MultiPolygon


class BoundingBox(Polygon, Nullable):
    
    @classmethod
    def from_extrema(
        cls,
        xmin: float,
        xmax: float,
        ymin: float,
        ymax: float,
    ):
        points = [
            [
                (xmin, ymin),
                (xmax, ymin),
                (xmax, ymax),
                (xmin, ymax),
                (xmin, ymin),
            ]
        ]
        return cls(value=points)


class BoundingPolygon(Polygon, Nullable):
    pass


class Raster(Spatial, Nullable, Area):
    
    @classmethod
    def from_geometry(cls, geometry: Union[Polygon, MultiPolygon]):
        pass


class Embedding(Spatial, Nullable):
    pass

box = BoundingBox.from_extrema(0,1,0,1)
poly = BoundingPolygon(value=box.value)

print(box)
print(poly)