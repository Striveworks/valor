import numpy as np

from typing import Union, Optional, Any

from valor.symbolic.modifiers import Symbol, Spatial, Nullable
from valor.symbolic.atomics import String, Float
from valor.symbolic.geojson import Polygon, MultiPolygon

    
class Score(Float, Nullable):
    
    @staticmethod
    def supports(value: Any) -> bool:
        return (
            Float.supports(value)
            or value is None
        )


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
    
    @staticmethod
    def supports(value: Any) -> bool:
        return (
            Polygon.supports(value)
            or value is None
        )


class BoundingPolygon(Polygon, Nullable):
    
    @staticmethod
    def supports(value: Any) -> bool:
        return (
            Polygon.supports(value)
            or value is None
        )


class Raster(Spatial, Nullable):

    @classmethod
    def from_numpy(cls, array: np.ndarray):
        pass
    
    @classmethod
    def from_geometry(cls, geometry: Union[Polygon, MultiPolygon]):
        pass

    @staticmethod
    def supports(value: Any) -> bool:
        return (
            String.supports(value)
            or value is None
        )
    
    @staticmethod
    def encode(value: Any):
        pass

    def decode(self):
        pass

    @property
    def area(self):
        if not isinstance(self._value, Symbol):
            raise ValueError
        return Float.symbolic(name=self._value._name, attribute="area")
    
    @property
    def array(self) -> np.ndarray:
        raise ValueError


class Embedding(Spatial, Nullable):
    pass
