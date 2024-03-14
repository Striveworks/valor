from typing import Union, Optional, Any

from valor.symbolic.modifiers import Equatable, Spatial, Nullable
from valor.symbolic.attributes import Area
from valor.symbolic.atomics import String, Float
from valor.symbolic.geojson import Polygon, MultiPolygon


class Label(Equatable):

    @classmethod
    def create(cls, key: str, value: str, score: Optional[float] = None):
        label = {
            "key": key,
            "value": value,
            "score": score,
        }
        return cls(value=label)
    
    @staticmethod
    def supports(value: Any) -> bool:
        if isinstance(value, dict):
            return (
                set(value.keys()) == {"key", "value", "score"}
                and String.supports(value["key"])
                and String.supports(value["value"])
                and (
                    Float.supports(value["score"])
                    or value["score"] is None
                )
            )
        return type(value) is Label
    
    def __repr__(self):
        return f"Label(value={self._value})" if self._value else super().__repr__()
        
    def __str__(self):
        return f"Label(value={self._value})" if self._value else super().__str__()
    
    @property
    def key(self):
        if self.is_value():
            return self._value["key"] if self._value else None
        else:
            return String(name=self._name, attribute="key")
    
    @property
    def value(self):
        if self.is_value():
            return self._value["value"] if self._value else None
        else:
            return String(name=self._name, attribute="value")

    @property
    def score(self):
        if self.is_value():
            return self._value["score"] if self._value else None
        else:
            return String(name=self._name, attribute="score")


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
