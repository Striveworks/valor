import datetime
import numpy as np
from typing import Union, Any, Optional

from velour.schemas.constraints import (
    BoolMapper,
    NumericMapper,
    StringMapper,
    DatetimeMapper,
    GeometryMapper,
    GeospatialMapper,
    DictionaryMapper,
    LabelMapper,
)
from velour.types import GeoJSONType, MetadataType
from velour.schemas import geometry


def getter_factory(name: str, type_: type):
    def _getter(self) -> type_:
        return getattr(self, name)
    return _getter


def setter_factory(name: str, type_: type):
    def _setter(self, __value: type_):
        setattr(self, name, __value)
    return _setter


class _BaseProperty(property):
    type_ = Any

    def __init__(
        self,
        property_name: str,
        filter_name: Optional[str] = None,
        key: Optional[str] = None,
        prefix: str = "_",
    ):
        self.property_name = property_name
        self.attribute_name = f"{prefix}{property_name}"
        self.filter_name = filter_name if filter_name else property_name
        self.key = key

        # set up `property`
        fget = lambda _ : self
        fset = None
        if not self.key:
            fset = lambda instance, value : self.__convert_to_value__(instance, value)
        super().__init__(fget=fget, fset=fset)

    def __convert_to_value__(self, instance, value: Any):
        fget = getter_factory(self.attribute_name, type_=self.type_)
        fset = setter_factory(self.attribute_name, type_=self.type_)
        super().__init__(fget=fget, fset=fset)
        setattr(instance, self.attribute_name, value)


class BoolProperty(_BaseProperty, BoolMapper):
    type_ = bool


class NumericProperty(_BaseProperty, NumericMapper):
    type_ = Union[int, float, np.floating]
    

class StringProperty(_BaseProperty, StringMapper):
    type_ = str


class DatetimeProperty(_BaseProperty, DatetimeMapper):
    type_ = Union[
        datetime.datetime,
        datetime.date,
        datetime.time,
        datetime.timedelta,
    ]


class GeometryProperty(_BaseProperty, GeometryMapper):
    type_ = Union[
        geometry.BoundingBox,
        geometry.Polygon,
        geometry.MultiPolygon,
        geometry.Raster,
    ]


class GeospatialProperty(_BaseProperty, GeospatialMapper):
    type_ = GeoJSONType
    

class DictionaryProperty(_BaseProperty, DictionaryMapper):
    type_ = MetadataType


class LabelProperty(_BaseProperty, LabelMapper):
    type_ = Any
