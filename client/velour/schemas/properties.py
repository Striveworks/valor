from typing import Any, Optional, Union

import numpy as np

from velour.schemas.constraints import (
    BoolMapper,
    DatetimeMapper,
    DictionaryMapper,
    GeometryMapper,
    GeospatialMapper,
    LabelMapper,
    NumericMapper,
    StringMapper,
)
from velour.types import DatetimeType, GeoJSONType, GeometryType, MetadataType


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
        fget = lambda _: self  # noqa: E731
        fset = None
        if not self.key:
            fset = lambda instance, value: self.__convert_to_value__(  # noqa: E731
                instance, value
            )
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
    type_ = DatetimeType


class GeometryProperty(_BaseProperty, GeometryMapper):
    type_ = GeometryType


class GeospatialProperty(_BaseProperty, GeospatialMapper):
    type_ = GeoJSONType


class DictionaryProperty(_BaseProperty, DictionaryMapper):
    type_ = MetadataType


class LabelProperty(_BaseProperty, LabelMapper):
    type_ = Any
