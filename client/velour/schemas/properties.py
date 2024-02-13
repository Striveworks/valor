import datetime
from typing import Any, Optional, Union

import numpy as np

from velour.schemas.constraints import (
    BoolMapper,
    DatetimeMapper,
    DictionaryMapper,
    GeometryMapper,
    GeometryType,
    GeospatialMapper,
    LabelMapper,
    NumericMapper,
    StringMapper,
)
from velour.schemas.metadata import DictMetadataType, GeoJSONType

DatetimeType = Union[
    datetime.datetime,
    datetime.date,
    datetime.time,
    datetime.timedelta,
]


def getter_factory(name: str, type_: type):
    def _getter(self) -> type_:
        return getattr(self, name)

    return _getter


def setter_factory(name: str, type_: type):
    def _setter(self, __value: type_):
        setattr(self, name, __value)

    return _setter


class _BaseProperty(property):
    """
    Subclass of property that is meant to be used with a mapper object.

    Parameters
    ----------
    name : str
        The name of the filter property.
    key : str, optional
        If applicable, the key to retrieve this object.
    prefix : str, default="_"
        The prefix applied to the property name when storing a value.

    Attributes
    ----------
    type_ : type
        The type of value that this property stores.
    """

    type_ = object

    def __init__(
        self,
        name: str,
        key: Optional[str] = None,
        prefix: str = "_",
    ):
        self.name = name
        self.attribute_name = f"{prefix}{name}"
        self.key = key

        # set up `property`
        fget = lambda _: self  # noqa: E731
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
    """
    Boolean property.

    Parameters
    ----------
    name : str
        The name of the filter property.
    key : str, optional
        If applicable, the key to retrieve this object.
    prefix : str, default="_"
        The prefix applied to the property name when storing a value.

    Attributes
    ----------
    type_ : type
        The type of value that this property stores.

    Examples
    --------
    >>> class Example:
    ...     x = BoolProperty(
    ...         name="filter_name",
    ...     )
    ...     def __init__(self, x: bool):
    ...         self.x = x
    ...
    >>> type(Example.x)
    <class 'velour.schemas.properties.BoolProperty'>
    >>> Example.x == False
    BinaryExpression(name='filter_name', constraint=Constraint(value=False, operator='=='), key=None)

    As a class variable 'x' operates as a mapping object. However, if we instantiate
    the object, we will see that it no longer functions as a mapper and instead has
    become a property of the value type.

    >>> example = Example(x=False)
    >>> example.x
    False
    """

    type_ = bool


class NumericProperty(_BaseProperty, NumericMapper):
    """
    Numeric property.

    This property supports the following types.

    - int
    - float
    - numpy.floating

    Parameters
    ----------
    name : str
        The name of the filter property.
    key : str, optional
        If applicable, the key to retrieve this object.
    prefix : str, default="_"
        The prefix applied to the property name when storing a value.

    Attributes
    ----------
    type_ : type
        The type of value that this property stores.

    Examples
    --------
    >>> class Example:
    ...     x = NumericProperty(
    ...         name="filter_name",
    ...     )
    ...     def __init__(self, x: float):
    ...         self.x = x
    ...
    >>> type(Example.x)
    <class 'velour.schemas.properties.NumericProperty'>
    >>> Example.x >= 1.4
    BinaryExpression(name='filter_name', constraint=Constraint(value=1.4, operator='>='), key=None)

    As a class variable 'x' operates as a mapping object. However, if we instantiate
    the object, we will see that it no longer functions as a mapper and instead has
    become a property of the value type.

    >>> example = Example(x=1.4)
    >>> example.x
    1.4
    """

    type_ = Union[int, float, np.floating]


class StringProperty(_BaseProperty, StringMapper):
    """
    String property.

    Parameters
    ----------
    name : str
        The name of the filter property.
    key : str, optional
        If applicable, the key to retrieve this object.
    prefix : str, default="_"
        The prefix applied to the property name when storing a value.

    Attributes
    ----------
    type_ : type
        The type of value that this property stores.

    Examples
    --------
    >>> class Example:
    ...     x = StringProperty(
    ...         name="filter_name",
    ...     )
    ...     def __init__(self, x: str):
    ...         self.x = x
    ...
    >>> type(Example.x)
    <class 'velour.schemas.properties.StringProperty'>
    >>> Example.x == "hello world"
    BinaryExpression(name='filter_name', constraint=Constraint(value="hello_world", operator='=='), key=None)

    As a class variable 'x' operates as a mapping object. However, if we instantiate
    the object, we will see that it no longer functions as a mapper and instead has
    become a property of the value type.

    >>> example = Example(x="hello world")
    >>> example.x
    'hello world'
    """

    type_ = str


class DatetimeProperty(_BaseProperty, DatetimeMapper):
    """
    Datetime property.

    This property supports the following objects from the datetime library.

    - datetime.datetime
    - datetime.date
    - datetime.time
    - datetime.timedelta

    Parameters
    ----------
    name : str
        The name of the filter property.
    key : str, optional
        If applicable, the key to retrieve this object.
    prefix : str, default="_"
        The prefix applied to the property name when storing a value.

    Attributes
    ----------
    type_ : type
        The type of value that this property stores.

    Examples
    --------
    >>> class Example:
    ...     x = DatetimeProperty(
    ...         name="filter_name",
    ...     )
    ...     def __init__(self, x: datetime):
    ...         self.x = x
    ...
    >>> type(Example.x)
    <class 'velour.schemas.properties.DatetimeProperty'>
    >>> Example.x == timedelta(days=1)
    BinaryExpression(name='filter_name', constraint=Constraint(value={'duration': '86400.0'}, operator='=='), key=None)

    As a class variable 'x' operates as a mapping object. However, if we instantiate
    the object, we will see that it no longer functions as a mapper and instead has
    become a property of the value type.

    >>> example = Example(x=timedelta(days=1))
    >>> type(example.x)
    <class 'datetime.timedelta'>
    >>> example.x
    datetime.timedelta(days=1)
    """

    type_ = DatetimeType


class GeometryProperty(_BaseProperty, GeometryMapper):
    """
    Geometry property.

    This property supports the following types.

    - velour.schemas.BoundingBox
    - velour.schemas.Polygon
    - velour.schemas.MultiPolygon
    - velour.schemas.Raster

    Parameters
    ----------
    name : str
        The name of the filter property.
    key : str, optional
        If applicable, the key to retrieve this object.
    prefix : str, default="_"
        The prefix applied to the property name when storing a value.

    Attributes
    ----------
    type_ : type
        The type of value that this property stores.

    Examples
    --------
    >>> class Example:
    ...     x = GeometryProperty(
    ...         name="filter_name",
    ...     )
    ...     def __init__(self, x: BoundingBox):
    ...         self.x = x
    ...
    >>> type(Example.x)
    <class 'velour.schemas.properties.GeometryProperty'>
    >>> Example.x.exists()
    BinaryExpression(name='filter_name', constraint=Constraint(value=None, operator='exists'), key=None)

    As a class variable 'x' operates as a mapping object. However, if we instantiate
    the object, we will see that it no longer functions as a mapper and instead has
    become a property of the value type.

    >>> example = Example(x=BoundingBox.from_extrema(0,1,0,1))
    >>> example.x
    BoundingBox(polygon=BasicPolygon(points=[Point(x=0.0, y=0.0), Point(x=1.0, y=0.0), Point(x=1.0, y=1.0), Point(x=0.0, y=1.0)]))
    """

    type_ = GeometryType


class GeospatialProperty(_BaseProperty, GeospatialMapper):
    """
    Geospatial property.

    This property supports GeoJSON defined in a dictionary.

    Parameters
    ----------
    name : str
        The name of the filter property.
    key : str, optional
        If applicable, the key to retrieve this object.
    prefix : str, default="_"
        The prefix applied to the property name when storing a value.

    Attributes
    ----------
    type_ : type
        The type of value that this property stores.
    """

    type_ = GeoJSONType


class DictionaryProperty(_BaseProperty, DictionaryMapper):
    """
    Dictionary property.

    This property supports values of the following types.

    - bool
    - str
    - numerics (int, float, np.floating)
    - datetime (datetime, date, time, timedelta)

    Parameters
    ----------
    name : str
        The name of the filter property.
    key : str, optional
        If applicable, the key to retrieve this object.
    prefix : str, default="_"
        The prefix applied to the property name when storing a value.

    Attributes
    ----------
    type_ : type
        The type of value that this property stores.

    Examples
    --------
    >>> class Example:
    ...     metadata = DictionaryProperty(
    ...         name="filter_name",
    ...     )
    ...     def __init__(self, metadata: MetadataType):
    ...         self.metadata = metadata
    ...
    >>> type(Example.metadata)
    <class 'velour.schemas.properties.DictionaryProperty'>
    >>> Example.metadata["some_key"] == "some_str"
    BinaryExpression(name='filter_name', constraint=Constraint(value='some_str', operator='=='), key='some_key')
    >>> Example.metadata["some_key"] >= 3.14
    BinaryExpression(name='filter_name', constraint=Constraint(value=3.14, operator='>='), key='some_key')

    As a class variable 'x' operates as a mapping object. However, if we instantiate
    the object, we will see that it no longer functions as a mapper and instead has
    become a property of the value type.

    >>> example = Example(metadata={"some_float": 3.14, "some_str": "hello"})
    >>> type(example.metadata)
    <class 'dict'>
    >>> example.metadata
    {'some_float': 3.14, 'some_str': 'hello'}
    """

    type_ = DictMetadataType


class LabelProperty(_BaseProperty, LabelMapper):
    """
    String property.

    Parameters
    ----------
    name : str
        The name of the filter property.
    key : str, optional
        If applicable, the key to retrieve this object.
    prefix : str, default="_"
        The prefix applied to the property name when storing a value.

    Attributes
    ----------
    type_ : type
        The type of value that this property stores.

    Examples
    --------
    >>> class Example:
    ...     x = LabelProperty(
    ...         name="filter_name",
    ...     )
    ...     def __init__(self, x: str):
    ...         self.x = x
    ...
    >>> type(Example.x)
    <class 'velour.schemas.properties.LabelProperty'>
    >>> Example.x == Label(key="k1", value="v1")
    BinaryExpression(name='filter_name', constraint=Constraint(value={'k1': 'v1'}, operator='=='), key=None)

    As a class variable 'x' operates as a mapping object. However, if we instantiate
    the object, we will see that it no longer functions as a mapper and instead has
    become a property of the value type.

    >>> example = Example(x=Label(key="k1", value="v1"))
    >>> type(example.x)
    <class 'velour.coretypes.Label'>
    >>> str(example.x.to_dict())
    "{'key': 'k1', 'value': 'v1', 'score': None}"
    """

    type_ = Any
