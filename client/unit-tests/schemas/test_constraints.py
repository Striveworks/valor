import datetime
import numpy
from typing import Dict, List, Union, Any

import pytest

from velour import Label
from velour.schemas.constraints import (
    NumericMapper,
    StringMapper,
    GeometryMapper,
    GeospatialMapper,
    DatetimeMapper,
    DictionaryMapper,
    LabelMapper,
)
from velour.schemas.geometry import (
    Point,
    BasicPolygon,
    Polygon,
    BoundingBox,
    MultiPolygon,
    Raster,
)


def _test_mapper(mapper, operator, value, retval=None):
    name = "name"
    key = "key"

    if operator == "==":
        expr = mapper(name, key) == value
    elif operator == "!=":
        expr = mapper(name, key) != value        
    elif operator == ">=":
        expr = mapper(name, key) >= value        
    elif operator == "<=":
        expr = mapper(name, key) <= value
    elif operator == ">":
        expr = mapper(name, key) > value        
    elif operator == "<":
        expr = mapper(name, key) < value
    elif operator == "contains":
        expr = mapper(name, key).contains(value)
    elif operator == "intersect":
        expr = mapper(name, key).intersect(value)
    elif operator == "inside":
        expr = mapper(name, key).inside(value)
    elif operator == "outside":
        expr = mapper(name, key).outside(value)
    elif operator == "is_none":
        expr = mapper(name, key).is_none()
        assert expr.name == name
        value = None
    elif operator == "exists":
        expr = mapper(name, key).exists()
        value = None
    elif operator == "in_":
        expr = mapper(name, key).in_([value])[0]
        operator = "=="
    else:
        raise NotImplementedError
        
    assert expr.name == name
    assert expr.key == key
    assert expr.constraint.operator == operator
    
    if retval:
        assert expr.constraint.value == retval
    elif value is None:
        assert expr.constraint.value is None
    else:
        assert expr.constraint.value == value


def test_numeric_mapper():

    # test operators on `int` and `float`
    for value in [int(123), float(0.123)]:
        _test_numeric_mapper = lambda operator : _test_mapper(NumericMapper, operator, value)
    
        _test_numeric_mapper("==")
        _test_numeric_mapper("!=")
        _test_numeric_mapper(">=")
        _test_numeric_mapper("<=")
        _test_numeric_mapper(">")
        _test_numeric_mapper("<")
        _test_numeric_mapper("in_")
        
        with pytest.raises(ValueError):
            _test_numeric_mapper("is_none")
        with pytest.raises(ValueError):
            _test_numeric_mapper("exists")
        with pytest.raises(ValueError):
            _test_numeric_mapper("contains")
        with pytest.raises(ValueError):
            _test_numeric_mapper("intersect")
        with pytest.raises(ValueError):
            _test_numeric_mapper("inside")
        with pytest.raises(ValueError):
            _test_numeric_mapper("outside")

    # test invalid type
    with pytest.raises(TypeError):
        _test_mapper(NumericMapper, "==", "some_str")


def test_string_mapper():
    
    # test operators on `str`
    _test_string_mapper = lambda operator : _test_mapper(StringMapper, operator, "some_str")
    
    _test_string_mapper("==")
    _test_string_mapper("!=")
    _test_string_mapper("in_")
    
    with pytest.raises(ValueError):
        _test_string_mapper(">=")
    with pytest.raises(ValueError):
        _test_string_mapper("<=")
    with pytest.raises(ValueError):
        _test_string_mapper(">")
    with pytest.raises(ValueError):
        _test_string_mapper("<")
    with pytest.raises(ValueError):
        _test_string_mapper("is_none")
    with pytest.raises(ValueError):
        _test_string_mapper("exists")
    with pytest.raises(ValueError):
        _test_string_mapper("contains")
    with pytest.raises(ValueError):
        _test_string_mapper("intersect")
    with pytest.raises(ValueError):
        _test_string_mapper("inside")
    with pytest.raises(ValueError):
        _test_string_mapper("outside")

    # test invalid type
    with pytest.raises(TypeError):
        _test_mapper(StringMapper, "==", 123)


def test_datetime_mapper():

    # test operators on `datetime` objects
    datetime_ = datetime.datetime.now()
    date_ = datetime.date.today()
    time_ = datetime.datetime.now().time()
    timedelta_ = datetime.timedelta(days=1)

    datetime_filter = {"datetime" : str(datetime_.isoformat())}
    date_filter = {"date" : str(date_.isoformat())}
    time_filter = {"time" : str(time_.isoformat())}
    timedelta_filter = {"duration" : str(timedelta_.total_seconds())}

    for value, retval in [
        (datetime_, datetime_filter),
        (date_, date_filter),
        (time_, time_filter),
        (timedelta_, timedelta_filter),
    ]:
        _test_datetime_mapper = lambda operator : _test_mapper(DatetimeMapper, operator, value, retval=retval)
    
        _test_datetime_mapper("==")
        _test_datetime_mapper("!=")
        _test_datetime_mapper(">=")
        _test_datetime_mapper("<=")
        _test_datetime_mapper(">")
        _test_datetime_mapper("<")
        _test_datetime_mapper("in_")
        
        with pytest.raises(ValueError):
            _test_datetime_mapper("is_none")
        with pytest.raises(ValueError):
            _test_datetime_mapper("exists")
        with pytest.raises(ValueError):
            _test_datetime_mapper("contains")
        with pytest.raises(ValueError):
            _test_datetime_mapper("intersect")
        with pytest.raises(ValueError):
            _test_datetime_mapper("inside")
        with pytest.raises(ValueError):
            _test_datetime_mapper("outside")

    # test invalid type
    with pytest.raises(TypeError):
        _test_mapper(DatetimeMapper, "==", 1234)


def test_geometry_mapper():

    # test attribute mappers
    assert type(GeometryMapper("name", "key").area) is NumericMapper

    # test operators on geometry objects
    point = Point(1,1)
    basic_polygon = BasicPolygon(
        points=[
            Point(0,0),
            Point(0,1),
            Point(1,0),
        ]
    )
    polygon = Polygon(
        boundary=basic_polygon,
    )
    bbox = BoundingBox.from_extrema(xmin=0, xmax=1, ymin=0, ymax=1)
    multipolygon = MultiPolygon(polygons=[polygon])
    raster = Raster.from_numpy(numpy.zeros((10,10)) == True)

    for value in [
        point,
        bbox,
        polygon,
        multipolygon,
        raster,
    ]:
        _test_spatial_mapper = lambda operator : _test_mapper(GeometryMapper, operator, value)

        _test_spatial_mapper("is_none")
        _test_spatial_mapper("exists")

        with pytest.raises(NotImplementedError):
            _test_spatial_mapper("contains")
        with pytest.raises(NotImplementedError):
            _test_spatial_mapper("intersect")
        with pytest.raises(NotImplementedError):
            _test_spatial_mapper("inside")
        with pytest.raises(NotImplementedError):
            _test_spatial_mapper("outside")
    
        with pytest.raises(ValueError):
            _test_spatial_mapper("==")
        with pytest.raises(ValueError):
            _test_spatial_mapper("!=")
        with pytest.raises(ValueError):
            _test_spatial_mapper(">=")
        with pytest.raises(ValueError):
            _test_spatial_mapper("<=")
        with pytest.raises(ValueError):
            _test_spatial_mapper(">")
        with pytest.raises(ValueError):
            _test_spatial_mapper("<")
        with pytest.raises(ValueError):
            _test_spatial_mapper("in_")      

    # test unsupported geometry
    with pytest.raises(TypeError):
        _test_mapper(GeometryMapper, "intersect", "intersect")
    
    # test invalid type
    with pytest.raises(TypeError):
        _test_mapper(GeometryMapper, "intersect", "some_str")


def test_geospatial_mapper():

    # test attribute mappers
    assert type(GeospatialMapper("name", "key").area) is NumericMapper

    # test operators on geojson
    point = {"type": "Point", "coordinates": [30.0, 10.0]}

    for value in [
        point,
    ]:
        _test_spatial_mapper = lambda operator : _test_mapper(GeospatialMapper, operator, value)

        _test_spatial_mapper("intersect")
        _test_spatial_mapper("inside")
        _test_spatial_mapper("outside")


        with pytest.raises(NotImplementedError):
            _test_spatial_mapper("contains")
        with pytest.raises(NotImplementedError):
            _test_spatial_mapper("is_none")            
        with pytest.raises(NotImplementedError):
            _test_spatial_mapper("exists")
    
        with pytest.raises(ValueError):
            _test_spatial_mapper("==")
        with pytest.raises(ValueError):
            _test_spatial_mapper("!=")
        with pytest.raises(ValueError):
            _test_spatial_mapper(">=")
        with pytest.raises(ValueError):
            _test_spatial_mapper("<=")
        with pytest.raises(ValueError):
            _test_spatial_mapper(">")
        with pytest.raises(ValueError):
            _test_spatial_mapper("<")
        with pytest.raises(ValueError):
            _test_spatial_mapper("in_")      
    
    # test invalid type
    with pytest.raises(TypeError):
        _test_mapper(GeospatialMapper, "intersect", "some_str")


def test_dictionary_mapper():

    # test operators on `int` and `float`
    for value in [int(123), float(0.123)]:
        mapper = lambda operator, key : DictionaryMapper("name")[key]
        _test_dict_mapper = lambda operator : _test_mapper(mapper, operator, value)
    
        _test_dict_mapper("is_none")
        _test_dict_mapper("exists")
        _test_dict_mapper("==")
        _test_dict_mapper("!=")
        _test_dict_mapper(">=")
        _test_dict_mapper("<=")
        _test_dict_mapper(">")
        _test_dict_mapper("<")
        _test_dict_mapper("in_")
        
        with pytest.raises(ValueError):
            _test_dict_mapper("contains")
        with pytest.raises(ValueError):
            _test_dict_mapper("intersect")
        with pytest.raises(ValueError):
            _test_dict_mapper("inside")
        with pytest.raises(ValueError):
            _test_dict_mapper("outside")

    # test operators on `str`
    value = "some_str"
    mapper = lambda operator, key : DictionaryMapper("name")[key]
    _test_dict_mapper = lambda operator : _test_mapper(mapper, operator, value)
    
    _test_dict_mapper("is_none")
    _test_dict_mapper("exists")
    _test_dict_mapper("==")
    _test_dict_mapper("!=")
    _test_dict_mapper("in_")
    
    with pytest.raises(ValueError):
        _test_dict_mapper(">=")
    with pytest.raises(ValueError):
        _test_dict_mapper("<=")
    with pytest.raises(ValueError):
        _test_dict_mapper(">")
    with pytest.raises(ValueError):
        _test_dict_mapper("<")
    with pytest.raises(ValueError):
        _test_dict_mapper("contains")
    with pytest.raises(ValueError):
        _test_dict_mapper("intersect")
    with pytest.raises(ValueError):
        _test_dict_mapper("inside")
    with pytest.raises(ValueError):
        _test_dict_mapper("outside")

    # test operators on `datetime` objects
    datetime_ = datetime.datetime.now()
    date_ = datetime.date.today()
    time_ = datetime.datetime.now().time()
    timedelta_ = datetime.timedelta(days=1)

    datetime_filter = {"datetime" : str(datetime_.isoformat())}
    date_filter = {"date" : str(date_.isoformat())}
    time_filter = {"time" : str(time_.isoformat())}
    timedelta_filter = {"duration" : str(timedelta_.total_seconds())}

    for value, retval in [
        (datetime_, datetime_filter),
        (date_, date_filter),
        (time_, time_filter),
        (timedelta_, timedelta_filter),
    ]:
        mapper = lambda operator, key : DictionaryMapper("name")[key]
    
        _test_dict_mapper = lambda operator : _test_mapper(mapper, operator, value)
        _test_dict_mapper("is_none")
        _test_dict_mapper("exists")

        _test_dict_mapper = lambda operator : _test_mapper(mapper, operator, value, retval=retval)
        _test_dict_mapper("==")
        _test_dict_mapper("!=")
        _test_dict_mapper(">=")
        _test_dict_mapper("<=")
        _test_dict_mapper(">")
        _test_dict_mapper("<")
        _test_dict_mapper("in_")
        
        with pytest.raises(ValueError):
            _test_dict_mapper("contains")
        with pytest.raises(ValueError):
            _test_dict_mapper("intersect")
        with pytest.raises(ValueError):
            _test_dict_mapper("inside")
        with pytest.raises(ValueError):
            _test_dict_mapper("outside")

    # test invalid type
    with pytest.raises(NotImplementedError):
        DictionaryMapper("name")["key"] == Point(1,1)


def test_label_mapper():

    # test operators on `Label`
    label = Label(key="k1", value="v1")
    _test_label_mapper = lambda operator : _test_mapper(LabelMapper, operator, label, retval={'k1':'v1'})
    
    _test_label_mapper("==")
    _test_label_mapper("!=")
    _test_label_mapper("in_")
    
    with pytest.raises(ValueError):
        _test_label_mapper(">=")
    with pytest.raises(ValueError):
        _test_label_mapper("<=")
    with pytest.raises(ValueError):
        _test_label_mapper(">")
    with pytest.raises(ValueError):
        _test_label_mapper("<")
    with pytest.raises(ValueError):
        _test_label_mapper("is_none")
    with pytest.raises(ValueError):
        _test_label_mapper("exists")
    with pytest.raises(ValueError):
        _test_label_mapper("contains")
    with pytest.raises(ValueError):
        _test_label_mapper("intersect")
    with pytest.raises(ValueError):
        _test_label_mapper("inside")
    with pytest.raises(ValueError):
        _test_label_mapper("outside")

    # test invalid type
    with pytest.raises(TypeError):
        _test_mapper(LabelMapper, "==", 123)
