import datetime

import numpy
import pytest

from valor import Label
from valor.schemas.constraints import (
    DatetimeMapper,
    DictionaryMapper,
    GeometryMapper,
    GeospatialMapper,
    LabelMapper,
    NumericMapper,
    StringMapper,
    _DictionaryValueMapper,
)
from valor.schemas.geometry import (
    BasicPolygon,
    BoundingBox,
    MultiPolygon,
    Point,
    Polygon,
    Raster,
)


def _test_mapper(mapper, operator, value, retval=None, name_prefix=""):
    name = "name"
    key = "key"

    ret_name = f"{name_prefix}{name}"

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
        value = None
    elif operator == "exists":
        expr = mapper(name, key).exists()
        value = None
    elif operator == "in_":
        expr = mapper(name, key).in_([value])[0]
        operator = "=="
    else:
        raise NotImplementedError

    assert expr.name == ret_name
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

        def _test_numeric_mapper(operator):
            _test_mapper(NumericMapper, operator, value)

        _test_numeric_mapper("==")
        _test_numeric_mapper("!=")
        _test_numeric_mapper(">=")
        _test_numeric_mapper("<=")
        _test_numeric_mapper(">")
        _test_numeric_mapper("<")
        _test_numeric_mapper("in_")

        with pytest.raises(AttributeError):
            _test_numeric_mapper("is_none")
        with pytest.raises(AttributeError):
            _test_numeric_mapper("exists")
        with pytest.raises(AttributeError):
            _test_numeric_mapper("contains")
        with pytest.raises(AttributeError):
            _test_numeric_mapper("intersect")
        with pytest.raises(AttributeError):
            _test_numeric_mapper("inside")
        with pytest.raises(AttributeError):
            _test_numeric_mapper("outside")

    # test invalid type
    with pytest.raises(TypeError):
        _test_mapper(NumericMapper, "==", "some_str")


def test_string_mapper():

    # test operators on `str`
    def _test_string_mapper(operator):
        _test_mapper(StringMapper, operator, "some_str")

    _test_string_mapper("==")
    _test_string_mapper("!=")
    _test_string_mapper("in_")

    with pytest.raises(AttributeError):
        _test_string_mapper(">=")
    with pytest.raises(AttributeError):
        _test_string_mapper("<=")
    with pytest.raises(AttributeError):
        _test_string_mapper(">")
    with pytest.raises(AttributeError):
        _test_string_mapper("<")
    with pytest.raises(AttributeError):
        _test_string_mapper("is_none")
    with pytest.raises(AttributeError):
        _test_string_mapper("exists")
    with pytest.raises(AttributeError):
        _test_string_mapper("contains")
    with pytest.raises(AttributeError):
        _test_string_mapper("intersect")
    with pytest.raises(AttributeError):
        _test_string_mapper("inside")
    with pytest.raises(AttributeError):
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

    datetime_filter = {"datetime": str(datetime_.isoformat())}
    date_filter = {"date": str(date_.isoformat())}
    time_filter = {"time": str(time_.isoformat())}
    timedelta_filter = {"duration": str(timedelta_.total_seconds())}

    for value, retval in [
        (datetime_, datetime_filter),
        (date_, date_filter),
        (time_, time_filter),
        (timedelta_, timedelta_filter),
    ]:

        def _test_datetime_mapper(operator):
            _test_mapper(DatetimeMapper, operator, value, retval=retval)

        _test_datetime_mapper("==")
        _test_datetime_mapper("!=")
        _test_datetime_mapper(">=")
        _test_datetime_mapper("<=")
        _test_datetime_mapper(">")
        _test_datetime_mapper("<")
        _test_datetime_mapper("in_")

        with pytest.raises(AttributeError):
            _test_datetime_mapper("is_none")
        with pytest.raises(AttributeError):
            _test_datetime_mapper("exists")
        with pytest.raises(AttributeError):
            _test_datetime_mapper("contains")
        with pytest.raises(AttributeError):
            _test_datetime_mapper("intersect")
        with pytest.raises(AttributeError):
            _test_datetime_mapper("inside")
        with pytest.raises(AttributeError):
            _test_datetime_mapper("outside")

    # test invalid type
    with pytest.raises(TypeError):
        _test_mapper(DatetimeMapper, "==", 1234)


def test_geometry_mapper():

    # test attribute mappers
    assert type(GeometryMapper("name", "key").area) is NumericMapper

    # test operators on geometry objects
    point = Point(1, 1)
    basic_polygon = BasicPolygon(
        points=[
            Point(0, 0),
            Point(0, 1),
            Point(1, 0),
        ]
    )
    polygon = Polygon(
        boundary=basic_polygon,
    )
    bbox = BoundingBox.from_extrema(xmin=0, xmax=1, ymin=0, ymax=1)
    multipolygon = MultiPolygon(polygons=[polygon])
    raster = Raster.from_numpy(numpy.zeros((10, 10)) == True)  # noqa 712

    for value in [
        point,
        bbox,
        polygon,
        multipolygon,
        raster,
    ]:

        def _test_spatial_mapper(operator, name_prefix=""):
            _test_mapper(
                GeometryMapper, operator, value, name_prefix=name_prefix
            )

        _test_spatial_mapper("is_none", "require_")
        _test_spatial_mapper("exists", "require_")

        with pytest.raises(NotImplementedError):
            _test_spatial_mapper("contains")
        with pytest.raises(NotImplementedError):
            _test_spatial_mapper("intersect")
        with pytest.raises(NotImplementedError):
            _test_spatial_mapper("inside")
        with pytest.raises(NotImplementedError):
            _test_spatial_mapper("outside")

        with pytest.raises(AttributeError):
            _test_spatial_mapper("==")
        with pytest.raises(AttributeError):
            _test_spatial_mapper("!=")
        with pytest.raises(AttributeError):
            _test_spatial_mapper(">=")
        with pytest.raises(AttributeError):
            _test_spatial_mapper("<=")
        with pytest.raises(AttributeError):
            _test_spatial_mapper(">")
        with pytest.raises(AttributeError):
            _test_spatial_mapper("<")
        with pytest.raises(AttributeError):
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

        def _test_spatial_mapper(operator):
            _test_mapper(GeospatialMapper, operator, value)

        _test_spatial_mapper("intersect")
        _test_spatial_mapper("inside")
        _test_spatial_mapper("outside")

        with pytest.raises(NotImplementedError):
            _test_spatial_mapper("contains")
        with pytest.raises(NotImplementedError):
            _test_spatial_mapper("is_none")
        with pytest.raises(NotImplementedError):
            _test_spatial_mapper("exists")

        with pytest.raises(AttributeError):
            _test_spatial_mapper("==")
        with pytest.raises(AttributeError):
            _test_spatial_mapper("!=")
        with pytest.raises(AttributeError):
            _test_spatial_mapper(">=")
        with pytest.raises(AttributeError):
            _test_spatial_mapper("<=")
        with pytest.raises(AttributeError):
            _test_spatial_mapper(">")
        with pytest.raises(AttributeError):
            _test_spatial_mapper("<")
        with pytest.raises(AttributeError):
            _test_spatial_mapper("in_")

    # test invalid type
    with pytest.raises(TypeError):
        _test_mapper(GeospatialMapper, "intersect", "some_str")


def test_dictionary_mapper():
    def mapper(operator, key) -> _DictionaryValueMapper:
        return DictionaryMapper("name")[key]

    # test operators on `int` and `float`
    for value in [int(123), float(0.123)]:

        def _test_dict_mapper(operator):
            _test_mapper(mapper, operator, value)

        _test_dict_mapper("is_none")
        _test_dict_mapper("exists")
        _test_dict_mapper("==")
        _test_dict_mapper("!=")
        _test_dict_mapper(">=")
        _test_dict_mapper("<=")
        _test_dict_mapper(">")
        _test_dict_mapper("<")
        _test_dict_mapper("in_")

        with pytest.raises(AttributeError):
            _test_dict_mapper("contains")
        with pytest.raises(AttributeError):
            _test_dict_mapper("intersect")
        with pytest.raises(AttributeError):
            _test_dict_mapper("inside")
        with pytest.raises(AttributeError):
            _test_dict_mapper("outside")

    # test operators on `str`
    value = "some_str"

    def _test_dict_mapper(operator):
        _test_mapper(mapper, operator, value)

    _test_dict_mapper("is_none")
    _test_dict_mapper("exists")
    _test_dict_mapper("==")
    _test_dict_mapper("!=")
    _test_dict_mapper("in_")

    with pytest.raises(AttributeError):
        _test_dict_mapper(">=")
    with pytest.raises(AttributeError):
        _test_dict_mapper("<=")
    with pytest.raises(AttributeError):
        _test_dict_mapper(">")
    with pytest.raises(AttributeError):
        _test_dict_mapper("<")
    with pytest.raises(AttributeError):
        _test_dict_mapper("contains")
    with pytest.raises(AttributeError):
        _test_dict_mapper("intersect")
    with pytest.raises(AttributeError):
        _test_dict_mapper("inside")
    with pytest.raises(AttributeError):
        _test_dict_mapper("outside")

    # test operators on `datetime` objects
    datetime_ = datetime.datetime.now()
    date_ = datetime.date.today()
    time_ = datetime.datetime.now().time()
    timedelta_ = datetime.timedelta(days=1)

    datetime_filter = {"datetime": str(datetime_.isoformat())}
    date_filter = {"date": str(date_.isoformat())}
    time_filter = {"time": str(time_.isoformat())}
    timedelta_filter = {"duration": str(timedelta_.total_seconds())}

    for value, retval in [
        (datetime_, datetime_filter),
        (date_, date_filter),
        (time_, time_filter),
        (timedelta_, timedelta_filter),
    ]:

        def _test_dict_mapper_no_key(operator):
            _test_mapper(mapper, operator, value)

        _test_dict_mapper_no_key("is_none")
        _test_dict_mapper_no_key("exists")

        def _test_dict_mapper(operator):
            _test_mapper(mapper, operator, value, retval=retval)

        _test_dict_mapper("==")
        _test_dict_mapper("!=")
        _test_dict_mapper(">=")
        _test_dict_mapper("<=")
        _test_dict_mapper(">")
        _test_dict_mapper("<")
        _test_dict_mapper("in_")

        with pytest.raises(AttributeError):
            _test_dict_mapper("contains")
        with pytest.raises(AttributeError):
            _test_dict_mapper("intersect")
        with pytest.raises(AttributeError):
            _test_dict_mapper("inside")
        with pytest.raises(AttributeError):
            _test_dict_mapper("outside")

    # test invalid type
    with pytest.raises(NotImplementedError):
        DictionaryMapper("name")["key"] == Point(1, 1)  # type: ignore


def test_label_mapper():

    # test operators on `Label`
    label = Label(key="k1", value="v1")

    def _test_label_mapper(operator):
        _test_mapper(LabelMapper, operator, label, retval={"k1": "v1"})

    _test_label_mapper("==")
    _test_label_mapper("!=")
    _test_label_mapper("in_")

    with pytest.raises(AttributeError):
        _test_label_mapper(">=")
    with pytest.raises(AttributeError):
        _test_label_mapper("<=")
    with pytest.raises(AttributeError):
        _test_label_mapper(">")
    with pytest.raises(AttributeError):
        _test_label_mapper("<")
    with pytest.raises(AttributeError):
        _test_label_mapper("is_none")
    with pytest.raises(AttributeError):
        _test_label_mapper("exists")
    with pytest.raises(AttributeError):
        _test_label_mapper("contains")
    with pytest.raises(AttributeError):
        _test_label_mapper("intersect")
    with pytest.raises(AttributeError):
        _test_label_mapper("inside")
    with pytest.raises(AttributeError):
        _test_label_mapper("outside")

    # test invalid type
    with pytest.raises(TypeError):
        _test_mapper(LabelMapper, "==", 123)
