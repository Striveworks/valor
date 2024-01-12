import datetime
from typing import Dict, List, Union

import pytest

from velour import Label
from velour.schemas.filters import (
    DeclarativeMapper,
    Filter,
    GeospatialFilter,
    ValueFilter,
)


def test_labels_filter():
    f = Filter.create(
        [
            Label.label.in_(
                [Label(key="k1", value="v1"), Label(key="k2", value="v2")]
            )
        ]
    )
    assert f.labels is not None
    assert {"k1": "v1"} in f.labels
    assert {"k2": "v2"} in f.labels


def test_value_filter():
    def _test_numeric(value):
        for operator in ["==", "!=", ">=", "<=", ">", "<"]:
            ValueFilter(value=value, operator=operator)
        for operator in ["inside", "outside", "intersect", "@"]:
            with pytest.raises(ValueError):
                ValueFilter(value=value, operator=operator)

    def _test_string_or_bool(value):
        for operator in ["==", "!="]:
            ValueFilter(value=value, operator=operator)
        for operator in [
            ">=",
            "<=",
            ">",
            "<",
            "inside",
            "outside",
            "intersect",
            "@",
        ]:
            with pytest.raises(ValueError):
                ValueFilter(value=value, operator=operator)

    # int
    _test_numeric(int(123))

    # float
    _test_numeric(float(123))

    # string
    _test_string_or_bool(str(123))

    # bool
    _test_string_or_bool(True)

    # datetime.datetime
    _test_numeric(datetime.datetime.now())

    # datetime.date
    _test_numeric(datetime.date.today())

    # datetime.time
    _test_numeric(datetime.datetime.now().time())

    # datetime.timedelta
    _test_numeric(datetime.timedelta(days=1))

    # unsupported type
    class SomeUnsupportedType:
        pass

    with pytest.raises(TypeError):
        _test_numeric(SomeUnsupportedType())
    with pytest.raises(TypeError):
        _test_string_or_bool(SomeUnsupportedType())


def test_geospatial_schema():
    GeospatialFilter(value={}, operator="intersect")
    GeospatialFilter(value={}, operator="inside")
    GeospatialFilter(value={}, operator="outside")

    for operator in ["==", "!=", ">=", "<=", ">", "<", "@"]:
        with pytest.raises(ValueError):
            GeospatialFilter(value={}, operator=operator)


def test_declarative_mapper_string():
    expression = (
        DeclarativeMapper(
            name="name",
            object_type=str,
        )
        == "value"
    )
    assert expression.name == "name"
    assert expression.operator == "=="
    assert expression.value == "value"
    assert expression.key is None

    expression = (
        DeclarativeMapper(name="name", object_type=str, key="key") != "value"
    )
    assert expression.name == "name"
    assert expression.operator == "!="
    assert expression.value == "value"
    assert expression.key == "key"

    expression = DeclarativeMapper(
        name="name",
        object_type=str,
    ).in_(["value1", "value2"])
    assert expression[0].name == "name"
    assert expression[0].operator == "=="
    assert expression[0].value == "value1"
    assert expression[0].key is None
    assert expression[1].name == "name"
    assert expression[1].operator == "=="
    assert expression[1].value == "value2"
    assert expression[1].key is None

    with pytest.raises(TypeError):
        expression = (
            DeclarativeMapper(
                name="name",
                object_type=str,
            )
            >= "value"
        )
    with pytest.raises(TypeError):
        expression = (
            DeclarativeMapper(
                name="name",
                object_type=str,
            )
            <= "value"
        )
    with pytest.raises(TypeError):
        expression = (
            DeclarativeMapper(
                name="name",
                object_type=str,
            )
            > "value"
        )
    with pytest.raises(TypeError):
        expression = (
            DeclarativeMapper(
                name="name",
                object_type=str,
            )
            < "value"
        )
    with pytest.raises(TypeError):
        expression = DeclarativeMapper(
            name="name",
            object_type=str,
        ).intersect(
            "value"
        )  # type: ignore
    with pytest.raises(TypeError):
        expression = DeclarativeMapper(
            name="name",
            object_type=str,
        ).inside("value")
    with pytest.raises(TypeError):
        expression = DeclarativeMapper(
            name="name",
            object_type=str,
        ).outside("value")


def test_declarative_mapper_int():
    expression = (
        DeclarativeMapper(
            name="name",
            object_type=int,
        )
        == 1234
    )
    assert expression.name == "name"
    assert expression.operator == "=="
    assert expression.value == 1234
    assert expression.key is None

    expression = (
        DeclarativeMapper(name="name", object_type=int, key="key") != 1234
    )
    assert expression.name == "name"
    assert expression.operator == "!="
    assert expression.value == 1234
    assert expression.key == "key"

    expression = DeclarativeMapper(
        name="name",
        object_type=int,
    ).in_([123, 987])
    assert expression[0].name == "name"
    assert expression[0].operator == "=="
    assert expression[0].value == 123
    assert expression[0].key is None
    assert expression[1].name == "name"
    assert expression[1].operator == "=="
    assert expression[1].value == 987
    assert expression[1].key is None

    expression = (
        DeclarativeMapper(
            name="name",
            object_type=int,
        )
        >= 1234
    )
    assert expression.name == "name"
    assert expression.operator == ">="
    assert expression.value == 1234
    assert expression.key is None

    expression = (
        DeclarativeMapper(
            name="name",
            object_type=int,
        )
        <= 1234
    )
    assert expression.name == "name"
    assert expression.operator == "<="
    assert expression.value == 1234
    assert expression.key is None

    expression = (
        DeclarativeMapper(
            name="name",
            object_type=int,
        )
        > 1234
    )
    assert expression.name == "name"
    assert expression.operator == ">"
    assert expression.value == 1234
    assert expression.key is None

    expression = (
        DeclarativeMapper(
            name="name",
            object_type=int,
        )
        < 1234
    )
    assert expression.name == "name"
    assert expression.operator == "<"
    assert expression.value == 1234
    assert expression.key is None

    with pytest.raises(TypeError):
        expression = DeclarativeMapper(
            name="name",
            object_type=int,
        ).intersect(
            1234
        )  # type: ignore
    with pytest.raises(TypeError):
        expression = DeclarativeMapper(
            name="name",
            object_type=int,
        ).inside(1234)
    with pytest.raises(TypeError):
        expression = DeclarativeMapper(
            name="name",
            object_type=int,
        ).outside(1234)


def test_declarative_mapper_float():
    expression = (
        DeclarativeMapper(
            name="name",
            object_type=float,
        )
        == 12.34
    )
    assert expression.name == "name"
    assert expression.operator == "=="
    assert expression.value == 12.34
    assert expression.key is None

    expression = (
        DeclarativeMapper(name="name", object_type=float, key="key") != 12.34
    )
    assert expression.name == "name"
    assert expression.operator == "!="
    assert expression.value == 12.34
    assert expression.key == "key"

    expression = DeclarativeMapper(
        name="name",
        object_type=float,
    ).in_([12.3, 98.7])
    assert expression[0].name == "name"
    assert expression[0].operator == "=="
    assert expression[0].value == 12.3
    assert expression[0].key is None
    assert expression[1].name == "name"
    assert expression[1].operator == "=="
    assert expression[1].value == 98.7
    assert expression[1].key is None

    expression = (
        DeclarativeMapper(
            name="name",
            object_type=float,
        )
        >= 12.34
    )
    assert expression.name == "name"
    assert expression.operator == ">="
    assert expression.value == 12.34
    assert expression.key is None

    expression = (
        DeclarativeMapper(
            name="name",
            object_type=float,
        )
        <= 12.34
    )
    assert expression.name == "name"
    assert expression.operator == "<="
    assert expression.value == 12.34
    assert expression.key is None

    expression = (
        DeclarativeMapper(
            name="name",
            object_type=float,
        )
        > 12.34
    )
    assert expression.name == "name"
    assert expression.operator == ">"
    assert expression.value == 12.34
    assert expression.key is None

    expression = (
        DeclarativeMapper(
            name="name",
            object_type=float,
        )
        < 12.34
    )
    assert expression.name == "name"
    assert expression.operator == "<"
    assert expression.value == 12.34
    assert expression.key is None

    with pytest.raises(TypeError):
        expression = DeclarativeMapper(
            name="name",
            object_type=float,
        ).intersect(
            12.34
        )  # type: ignore
    with pytest.raises(TypeError):
        expression = DeclarativeMapper(
            name="name",
            object_type=float,
        ).inside(12.34)
    with pytest.raises(TypeError):
        expression = DeclarativeMapper(
            name="name",
            object_type=float,
        ).outside(12.34)


def test_declarative_mapper_datetime_objects():
    def _test_datetime_object(datetime_object):
        # positive
        assert (
            DeclarativeMapper(
                name="some_name",
                object_type=type(datetime_object),
            )
            == datetime_object
        )
        assert (
            DeclarativeMapper(
                name="some_name",
                object_type=type(datetime_object),
            )
            != datetime_object
        )
        assert (
            DeclarativeMapper(
                name="some_name",
                object_type=type(datetime_object),
            )
            >= datetime_object
        )
        assert (
            DeclarativeMapper(
                name="some_name",
                object_type=type(datetime_object),
            )
            <= datetime_object
        )
        assert (
            DeclarativeMapper(
                name="some_name",
                object_type=type(datetime_object),
            )
            > datetime_object
        )
        assert (
            DeclarativeMapper(
                name="some_name",
                object_type=type(datetime_object),
            )
            < datetime_object
        )
        assert DeclarativeMapper(
            name="some_name",
            object_type=type(datetime_object),
        ).in_([datetime_object, datetime_object])

        # negative
        with pytest.raises(TypeError):
            DeclarativeMapper(
                name="some_name",
                object_type=type(datetime_object),
            ).intersect(
                datetime_object, datetime_object
            )  # type: ignore
        with pytest.raises(TypeError):
            DeclarativeMapper(
                name="some_name",
                object_type=type(datetime_object),
            ).inside(
                datetime_object, datetime_object
            )  # type: ignore
        with pytest.raises(TypeError):
            DeclarativeMapper(
                name="some_name",
                object_type=type(datetime_object),
            ).outside(
                datetime_object, datetime_object
            )  # type: ignore

    _test_datetime_object(datetime.datetime.now())
    _test_datetime_object(datetime.date.today())
    _test_datetime_object(datetime.datetime.now().time())
    _test_datetime_object(datetime.timedelta(days=1))


def test_declarative_mapper_geospatial():
    point = {"type": "Point", "coordinates": [30.0, 10.0]}
    object_type = Dict[
        str,
        Union[
            List[List[List[List[Union[float, int]]]]],
            List[List[List[Union[float, int]]]],
            List[Union[float, int]],
            str,
        ],
    ]

    DeclarativeMapper(
        name="name",
        object_type=object_type,
    ).intersect(point)
    DeclarativeMapper(
        name="name",
        object_type=object_type,
    ).inside(point)
    DeclarativeMapper(
        name="name",
        object_type=object_type,
    ).outside(point)

    with pytest.raises(TypeError):
        (
            DeclarativeMapper(
                name="name",
                object_type=object_type,
            )
            == point
        )
    with pytest.raises(TypeError):
        (
            DeclarativeMapper(
                name="name",
                object_type=object_type,
            )
            != point
        )
    with pytest.raises(TypeError):
        (
            DeclarativeMapper(
                name="name",
                object_type=object_type,
            )
            >= point
        )
    with pytest.raises(TypeError):
        (
            DeclarativeMapper(
                name="name",
                object_type=object_type,
            )
            <= point
        )
    with pytest.raises(TypeError):
        (
            DeclarativeMapper(
                name="name",
                object_type=object_type,
            )
            > point
        )
    with pytest.raises(TypeError):
        (
            DeclarativeMapper(
                name="name",
                object_type=object_type,
            )
            < point
        )
    with pytest.raises(TypeError):
        DeclarativeMapper(
            name="name",
            object_type=object_type,
        ).in_([point, point])
