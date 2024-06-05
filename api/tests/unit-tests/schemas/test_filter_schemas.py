import datetime

import pytest

from valor_api.schemas import DateTime
from valor_api.schemas.filters import (
    BooleanFilter,
    DateTimeFilter,
    Equal,
    GeospatialFilter,
    GreaterThan,
    GreaterThanEqual,
    Inside,
    Intersects,
    LessThan,
    LessThanEqual,
    NotEqual,
    NumericFilter,
    Operands,
    Outside,
    StringFilter,
    Symbol,
    Value,
    validate_type_symbol,
)
from valor_api.schemas.geometry import GeoJSON


def test_validate_type_symbol():
    x = Symbol(type="string", name="name")
    y = Value(type="string", value="name")
    validate_type_symbol(x)
    with pytest.raises(TypeError):
        validate_type_symbol(y)


def test_string_filter_to_function():
    name = "owner.name"
    value = "hello world"
    operands = Operands(
        lhs=Symbol(type="string", name=name),
        rhs=Value(type="string", value=value),
    )

    assert StringFilter(value=value, operator="==",).to_function(
        name
    ) == Equal(eq=operands)

    assert StringFilter(value=value, operator="!=",).to_function(
        name
    ) == NotEqual(ne=operands)


def test_boolean_filter_to_function():
    type_str = "boolean"
    name = "owner.name"
    value = True
    operands = Operands(
        lhs=Symbol(type=type_str, name=name),
        rhs=Value(type=type_str, value=value),
    )

    assert BooleanFilter(value=value, operator="==",).to_function(
        name
    ) == Equal(eq=operands)

    assert BooleanFilter(value=value, operator="!=",).to_function(
        name
    ) == NotEqual(ne=operands)


def test_numeric_filter_to_function():
    type_str = "float"
    name = "owner.name"
    value = 0.123
    operands = Operands(
        lhs=Symbol(type=type_str, name=name),
        rhs=Value(type=type_str, value=value),
    )

    assert NumericFilter(value=value, operator="==",).to_function(
        name
    ) == Equal(eq=operands)

    assert NumericFilter(value=value, operator="!=",).to_function(
        name
    ) == NotEqual(ne=operands)

    assert NumericFilter(value=value, operator=">",).to_function(
        name
    ) == GreaterThan(gt=operands)

    assert NumericFilter(value=value, operator=">=",).to_function(
        name
    ) == GreaterThanEqual(ge=operands)

    assert NumericFilter(value=value, operator="<",).to_function(
        name
    ) == LessThan(lt=operands)

    assert NumericFilter(value=value, operator="<=",).to_function(
        name
    ) == LessThanEqual(le=operands)


def test_datetime_filter_to_function():
    type_str = "datetime"
    name = "owner.name"
    value = DateTime(value=datetime.datetime.now().isoformat())
    operands = Operands(
        lhs=Symbol(type=type_str, name=name),
        rhs=Value(type=type_str, value=value.value),
    )

    assert DateTimeFilter(value=value, operator="==",).to_function(
        name
    ) == Equal(eq=operands)

    assert DateTimeFilter(value=value, operator="!=",).to_function(
        name
    ) == NotEqual(ne=operands)

    assert DateTimeFilter(value=value, operator=">",).to_function(
        name
    ) == GreaterThan(gt=operands)

    assert DateTimeFilter(value=value, operator=">=",).to_function(
        name
    ) == GreaterThanEqual(ge=operands)

    assert DateTimeFilter(value=value, operator="<",).to_function(
        name
    ) == LessThan(lt=operands)

    assert DateTimeFilter(value=value, operator="<=",).to_function(
        name
    ) == LessThanEqual(le=operands)


def test_geospatial_filter_to_function():
    type_str = "geojson"
    name = "owner.name"
    value = GeoJSON(type="Point", coordinates=[0.1, 0.1])
    operands = Operands(
        lhs=Symbol(type=type_str, name=name),
        rhs=Value(type=type_str, value=value.geometry.to_json()),
    )

    assert GeospatialFilter(value=value, operator="inside",).to_function(
        name
    ) == Inside(inside=operands)

    assert GeospatialFilter(value=value, operator="outside",).to_function(
        name
    ) == Outside(outside=operands)

    assert GeospatialFilter(value=value, operator="intersect",).to_function(
        name
    ) == Intersects(intersects=operands)
