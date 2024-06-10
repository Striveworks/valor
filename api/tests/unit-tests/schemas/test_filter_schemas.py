import datetime

import pytest

from valor_api.schemas import DateTime
from valor_api.schemas.filters import (
    Equal,
    GreaterThan,
    GreaterThanEqual,
    Inside,
    Intersects,
    LessThan,
    LessThanEqual,
    NotEqual,
    Operands,
    Outside,
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
