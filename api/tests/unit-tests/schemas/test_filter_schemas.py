import pytest

from valor_api.schemas.filters import (
    Condition,
    FilterOperator,
    LogicalFunction,
    LogicalOperator,
    Symbol,
    Value,
    soft_and,
    soft_or,
)


@pytest.fixture
def condition() -> Condition:
    return Condition(
        lhs=Symbol.DATASET_NAME,
        rhs=Value.infer("name"),
        op=FilterOperator.EQ,
    )


def test_soft_and(condition: Condition):

    # raises value error if list is empty
    with pytest.raises(ValueError):
        soft_and([])

    # raises value error if list is empty
    with pytest.raises(ValueError):
        soft_and([None, None, None])

    # if list has length of 1, return the contents
    assert soft_and([condition]) == condition

    # if list has length > 1, return the logical combination
    assert soft_and([condition, condition]) == LogicalFunction(
        args=[condition, condition],
        op=LogicalOperator.AND,
    )


def test_soft_or(condition: Condition):

    # raises value error if list is empty
    with pytest.raises(ValueError):
        soft_or([])

    # raises value error if list is empty
    with pytest.raises(ValueError):
        soft_or([None, None, None])

    # if list has length of 1, return the contents
    assert soft_or([condition]) == condition

    # if list has length > 1, return the logical combination
    assert soft_or([condition, condition]) == LogicalFunction(
        args=[condition, condition],
        op=LogicalOperator.OR,
    )
