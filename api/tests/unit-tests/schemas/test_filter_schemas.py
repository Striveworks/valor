import pytest

from valor_api.schemas.filters import (
    Condition,
    FilterOperator,
    LogicalFunction,
    LogicalOperator,
    SupportedSymbol,
    Symbol,
    Value,
)


@pytest.fixture
def condition() -> Condition:
    return Condition(
        lhs=Symbol(name=SupportedSymbol.DATASET_NAME),
        rhs=Value.infer("name"),
        op=FilterOperator.EQ,
    )


def test_logical_and(condition: Condition):

    # raises value error if list is empty
    with pytest.raises(ValueError):
        LogicalFunction.and_()

    # raises value error if list is empty
    with pytest.raises(ValueError):
        LogicalFunction.and_(*[None, None, None])

    # if list has length of 1, return the contents
    assert LogicalFunction.and_(*[condition]) == condition

    # if list has length > 1, return the logical combination
    assert LogicalFunction.and_(*[condition, condition]) == LogicalFunction(
        args=[condition, condition],
        op=LogicalOperator.AND,
    )


def test_logical_or(condition: Condition):

    # raises value error if list is empty
    with pytest.raises(ValueError):
        LogicalFunction.or_()

    # raises value error if list is empty
    with pytest.raises(ValueError):
        LogicalFunction.or_(*[None, None, None])

    # if list has length of 1, return the contents
    assert LogicalFunction.or_(*[condition]) == condition

    # if list has length > 1, return the logical combination
    assert LogicalFunction.or_(*[condition, condition]) == LogicalFunction(
        args=[condition, condition],
        op=LogicalOperator.OR,
    )


def test_logical_not(condition: Condition):

    # raises value error if list is empty
    with pytest.raises(TypeError):
        LogicalFunction.not_()  # type: ignore - testing

    # raises value error if list is empty
    with pytest.raises(ValueError):
        LogicalFunction.not_(None)  # type: ignore - testing

    # if list has length of 1, return the negation
    assert LogicalFunction.not_(condition) == LogicalFunction(
        args=condition,
        op=LogicalOperator.NOT,
    )

    # double negation should return the original condition
    assert LogicalFunction.not_(LogicalFunction.not_(condition)) == condition

    # not function cannot be passed more than one argument
    with pytest.raises(TypeError):
        assert LogicalFunction.not_(condition, condition)  # type: ignore - testing
