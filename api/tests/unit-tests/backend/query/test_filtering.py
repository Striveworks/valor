import pytest

from valor_api.backend import models
from valor_api.backend.query.filtering import (
    _recursive_search_logic_tree,
    create_cte,
)
from valor_api.schemas.filters import (
    And,
    Equal,
    IsNull,
    Not,
    Operands,
    Symbol,
    Value,
)


def test_create_cte_validation():
    with pytest.raises(ValueError):
        create_cte(
            opstr="eq",
            symbol="symbol",  # type: ignore - testing
            value=Value(type="string", value="some_name"),
        )
    with pytest.raises(ValueError):
        create_cte(
            opstr="eq",
            symbol=Symbol(type="string", name="dataset.name"),
            value="value",  # type: ignore - testing
        )
    with pytest.raises(TypeError):
        create_cte(
            opstr="eq",
            symbol=Symbol(type="string", name="dataset.name"),
            value=Value(type="integer", value=1),
        )


def test__recursive_search_logic_tree():

    # test one arg function
    tree, _, tables = _recursive_search_logic_tree(
        func=Not(
            logical_not=IsNull(
                isnull=Symbol(type="box", name="annotation.bounding_box")
            )
        )
    )
    assert tables == [models.Annotation]
    assert tree == {"not": 0}

    # test two arg function
    tree, _, tables = _recursive_search_logic_tree(
        func=Equal(
            eq=Operands(
                lhs=Symbol(type="string", name="dataset.name"),
                rhs=Value(type="string", value="some_name"),
            )
        )
    )
    assert tables == [models.Dataset]
    assert tree == 0

    # test n arg function
    tree, _, tables = _recursive_search_logic_tree(
        func=And(
            logical_and=[
                IsNull(
                    isnull=Symbol(type="box", name="annotation.bounding_box")
                ),
                Equal(
                    eq=Operands(
                        lhs=Symbol(type="string", name="dataset.name"),
                        rhs=Value(type="string", value="some_name"),
                    )
                ),
            ]
        )
    )
    assert tables == [models.Annotation, models.Dataset]
    assert tree == {"and": [0, 1]}
