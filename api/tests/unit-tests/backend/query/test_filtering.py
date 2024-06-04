from valor_api.backend.query.filtering import create_cte
from valor_api.schemas.filters import (
    And,
    Equal,
    IsNull,
    Operands,
    Or,
    Symbol,
    Value,
)


def test_create_cte():
    cte = create_cte(
        opstr="equal",
        symbol=Symbol(
            name="dataset.metadata",
            key="key1",
            attribute="area",
            type="polygon",
        ),
        value=Value(
            type="polygon",
            value=[
                [
                    [0, 0],
                    [0, 1],
                    [2, 0],
                    [0, 0],
                ]
            ],
        ),
    )
    # print(cte)


def test__recursive_search_logic_tree():
    f = And(
        logical_and=[
            IsNull(
                isnull=Symbol(
                    name="annotation.polygon",
                    type="polygon",
                )
            ),
            Equal(
                eq=Operands(
                    lhs=Symbol(
                        name="dataset.metadata",
                        key="some_string",
                        attribute=None,
                        type="string",
                    ),
                    rhs=Value(
                        type="string",
                        value="hello world",
                    ),
                )
            ),
            Or(
                logical_or=[
                    IsNull(
                        isnull=Symbol(
                            name="annotation.polygon",
                            type="polygon",
                        )
                    ),
                    Equal(
                        eq=Operands(
                            lhs=Symbol(
                                name="dataset.metadata",
                                key="some_string",
                                attribute=None,
                                type="string",
                            ),
                            rhs=Value(
                                type="string",
                                value="hello world",
                            ),
                        )
                    ),
                ]
            ),
        ]
    )
