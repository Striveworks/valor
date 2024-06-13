import pytest

from valor_api.backend import models
from valor_api.backend.query.filtering import (
    _recursive_search_logic_tree,
    create_cte,
    generate_logical_expression,
    map_filter_to_tables,
    map_keyed_symbol_to_resources,
    map_opstr_to_operator,
    map_symbol_to_resources,
    map_type_to_jsonb_type_cast,
    map_type_to_type_cast,
)
from valor_api.schemas.filters import (
    Condition,
    Filter,
    FilterOperator,
    LogicalFunction,
    LogicalOperator,
    SupportedSymbol,
    SupportedType,
    Symbol,
    Value,
)


def test_map_to_resources():
    for symbol in SupportedSymbol:
        # test that there is a singular mapping for each symbol
        assert (symbol in map_symbol_to_resources) != (
            symbol in map_keyed_symbol_to_resources
        )


def test_map_to_operator():
    for op in FilterOperator:
        # test that each op has an associated function
        assert op in map_opstr_to_operator


def test_map_to_type_cast():
    for type_ in SupportedType:
        # value type cast
        assert type_ in map_type_to_type_cast
        # jsonb type cast
        assert type_ in map_type_to_jsonb_type_cast


def test_create_cte_validation():
    with pytest.raises(ValueError):
        create_cte(
            Condition(
                lhs="symbol",  # type: ignore - testing
                rhs=Value(type=SupportedType.STRING, value="some_name"),
                op=FilterOperator.EQ,
            )
        )
    with pytest.raises(ValueError):
        create_cte(
            Condition(
                lhs=Symbol(name=SupportedSymbol.DATASET_NAME),
                rhs="value",  # type: ignore - testing
                op=FilterOperator.EQ,
            )
        )
    with pytest.raises(TypeError):
        create_cte(
            Condition(
                lhs=Symbol(name=SupportedSymbol.DATASET_NAME),
                rhs=Value(type=SupportedType.INTEGER, value=1),
                op=FilterOperator.EQ,
            )
        )


def test__recursive_search_logic_tree():

    # validation
    with pytest.raises(TypeError):
        _recursive_search_logic_tree(func="string")  # type: ignore - testing

    # test one arg function
    tree, _, tables = _recursive_search_logic_tree(
        func=LogicalFunction(
            args=Condition(
                lhs=Symbol(name=SupportedSymbol.BOX),
                op=FilterOperator.ISNULL,
            ),
            op=LogicalOperator.NOT,
        )
    )
    assert tables == [models.Annotation]
    assert tree == {"not": 0}

    # test two arg function
    tree, _, tables = _recursive_search_logic_tree(
        func=Condition(
            lhs=Symbol(name=SupportedSymbol.DATASET_NAME),
            rhs=Value.infer("some_name"),
            op=FilterOperator.EQ,
        )
    )
    assert tables == [models.Dataset]
    assert tree == 0

    # test n arg function
    tree, _, tables = _recursive_search_logic_tree(
        func=LogicalFunction(
            args=[
                Condition(
                    lhs=Symbol(name=SupportedSymbol.BOX),
                    op=FilterOperator.ISNULL,
                ),
                Condition(
                    lhs=Symbol(name=SupportedSymbol.DATASET_NAME),
                    rhs=Value(type=SupportedType.STRING, value="some_name"),
                    op=FilterOperator.EQ,
                ),
            ],
            op=LogicalOperator.AND,
        )
    )
    assert tables == [models.Annotation, models.Dataset]
    assert tree == {"and": [0, 1]}


def test_map_filter_to_labels():

    fn = Condition(
        lhs=Symbol(name=SupportedSymbol.BOX),
        op=FilterOperator.ISNULL,
    )

    filters = Filter(
        datasets=fn,
        models=fn,
        datums=fn,
        annotations=fn,
        groundtruths=fn,
        predictions=fn,
        labels=fn,
        embeddings=fn,
    )

    assert map_filter_to_tables(filters, label_source=models.Annotation) == {
        models.Dataset,
        models.Model,
        models.Datum,
        models.Annotation,
        models.GroundTruth,
        models.Prediction,
        models.Label,
        models.Embedding,
    }
    assert map_filter_to_tables(filters, label_source=models.GroundTruth) == {
        models.Dataset,
        models.Model,
        models.Datum,
        models.Annotation,
        models.GroundTruth,
        models.Label,
        models.Embedding,
    }
    assert map_filter_to_tables(filters, label_source=models.Prediction) == {
        models.Dataset,
        models.Model,
        models.Datum,
        models.Annotation,
        models.Prediction,
        models.Label,
        models.Embedding,
    }


def test_generate_logical_expression_validation():
    from sqlalchemy import select

    # tree should be an int or a dict
    with pytest.raises(ValueError):
        generate_logical_expression(
            root=select(models.Label.id).cte(),
            tree=[0, 1],  # type: ignore - testing
            prefix="cte",
        )

    # n-arg expressions should be represented by a list
    with pytest.raises(ValueError):
        generate_logical_expression(
            root=select(models.Label.id).cte(), tree={"and": 0}, prefix="cte"
        )
