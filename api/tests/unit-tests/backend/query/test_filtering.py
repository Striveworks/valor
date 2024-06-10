import pytest

from valor_api.backend import models
from valor_api.backend.query.filtering import (
    _recursive_search_logic_tree,
    create_cte,
    generate_logical_expression,
    map_filter_to_tables,
)
from valor_api.schemas.filters import AdvancedFilter as Filter
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

    # validation
    with pytest.raises(TypeError):
        _recursive_search_logic_tree(func="string")  # type: ignore - testing

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


def test_map_filter_to_labels():

    fn = IsNull(isnull=Symbol(type="box", name="annotation.bounding_box"))

    filter_ = Filter(
        datasets=fn,
        models=fn,
        datums=fn,
        annotations=fn,
        groundtruths=fn,
        predictions=fn,
        labels=fn,
        embeddings=fn,
    )

    assert map_filter_to_tables(filter_, label_source=models.Annotation) == {
        models.Dataset,
        models.Model,
        models.Datum,
        models.Annotation,
        models.GroundTruth,
        models.Prediction,
        models.Label,
        models.Embedding,
    }
    assert map_filter_to_tables(filter_, label_source=models.GroundTruth) == {
        models.Dataset,
        models.Model,
        models.Datum,
        models.Annotation,
        models.GroundTruth,
        models.Label,
        models.Embedding,
    }
    assert map_filter_to_tables(filter_, label_source=models.Prediction) == {
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
