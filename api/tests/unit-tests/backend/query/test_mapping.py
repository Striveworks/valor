import pytest
from sqlalchemy import Column, Integer, MetaData, Table, case, distinct, func

from valor_api.backend import models
from valor_api.backend.database import Base
from valor_api.backend.query.mapping import (
    _map_name_to_table,
    _recursive_select_to_table_names,
)


def test__map_name_to_table():
    assert _map_name_to_table(models.Dataset.__tablename__) == models.Dataset
    assert _map_name_to_table(models.Model.__tablename__) == models.Model
    assert _map_name_to_table(models.Datum.__tablename__) == models.Datum
    assert (
        _map_name_to_table(models.Annotation.__tablename__)
        == models.Annotation
    )
    assert (
        _map_name_to_table(models.GroundTruth.__tablename__)
        == models.GroundTruth
    )
    assert (
        _map_name_to_table(models.Prediction.__tablename__)
        == models.Prediction
    )
    assert _map_name_to_table(models.Label.__tablename__) == models.Label

    with pytest.raises(ValueError):
        _map_name_to_table("random_str")


def test__recursive_select_to_table_names():

    # some extra tables for testing
    table = Table("test", MetaData(), Column("id", Integer, primary_key=True))

    class Other(Base):
        __table__ = table

    # test passing generic table (Table)
    assert _recursive_select_to_table_names(table) == ["test"]

    # test passing valor table (TableTypeAlias, DeclarativeMeta)
    assert _recursive_select_to_table_names(models.Annotation) == [
        "annotation"
    ]
    assert _recursive_select_to_table_names(Other) == ["test"]

    # test passing valor table
    assert _recursive_select_to_table_names(table.c.id) == ["test"]

    # test (InstrumentedAttribute)
    assert _recursive_select_to_table_names(models.Annotation.id) == [
        "annotation"
    ]

    # test (Function, ClauseList, ColumnClause)
    assert _recursive_select_to_table_names(
        func.count(models.Annotation.id)
    ) == ["annotation"]
    assert _recursive_select_to_table_names(func.count()) == []

    # test (UnaryExpression, ColumnClause)
    assert _recursive_select_to_table_names(distinct(models.Datum.uid)) == [
        "datum"
    ]

    # test (Case)
    assert (
        _recursive_select_to_table_names(
            case(
                (
                    models.Prediction.annotation_id == models.Annotation.id,
                    models.Annotation.id,
                ),
                else_=models.Datum.id,
            )
        )
        == []
    )

    # test not implemented
    with pytest.raises(NotImplementedError):
        _recursive_select_to_table_names("hello")  # type: ignore - testing
