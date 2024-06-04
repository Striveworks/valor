from sqlalchemy import Function
from sqlalchemy.orm.attributes import InstrumentedAttribute
from sqlalchemy.orm.decl_api import DeclarativeMeta
from sqlalchemy.sql.elements import (
    Case,
    ClauseElement,
    ClauseList,
    ColumnClause,
    UnaryExpression,
)
from sqlalchemy.sql.expression import Label
from sqlalchemy.sql.schema import Table

from valor_api.backend import models
from valor_api.backend.query.types import TableTypeAlias


def _map_name_to_table(table_name: str) -> TableTypeAlias | None:
    """
    Returns a sqlalchemy table with matching name.

    Parameters
    ----------
    table_name : str
        The name of a table.

    Returns
    -------
    TableTypeAlias | None
        The corresponding table or 'None' if it doesn't exist.
    """
    match table_name:
        case models.Dataset.__tablename__:
            return models.Dataset
        case models.Model.__tablename__:
            return models.Model
        case models.Datum.__tablename__:
            return models.Datum
        case models.Annotation.__tablename__:
            return models.Annotation
        case models.GroundTruth.__tablename__:
            return models.GroundTruth
        case models.Prediction.__tablename__:
            return models.Prediction
        case models.Label.__tablename__:
            return models.Label
        case _:
            raise ValueError(f"Unsupported table name '{table_name}'.")


def _recursive_select_to_table_names(
    argument: TableTypeAlias
    | DeclarativeMeta
    | InstrumentedAttribute
    | UnaryExpression
    | Function
    | ColumnClause
    | ClauseElement
    | Label
    | Case
    | Table,
) -> list[str]:
    """
    Recursively extract table names from sqlalchemy arguments.

    Recursion is necessary as statements can have deep nesting.
    """
    if isinstance(argument, Table):
        return [argument.name]
    elif isinstance(argument, TableTypeAlias):
        return [argument.__tablename__]
    elif isinstance(argument, DeclarativeMeta):
        if "__tablename__" not in argument.__dict__:
            raise AttributeError(
                f"DeclarativeMeta object '{argument}' missing __tablename__ attribute."
            )
        return [argument.__dict__["__tablename__"]]
    elif isinstance(argument, InstrumentedAttribute):
        return _recursive_select_to_table_names(argument.table)
    elif isinstance(argument, UnaryExpression):
        return _recursive_select_to_table_names(argument.element)
    elif isinstance(argument, ColumnClause):
        if argument.table is None:
            return []
        return _recursive_select_to_table_names(argument.table)
    elif isinstance(argument, Case):
        if argument.value is None:
            return []
        return _recursive_select_to_table_names(argument.value)
    elif isinstance(argument, Label):
        return _recursive_select_to_table_names(argument._element)
    elif isinstance(argument, Function):
        if not argument._has_args:
            return []
        return _recursive_select_to_table_names(argument.clause_expr.element)
    elif isinstance(argument, ClauseList):
        table_names = []
        for clause in argument.clauses:
            table_names.extend(_recursive_select_to_table_names(clause))
        return table_names
    else:
        raise NotImplementedError(
            f"Unsupported table type '{type(argument)}'."
        )


def map_arguments_to_tables(*args) -> set[TableTypeAlias]:
    """
    Finds all dependencies of a sql selection.

    Parameters
    ----------
    args : tuple[Any]
        A tuple of arguments from a selection statement.

    Returns
    -------
    set[Declarative]
        The set of tables required for the selection statement.
    """
    tables = set()
    for argument in args:
        table_names = _recursive_select_to_table_names(argument)
        tables.update(
            [
                _map_name_to_table(name)
                for name in table_names
                if name is not None
            ]
        )
    return tables
