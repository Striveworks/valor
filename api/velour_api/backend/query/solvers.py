from typing import Callable

from sqlalchemy import Select, and_, or_, select
from sqlalchemy.orm.attributes import InstrumentedAttribute
from sqlalchemy.orm.decl_api import DeclarativeMeta
from sqlalchemy.sql.elements import (
    BinaryExpression,
    ColumnElement,
    UnaryExpression,
)

from velour_api.backend.models import (
    Annotation,
    Dataset,
    Datum,
    GroundTruth,
    Label,
    Model,
    Prediction,
)


def _create_where_expression(
    table_set: set[DeclarativeMeta],
    expressions: dict[
        DeclarativeMeta, list[ColumnElement[bool] | BinaryExpression]
    ],
) -> ColumnElement[bool] | BinaryExpression | None:
    expr_agg = []
    for table in table_set:
        if table in expressions:
            expr_agg.extend(expressions[table])
    if len(expr_agg) == 1:
        return expr_agg[0]
    elif len(expr_agg) > 1:
        return and_(*expr_agg)
    else:
        return None


def _trim_extremities(
    graph: list[DeclarativeMeta],
    joint_set: set[DeclarativeMeta],
) -> list[DeclarativeMeta]:
    """trim graph extremities of unused nodes"""
    lhi = 0
    rhi = len(graph)
    for idx, table in enumerate(graph):
        if table in joint_set:
            lhi = idx
            break
    for idx, table in enumerate(reversed(graph)):
        if table in joint_set:
            rhi = rhi - idx
            break
    return graph[lhi:rhi]


def _solve_groundtruth_graph(
    args: tuple[DeclarativeMeta | InstrumentedAttribute | UnaryExpression],
    selected: set[DeclarativeMeta],
    filtered: set[DeclarativeMeta],
    expressions: dict[
        DeclarativeMeta, list[ColumnElement[bool] | BinaryExpression]
    ],
):
    """
    groundtruth_graph = [Dataset, Datum, Annotation, GroundTruth, Label]
    """
    # Graph defintion
    graph = [
        Dataset,
        Datum,
        Annotation,
        GroundTruth,
        Label,
    ]
    connections = {
        Datum: Datum.dataset_id == Dataset.id,
        Annotation: Annotation.datum_id == Datum.id,
        GroundTruth: GroundTruth.annotation_id == Annotation.id,
        Label: Label.id == GroundTruth.label_id,
    }

    # set of tables required to construct query
    joint_set = selected.union(filtered)

    # generate query statement
    query = None
    for table in _trim_extremities(graph, joint_set):
        if query is None:
            query = select(*args).select_from(table)
        else:
            query = query.join(table, connections[table])  # type: ignore

    # generate where statement
    expression = _create_where_expression(joint_set, expressions)
    if query is None:
        raise RuntimeError("Query is unexpectedly None.")

    if expression is not None:
        query = query.where(expression)

    return query


def _solve_model_graph(
    args: tuple[DeclarativeMeta | InstrumentedAttribute | UnaryExpression],
    selected: set[DeclarativeMeta],
    filtered: set[DeclarativeMeta],
    expressions: dict[
        DeclarativeMeta, list[ColumnElement[bool] | BinaryExpression]
    ],
):
    """
    model_graph = [[Model, Annotation, Prediction, Label], [Model, Annotation, Datum, Dataset]]
    """
    subgraph1 = [
        Model,
        Annotation,
        Prediction,
        Label,
    ]
    subgraph2 = [
        Model,
        Annotation,
        Datum,
        Dataset,
    ]
    connections = {
        Annotation: Annotation.model_id == Model.id,
        Datum: Datum.id == Annotation.datum_id,
        Dataset: Dataset.id == Datum.dataset_id,
        Prediction: Prediction.annotation_id == Annotation.id,
        Label: Label.id == Prediction.label_id,
    }

    joint_set = selected.union(filtered)  # type: ignore

    # generate query statement
    graph = _trim_extremities(subgraph1, joint_set)
    graph.extend(_trim_extremities(subgraph2, joint_set))
    query = select(*args).select_from(Model)
    repeated_set = {Model}
    for table in graph:
        if table not in repeated_set:
            query = query.join(table, connections[table])  # type: ignore
            repeated_set.add(table)  # type: ignore

    # generate where statement
    expression = _create_where_expression(joint_set, expressions)
    if expression is not None:
        query = query.where(expression)

    return query


def _solve_prediction_graph(
    args: tuple[DeclarativeMeta | InstrumentedAttribute | UnaryExpression],
    selected: set[DeclarativeMeta],
    filtered: set[DeclarativeMeta],
    expressions: dict[
        DeclarativeMeta, list[ColumnElement[bool] | BinaryExpression]
    ],
):
    """
    prediction_graph = [Dataset, Datum, Annotation, Prediction, Label]
    """
    # Graph defintion
    graph = [
        Dataset,
        Datum,
        Annotation,
        Prediction,
        Label,
    ]
    connections = {
        Datum: Datum.dataset_id == Dataset.id,
        Annotation: Annotation.datum_id == Datum.id,
        Prediction: Prediction.annotation_id == Annotation.id,
        Label: Label.id == Prediction.label_id,
    }

    # set of tables required to construct query
    joint_set = selected.union(filtered)  # type: ignore

    # generate query statement
    query = None
    for table in _trim_extremities(graph, joint_set):
        if query is None:
            query = select(*args).select_from(table)
        else:
            query = query.join(table, connections[table])  # type: ignore

    # generate where statement
    expression = _create_where_expression(joint_set, expressions)

    if query is None:
        raise RuntimeError("Query is unexpectedly None.")

    if expression is not None:
        query = query.where(expression)

    return query


def _solve_joint_graph(
    args: tuple[DeclarativeMeta | InstrumentedAttribute | UnaryExpression],
    selected: set[DeclarativeMeta],
    filtered: set[DeclarativeMeta],
    expressions: dict[
        DeclarativeMeta, list[ColumnElement[bool] | BinaryExpression]
    ],
):
    """
    joint_graph = [[Dataset, Datum, Annotation, Label]]
    """
    graph = [
        Dataset,
        Datum,
        Annotation,
        Label,
    ]  # excluding label as edge case due to forking at groundtruth, prediction
    connections = {
        Datum: Datum.dataset_id == Dataset.id,
        Annotation: Annotation.datum_id == Datum.id,
        GroundTruth: GroundTruth.annotation_id == Annotation.id,
        Prediction: Prediction.annotation_id == Annotation.id,
        Label: or_(
            Label.id == GroundTruth.label_id,
            Label.id == Prediction.label_id,
        ),
    }

    # set of tables required to construct query
    joint_set = selected.union(filtered)  # type: ignore

    # generate query statement
    query = None
    for table in _trim_extremities(graph, joint_set):
        if query is None:
            query = select(*args).select_from(table)
        else:
            if table == Label:
                query = query.join(GroundTruth, connections[GroundTruth])
                query = query.join(
                    Prediction,
                    connections[Prediction],
                    full=True,
                )
                query = query.join(Label, connections[Label])
            else:
                query = query.join(table, connections[table])  # type: ignore

    # generate where statement
    expression = _create_where_expression(joint_set, expressions)

    if query is None:
        raise RuntimeError("Query is unexpectedly None.")

    if expression is not None:
        query = query.where(expression)

    return query


def _solve_nested_graphs(
    query_solver: Callable,
    subquery_solver: Callable,
    unique_set: (
        set[type[Model]] | set[type[Prediction]] | set[type[GroundTruth]]
    ),
    args: tuple[DeclarativeMeta | InstrumentedAttribute | UnaryExpression],
    selected: set[DeclarativeMeta],
    filtered: set[DeclarativeMeta],
    expressions: dict[
        DeclarativeMeta, list[ColumnElement[bool] | BinaryExpression]
    ],
    pivot_table: DeclarativeMeta | None = None,
):
    qset = (filtered - unique_set).union({Datum})
    query = query_solver(
        args=args,
        selected=selected,
        filtered=qset,
        expressions=expressions,
    )
    sub_qset = filtered.intersection(unique_set).union({Datum})
    subquery = (
        subquery_solver(
            args=[Datum.id],
            selected={Datum},
            filtered=sub_qset,
            expressions=expressions,
        )
        if pivot_table in filtered or sub_qset != {pivot_table}
        else None
    )
    return query, subquery


def solve_graph(
    select_args: tuple[
        DeclarativeMeta | InstrumentedAttribute | UnaryExpression
    ],
    selected_tables: set[DeclarativeMeta],
    filter_by_tables: set[DeclarativeMeta],
    expressions: dict[
        DeclarativeMeta, list[ColumnElement[bool] | BinaryExpression]
    ],
    pivot_table: DeclarativeMeta | None = None,
) -> tuple[Select | None, None]:
    """
    Selects best fitting graph to run query generation and returns tuple(query, subquery | None).

    Description
    ----------
    To construct complex queries it is necessary to describe the relationship between predictions and groundtruths.
    By splitting the underlying table relationships into four foundatational graphs the complex relationships can be described by
    sequental lists. From these sequential graphs it is possible to construct the minimum set of nodes required to generate a query.
    For queries that can be described by a single foundational graph, the solution is to trim both ends of the sequence until you
    reach nodes in the query set. The relationships of the remaining nodes can then be used to construct the query. Two foundational
    graphs are required for queries that include both groundtruth and prediction/model constraints. The solver inserts `Datum` as
    the linking point between these two graphs allowing the generation of a query and subquery.

    The four foundational graphs:
    groundtruth_graph   = [Dataset, Datum, Annotation, GroundTruth, Label]
    model_graph         = [[Model, Annotation, Prediction, Label], [Model, Annotation, Datum, Dataset]]
    prediction_graph    = [Dataset, Datum, Annotation, Prediction, Label],
    joint_graph         = [Dataset, Datum, Annotation, Label]

    Removing common nodes, these are reduced to:
    groundtruth_graph_unique    = {GroundTruth}
    prediction_graph_unique     = {Prediction, Model}
    model_graph_unique          = {Prediction}
    joint_graph_unique          = {}

    Descriptions:
    groundtruth_graph : All prediction or model related information removed.
    model_graph       : All groundtruth related information removed.
    prediction_graph  : Subgraph of model_graph. All groundtruth and model information removed.
    joint_graph       : Predictions and groundtruths combined under a full outer join.
    """

    # edge case - only one table specified in args and filters
    if len(selected_tables) == 1 and (
        len(filter_by_tables) == 0 or selected_tables == filter_by_tables
    ):
        if pivot_table:
            filter_by_tables.add(pivot_table)
        else:
            query = select(*select_args)
            expression = _create_where_expression(selected_tables, expressions)  # type: ignore
            if expression is not None:
                query = query.where(expression)
            return query, None

    # edge case - catch intersection that resolves into an empty return.
    # if {GroundTruth, Prediction}.issubset(selected_tables):
    #     raise RuntimeError(
    #         f"Cannot evaluate graph as invalid connection between queried tables. `{selected_tables}`"
    #     )

    # create joint (selected + filtered) table set
    joint_set = selected_tables.union(filter_by_tables)

    # create set of tables to select graph with
    graph_set = (
        {pivot_table}.union(joint_set)
        if isinstance(pivot_table, DeclarativeMeta)
        else joint_set
    )

    subquery_solver = None
    unique_set = None

    # solve groundtruth graph
    if (
        GroundTruth in graph_set
        and Prediction not in selected_tables
        and Model not in selected_tables
    ):
        query_solver = _solve_groundtruth_graph
        if Model in joint_set:
            subquery_solver = _solve_model_graph
            unique_set = {Model}
        elif Prediction in joint_set:
            subquery_solver = _solve_prediction_graph
            unique_set = {Prediction}
    # solve model or prediction graph
    elif GroundTruth not in selected_tables and (
        Model in graph_set or Prediction in graph_set
    ):
        query_solver = (
            _solve_model_graph
            if Model in graph_set
            else _solve_prediction_graph
        )
        if GroundTruth in joint_set:
            subquery_solver = _solve_groundtruth_graph
            unique_set = {GroundTruth}
    # solve joint graph
    else:
        query_solver = _solve_joint_graph

    # generate statement
    if subquery_solver is not None and unique_set is not None:
        return _solve_nested_graphs(
            query_solver=query_solver,
            subquery_solver=subquery_solver,
            unique_set=unique_set,
            args=select_args,
            selected=selected_tables,
            filtered=filter_by_tables,
            expressions=expressions,
            pivot_table=pivot_table,
        )
    else:
        query = query_solver(
            select_args,
            selected_tables,
            filter_by_tables,
            expressions,
        )
        subquery = None
        return query, subquery
