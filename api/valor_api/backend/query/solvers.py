from typing import Any, Callable

from sqlalchemy import Select, Subquery, alias, case, or_, select
from sqlalchemy.orm import InstrumentedAttribute, Query
from sqlalchemy.sql.elements import UnaryExpression

from valor_api.backend.models import (
    Annotation,
    Dataset,
    Datum,
    Embedding,
    GroundTruth,
    Label,
    Model,
    Prediction,
)
from valor_api.backend.query.filtering import (
    generate_dependencies,
    generate_logical_expression,
    map_filter_to_tables,
)
from valor_api.backend.query.mapping import map_arguments_to_tables
from valor_api.backend.query.types import LabelSourceAlias, TableTypeAlias
from valor_api.schemas.filters import Filter, FunctionType


def _join_label_to_annotation(selection: Select) -> Select[Any]:
    """
    Joins Label to Annotation.

    Aliases GroundTruth and Prediction so that the join does not affect other operations.
    """
    groundtruth = alias(GroundTruth)
    prediction = alias(Prediction)
    return (
        selection.join(
            groundtruth,
            groundtruth.c.annotation_id == Annotation.id,
            isouter=True,
        )
        .join(
            prediction,
            prediction.c.annotation_id == Annotation.id,
            isouter=True,
        )
        .join(
            Label,
            or_(
                Label.id == groundtruth.c.label_id,
                Label.id == prediction.c.label_id,
            ),
        )
    )


def _join_annotation_to_label(selection: Select) -> Select[Any]:
    """
    Joins Annotation to Label.

    Aliases GroundTruth and Prediction so that the join does not affect other operations.
    """
    groundtruth = alias(GroundTruth)
    prediction = alias(Prediction)
    return (
        selection.join(
            groundtruth,
            groundtruth.c.label_id == Label.id,
            isouter=True,
        )
        .join(prediction, prediction.c.label_id == Label.id, isouter=True)
        .join(
            Annotation,
            or_(
                Annotation.id == groundtruth.c.annotation_id,
                Annotation.id == prediction.c.annotation_id,
            ),
        )
    )


def _join_prediction_to_datum(selection: Select) -> Select[Any]:
    """
    Joins Prediction to Datum.

    Aliases Annotation so that the join does not affect other operations.
    """
    annotation = alias(Annotation)
    return selection.join(
        annotation, annotation.c.datum_id == Datum.id, isouter=True
    ).join(Prediction, Prediction.annotation_id == annotation.c.id)


def _join_datum_to_prediction(selection: Select) -> Select[Any]:
    """
    Joins Datum to Prediction.

    Aliases Annotation so that the join does not affect other operations.
    """
    annotation = alias(Annotation)
    return selection.join(
        annotation, annotation.c.id == Prediction.annotation_id, isouter=True
    ).join(Datum, Datum.id == annotation.c.datum_id)


def _join_groundtruth_to_datum(selection: Select) -> Select[Any]:
    """
    Joins GroundTruth to Datum.

    Aliases Annotation so that the join does not affect other operations.
    """
    annotation = alias(Annotation)
    return selection.join(
        annotation, annotation.c.datum_id == Datum.id, isouter=True
    ).join(GroundTruth, GroundTruth.annotation_id == annotation.c.id)


def _join_datum_to_groundtruth(selection: Select) -> Select[Any]:
    """
    Joins Datum to GroundTruth.

    Aliases Annotation so that the join does not affect other operations.
    """
    annotation = alias(Annotation)
    return selection.join(
        annotation, annotation.c.id == GroundTruth.annotation_id, isouter=True
    ).join(Datum, Datum.id == annotation.c.datum_id)


def _join_datum_to_model(selection: Select) -> Select[Any]:
    """
    Joins Datum to Model.

    Aliases Annotation so that the join does not affect other operations.
    """
    annotation = alias(Annotation)
    return selection.join(
        annotation, annotation.c.model_id == Model.id, isouter=True
    ).join(Datum, Datum.id == annotation.c.datum_id)


def _join_model_to_datum(selection: Select) -> Select[Any]:
    """
    Joins Model to Datum.

    Aliases Annotation so that the join does not affect other operations.
    """
    annotation = alias(Annotation)
    return selection.join(
        annotation, annotation.c.datum_id == Datum.id, isouter=True
    ).join(Model, Model.id == annotation.c.model_id)


# Map table to neighbor joins. Annotation as label source.
table_joins_with_annotation_as_label_source = {
    Dataset: {Datum: lambda x: x.join(Datum, Datum.dataset_id == Dataset.id)},
    Model: {
        Annotation: lambda x: x.join(
            Annotation, Annotation.model_id == Model.id
        )
    },
    Datum: {
        Dataset: lambda x: x.join(Dataset, Dataset.id == Datum.dataset_id),
        Annotation: lambda x: x.join(
            Annotation, Annotation.datum_id == Datum.id
        ),
    },
    Annotation: {
        Datum: lambda x: x.join(Datum, Datum.id == Annotation.datum_id),
        Model: lambda x: x.join(Model, Model.id == Annotation.model_id),
        Embedding: lambda x: x.join(
            Embedding, Embedding.id == Annotation.embedding_id
        ),
        Label: _join_label_to_annotation,
    },
    GroundTruth: {
        Label: lambda x: x.join(Label, Label.id == GroundTruth.label_id)
    },
    Prediction: {
        Label: lambda x: x.join(Label, Label.id == Prediction.label_id)
    },
    Label: {
        Annotation: _join_annotation_to_label,
        GroundTruth: lambda x: x.join(
            GroundTruth, GroundTruth.label_id == Label.id
        ),
        Prediction: lambda x: x.join(
            Prediction, Prediction.label_id == Label.id
        ),
    },
    Embedding: {
        Annotation: lambda x: x.join(
            Annotation, Annotation.embedding_id == Embedding.id
        )
    },
}


# Map table to neighbor joins. GroundTruth as label source.
table_joins_with_groundtruth_as_label_source = {
    Dataset: {Datum: lambda x: x.join(Datum, Datum.dataset_id == Dataset.id)},
    Model: {Datum: _join_datum_to_model},
    Datum: {
        Dataset: lambda x: x.join(Dataset, Dataset.id == Datum.dataset_id),
        Model: _join_model_to_datum,
        Annotation: lambda x: x.join(
            Annotation, Annotation.datum_id == Datum.id
        ),
        Prediction: _join_prediction_to_datum,
    },
    Annotation: {
        Datum: lambda x: x.join(Datum, Datum.id == Annotation.datum_id),
        GroundTruth: lambda x: x.join(
            GroundTruth, GroundTruth.annotation_id == Annotation.id
        ),
        Embedding: lambda x: x.join(
            Embedding, Embedding.id == Annotation.embedding_id
        ),
    },
    GroundTruth: {
        Annotation: lambda x: x.join(
            Annotation, Annotation.id == GroundTruth.annotation_id
        ),
        Label: lambda x: x.join(Label, Label.id == GroundTruth.label_id),
    },
    Prediction: {
        Datum: _join_datum_to_prediction,
    },
    Label: {
        GroundTruth: lambda x: x.join(
            GroundTruth, GroundTruth.label_id == Label.id
        ),
    },
    Embedding: {
        Annotation: lambda x: x.join(
            Annotation, Annotation.embedding_id == Embedding.id
        )
    },
}


# Map table to neighbor joins. Prediction as label source.
table_joins_with_prediction_as_label_source = {
    Dataset: {Datum: lambda x: x.join(Datum, Datum.dataset_id == Dataset.id)},
    Model: {
        Annotation: lambda x: x.join(
            Annotation, Annotation.model_id == Model.id
        )
    },
    Datum: {
        Dataset: lambda x: x.join(Dataset, Dataset.id == Datum.dataset_id),
        Annotation: lambda x: x.join(
            Annotation, Annotation.datum_id == Datum.id
        ),
        GroundTruth: _join_groundtruth_to_datum,
    },
    Annotation: {
        Datum: lambda x: x.join(Datum, Datum.id == Annotation.datum_id),
        Model: lambda x: x.join(Model, Model.id == Annotation.model_id),
        Prediction: lambda x: x.join(
            Prediction, Prediction.annotation_id == Annotation.id
        ),
        Embedding: lambda x: x.join(
            Embedding, Embedding.id == Annotation.embedding_id
        ),
    },
    GroundTruth: {
        Datum: _join_datum_to_groundtruth,
    },
    Prediction: {
        Annotation: lambda x: x.join(
            Annotation, Annotation.id == Prediction.annotation_id
        ),
        Label: lambda x: x.join(Label, Label.id == Prediction.label_id),
    },
    Label: {
        Prediction: lambda x: x.join(
            Prediction, Prediction.label_id == Label.id
        ),
    },
    Embedding: {
        Annotation: lambda x: x.join(
            Annotation, Annotation.embedding_id == Embedding.id
        )
    },
}


# Maps label source to dictionaries containing neighor join mappings.
map_label_source_to_neighbor_joins = {
    Annotation: table_joins_with_annotation_as_label_source,
    GroundTruth: table_joins_with_groundtruth_as_label_source,
    Prediction: table_joins_with_prediction_as_label_source,
}


# Maps label source to dictionary containing neighbor mappings.
map_label_source_to_neighbor_tables = {
    Annotation: {
        Dataset: {Datum},
        Model: {Annotation},
        Datum: {Dataset, Annotation},
        Annotation: {Datum, Model, Embedding, Label},
        GroundTruth: {Label},
        Prediction: {Label},
        Embedding: {Annotation},
        Label: {Annotation, GroundTruth, Prediction},
    },
    GroundTruth: {
        Dataset: {Datum},
        Model: {Annotation},
        Datum: {Dataset, Annotation, Prediction},
        Annotation: {Datum, Model, GroundTruth, Embedding},
        Embedding: {Annotation},
        GroundTruth: {Annotation, Label},
        Prediction: {Datum},
        Label: {GroundTruth},
    },
    Prediction: {
        Dataset: {Datum},
        Model: {Annotation},
        Datum: {Dataset, Annotation, GroundTruth},
        Annotation: {Datum, Model, Prediction, Embedding},
        Embedding: {Annotation},
        GroundTruth: {Datum},
        Prediction: {Annotation, Label},
        Label: {Prediction},
    },
}


def _recursive_search(
    table: TableTypeAlias,
    target: TableTypeAlias,
    mapping: dict[TableTypeAlias, set[TableTypeAlias]],
    cache: list[TableTypeAlias] | None = None,
) -> list[TableTypeAlias]:
    """
    Depth-first search of table graph.

    Parameters
    ----------
    table : TableTypeAlias
        The starting node.
    target : TableTypeAlias
        The desired endpoint.
    mapping : dict[TableTypeAlias, set[TableTypeAlias]]
        A mapping of tables to their neighbors.
    cache : list[TableTypeAlias] | None, optional
        A cache of previously visited nodes.

    Returns
    -------
    list[TableTypeAlias]
        An ordered list of tables representing join order.
    """
    if cache is None:
        cache = [table]
    for neighbor in mapping[table]:
        if neighbor in cache:
            continue
        elif neighbor is target:
            cache.append(target)
            return cache
        elif retval := _recursive_search(
            neighbor, target=target, mapping=mapping, cache=[*cache, neighbor]
        ):
            return retval
    return []


def _solve_graph(
    select_from: TableTypeAlias,
    label_source: LabelSourceAlias,
    tables: set[TableTypeAlias],
) -> list[Callable]:
    """
    Returns a list of join operations that connect the 'select_from' table to all the provided 'tables'.

    Parameters
    ----------
    select_from : TableTypeAlias
        The table that is being selected from.
    label_source : LabelSourceAlias
        The table that is being used as the source of labels.
    tables : set[TableTypeAlias]
        The set of tables that need to be joined.

    Returns
    -------
    list[Callable]
        An ordered list of join operations.
    """
    if label_source not in {GroundTruth, Prediction, Annotation}:
        raise ValueError(
            "Label source must be either GroundTruth, Prediction or Annotation."
        )

    table_mapping = map_label_source_to_neighbor_tables[label_source]
    join_mapping = map_label_source_to_neighbor_joins[label_source]

    ordered_tables = [select_from]
    ordered_joins = []
    for target in tables:
        if select_from is target:
            continue
        solution = _recursive_search(
            table=select_from,
            target=target,
            mapping=table_mapping,
        )
        for idx in range(1, len(solution)):
            lhs = solution[idx - 1]
            rhs = solution[idx]
            if rhs not in ordered_tables:
                ordered_tables.append(rhs)
                ordered_joins.append(join_mapping[lhs][rhs])
    return ordered_joins


def generate_query(
    select_statement: Select[Any] | Query[Any],
    args: tuple[TableTypeAlias | InstrumentedAttribute | UnaryExpression],
    select_from: TableTypeAlias,
    label_source: LabelSourceAlias,
    filters: Filter | None = None,
) -> Select[Any] | Query[Any]:
    """
    Generates the main query.

    Includes all args-related and filter-related tables.

    Parameters
    ----------
    select_statement : Select[Any] | Query[Any]
        The select statement.
    args : tuple[TableTypeAlias | InstrumentedAttribute | UnaryExpression]
        The user's list of positional arguments.
    select_from : TableTypeAlias
        The table to center the query over.
    label_source : LabelSourceAlias
        The table to use as a source of labels.
    filters : Filter, optional
        An optional filter to apply to the query.

    Returns
    -------
    Select[Any] | Query[Any]
        The main body of the query. Does not include filter conditions.
    """
    if label_source not in {Annotation, GroundTruth, Prediction}:
        raise ValueError(f"Invalid label source '{label_source}'.")

    arg_tables = map_arguments_to_tables(*args)
    filter_tables = map_filter_to_tables(filters, label_source)

    tables = arg_tables.union(filter_tables)
    tables.discard(select_from)
    ordered_joins = _solve_graph(
        select_from=select_from, label_source=label_source, tables=tables
    )
    query = select_statement.select_from(select_from)
    for join in ordered_joins:
        query = join(query)
    return query


def generate_filter_subquery(
    conditions: FunctionType,
    select_from: TableTypeAlias,
    label_source: LabelSourceAlias,
    prefix: str,
) -> Subquery[Any]:
    """
    Generates the filtering subquery.

    Parameters
    ----------
    conditions : FunctionType
        The filtering function to apply.
    select_from : TableTypeAlias
        The table to center the query over.
    label_source : LabelSourceAlias
        The table to use as a source of labels.
    prefix : str
        The prefix to use in naming the CTE queries.

    Returns
    -------
    Subquery[Any]
        A filtering subquery.
    """
    if label_source not in {Annotation, GroundTruth, Prediction}:
        raise ValueError(f"Invalid label source '{label_source}'.")

    tree, ordered_ctes, ordered_tables = generate_dependencies(conditions)
    if tree is None:
        raise ValueError(f"Invalid function given as input. '{conditions}'")

    tables = set(ordered_tables)
    tables.discard(select_from)
    ordered_joins = _solve_graph(
        select_from=select_from,
        label_source=label_source,
        tables=tables,
    )

    expressions = [
        (table.id == cte.c.id)
        for table, cte in zip(ordered_tables, ordered_ctes)
    ]

    # construct query
    query = select(
        select_from.id.label("id"),
        *[
            case((expr, 1), else_=0).label(f"{prefix}{idx}")
            for idx, expr in enumerate(expressions)
        ],
    )
    query = query.select_from(select_from)
    for join in ordered_joins:
        query = join(query)
    for table, cte in zip(ordered_tables, ordered_ctes):
        query = query.join(cte, cte.c.id == table.id, isouter=True)
    query = query.distinct().cte()

    return (
        select(query.c.id.label("id"))
        .select_from(query)
        .where(generate_logical_expression(query, tree, prefix=prefix))
        .subquery()
    )


def generate_filter_queries(
    filters: Filter,
    label_source: LabelSourceAlias,
) -> list[tuple[Subquery[Any], TableTypeAlias]]:
    """
    Generates the filtering subqueries.

    For each attribute defined in the filter a subquery is created that implements it.

    Parameters
    ----------
    filters : Filter
        The filter to apply.
    label_source : LabelSourceAlias
        The table to use as a source of labels.

    Returns
    -------
    list[tuple[Subquery[Any], TableTypeAlias]]
        A list of tuples containing a filtering subquery and the table to join it on.
    """

    def _generator(
        conditions: FunctionType,
        select_from: TableTypeAlias,
        label_source: LabelSourceAlias,
        prefix: str,
    ):
        return generate_filter_subquery(
            conditions=conditions,
            select_from=select_from,
            label_source=label_source,
            prefix=f"{prefix}_cte",
        )

    queries = list()
    if filters.datasets:
        conditions = filters.datasets
        select_from = Dataset
        prefix = "ds"
        queries.append(
            (
                _generator(conditions, select_from, label_source, prefix),
                select_from,
            )
        )
    if filters.models:
        conditions = filters.models
        select_from = Model
        prefix = "md"
        queries.append(
            (
                _generator(conditions, select_from, label_source, prefix),
                select_from,
            )
        )
    if filters.datums:
        conditions = filters.datums
        select_from = Datum
        prefix = "dt"
        queries.append(
            (
                _generator(conditions, select_from, label_source, prefix),
                select_from,
            )
        )
    if filters.annotations:
        conditions = filters.annotations
        select_from = Annotation
        prefix = "an"
        queries.append(
            (
                _generator(conditions, select_from, label_source, prefix),
                select_from,
            )
        )
    if filters.groundtruths:
        conditions = filters.groundtruths
        select_from = GroundTruth if label_source is not Prediction else Datum
        prefix = "gt"
        queries.append(
            (
                _generator(conditions, select_from, GroundTruth, prefix),
                select_from,
            )
        )
    if filters.predictions:
        conditions = filters.predictions
        select_from = Prediction if label_source is not GroundTruth else Datum
        prefix = "pd"
        queries.append(
            (
                _generator(conditions, select_from, Prediction, prefix),
                select_from,
            )
        )
    if filters.labels:
        conditions = filters.labels
        select_from = Label
        prefix = "lb"
        queries.append(
            (
                _generator(conditions, select_from, label_source, prefix),
                select_from,
            )
        )
    if filters.embeddings:
        conditions = filters.embeddings
        select_from = Embedding
        prefix = "em"
        queries.append(
            (
                _generator(conditions, select_from, label_source, prefix),
                select_from,
            )
        )
    return queries


def solver(
    *args,
    stmt: Select[Any] | Query[Any],
    filters: Filter | None,
    label_source: LabelSourceAlias,
) -> Select[Any] | Query[Any]:
    """
    Solves and generates a query from the provided arguements.

    Description
    -----------
    To construct complex queries it is necessary to describe the relationship between predictions and groundtruths.
    By splitting the underlying table relationships into three foundatational graphs the complex relationships can be described by
    sequental lists. From these sequential graphs it is possible to construct the minimum set of nodes required to generate a query.
    For queries that can be described by a single foundational graph, the solution is to trim both ends of the sequence until you
    reach nodes in the query set. The relationships of the remaining nodes can then be used to construct the query. Depending on the
    configuration the solver will choose a table  as the linking point between these two graphs allowing the generation of a
    query and subquery. This configuration is handled by the 'label_source' parameter.

    The graph structure is determined soley by label source as the Label table is the only table that can create cycles in the
    graph (e.g. Annotation -> GroundTruth -> Label -> Prediction -> Annotation). Cycles are not ideal as a solver would not be able to
    determine stopping conditions or whether the chosen path is correct. For this reason, we choose either the Annotation, GroundTruth
    or Prediction table to be our 'label_source' and rearrange the graph structure accordingly.

    All information regarding the structure of the three graphs can be found above in the following dictionaries.
    - map_label_source_to_neighbor_joins
    - map_label_source_to_neighbor_tables

    See documentation for more information.

    Parameters
    ----------
    *args : tuple[Any]
        A list of select statement arguments.
    stmt : Select[Any] | Query[Any]
        A selection or query using the provided args.
    filters : Filter, optional
        An optional filter.
    label_source : LabelSourceAlias
        The table to use as a source of labels.

    Returns
    -------
    Select[Any] | Query[Any]
        An executable query that meets all conditions.
    """

    select_from = map_arguments_to_tables(args[0]).pop()
    if select_from is Label:
        select_from = label_source
    query = generate_query(
        select_statement=stmt,
        args=args,
        select_from=select_from,
        label_source=label_source,
        filters=filters,
    )
    if filters is not None:
        filter_subqueries = generate_filter_queries(
            filters=filters, label_source=label_source
        )
        for subquery, selected_from in filter_subqueries:
            query = query.join(subquery, subquery.c.id == selected_from.id)
    return query
