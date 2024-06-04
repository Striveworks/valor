from typing import Any

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
from valor_api.schemas.filters import AdvancedFilter as Filter
from valor_api.schemas.filters import FunctionType


def _join_label_to_annotation(selection: Select):
    """TODO"""
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


def _join_annotation_to_label(selection: Select):
    """TODO"""
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


def _join_prediction_to_datum(selection: Select):
    """TODO"""
    annotation = alias(Annotation)
    return selection.join(
        annotation, annotation.c.datum_id == Datum.id, isouter=True
    ).join(Prediction, Prediction.annotation_id == annotation.c.id)


def _join_datum_to_prediction(selection: Select):
    """TODO"""
    annotation = alias(Annotation)
    return selection.join(
        annotation, annotation.c.id == Prediction.annotation_id, isouter=True
    ).join(Datum, Datum.id == annotation.c.datum_id)


def _join_groundtruth_to_datum(selection: Select):
    """TODO"""
    annotation = alias(Annotation)
    return selection.join(
        annotation, annotation.c.datum_id == Datum.id, isouter=True
    ).join(GroundTruth, GroundTruth.annotation_id == annotation.c.id)


def _join_datum_to_groundtruth(selection: Select):
    """TODO"""
    annotation = alias(Annotation)
    return selection.join(
        annotation, annotation.c.id == GroundTruth.annotation_id, isouter=True
    ).join(Datum, Datum.id == annotation.c.datum_id)


map_label_source_to_neighbor_mapping = {
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


map_label_source_to_neighbor_joins = {
    Annotation: {
        Dataset: {
            Datum: lambda x: x.join(Datum, Datum.dataset_id == Dataset.id)
        },
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
    },
    GroundTruth: {
        Dataset: {
            Datum: lambda x: x.join(Datum, Datum.dataset_id == Dataset.id)
        },
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
            Prediction: _join_prediction_to_datum,
        },
        Annotation: {
            Datum: lambda x: x.join(Datum, Datum.id == Annotation.datum_id),
            Model: lambda x: x.join(Model, Model.id == Annotation.model_id),
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
    },
    Prediction: {
        Dataset: {
            Datum: lambda x: x.join(Datum, Datum.dataset_id == Dataset.id)
        },
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
    },
}


def _recursive_search(
    table: TableTypeAlias,
    target: TableTypeAlias,
    mapping: dict[TableTypeAlias, set[TableTypeAlias]],
    cache: list[TableTypeAlias] | None = None,
) -> list[TableTypeAlias]:
    """TODO"""
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
):
    """TODO"""
    if label_source is LabelSourceAlias:
        raise ValueError

    table_mapping = map_label_source_to_neighbor_mapping[label_source]
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
    filter_: Filter | None = None,
) -> Select[Any] | Query[Any]:
    """TODO"""
    if label_source not in {Annotation, GroundTruth, Prediction}:
        raise ValueError(f"Invalid label source '{label_source}'.")

    arg_tables = map_arguments_to_tables(*args)
    filter_tables = map_filter_to_tables(filter_, label_source)

    tables = arg_tables.union(filter_tables)
    tables.discard(select_from)
    ordered_joins = _solve_graph(
        select_from=select_from, label_source=label_source, tables=tables
    )
    query = select_statement.select_from(select_from)
    for join in ordered_joins:
        query = join(query)
    return query


def generate_filter_query(
    conditions: FunctionType,
    select_from: TableTypeAlias,
    label_source: LabelSourceAlias,
    prefix: str,
) -> Subquery[Any]:
    """TODO"""
    tree, ordered_ctes, ordered_tables = generate_dependencies(conditions)

    if tree is None:
        raise ValueError

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
        select_from.id.label("id"),  # type: ignore - sqlalchemy issue
        *[
            case((expr, 1), else_=0).label(f"{prefix}{idx}")  # type: ignore - sqlalchemy issue
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
    filter_: Filter,
    label_source: LabelSourceAlias,
) -> list[tuple[Subquery[Any], TableTypeAlias]]:
    """TODO"""

    def _generator(
        conditions: FunctionType,
        select_from: TableTypeAlias,
        label_source: LabelSourceAlias,
        prefix: str,
    ):
        return generate_filter_query(
            conditions=conditions,
            select_from=select_from,
            label_source=label_source,
            prefix=f"{prefix}_cte",
        )

    queries = list()
    if filter_.datasets:
        conditions = filter_.datasets
        select_from = Dataset
        prefix = "ds"
        queries.append(
            (
                _generator(conditions, select_from, label_source, prefix),
                select_from,
            )
        )
    if filter_.models:
        conditions = filter_.models
        select_from = Model
        prefix = "md"
        queries.append(
            (
                _generator(conditions, select_from, label_source, prefix),
                select_from,
            )
        )
    if filter_.datums:
        conditions = filter_.datums
        select_from = Datum
        prefix = "dt"
        queries.append(
            (
                _generator(conditions, select_from, label_source, prefix),
                select_from,
            )
        )
    if filter_.annotations:
        conditions = filter_.annotations
        select_from = Annotation
        prefix = "an"
        queries.append(
            (
                _generator(conditions, select_from, label_source, prefix),
                select_from,
            )
        )
    if filter_.groundtruths:
        conditions = filter_.groundtruths
        select_from = GroundTruth if label_source is not Prediction else Datum
        prefix = "gt"
        queries.append(
            (
                _generator(conditions, select_from, GroundTruth, prefix),
                select_from,
            )
        )
    if filter_.predictions:
        conditions = filter_.predictions
        select_from = Prediction if label_source is not GroundTruth else Datum
        prefix = "pd"
        queries.append(
            (
                _generator(conditions, select_from, Prediction, prefix),
                select_from,
            )
        )
    if filter_.labels:
        conditions = filter_.labels
        select_from = Label
        prefix = "lb"
        queries.append(
            (
                _generator(conditions, select_from, label_source, prefix),
                select_from,
            )
        )
    if filter_.embeddings:
        conditions = filter_.embeddings
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
    filter_: Filter | None,
    label_source: LabelSourceAlias,
) -> Select[Any] | Query[Any]:
    """TODO"""

    select_from = map_arguments_to_tables(args[0]).pop()
    if select_from is Label:
        select_from = label_source
    query = generate_query(
        select_statement=stmt,
        args=args,
        select_from=select_from,
        label_source=label_source,
        filter_=filter_,
    )
    if filter_ is not None:
        filter_subqueries = generate_filter_queries(
            filter_=filter_, label_source=label_source
        )
        for subquery, selected_from in filter_subqueries:
            query = query.join(subquery, subquery.c.id == selected_from.id)
    return query
