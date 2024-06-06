import operator
from typing import Callable

from sqlalchemy import TIMESTAMP, Boolean, Float, and_, cast, func, not_, or_
from sqlalchemy.dialects.postgresql import INTERVAL, TEXT
from sqlalchemy.sql.elements import BinaryExpression, ColumnElement

from valor_api.backend.models import (
    Annotation,
    Dataset,
    Datum,
    Label,
    Model,
    Prediction,
)
from valor_api.backend.query.types import TableTypeAlias
from valor_api.enums import TaskType
from valor_api.schemas import (
    BooleanFilter,
    DateTimeFilter,
    Duration,
    Filter,
    GeospatialFilter,
    NumericFilter,
    StringFilter,
    Time,
)


def _get_boolean_op(opstr) -> Callable:
    """Returns function if operator is valid for boolean comparison."""
    ops = {
        "==": operator.eq,
        "!=": operator.ne,
    }
    if opstr not in ops:
        raise ValueError(f"invalid boolean comparison operator `{opstr}`")
    return ops[opstr]


def _get_string_op(opstr) -> Callable:
    """Returns function if operator is valid for string comparison."""
    ops = {
        "==": operator.eq,
        "!=": operator.ne,
    }
    if opstr not in ops:
        raise ValueError(f"invalid string comparison operator `{opstr}`")
    return ops[opstr]


def _get_numeric_op(opstr) -> Callable:
    """Returns function if operator is valid for numeric comparison."""
    ops = {
        ">": operator.gt,
        "<": operator.lt,
        ">=": operator.ge,
        "<=": operator.le,
        "==": operator.eq,
        "!=": operator.ne,
    }
    if opstr not in ops:
        raise ValueError(f"invalid numeric comparison operator `{opstr}`")
    return ops[opstr]


def _get_spatial_op(opstr) -> Callable:
    """Returns function if operator is valid for spatial comparison."""
    ops = {
        "intersect": lambda lhs, rhs: func.ST_Intersects(lhs, rhs),
        "inside": lambda lhs, rhs: func.ST_Covers(rhs, lhs),
        "outside": lambda lhs, rhs: not_(func.ST_Covers(rhs, lhs)),
    }
    if opstr not in ops:
        raise ValueError(f"invalid spatial operator `{opstr}`")
    return ops[opstr]


def _flatten_expressions(
    expressions: list[list[BinaryExpression]],
) -> list[ColumnElement[bool] | BinaryExpression]:
    """Flattens a nested list of expressions."""
    return [
        or_(*exprlist) if len(exprlist) > 1 else exprlist[0]
        for exprlist in expressions
    ]


def _filter_by_metadatum(
    key: str,
    value_filter: (
        NumericFilter
        | StringFilter
        | BooleanFilter
        | DateTimeFilter
        | GeospatialFilter
    ),
    table: TableTypeAlias,
) -> BinaryExpression:
    """
    Filter by metadatum.

    Supports all existing filter types.
    """
    if isinstance(value_filter, NumericFilter):
        op = _get_numeric_op(value_filter.operator)
        lhs = table.meta[key].astext.cast(Float)
        rhs = value_filter.value
    elif isinstance(value_filter, StringFilter):
        op = _get_string_op(value_filter.operator)
        lhs = table.meta[key].astext
        rhs = value_filter.value
    elif isinstance(value_filter, BooleanFilter):
        op = _get_boolean_op(value_filter.operator)
        lhs = table.meta[key].astext.cast(Boolean)
        rhs = value_filter.value
    elif isinstance(value_filter, DateTimeFilter):
        lhs_operand = table.meta[key]["value"].astext
        rhs_operand = value_filter.value.value
        if isinstance(value_filter.value, Time):
            lhs = cast(lhs_operand, INTERVAL)
            rhs = cast(rhs_operand, INTERVAL)
        elif isinstance(value_filter.value, Duration):
            lhs = cast(lhs_operand, INTERVAL)
            rhs = cast(cast(rhs_operand, TEXT), INTERVAL)
        else:
            lhs = cast(lhs_operand, TIMESTAMP(timezone=True))
            rhs = cast(rhs_operand, TIMESTAMP(timezone=True))
        op = _get_numeric_op(value_filter.operator)
    elif isinstance(value_filter, GeospatialFilter):
        op = _get_spatial_op(value_filter.operator)
        lhs = func.ST_GeomFromGeoJSON(table.meta[key]["value"])
        rhs = func.ST_GeomFromGeoJSON(value_filter.value.model_dump_json())
    else:
        raise NotImplementedError(
            f"Filter with type `{type(value_filter)}` is currently not supported"
        )
    return op(lhs, rhs)


def _filter_by_metadata(
    metadata: dict[
        str,
        list[
            NumericFilter
            | StringFilter
            | BooleanFilter
            | DateTimeFilter
            | GeospatialFilter
        ],
    ],
    table: TableTypeAlias,
) -> list[ColumnElement[bool]] | list[BinaryExpression]:
    """
    Iterates through a dictionary containing metadata.
    """
    expressions = [
        _filter_by_metadatum(key, value, table)
        for key, f_list in metadata.items()
        for value in f_list
    ]
    if len(expressions) > 1:
        expressions = [and_(*expressions)]
    return expressions


def filter_by_dataset(
    filter_: Filter,
) -> list[ColumnElement[bool] | BinaryExpression]:
    """
    Constructs sqlalchemy expressions using dataset filters.

    Parameters
    ----------
    filter_ : schemas.Filter
        The filter to apply.

    Returns
    -------
    list[ColumnElement[bool] | BinaryExpression]
        A list of expressions that can be used in a WHERE clause.
    """
    expressions = []
    if filter_.dataset_names:
        expressions.append(
            [
                Dataset.name == name
                for name in filter_.dataset_names
                if isinstance(name, str)
            ]
        )
    if filter_.dataset_metadata:
        expressions.append(
            _filter_by_metadata(filter_.dataset_metadata, Dataset),
        )
    return _flatten_expressions(expressions)


def filter_by_model(
    filter_: Filter,
) -> list[ColumnElement[bool] | BinaryExpression]:
    """
    Constructs sqlalchemy expressions using model filters.

    Parameters
    ----------
    filter_ : schemas.Filter
        The filter to apply.

    Returns
    -------
    list[ColumnElement[bool] | BinaryExpression]
        A list of expressions that can be used in a WHERE clause.
    """
    expressions = []
    if filter_.model_names:
        expressions.append(
            [
                Model.name == name
                for name in filter_.model_names
                if isinstance(name, str)
            ],
        )
    if filter_.model_metadata:
        expressions.append(
            _filter_by_metadata(filter_.model_metadata, Model),
        )
    return _flatten_expressions(expressions)


def filter_by_datum(
    filter_: Filter,
) -> list[ColumnElement[bool] | BinaryExpression]:
    """
    Constructs sqlalchemy expressions using datum filters.

    Parameters
    ----------
    filter_ : schemas.Filter
        The filter to apply.

    Returns
    -------
    list[ColumnElement[bool] | BinaryExpression]
        A list of expressions that can be used in a WHERE clause.
    """
    expressions = []
    if filter_.datum_uids:
        expressions.append(
            [
                Datum.uid == uid
                for uid in filter_.datum_uids
                if isinstance(uid, str)
            ],
        )
    if filter_.datum_metadata:
        expressions.append(
            _filter_by_metadata(
                metadata=filter_.datum_metadata,
                table=Datum,
            ),
        )
    return _flatten_expressions(expressions)


def filter_by_annotation(
    filter_: Filter,
) -> list[ColumnElement[bool] | BinaryExpression]:
    """
    Constructs sqlalchemy expressions using annotation filters.

    Parameters
    ----------
    filter_ : schemas.Filter
        The filter to apply.

    Returns
    -------
    list[ColumnElement[bool] | BinaryExpression]
        A list of expressions that can be used in a WHERE clause.
    """
    expressions = []
    if filter_.task_types:
        expressions.append(
            [
                Annotation.implied_task_types.op("?")(task_type.value)
                for task_type in filter_.task_types
                if isinstance(task_type, TaskType)
            ]
        )
    if filter_.annotation_metadata:
        expressions.append(
            _filter_by_metadata(filter_.annotation_metadata, Annotation),
        )

    # geometry requirement
    if filter_.require_bounding_box is not None:
        if filter_.require_bounding_box:
            expressions.append([Annotation.box.isnot(None)])
        else:
            expressions.append([Annotation.box.is_(None)])
    if filter_.require_polygon is not None:
        if filter_.require_polygon:
            expressions.append([Annotation.polygon.isnot(None)])
        else:
            expressions.append([Annotation.polygon.is_(None)])
    if filter_.require_raster is not None:
        if filter_.require_raster:
            expressions.append([Annotation.raster.isnot(None)])
        else:
            expressions.append([Annotation.raster.is_(None)])

    # geometric area - AND like-typed filter_, OR different-typed filter_
    area_expr = []
    if filter_.bounding_box_area:
        bounding_box_area_expr = []
        for area_filter in filter_.bounding_box_area:
            op = _get_numeric_op(area_filter.operator)
            bounding_box_area_expr.append(
                op(func.ST_Area(Annotation.box), area_filter.value)
            )
        if len(bounding_box_area_expr) > 1:
            area_expr.append(and_(*bounding_box_area_expr))
        else:
            area_expr.append(bounding_box_area_expr[0])
    if filter_.polygon_area:
        polygon_area_expr = []
        for area_filter in filter_.polygon_area:
            op = _get_numeric_op(area_filter.operator)
            polygon_area_expr.append(
                op(
                    func.ST_Area(Annotation.polygon),
                    area_filter.value,
                )
            )
        if len(polygon_area_expr) > 1:
            area_expr.append(and_(*polygon_area_expr))
        else:
            area_expr.append(polygon_area_expr[0])
    if filter_.raster_area:
        raster_area_expr = []
        for area_filter in filter_.raster_area:
            op = _get_numeric_op(area_filter.operator)
            raster_area_expr.append(
                op(
                    func.ST_Count(Annotation.raster),
                    area_filter.value,
                )
            )
        if len(raster_area_expr) > 1:
            area_expr.append(and_(*raster_area_expr))
        else:
            area_expr.append(raster_area_expr[0])
    if area_expr:
        expressions.append(area_expr)

    return _flatten_expressions(expressions)


def filter_by_label(
    filter_: Filter,
) -> list[ColumnElement[bool] | BinaryExpression]:
    """
    Constructs sqlalchemy expressions using label filters.

    Parameters
    ----------
    filter_ : schemas.Filter
        The filter to apply.

    Returns
    -------
    list[ColumnElement[bool] | BinaryExpression]
        A list of expressions that can be used in a WHERE clause.
    """
    expressions = []
    if filter_.label_ids:
        expressions.append(
            [
                Label.id == id
                for id in filter_.label_ids
                if isinstance(id, int)
            ],
        )
    if filter_.labels:
        expressions.append(
            [
                and_(
                    Label.key == key,
                    Label.value == value,
                )
                for label in filter_.labels
                if (isinstance(label, dict) and len(label) == 1)
                for key, value in label.items()
            ],
        )
    if filter_.label_keys:
        expressions.append(
            [
                Label.key == key
                for key in filter_.label_keys
                if isinstance(key, str)
            ],
        )

    return _flatten_expressions(expressions)


def filter_by_prediction(
    filter_: Filter,
) -> list[ColumnElement[bool] | BinaryExpression]:
    """
    Constructs sqlalchemy expressions using prediction filters.

    Parameters
    ----------
    filter_ : schemas.Filter
        The filter to apply.

    Returns
    -------
    list[ColumnElement[bool] | BinaryExpression]
        A list of expressions that can be used in a WHERE clause.
    """
    expressions = []
    if filter_.label_scores:
        for score_filter in filter_.label_scores:
            op = _get_numeric_op(score_filter.operator)
            expressions.append(
                [op(Prediction.score, score_filter.value)],
            )
    return _flatten_expressions(expressions)
