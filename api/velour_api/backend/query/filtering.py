import operator
from typing import Callable

from sqlalchemy import TIMESTAMP, Boolean, Float, and_, cast, func, not_, or_
from sqlalchemy.dialects.postgresql import INTERVAL
from sqlalchemy.orm.decl_api import DeclarativeMeta
from sqlalchemy.sql.elements import BinaryExpression, ColumnElement

from velour_api import enums
from velour_api.backend.models import (
    Annotation,
    Dataset,
    Datum,
    Label,
    Model,
    Prediction,
)
from velour_api.schemas import (
    BooleanFilter,
    DateTimeFilter,
    Duration,
    Filter,
    GeospatialFilter,
    NumericFilter,
    StringFilter,
    Time,
)


def _get_numeric_op(opstr) -> Callable:
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


def _get_boolean_op(opstr) -> Callable:
    ops = {"==": operator.eq, "!=": operator.ne}
    if opstr not in ops:
        raise ValueError(f"invalid boolean comparison operator `{opstr}`")
    return ops[opstr]


def _get_string_op(opstr) -> Callable:
    ops = {
        "==": operator.eq,
        "!=": operator.ne,
    }
    if opstr not in ops:
        raise ValueError(f"invalid string comparison operator `{opstr}`")
    return ops[opstr]


def _get_spatial_op(opstr) -> Callable:
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
    value_filter: NumericFilter
    | StringFilter
    | BooleanFilter
    | DateTimeFilter
    | GeospatialFilter,
    table: DeclarativeMeta,
) -> BinaryExpression:
    if isinstance(value_filter, NumericFilter):
        op = _get_numeric_op(value_filter.operator)
        lhs = table.meta[key].astext.cast(Float)  # type: ignore - SQLAlchemy type issue
        rhs = value_filter.value
    elif isinstance(value_filter, StringFilter):
        op = _get_string_op(value_filter.operator)
        lhs = table.meta[key].astext  # type: ignore - SQLAlchemy type issue
        rhs = value_filter.value
    elif isinstance(value_filter, BooleanFilter):
        op = _get_boolean_op(value_filter.operator)
        lhs = table.meta[key].astext.cast(Boolean)  # type: ignore - SQLAlchemy type issue
        rhs = value_filter.value
    elif isinstance(value_filter, DateTimeFilter):
        if isinstance(value_filter.value, Time) or isinstance(
            value_filter.value, Duration
        ):
            cast_type = INTERVAL
        else:
            cast_type = TIMESTAMP(timezone=True)
        op = _get_numeric_op(value_filter.operator)
        lhs = cast(
            table.meta[key][value_filter.value.key].astext,  # type: ignore - SQLAlchemy type issue
            cast_type,  # type: ignore - SQLAlchemy type issue
        )
        rhs = cast(
            value_filter.value.value,
            cast_type,  # type: ignore - SQLAlchemy type issue
        )
    elif isinstance(value_filter, GeospatialFilter):
        op = _get_spatial_op(value_filter.operator)
        lhs = func.ST_GeomFromGeoJSON(table.meta[key]["geojson"])  # type: ignore - SQLAlchemy type issue
        rhs = func.ST_GeomFromGeoJSON(value_filter.value.model_dump_json())
    else:
        raise NotImplementedError(
            f"metadatum value of type `{type(value_filter.value)}` is currently not supported"
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
    table: DeclarativeMeta,
) -> list[BinaryExpression]:
    expressions = [
        _filter_by_metadatum(key, value, table)
        for key, f_list in metadata.items()
        for value in f_list
    ]
    if len(expressions) > 1:
        expressions = [and_(*expressions)]
    return expressions  # type: ignore - SQLAlchemy type issue


def filter_by_dataset(
    filter_: Filter,
) -> list[ColumnElement[bool] | BinaryExpression]:
    expressions = []
    if filter_.dataset_names:
        expressions.append(
            [
                Dataset.name == name
                for name in filter_.dataset_names
                if isinstance(name, str)
            ]  # type: ignore - SQLAlchemy type issue
        )
    if filter_.dataset_metadata:
        expressions.append(
            _filter_by_metadata(filter_.dataset_metadata, Dataset),
        )
    return _flatten_expressions(expressions)


def filter_by_model(
    filter_: Filter,
) -> list[ColumnElement[bool] | BinaryExpression]:
    expressions = []
    if filter_.model_names:
        expressions.append(
            [
                Model.name == name
                for name in filter_.model_names
                if isinstance(name, str)
            ],  # type: ignore - SQLAlchemy type issue
        )
    if filter_.model_metadata:
        expressions.append(
            _filter_by_metadata(filter_.model_metadata, Model),
        )
    return _flatten_expressions(expressions)


def filter_by_datum(
    filter_: Filter,
) -> list[ColumnElement[bool] | BinaryExpression]:
    expressions = []
    if filter_.datum_uids:
        expressions.append(
            [
                Datum.uid == uid
                for uid in filter_.datum_uids
                if isinstance(uid, str)
            ],  # type: ignore - SQLAlchemy type issue
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
    expressions = []
    if filter_.task_types:
        expressions.append(
            [
                Annotation.task_type == task_type.value
                for task_type in filter_.task_types
                if isinstance(task_type, enums.TaskType)
            ],  # type: ignore - SQLAlchemy type issue
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
    if filter_.require_multipolygon is not None:
        if filter_.require_multipolygon:
            expressions.append([Annotation.multipolygon.isnot(None)])
        else:
            expressions.append([Annotation.multipolygon.is_(None)])
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
    if filter_.multipolygon_area:
        multipolygon_area_expr = []
        for area_filter in filter_.multipolygon_area:
            op = _get_numeric_op(area_filter.operator)
            multipolygon_area_expr.append(
                op(
                    func.ST_Area(Annotation.multipolygon),
                    area_filter.value,
                )
            )
        if len(multipolygon_area_expr) > 1:
            area_expr.append(and_(*multipolygon_area_expr))
        else:
            area_expr.append(multipolygon_area_expr[0])
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
    expressions = []
    if filter_.label_ids:
        expressions.append(
            [
                Label.id == id
                for id in filter_.label_ids
                if isinstance(id, int)
            ],  # type: ignore
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
            ],  # type: ignore
        )
    if filter_.label_keys:
        expressions.append(
            [
                Label.key == key
                for key in filter_.label_keys
                if isinstance(key, str)
            ],  # type: ignore
        )

    return _flatten_expressions(expressions)


def filter_by_prediction(
    filter_: Filter,
) -> list[ColumnElement[bool] | BinaryExpression]:
    expressions = []
    if filter_.label_scores:
        for score_filter in filter_.label_scores:
            op = _get_numeric_op(score_filter.operator)
            expressions.append(
                [op(Prediction.score, score_filter.value)],
            )
    return _flatten_expressions(expressions)
