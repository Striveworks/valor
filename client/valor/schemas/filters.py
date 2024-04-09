from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Union

from valor.enums import TaskType
from valor.schemas.symbolic.operators import (
    And,
    AppendableFunction,
    Function,
    Inside,
    Intersects,
    IsNotNull,
    IsNull,
    Negate,
    OneArgumentFunction,
    Or,
    Outside,
    TwoArgumentFunction,
    Xor,
)
from valor.schemas.symbolic.types import (
    Date,
    DateTime,
    Duration,
    LineString,
    MultiLineString,
    MultiPoint,
    MultiPolygon,
    Point,
    Polygon,
    Time,
    Variable,
)


@dataclass
class Constraint:
    """
    Represents a constraint with a value and an operator.

    Attributes:
        value : Any
            The value associated with the constraint.
        operator : str
            The operator used to define the constraint.
    """

    value: Any
    operator: str


def _convert_symbol_to_attribute_name(symbol_name):
    map_sym_to_attr = {
        "dataset.name": "dataset_names",
        "dataset.metadata": "dataset_metadata",
        "model.name": "model_names",
        "model.metadata": "model_metadata",
        "datum.uid": "datum_uids",
        "datum.metadata": "datum_metadata",
        "annotation.task_type": "task_types",
        "annotation.metadata": "annotation_metadata",
        "annotation.bounding_box": "require_bounding_box",
        "annotation.bounding_box.area": "bounding_box_area",
        "annotation.polygon": "require_polygon",
        "annotation.polygon.area": "polygon_area",
        "annotation.raster": "require_raster",
        "annotation.raster.area": "raster_area",
        "annotation.labels": "labels",
        "label.id": "label_ids",
        "label.key": "label_keys",
        "label.score": "label_scores",
    }
    return map_sym_to_attr[symbol_name]


def _convert_expression_to_constraint(expr: Function):
    # extract value
    if isinstance(expr, TwoArgumentFunction):
        variable = expr.rhs
        if isinstance(
            variable,
            (
                Point,
                MultiPoint,
                LineString,
                MultiLineString,
                Polygon,
                MultiPolygon,
            ),
        ):
            value = {
                "type": type(variable).__name__,
                "coordinates": variable.get_value(),
            }
        elif isinstance(variable, (DateTime, Date, Time, Duration)):
            value = {type(variable).__name__.lower(): variable.encode_value()}
        else:
            value = variable.encode_value()
    else:
        value = None

    # extract operator
    if hasattr(expr, "_operator") and expr._operator is not None:
        op = expr._operator
    elif isinstance(expr, Inside):
        op = "inside"
    elif isinstance(expr, Intersects):
        op = "intersect"
    elif isinstance(expr, Outside):
        op = "outside"
    elif isinstance(expr, IsNotNull):
        op = "exists"
    elif isinstance(expr, IsNull):
        op = "is_none"
    else:
        raise NotImplementedError(
            f"Function '{type(expr)}' has not been implemented by the API."
        )

    return Constraint(value=value, operator=op)


def _scan_one_arg_function(fn: OneArgumentFunction):
    if not fn.arg.is_symbolic:
        raise ValueError(
            "Single argument functions should take a symbol as input."
        )


def _scan_two_arg_function(fn: TwoArgumentFunction):
    if not isinstance(fn.lhs, Variable) or not isinstance(fn.rhs, Variable):
        raise ValueError("Nested arguments are currently unsupported.")
    elif not fn.lhs.is_symbolic:
        raise ValueError(
            f"Values on the lhs of an operator are currently unsupported. {fn.lhs}"
        )
    elif not fn.rhs.is_value:
        raise ValueError(
            f"Symbols on the rhs of an operator are currently unsupported. {fn.rhs}"
        )


def _scan_appendable_function(fn: AppendableFunction):
    if not isinstance(fn, (And, Or)):
        raise ValueError(
            f"Operation '{type(fn)}' is currently unsupported by the API."
        )

    symbols = set()
    for arg in fn._args:
        if not isinstance(fn, Function):
            raise ValueError(
                f"Expected a function but received value with type '{type(fn)}'"
            )

        # scan for nested logic
        if isinstance(arg, (Or, And, Xor, Negate)):
            raise NotImplementedError

        # scan for symbol/value positioning
        if isinstance(arg, OneArgumentFunction):
            _scan_one_arg_function(arg)
        elif isinstance(arg, TwoArgumentFunction):
            _scan_two_arg_function(arg)

        symbols.add(arg._args[0].get_symbol())

    # check that only one symbol was defined per statement
    if len(symbols) > 1:
        raise ValueError(
            f"Defining more than one variable per statement is currently unsupported. {symbols}"
        )
    symbol = list(symbols)[0]

    # check that symbol is compatible with the logical operation
    if isinstance(fn, And) and not (
        symbol._name in Filter._supports_and()
        or symbol._attribute in Filter._supports_and()
    ):
        raise ValueError(
            f"Symbol '{str(symbol)}' currently does not support the 'AND' operation."
        )
    elif isinstance(fn, Or) and not (
        symbol._name in Filter._supports_or()
        or symbol._attribute in Filter._supports_or()
    ):
        raise ValueError(
            f"Symbol '{str(symbol)}' currently does not support the 'AND' operation."
        )


def _parse_listed_expressions(flist):
    expressions = {}
    for row in flist:
        if not isinstance(row, Function):
            raise ValueError(
                f"Expected a function but received value with type '{type(row)}'"
            )
        elif isinstance(row, AppendableFunction):
            _scan_appendable_function(row)
            symbol = row._args[0]._args[0].get_symbol()
            constraints = [
                _convert_expression_to_constraint(arg) for arg in row._args
            ]
        elif isinstance(row, TwoArgumentFunction):
            _scan_two_arg_function(row)
            symbol = row.lhs.get_symbol()
            constraints = [_convert_expression_to_constraint(row)]
        elif isinstance(row, OneArgumentFunction):
            _scan_one_arg_function(row)
            symbol = row.arg.get_symbol()
            constraints = [_convert_expression_to_constraint(row)]
        else:
            raise NotImplementedError

        symbol_name = f"{symbol._owner}.{symbol._name}"
        if symbol._attribute:
            symbol_name += f".{symbol._attribute}"
        attribute_name = _convert_symbol_to_attribute_name(symbol_name)
        if symbol._key:
            if attribute_name not in expressions:
                expressions[attribute_name] = dict()
            if symbol._key not in expressions[attribute_name]:
                expressions[attribute_name][symbol._key] = list()
            expressions[attribute_name][symbol._key] += constraints
        else:
            if attribute_name not in expressions:
                expressions[attribute_name] = list()
            expressions[attribute_name] += constraints

    return expressions


@dataclass
class Filter:
    """
    Used to filter Evaluations according to specific, user-defined criteria.

    Attributes
    ----------
    dataset_names : List[str], optional
        A list of `Dataset` names to filter on.
    dataset_metadata : Dict[str, List[Constraint]], optional
        A dictionary of `Dataset` metadata to filter on.
    model_names : List[str], optional
        A list of `Model` names to filter on.
    model_metadata : Dict[str, List[Constraint]], optional
        A dictionary of `Model` metadata to filter on.
    datum_uids : List[str], optional
        A list of `Datum` UIDs to filter on.
    datum_metadata : Dict[str, List[Constraint]], optional
        A dictionary of `Datum` metadata to filter on.
    task_types : List[TaskType], optional
        A list of task types to filter on.
    annotation_metadata : Dict[str, List[Constraint]], optional
        A dictionary of `Annotation` metadata to filter on.
    require_box : bool, optional
        A toggle for filtering by bounding boxes.
    box_area : bool, optional
        An optional constraint to filter by bounding box area.
    require_polygon : bool, optional
        A toggle for filtering by polygons.
    polygon_area : bool, optional
        An optional constraint to filter by polygon area.
    require_raster : bool, optional
        A toggle for filtering by rasters.
    raster_area : bool, optional
        An optional constraint to filter by raster area.
    labels : List[Label], optional
        A list of `Labels' to filter on.
    label_ids : List[int], optional
        A list of label row id's.
    label_keys : List[str], optional
        A list of `Label` keys to filter on.
    label_scores : List[Constraint], optional
        A list of `Constraints` which are used to filter `Evaluations` according to the `Model`'s prediction scores.

    Raises
    ------
    TypeError
        If `value` isn't of the correct type.
    ValueError
        If the `operator` doesn't match one of the allowed patterns.
    """

    # datasets
    dataset_names: Optional[List[str]] = None
    dataset_metadata: Optional[Dict[str, List[Constraint]]] = None

    # models
    model_names: Optional[List[str]] = None
    model_metadata: Optional[Dict[str, List[Constraint]]] = None

    # datums
    datum_uids: Optional[List[str]] = None
    datum_metadata: Optional[Dict[str, List[Constraint]]] = None

    # annotations
    task_types: Optional[List[TaskType]] = None
    annotation_metadata: Optional[Dict[str, List[Constraint]]] = None

    # geometries
    require_box: Optional[bool] = None
    box_area: Optional[List[Constraint]] = None
    require_polygon: Optional[bool] = None
    polygon_area: Optional[List[Constraint]] = None
    require_raster: Optional[bool] = None
    raster_area: Optional[List[Constraint]] = None

    # labels
    labels: Optional[List[Dict[str, str]]] = None
    label_ids: Optional[List[int]] = None
    label_keys: Optional[List[str]] = None
    label_scores: Optional[List[Constraint]] = None

    @staticmethod
    def _supports_and():
        return {
            "area",
            "score",
            "metadata",
        }

    @staticmethod
    def _supports_or():
        return {
            "name",
            "uid",
            "task_type",
            "labels",
            "keys",
        }

    def __post_init__(self):
        def _unpack_metadata(metadata: Optional[dict]) -> Union[dict, None]:
            if metadata is None:
                return None
            for k, vlist in metadata.items():
                metadata[k] = [
                    v if isinstance(v, Constraint) else Constraint(**v)
                    for v in vlist
                ]
            return metadata

        # unpack metadata
        self.dataset_metadata = _unpack_metadata(self.dataset_metadata)
        self.model_metadata = _unpack_metadata(self.model_metadata)
        self.datum_metadata = _unpack_metadata(self.datum_metadata)
        self.annotation_metadata = _unpack_metadata(self.annotation_metadata)

        def _unpack_list(
            vlist: Optional[list], object_type: type
        ) -> Optional[list]:
            def _handle_conversion(v, object_type):
                if object_type is Constraint:
                    return object_type(**v)
                else:
                    return object_type(v)

            if vlist is None:
                return None

            return [
                (
                    v
                    if isinstance(v, object_type)
                    else _handle_conversion(v=v, object_type=object_type)
                )
                for v in vlist
            ]

        # unpack tasktypes
        self.task_types = _unpack_list(self.task_types, TaskType)

        # unpack area
        self.box_area = _unpack_list(self.box_area, Constraint)
        self.polygon_area = _unpack_list(self.polygon_area, Constraint)
        self.raster_area = _unpack_list(self.raster_area, Constraint)

        # scores
        self.label_scores = _unpack_list(self.label_scores, Constraint)

    @classmethod
    def create(cls, expressions: List[Any]):
        """
        Parses a list of `BinaryExpression` to create a `schemas.Filter` object.

        Parameters
        ----------
        expressions: Sequence[Union[BinaryExpression, Sequence[BinaryExpression]]]
            A list of (lists of) `BinaryExpressions' to parse into a `Filter` object.
        """

        constraints = _parse_listed_expressions(expressions)

        # create filter
        filter_request = cls()

        # metadata constraints
        for attr in [
            "dataset_metadata",
            "model_metadata",
            "datum_metadata",
            "annotation_metadata",
            "box_area",
            "polygon_area",
            "raster_area",
            "label_scores",
        ]:
            if attr in constraints:
                setattr(filter_request, attr, constraints[attr])

        # boolean constraints
        for attr in [
            "require_box",
            "require_polygon",
            "require_raster",
        ]:
            if attr in constraints:
                for constraint in constraints[attr]:
                    if constraint.operator == "exists":
                        setattr(filter_request, attr, True)
                    elif constraint.operator == "is_none":
                        setattr(filter_request, attr, False)

        # equality constraints
        for attr in [
            "dataset_names",
            "model_names",
            "datum_uids",
            "task_types",
            "label_keys",
        ]:
            if attr in constraints:
                setattr(
                    filter_request,
                    attr,
                    [expr.value for expr in constraints[attr]],
                )

        # edge case - label list
        if "labels" in constraints:
            setattr(
                filter_request,
                "labels",
                [
                    {label["key"]: label["value"]}
                    for labels in constraints["labels"]
                    for label in labels.value
                ],
            )

        return filter_request
