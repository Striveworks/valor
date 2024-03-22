from dataclasses import dataclass
from typing import Any, Dict, Iterable, Iterator, List, Optional, Union

from valor.enums import TaskType


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


@dataclass
class BinaryExpression:
    """
    Stores a conditional relationship.

    Attributes
    ----------
    name : str
        The name of the filter property.
    constraint : valor.schemas.Constraint
        The operation that is performed.
    key : str, optional
        An optional key used for object retrieval.
    """

    name: str
    constraint: Constraint
    key: Union[str, None] = None


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
    require_bounding_box : bool, optional
        A toggle for filtering by bounding boxes.
    bounding_box_area : bool, optional
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
    require_bounding_box: Optional[bool] = None
    bounding_box_area: Optional[List[Constraint]] = None
    require_polygon: Optional[bool] = None
    polygon_area: Optional[List[Constraint]] = None
    require_raster: Optional[bool] = None
    raster_area: Optional[List[Constraint]] = None

    # labels
    labels: Optional[List[Dict[str, str]]] = None
    label_ids: Optional[List[int]] = None
    label_keys: Optional[List[str]] = None
    label_scores: Optional[List[Constraint]] = None

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
        self.bounding_box_area = _unpack_list(
            self.bounding_box_area, Constraint
        )
        self.polygon_area = _unpack_list(self.polygon_area, Constraint)
        self.raster_area = _unpack_list(self.raster_area, Constraint)

        # scores
        self.label_scores = _unpack_list(self.label_scores, Constraint)

    @classmethod
    def create(
        cls, expressions: List[Union[BinaryExpression, List[BinaryExpression]]]
    ):
        """
        Parses a list of `BinaryExpression` to create a `schemas.Filter` object.

        Parameters
        ----------
        expressions: Sequence[Union[BinaryExpression, Sequence[BinaryExpression]]]
            A list of (lists of) `BinaryExpressions' to parse into a `Filter` object.
        """

        def flatten(
            t: Iterable[Union[BinaryExpression, Iterable[BinaryExpression]]]
        ) -> Iterator[BinaryExpression]:
            """Flatten a nested iterable of BinaryExpressions."""
            for item in t:
                if isinstance(item, BinaryExpression):
                    yield item
                else:
                    yield from flatten(item)

        # create dict using expr names as keys
        expression_dict = {}
        for expr in flatten(expressions):
            if expr.name not in expression_dict:
                expression_dict[expr.name] = []
            expression_dict[expr.name].append(expr)

        # create filter
        filter_request = cls()

        # metadata constraints
        for attr in [
            "dataset_metadata",
            "model_metadata",
            "datum_metadata",
            "annotation_metadata",
        ]:
            if attr in expression_dict:
                for expr in expression_dict[attr]:
                    if not getattr(filter_request, attr):
                        setattr(filter_request, attr, {})
                    __value = getattr(filter_request, attr)
                    if expr.key not in __value:
                        __value[expr.key] = []
                    __value[expr.key].append(expr.constraint)
                    setattr(filter_request, attr, __value)

        # numeric constraints
        for attr in [
            "bounding_box_area",
            "polygon_area",
            "raster_area",
            "label_scores",
        ]:
            if attr in expression_dict:
                setattr(
                    filter_request,
                    attr,
                    [expr.constraint for expr in expression_dict[attr]],
                )

        # boolean constraints
        for attr in [
            "require_bounding_box",
            "require_polygon",
            "require_raster",
        ]:
            if attr in expression_dict:
                for expr in expression_dict[attr]:
                    if expr.constraint.operator == "exists":
                        setattr(filter_request, attr, True)
                    elif expr.constraint.operator == "is_none":
                        setattr(filter_request, attr, False)

        # equality constraints
        for attr in [
            "dataset_names",
            "model_names",
            "datum_uids",
            "task_types",
            "labels",
            "label_keys",
        ]:
            if attr in expression_dict:
                setattr(
                    filter_request,
                    attr,
                    [expr.constraint.value for expr in expression_dict[attr]],
                )

        return filter_request
