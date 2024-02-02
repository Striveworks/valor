from dataclasses import dataclass
from typing import Dict, Iterable, Iterator, List, Optional, Sequence, Union

from velour.enums import TaskType
from velour.schemas.constraints import BinaryExpression, Constraint

FilterExpressionsType = Sequence[
    Union[BinaryExpression, Sequence[BinaryExpression]]
]


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
    dataset_geospatial : List[Constraint], optional
        A list of `Dataset` geospatial filters to filter on.
    model_names : List[str], optional
        A list of `Model` names to filter on.
    model_metadata : Dict[str, List[Constraint]], optional
        A dictionary of `Model` metadata to filter on.
    model_geospatial : List[Constraint], optional
        A list of `Model` geospatial filters to filter on.
    datum_uids : List[str], optional
        A list of `Datum` UIDs to filter on.
    datum_metadata : Dict[str, List[Constraint]], optional
        A dictionary of `Datum` metadata to filter on.
    datum_geospatial : List[Constraint], optional
        A list of `Datum` geospatial filters to filter on.
    task_types : List[TaskType], optional
        A list of task types to filter on.
    annotation_metadata : Dict[str, List[Constraint]], optional
        A dictionary of `Annotation` metadata to filter on.
    bounding_box : bool, optional
        A toggle for filtering by bounding boxes.
    bounding_box_area : bool, optional
        A optional constraint to filter by bounding box area.
    polygon : bool, optional
        A toggle for filtering by polygons.
    polygon_area : bool, optional
        A optional constraint to filter by polygon area.
    multipolygon : bool, optional
        A toggle for filtering by multipolygons.
    multipolygon_area : bool, optional
        A optional constraint to filter by multipolygon area.
    raster : bool, optional
        A toggle for filtering by rasters.
    raster_area : bool, optional
        A optional constraint to filter by raster area.
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
    dataset_geospatial: Optional[List[Constraint]] = None

    # models
    model_names: Optional[List[str]] = None
    model_metadata: Optional[Dict[str, List[Constraint]]] = None
    model_geospatial: Optional[List[Constraint]] = None

    # datums
    datum_uids: Optional[List[str]] = None
    datum_metadata: Optional[Dict[str, List[Constraint]]] = None
    datum_geospatial: Optional[List[Constraint]] = None

    # annotations
    task_types: Optional[List[TaskType]] = None
    annotation_metadata: Optional[Dict[str, List[Constraint]]] = None
    annotation_geospatial: Optional[List[Constraint]] = None

    # geometries
    bounding_box: Optional[bool] = None
    bounding_box_area: Optional[List[Constraint]] = None
    polygon: Optional[bool] = None
    polygon_area: Optional[List[Constraint]] = None
    multipolygon: Optional[bool] = None
    multipolygon_area: Optional[List[Constraint]] = None
    raster: Optional[bool] = None
    raster_area: Optional[List[Constraint]] = None

    # labels
    labels: Optional[List[Dict[str, str]]] = None
    label_ids: Optional[List[int]] = None
    label_keys: Optional[List[str]] = None
    label_scores: Optional[List[Constraint]] = None

    @classmethod
    def create(cls, expressions: FilterExpressionsType):
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

        # export full constraints
        for attr in [
            "dataset_geospatial",
            "model_geospatial",
            "datum_geospatial",
            "annotation_geospatial",
            "bounding_box_area",
            "polygon_area",
            "multipolygon_area",
            "raster_area",
            "label_scores",
        ]:
            if attr in expression_dict:
                setattr(
                    filter_request,
                    attr,
                    [expr.constraint for expr in expression_dict[attr]],
                )

        # export list of equality constraints
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

        # export metadata constraints
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

        # bool cases
        for attr in [
            "bounding_box",
            "polygon",
            "multipolygon",
            "raster",
        ]:
            if attr in expression_dict:
                for expr in expression_dict[attr]:
                    if expr.constraint.operator == "exists":
                        setattr(filter_request, attr, True)
                    elif expr.constraint.operator == "is_none":
                        setattr(filter_request, attr, False)

        return filter_request
