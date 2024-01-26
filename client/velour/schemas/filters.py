from dataclasses import dataclass
from typing import Dict, List

from velour.enums import AnnotationType, TaskType
from velour.schemas.constraints import (
    Constraint,
    BinaryExpression,
)


@dataclass
class Filter:
    """
    Used to filter Evaluations according to specific, user-defined criteria.

    Attributes
    ----------
    dataset_names: List[str]
        A list of `Dataset` names to filter on.
    dataset_metadata: Dict[str, List[Constraint]]
        A dictionary of `Dataset` metadata to filter on.
    dataset_geospatial: List[Constraint].
        A list of `Dataset` geospatial filters to filter on.
    model_names: List[str]
        A list of `Model` names to filter on.
    model_metadata: Dict[str, List[Constraint]]
        A dictionary of `Model` metadata to filter on.
    model_geospatial: List[Constraint]
        A list of `Model` geospatial filters to filter on.
    datum_uids: List[str]
        A list of `Datum` UIDs to filter on.
    datum_metadata: Dict[str, List[Constraint]] = None
        A dictionary of `Datum` metadata to filter on.
    datum_geospatial: List[Constraint]
        A list of `Datum` geospatial filters to filter on.
    task_types: List[TaskType]
        A list of task types to filter on.
    annotation_types: List[AnnotationType]
        A list of `Annotation` types to filter on.
    annotation_geometric_area: List[Constraint]
        A list of `Constraints` which are used to filter `Evaluations` according to the `Annotation`'s geometric area.
    annotation_metadata: Dict[str, List[Constraint]]
        A dictionary of `Annotation` metadata to filter on.
    annotation_geospatial: List[Constraint]
        A list of `Annotation` geospatial filters to filter on.
    prediction_scores: List[Constraint]
        A list of `Constraints` which are used to filter `Evaluations` according to the `Model`'s prediction scores.
    labels: List[Label]
        A list of `Labels' to filter on.
    label_ids: List[int]
        A list of `Label` IDs to filter on.
    label_keys: List[str] = None
        A list of `Label` keys to filter on.


    Raises
    ------
    TypeError
        If `value` isn't of the correct type.
    ValueError
        If the `operator` doesn't match one of the allowed patterns.
    """

    # datasets
    dataset_names: List[str] = None
    dataset_metadata: Dict[str, List[Constraint]] = None
    dataset_geospatial: List[Constraint] = None

    # models
    model_names: List[str] = None
    model_metadata: Dict[str, List[Constraint]] = None
    model_geospatial: List[Constraint] = None

    # datums
    datum_uids: List[str] = None
    datum_metadata: Dict[str, List[Constraint]] = None
    datum_geospatial: List[Constraint] = None

    # annotations
    task_types: List[TaskType] = None
    annotation_types: List[AnnotationType] = None
    annotation_geometric_area: List[Constraint] = None
    annotation_metadata: Dict[str, List[Constraint]] = None
    annotation_geospatial: List[Constraint] = None

    # predictions
    prediction_scores: List[Constraint] = None

    # labels
    labels: List[Dict[str, str]] = None
    label_ids: List[int] = None
    label_keys: List[str] = None

    @classmethod
    def create(cls, expressions: List[BinaryExpression]):
        """
        Parses a list of `BinaryExpression` to create a `schemas.Filter` object.

        Parameters
        ----------
        expressions: List[BinaryExpression]
            A list of `BinaryExpressions' to parse into a `Filter` object.
        """

        # expand nested expressions
        expression_list = [
            expr for expr in expressions if isinstance(expr, BinaryExpression)
        ] + [
            expr_
            for expr in expressions
            if isinstance(expr, list)
            for expr_ in expr
            if isinstance(expr_, BinaryExpression)
        ]

        # create dict using expr names as keys
        expression_dict = {}
        for expr in expression_list:
            if expr.name not in expression_dict:
                expression_dict[expr.name] = []
            expression_dict[expr.name].append(expr)

        # create filter
        filter_request = cls()

        # export full constraints
        for attr in [
            "annotation_geometric_area",
            "prediction_scores",
            "dataset_geospatial",
            "model_geospatial",
            "datum_geospatial",
            "annotation_geospatial",
        ]:
            if attr in expression_dict:
                setattr(
                    filter_request, 
                    attr, 
                    [
                        expr.constraint     
                        for expr in expression_dict[attr]
                    ],
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
                    [
                        expr.constraint.value
                        for expr in expression_dict[attr]
                    ]
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

        # edge cases
        for attr, atype in [
            ("annotation_bounding_box", AnnotationType.BOX),
            ("annotation_polygon", AnnotationType.POLYGON),
            ("annotation_multipolygon", AnnotationType.MULTIPOLYGON),
            ("annotation_raster", AnnotationType.RASTER),
        ]:
            if attr in expression_dict:
                for expr in expression_dict[attr]:
                    if expr.constraint.operator == "exists":
                        if not filter_request.annotation_types:
                            filter_request.annotation_types = []
                        filter_request.annotation_types.append(atype)

        for attr in [
            "annotation_bounding_box_area",
            "annotation_polygon_area",
            "annotation_multipolygon_area",
            "annotation_raster_area",
        ]:
            if attr in expression_dict:
                setattr(
                    filter_request,
                    "annotation_geometric_area",
                    [
                        expr.constraint
                        for expr in expression_dict[attr]
                    ]
                )

        return filter_request
