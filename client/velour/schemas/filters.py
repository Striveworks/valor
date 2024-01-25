import datetime
from dataclasses import dataclass
from enum import Enum
from typing import Dict, List, Union, Any, Optional

from velour.types import ValueType, GeoJSONType, GeometryType
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

        # datasets
        if "dataset_names" in expression_dict:
            filter_request.dataset_names = [
                expr.value for expr in expression_dict["dataset_names"]
            ]
        if "dataset_metadata" in expression_dict:
            for expr in expression_dict["dataset_metadata"]:
                if not filter_request.dataset_metadata:
                    filter_request.dataset_metadata = {}
                if expr.key not in filter_request.dataset_metadata:
                    filter_request.dataset_metadata[expr.key] = []
                filter_request.dataset_metadata[expr.key].append(
                    Constraint(
                        value=expr.value,
                        operator=expr.operator,
                    )
                )
        if "dataset_geospatial" in expression_dict:
            filter_request.dataset_geospatial = [
                Constraint(
                    value=expr.value,
                    operator=expr.operator,
                )
                for expr in expression_dict["dataset_geospatial"]
            ]
        # models
        if "model_names" in expression_dict:
            filter_request.model_names = [
                expr.value for expr in expression_dict["model_names"]
            ]
        if "model_metadata" in expression_dict:
            for expr in expression_dict["model_metadata"]:
                if not filter_request.model_metadata:
                    filter_request.model_metadata = {}
                if expr.key not in filter_request.model_metadata:
                    filter_request.model_metadata[expr.key] = []
                filter_request.model_metadata[expr.key].append(
                    Constraint(
                        value=expr.value,
                        operator=expr.operator,
                    )
                )
        if "model_geospatial" in expression_dict:
            filter_request.model_geospatial = [
                Constraint(
                    value=expr.value,
                    operator=expr.operator,
                )
                for expr in expression_dict["model_geospatial"]
            ]
        # datums
        if "datum_uids" in expression_dict:
            filter_request.datum_uids = [
                expr.value for expr in expression_dict["datum_uids"]
            ]
        if "datum_metadata" in expression_dict:
            for expr in expression_dict["datum_metadata"]:
                if not filter_request.datum_metadata:
                    filter_request.datum_metadata = {}
                if expr.key not in filter_request.datum_metadata:
                    filter_request.datum_metadata[expr.key] = []
                filter_request.datum_metadata[expr.key].append(
                    Constraint(
                        value=expr.value,
                        operator=expr.operator,
                    )
                )
        if "datum_geospatial" in expression_dict:
            filter_request.datum_geospatial = [
                Constraint(
                    value=expr.value,
                    operator=expr.operator,
                )
                for expr in expression_dict["datum_geospatial"]
            ]

        # annotations
        if "task_types" in expression_dict:
            filter_request.task_types = [
                expr.value for expr in expression_dict["task_types"]
            ]
        if "annotation_types" in expression_dict:
            filter_request.annotation_types = [
                expr.value for expr in expression_dict["annotation_types"]
            ]
        if "annotation_geometric_area" in expression_dict:
            filter_request.annotation_geometric_area = [
                Constraint(
                    value=expr.value,
                    operator=expr.operator,
                )
                for expr in expression_dict["annotation_geometric_area"]
            ]
        if "annotation_metadata" in expression_dict:
            for expr in expression_dict["annotation_metadata"]:
                if not filter_request.annotation_metadata:
                    filter_request.annotation_metadata = {}
                if expr.key not in filter_request.annotation_metadata:
                    filter_request.annotation_metadata[expr.key] = []
                filter_request.annotation_metadata[expr.key].append(
                    Constraint(
                        value=expr.value,
                        operator=expr.operator,
                    )
                )
        if "annotation_geospatial" in expression_dict:
            filter_request.annotation_geospatial = [
                Constraint(
                    value=expr.value,
                    operator=expr.operator,
                )
                for expr in expression_dict["annotation_geospatial"]
            ]
        # predictions
        if "prediction_scores" in expression_dict:
            filter_request.prediction_scores = [
                Constraint(
                    value=expr.value,
                    operator=expr.operator,
                )
                for expr in expression_dict["prediction_scores"]
            ]

        # labels
        if "label_ids" in expression_dict:
            filter_request.label_ids = [
                expr.value for expr in expression_dict["label_ids"]
            ]
        if "labels" in expression_dict:
            filter_request.labels = [
                {expr.value.key: expr.value.value}
                for expr in expression_dict["labels"]
            ]
        if "label_keys" in expression_dict:
            filter_request.label_keys = [
                expr.value for expr in expression_dict["label_keys"]
            ]

        return filter_request



