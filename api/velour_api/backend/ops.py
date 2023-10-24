import operator

from sqlalchemy import Float, and_, func, or_, select
from sqlalchemy.orm.decl_api import DeclarativeMeta
from sqlalchemy.sql.elements import BinaryExpression

from velour_api import enums, schemas
from velour_api.backend import models


class Query:
    """
    Query generator object.

    Attributes
    ----------
    table : velour_api.backend.database.Base
        sqlalchemy table
    groundtruth_target : Node, optional
        groundtruth Node is targeted if not None
    prediction_target : Node, optional
        prediction Node is targeted if not None
    filter_by : dict[Node, list[BinaryExpression]]
        List of BinaryExpressions to apply per node.
    """

    def __init__(self, *args):
        self._args = args
        self._where: list[BinaryExpression] = []
        self._tables: set[str] = set(
            [
                str(arg.__tablename__)
                if isinstance(arg, DeclarativeMeta)
                else str(arg.table)
                for arg in self._args
            ]
        )

    """ Private Methods """

    def _add_expressions(self, table, expressions: list[BinaryExpression]):
        self._tables.add(table.__tablename__)
        if len(expressions) == 1:
            self._where.extend(expressions)
        elif len(expressions) > 1:
            self._where.append(or_(*expressions))

    def _check_simple_graph(self):
        joint_table_set = {
            models.Dataset.__tablename__,
            models.Model.__tablename__,
            models.Datum.__tablename__,
            models.Annotation.__tablename__,
        }
        return self._tables.issubset(joint_table_set)

    def _check_joint_graph(self):
        joint_table_set = {
            models.Dataset.__tablename__,
            models.Datum.__tablename__,
            models.Annotation.__tablename__,
            models.Label.__tablename__,
        }
        return self._tables.issubset(joint_table_set)

    def _check_groundtruth_graph(self):
        groundtruth_table_set = {
            models.Dataset.__tablename__,
            models.Datum.__tablename__,
            models.Annotation.__tablename__,
            models.GroundTruth.__tablename__,
            models.Label.__tablename__,
        }
        return self._tables.issubset(groundtruth_table_set)

    def _check_prediction_graph(self) -> bool:
        prediction_table_set = {
            models.Dataset.__tablename__,
            models.Model.__tablename__,
            models.Datum.__tablename__,
            models.Annotation.__tablename__,
            models.Prediction.__tablename__,
            models.Label.__tablename__,
        }
        return self._tables.issubset(prediction_table_set)

    def _get_numeric_op(self, opstr) -> operator:
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

    def _get_string_op(self, opstr) -> operator:
        ops = {
            "==": operator.eq,
            "!=": operator.ne,
        }
        if opstr not in ops:
            raise ValueError(f"invalid string comparison operator `{opstr}`")
        return ops[opstr]

    """ Public methods """

    def query(self):
        # TODO: Add more graph definitions, query should test graph from simplest to most complex.
        if self._check_simple_graph():
            return self.simple()
        elif self._check_joint_graph():
            return self.joint()
        elif self._check_groundtruth_graph():
            return self.groundtruths()
        elif self._check_prediction_graph():
            return self.predictions()
        else:
            raise RuntimeError(
                "Query does not conform to any available table graph."
            )

    def simple(self):
        if not self._check_simple_graph():
            raise RuntimeError("Query does not conform to simple table graph.")
        return (
            select(*self._args)
            .select_from(models.Dataset)
            .join(models.Datum, models.Datum.dataset_id == models.Dataset.id)
            .join(
                models.Annotation,
                models.Annotation.datum_id == models.Datum.id,
            )
            .join(models.Model, models.Model.id == models.Annotation.model_id)
            .where(and_(*self._where))
            .subquery("generated_query")
        )

    def joint(self):
        if not self._check_joint_graph():
            raise RuntimeError("Query does not conform to joint table graph.")

        return (
            select(*self._args)
            .select_from(models.Dataset)
            .join(models.Datum, models.Datum.dataset_id == models.Dataset.id)
            .join(
                models.Annotation,
                models.Annotation.datum_id == models.Datum.id,
            )
            .join(
                models.GroundTruth,
                models.GroundTruth.annotation_id == models.Annotation.id,
            )
            .join(
                models.Prediction,
                models.Prediction.annotation_id == models.Annotation.id,
                full=True,
            )
            .join(
                models.Label,
                or_(
                    models.Label.id == models.GroundTruth.label_id,
                    models.Label.id == models.Prediction.label_id,
                ),
            )
            .where(and_(*self._where))
            .subquery("generated_query")
        )

    def groundtruths(self):
        if not self._check_groundtruth_graph():
            raise RuntimeError(
                "Query does not conform to groundtruths table graph."
            )

        return (
            select(*self._args)
            .select_from(models.Dataset)
            .join(models.Datum, models.Datum.dataset_id == models.Dataset.id)
            .join(
                models.Annotation,
                and_(
                    models.Annotation.datum_id == models.Datum.id,
                    models.Annotation.model_id.is_(None),
                ),
            )
            .join(
                models.GroundTruth,
                models.GroundTruth.annotation_id == models.Annotation.id,
            )
            .join(models.Label, models.Label.id == models.GroundTruth.label_id)
            .where(and_(*self._where))
            .subquery("generated_query")
        )

    def predictions(self):
        if not self._check_prediction_graph:
            raise RuntimeError(
                "Query does not conform to predictions table graph."
            )

        return (
            select(*self._args)
            .select_from(models.Dataset)
            .join(models.Datum, models.Datum.dataset_id == models.Dataset.id)
            .join(
                models.Annotation,
                and_(
                    models.Annotation.datum_id == models.Datum.id,
                    models.Annotation.model_id.is_not(None),
                ),
            )
            .join(models.Model, models.Model.id == models.Annotation.model_id)
            .join(
                models.Prediction,
                models.Prediction.annotation_id == models.Annotation.id,
            )
            .join(models.Label, models.Label.id == models.Prediction.label_id)
            .where(and_(*self._where))
            .subquery("generated_query")
        )

    def filter(self, filters: schemas.Filter):
        """Parses `schemas.Filter`"""
        if filters is None:
            return self
        if not isinstance(filters, schemas.Filter):
            raise TypeError(
                "filters should be of type `schemas.Filter` or `None`"
            )

        # dataset filters
        if filters.datasets:
            self.filter_by_dataset(filters.datasets)

        # models
        if filters.models:
            self.filter_by_model(filters.models)

        # datums
        if filters.datums:
            self.filter_by_datum(filters.datums)

        # annotations
        if filters.annotations:
            self.filter_by_annotation(filters.annotations)

        # prediction
        if filters.predictions:
            self.filter_by_prediction(filters.predictions)

        # labels
        if filters.labels:
            self.filter_by_label(filters.labels)

        return self

    """ dataset filter """

    def filter_by_dataset(self, filters: schemas.DatasetFilter):
        if filters.ids:
            self._add_expressions(
                models.Dataset,
                [
                    models.Dataset.id == id
                    for id in filters.ids
                    if isinstance(id, int)
                ],
            )
        if filters.names:
            self._add_expressions(
                models.Dataset,
                [
                    models.Dataset.name == name
                    for name in filters.names
                    if isinstance(name, str)
                ],
            )
        if filters.metadata:
            self._add_expressions(
                models.Dataset,
                [
                    self._filter_by_metadatum(metadatum, models.Dataset)
                    for metadatum in filters.metadata
                ],
            )
        return self

    """ model filter """

    def filter_by_model(self, filters: schemas.ModelFilter):
        if filters.ids:
            self._add_expressions(
                models.Model,
                [
                    models.Model.id == id
                    for id in filters.ids
                    if isinstance(id, int)
                ],
            )
        if filters.names:
            self._add_expressions(
                models.Model,
                [
                    models.Model.name == name
                    for name in filters.names
                    if isinstance(name, str)
                ],
            )
        if filters.metadata:
            self._add_expressions(
                models.Model,
                [
                    self._filter_by_metadatum(metadatum, models.Model)
                    for metadatum in filters.metadata
                ],
            )
        return self

    """ datum filter """

    def filter_by_datum(self, filters: schemas.DatumFilter):
        if filters.ids:
            self._add_expressions(
                models.Datum,
                [
                    models.Datum.id == id
                    for id in filters.ids
                    if isinstance(id, int)
                ],
            )
        if filters.uids:
            self._add_expressions(
                models.Datum,
                [
                    models.Datum.uid == uid
                    for uid in filters.uids
                    if isinstance(uid, str)
                ],
            )
        if filters.metadata:
            self._add_expressions(
                models.Datum,
                [
                    self._filter_by_metadatum(metadatum, models.Datum)
                    for metadatum in filters.metadata
                ],
            )
        return self

    """ filter by annotation """

    def filter_by_annotation(
        self,
        filters: schemas.AnnotationFilter,
    ):
        if filters.task_types:
            self._add_expressions(
                models.Annotation,
                [
                    models.Annotation.task_type == task_type.value
                    for task_type in filters.task_types
                    if isinstance(task_type, enums.TaskType)
                ],
            )
        if filters.annotation_types:
            if enums.AnnotationType.NONE in filters.annotation_types:
                self._add_expressions(
                    models.Annotation,
                    [
                        and_(
                            models.Annotation.box.is_(None),
                            models.Annotation.polygon.is_(None),
                            models.Annotation.multipolygon.is_(None),
                            models.Annotation.raster.is_(None),
                        )
                    ],
                )
            else:
                expressions = []
                if enums.AnnotationType.BOX in filters.annotation_types:
                    expressions.append(models.Annotation.box.isnot(None))
                if enums.AnnotationType.POLYGON in filters.annotation_types:
                    expressions.append(models.Annotation.polygon.isnot(None))
                if (
                    enums.AnnotationType.MULTIPOLYGON
                    in filters.annotation_types
                ):
                    expressions.append(
                        models.Annotation.multipolygon.isnot(None)
                    )
                if enums.AnnotationType.RASTER in filters.annotation_types:
                    expressions.append(models.Annotation.raster.isnot(None))
                self._add_expressions(models.Annotation, expressions)
        if filters.geometry:
            match filters.geometry.type:
                case enums.AnnotationType.BOX:
                    geom = models.Annotation.box
                case enums.AnnotationType.POLYGON:
                    geom = models.Annotation.polygon
                case enums.AnnotationType.MULTIPOLYGON:
                    geom = models.Annotation.multipolygon
                case enums.AnnotationType.RASTER:
                    geom = models.Annotation.raster
                case _:
                    raise RuntimeError
            if filters.geometry.area:
                op = self._get_numeric_op(filters.geometry.area.operator)
                self._add_expressions(
                    models.Annotation,
                    [op(func.ST_Area(geom), filters.geometry.area.value)],
                )
            if filters.geometry.height:
                op = self._get_numeric_op(filters.geometry.height.operator)
                self._add_expressions(
                    models.Annotation,
                    [op(func.ST_Area(geom), filters.geometry.height.value)],
                )
            if filters.geometry.width:
                op = self._get_numeric_op(filters.geometry.width.operator)
                self._add_expressions(
                    models.Annotation,
                    [op(func.ST_Area(geom), filters.geometry.width.value)],
                )
        if filters.metadata:
            self._add_expressions(
                models.Annotation,
                [
                    self._filter_by_metadatum(metadatum, models.Annotation)
                    for metadatum in filters.metadata
                ],
            )
        return self

    """ filter by prediction """

    def filter_by_prediction(self, filters: schemas.PredictionFilter):
        if filters.score:
            op = self._get_numeric_op(filters.score.operator)
            self._add_expressions(
                models.Prediction,
                [op(models.Prediction.score, filters.score.value)],
            )
        return self

    """ filter by label """

    def filter_by_label(
        self,
        filters: schemas.LabelFilter,
    ):
        if filters.labels:
            self._add_expressions(
                models.Label,
                [
                    and_(
                        models.Label.key == label.key,
                        models.Label.value == label.value,
                    )
                    for label in filters.labels
                    if isinstance(label, schemas.Label)
                ],
            )
        if filters.keys:
            self._add_expressions(
                models.Label,
                [
                    models.Label.key == key
                    for key in filters.keys
                    if isinstance(key, str)
                ],
            )
        return self

    """ filter by metadata """

    def _filter_by_metadatum(
        self,
        metadatum: schemas.MetadatumFilter,
        table,
    ):
        if not isinstance(metadatum, schemas.MetadatumFilter):
            raise TypeError("metadatum should be of type `schemas.Metadatum`")

        if isinstance(metadatum.comparison.value, str):
            op = self._get_string_op(metadatum.comparison.operator)
            lhs = table.meta[metadatum.key].astext
        elif isinstance(metadatum.comparison.value, float):
            op = self._get_numeric_op(metadatum.comparison.operator)
            lhs = table.meta[metadatum.key].astext.cast(Float)
        else:
            raise NotImplementedError(
                f"metadatum value of type `{type(metadatum.comparison.value)}` is currently not supported"
            )

        return op(lhs, metadatum.comparison.value)
