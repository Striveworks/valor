import operator

from sqlalchemy import Float, and_, func, or_, select
from sqlalchemy.orm.attributes import InstrumentedAttribute
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
        self._where: dict[DeclarativeMeta, list[BinaryExpression]] = {}
        self._selected: set[DeclarativeMeta] = set(
            [
                self._map_attribute_to_table(argument)
                for argument in args
                if (
                    isinstance(argument, DeclarativeMeta)
                    or isinstance(argument, InstrumentedAttribute)
                )
            ]
        )
        self._filtered = set()

    """ Private Methods """

    def _add_expressions(self, table, expressions: list[BinaryExpression]):
        self._filtered.add(table)
        if table not in self._where:
            self._where[table] = []
        if len(expressions) == 1:
            self._where[table].extend(expressions)
        elif len(expressions) > 1:
            self._where[table].append(or_(*expressions))

    def _map_attribute_to_table(
        self, attr: InstrumentedAttribute | DeclarativeMeta
    ) -> DeclarativeMeta | None:
        if isinstance(attr, DeclarativeMeta):
            return attr
        if not isinstance(attr, InstrumentedAttribute):
            return None
        match attr.table.name:
            case models.Dataset.__tablename__:
                return models.Dataset
            case models.Model.__tablename__:
                return models.Model
            case models.Datum.__tablename__:
                return models.Datum
            case models.Annotation.__tablename__:
                return models.Annotation
            case models.GroundTruth.__tablename__:
                return models.GroundTruth
            case models.Prediction.__tablename__:
                return models.Prediction
            case models.Label.__tablename__:
                return models.Label
            case _:
                return None

    def _trim_extremities(
        self, graph: list[DeclarativeMeta], joint_set: set[DeclarativeMeta]
    ) -> list[DeclarativeMeta]:
        """trim graph extremities of unused nodes"""
        lhi = 0
        rhi = len(graph)
        for idx, table in enumerate(graph):
            if table in joint_set:
                lhi = idx
                break
        for idx, table in enumerate(reversed(graph)):
            if table in joint_set:
                rhi = rhi - idx
                break
        return graph[lhi:rhi]

    def _g1_solver(
        self,
        args: list[DeclarativeMeta | InstrumentedAttribute],
        selected: list[DeclarativeMeta],
        filtered: list[DeclarativeMeta],
    ):
        """
        g1 = [models.Dataset, models.Datum, models.Annotation, models.GroundTruth, models.Label]
        """
        # Graph defintion
        graph = [
            models.Dataset,
            models.Datum,
            models.Annotation,
            models.GroundTruth,
            models.Label,
        ]
        connections = {
            models.Datum: models.Datum.dataset_id == models.Dataset.id,
            models.Annotation: models.Annotation.datum_id == models.Datum.id,
            models.GroundTruth: models.GroundTruth.annotation_id
            == models.Annotation.id,
            models.Label: models.Label.id == models.GroundTruth.label_id,
        }

        # set of tables required to construct query
        joint_set = selected.union(filtered)

        # generate query statement
        query = None
        for table in self._trim_extremities(graph, joint_set):
            if query is None:
                query = select(*args).select_from(table)
            else:
                query = query.join(table, connections[table])

        # generate where statement
        expressions = [
            expression
            for table in joint_set.intersection(set(self._where.keys()))
            for expression in self._where[table]
        ]
        return query.where(and_(*expressions))

    def _g2_solver(
        self,
        args: list[DeclarativeMeta | InstrumentedAttribute],
        selected: list[DeclarativeMeta],
        filtered: list[DeclarativeMeta],
    ):
        """
        g2 = [[models.Model, models.Annotation, models.Prediction, models.Label], [models.Model, models.Annotation, models.Datum, models.Dataset]]
        """
        subgraph1 = [
            models.Model,
            models.Annotation,
            models.Prediction,
            models.Label,
        ]
        subgraph2 = [
            models.Model,
            models.Annotation,
            models.Datum,
            models.Dataset,
        ]
        connections = {
            models.Annotation: models.Annotation.model_id == models.Model.id,
            models.Datum: models.Datum.id == models.Annotation.datum_id,
            models.Dataset: models.Dataset.id == models.Datum.dataset_id,
            models.Prediction: models.Prediction.annotation_id
            == models.Annotation.id,
            models.Label: models.Label.id == models.Prediction.label_id,
        }

        joint_set = selected.union(filtered)

        # generate query statement
        graph = self._trim_extremities(subgraph1, joint_set)
        graph.extend(self._trim_extremities(subgraph2, joint_set))
        repeated_set = set()
        query = select(*args).select_from(models.Model)
        for table in graph:
            if table not in repeated_set:
                query = query.join(table, connections[table])
                repeated_set.add(table)

        # generate where statement
        expressions = [
            expression
            for table in joint_set.intersection(set(self._where.keys()))
            for expression in self._where[table]
        ]
        return query.where(and_(*expressions))

    def _g3_solver(
        self,
        args: list[DeclarativeMeta | InstrumentedAttribute],
        selected: list[DeclarativeMeta],
        filtered: list[DeclarativeMeta],
    ):
        """
        g3 = [models.Dataset, models.Datum, models.Annotation, models.Prediction, models.Label]
        """
        # Graph defintion
        graph = [
            models.Dataset,
            models.Datum,
            models.Annotation,
            models.Prediction,
            models.Label,
        ]
        connections = {
            models.Datum: models.Datum.dataset_id == models.Dataset.id,
            models.Annotation: models.Annotation.datum_id == models.Datum.id,
            models.Prediction: models.Prediction.annotation_id
            == models.Annotation.id,
            models.Label: models.Label.id == models.Prediction.label_id,
        }

        # set of tables required to construct query
        joint_set = selected.union(filtered)

        # generate query statement
        query = None
        for table in self._trim_extremities(graph, joint_set):
            if query is None:
                query = select(*args).select_from(table)
            else:
                query = query.join(table, connections[table])

        # generate where statement
        expressions = [
            expression
            for table in joint_set.intersection(set(self._where.keys()))
            for expression in self._where[table]
        ]
        return query.where(and_(*expressions))

    def _g4_solver(
        self,
        args: list[DeclarativeMeta | InstrumentedAttribute],
        selected: list[DeclarativeMeta],
        filtered: list[DeclarativeMeta],
    ):
        """
        g4 = [[models.Dataset, models.Datum, models.Annotation, models.Label]]
        """
        graph = [
            models.Dataset,
            models.Datum,
            models.Annotation,
            models.Label,
        ]  # excluding label as edge case due to forking at groundtruth, prediction
        connections = {
            models.Datum: models.Datum.dataset_id == models.Dataset.id,
            models.Annotation: models.Annotation.datum_id == models.Datum.id,
            models.GroundTruth: models.GroundTruth.annotation_id
            == models.Annotation.id,
            models.Prediction: models.Prediction.annotation_id
            == models.Annotation.id,
            models.Label: or_(
                models.Label.id == models.GroundTruth.label_id,
                models.Label.id == models.Prediction.label_id,
            ),
        }

        # set of tables required to construct query
        joint_set = selected.union(filtered)

        # generate query statement
        query = None
        for table in self._trim_extremities(graph, joint_set):
            if query is None:
                query = select(*args).select_from(table)
            else:
                if table == models.Label:
                    query = query.join(
                        models.GroundTruth, connections[models.GroundTruth]
                    )
                    query = query.join(
                        models.Prediction,
                        connections[models.Prediction],
                        full=True,
                    )
                    query = query.join(models.Label, connections[models.Label])
                else:
                    query = query.join(table, connections[table])

        # generate where statement
        expressions = [
            expression
            for table in joint_set.intersection(set(self._where.keys()))
            for expression in self._where[table]
        ]
        return query.where(and_(*expressions))

    def _graph_solver(self, additional_table: DeclarativeMeta | None = None):
        """
        There are 4 foundational graphs.
        g1 = [models.Dataset, models.Datum, models.Annotation, models.GroundTruth, models.Label]
        g2 = [[models.Model, models.Annotation, models.Prediction, models.Label], [models.Model, models.Annotation, models.Datum, models.Dataset]]
        g3 = [models.Dataset, models.Datum, models.Annotation, models.Prediction, models.Label],
        g4 = [models.Dataset, models.Datum, models.Annotation, models.Label]

        Removing common nodes, these are reduced to:
        g1_unique = [models.Groundtruth]
        g3_unique = [models.Prediction, models.Model]
        g2_unique = [models.Prediction]
        g4_unique = []
        """

        g1_unique = {models.GroundTruth}
        g2_unique = {models.Model}
        g3_unique = {models.Prediction}

        if self._selected.intersection(
            g1_unique
        ) and self._selected.intersection(g2_unique):
            raise RuntimeError(
                f"Cannot evaluate graph as invalid connection between queried tables. `{self._selected}`"
            )

        joint_set = self._selected.union(self._filtered)
        if isinstance(additional_table, DeclarativeMeta):
            joint_set.add(additional_table)

        # query - graph g1
        if g1_unique.issubset(joint_set):
            if g2_unique.issubset(joint_set):
                query = self._g1_solver(
                    self._args,
                    self._selected,
                    (self._filtered - g2_unique).union({models.Datum}),
                )
                subquery = self._g2_solver(
                    [models.Datum.id],
                    {models.Datum},
                    self._filtered.intersection(g2_unique).union(
                        {models.Datum}
                    ),
                )
            elif g3_unique.issubset(joint_set):
                query = self._g1_solver(
                    self._args,
                    self._selected,
                    (self._filtered - g3_unique).union({models.Datum}),
                )
                subquery = self._g3_solver(
                    [models.Datum.id],
                    {models.Datum},
                    self._filtered.intersection(g3_unique).union(
                        {models.Datum}
                    ),
                )
            else:
                query = self._g1_solver(
                    self._args, self._selected, self._filtered
                )
                subquery = None

        # query - graph g2
        elif g2_unique.issubset(joint_set):
            if not g1_unique.issubset(joint_set):
                query = self._g2_solver(
                    self._args, self._selected, self._filtered
                )
                subquery = None
            else:
                query = self._g2_solver(
                    self._args,
                    self._selected,
                    (self._filtered - g1_unique).union({models.Datum}),
                )
                subquery = self._g1_solver(
                    [models.Datum.id],
                    {models.Datum},
                    self._filtered.intersection(g1_unique).union(
                        {models.Datum}
                    ),
                )

        # query - graph g3
        elif g3_unique.issubset(joint_set):
            if not g1_unique.issubset(joint_set):
                query = self._g3_solver(
                    self._args, self._selected, self._filtered
                )
                subquery = None
            else:
                query = self._g3_solver(
                    self._args,
                    self._selected,
                    (self._filtered - g1_unique).union({models.Datum}),
                )
                subquery = self._g1_solver(
                    [models.Datum.id],
                    {models.Datum},
                    self._filtered.intersection(g1_unique).union(
                        {models.Datum}
                    ),
                )

        # query - graph g3
        else:
            query = self._g4_solver(self._args, self._selected, self._filtered)
            subquery = None

        return query, subquery

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

    def any(self, *_, _table: DeclarativeMeta | None = None):
        query, subquery = self._graph_solver(_table)
        if subquery is not None:
            query = query.where(models.Datum.id.in_(subquery))
        return query.subquery("generated_query")

    def groundtruths(self):
        return self.any(_table=models.GroundTruth)

    def predictions(self):
        return self.any(_table=models.Prediction)

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

    def filter_by_prediction(self, filters: schemas.PredictionFilter):
        if filters.score:
            op = self._get_numeric_op(filters.score.operator)
            self._add_expressions(
                models.Prediction,
                [op(models.Prediction.score, filters.score.value)],
            )
        return self

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


f = schemas.Filter(
    datasets=schemas.DatasetFilter(names=["dset"]),
    predictions=schemas.PredictionFilter(
        score=schemas.NumericFilter(value=0.5, operator=">=")
    ),
)

q, s = Query(models.Label).filter(f)._graph_solver(models.GroundTruth)
print(q)
print()
print(s)
print()


qg = Query(models.Label).filter(f).groundtruths()

# print(qa)
print()
print(qg)
