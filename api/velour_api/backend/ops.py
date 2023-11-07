import operator

from sqlalchemy import Float, and_, func, or_, select
from sqlalchemy.orm.attributes import InstrumentedAttribute
from sqlalchemy.orm.decl_api import DeclarativeMeta
from sqlalchemy.sql.elements import BinaryExpression

from velour_api import enums
from velour_api.backend import models
from velour_api.schemas import Filter, NumericFilter, StringFilter


class Query:
    """
    Query generator object.

    Attributes
    ----------
    *args : DeclarativeMeta | InstrumentedAttribute
        args is a list of models or model attributes. (e.g. models.Label or models.Label.key)

    Examples
    ----------
    Querying models.
    >>> f = schemas.Filter(...)
    >>> q = Query(models.Label).filter(f).any()

    Querying model attributes.
    >>> f = schemas.Filter(...)
    >>> q = Query(models.Label.key).filter(f).any()
    """

    def __init__(self, *args):
        self._args = args
        self._expressions: dict[DeclarativeMeta, list[BinaryExpression]] = {}
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
        if table not in self._expressions:
            self._expressions[table] = []
        if len(expressions) == 1:
            self._expressions[table].extend(expressions)
        elif len(expressions) > 1:
            self._expressions[table].append(or_(*expressions))

    def _expression(self, table_set: set[DeclarativeMeta]) -> BinaryExpression:
        expressions = []
        for table in table_set:
            if table in self._expressions:
                expressions.extend(self._expressions[table])
        if len(expressions) == 1:
            return expressions[0]
        elif len(expressions) > 1:
            return and_(*expressions)
        else:
            return None

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

    def _solve_groundtruth_graph(
        self,
        args: list[DeclarativeMeta | InstrumentedAttribute],
        selected: list[DeclarativeMeta],
        filtered: list[DeclarativeMeta],
    ):
        """
        groundtruth_graph = [models.Dataset, models.Datum, models.Annotation, models.GroundTruth, models.Label]
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
        expression = self._expression(joint_set)
        if expression is not None:
            query = query.where(expression)

        return query

    def _solve_model_graph(
        self,
        args: list[DeclarativeMeta | InstrumentedAttribute],
        selected: list[DeclarativeMeta],
        filtered: list[DeclarativeMeta],
    ):
        """
        model_graph = [[models.Model, models.Annotation, models.Prediction, models.Label], [models.Model, models.Annotation, models.Datum, models.Dataset]]
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
        query = select(*args).select_from(models.Model)
        repeated_set = {models.Model}
        for table in graph:
            if table not in repeated_set:
                query = query.join(table, connections[table])
                repeated_set.add(table)

        # generate where statement
        expression = self._expression(joint_set)
        if expression is not None:
            query = query.where(expression)

        return query

    def _solve_prediction_graph(
        self,
        args: list[DeclarativeMeta | InstrumentedAttribute],
        selected: list[DeclarativeMeta],
        filtered: list[DeclarativeMeta],
    ):
        """
        prediction_graph = [models.Dataset, models.Datum, models.Annotation, models.Prediction, models.Label]
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
        expression = self._expression(joint_set)
        if expression is not None:
            query = query.where(expression)

        return query

    def _solve_joint_graph(
        self,
        args: list[DeclarativeMeta | InstrumentedAttribute],
        selected: list[DeclarativeMeta],
        filtered: list[DeclarativeMeta],
    ):
        """
        joint_graph = [[models.Dataset, models.Datum, models.Annotation, models.Label]]
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
        expression = self._expression(joint_set)
        if expression is not None:
            query = query.where(expression)

        return query

    def _solve_nested_graphs(
        self,
        query_solver: callable,
        subquery_solver: callable,
        unique_set: set[DeclarativeMeta],
        pivot_table: DeclarativeMeta | None = None,
    ):
        qset = (self._filtered - unique_set).union({models.Datum})
        query = query_solver(
            self._args,
            self._selected,
            qset,
        )
        sub_qset = self._filtered.intersection(unique_set).union(
            {models.Datum}
        )
        subquery = (
            subquery_solver(
                [models.Datum.id],
                {models.Datum},
                sub_qset,
            )
            if pivot_table in self._filtered or sub_qset != {pivot_table}
            else None
        )
        return query, subquery

    def _select_graph(self, pivot_table: DeclarativeMeta | None = None):
        """
        Selects best fitting graph to run query generation and returns tuple(query, subquery | None).

        Description
        ----------
        To construct complex queries it is necessary to describe the relationship between predictions and groundtruths.
        By splitting the underlying table relationships into four foundatational graphs the complex relationships can be described by
        sequental lists without branches (with the exception of model_graph). From these sequential graphs it is possible to construct
        the minimum set of nodes required to generate a query. For queries that can be described by a single foundational graph,
        the solution is to trim both ends of the sequence until you reach nodes in the query set. The relationships of the
        remaining nodes can then be used to construct the query. Two foundational graphs are required for queries that include both
        groundtruth and prediction/model constraints. The solver inserts `models.Datum` as the linking point between these two graphs
        allowing the generation of a query and subquery.

        The four foundational graphs:
        groundtruth_graph   = [models.Dataset, models.Datum, models.Annotation, models.GroundTruth, models.Label]
        model_graph         = [[models.Model, models.Annotation, models.Prediction, models.Label], [models.Model, models.Annotation, models.Datum, models.Dataset]]
        prediction_graph    = [models.Dataset, models.Datum, models.Annotation, models.Prediction, models.Label],
        joint_graph         = [models.Dataset, models.Datum, models.Annotation, models.Label]

        Removing common nodes, these are reduced to:
        groundtruth_graph_unique    = {models.Groundtruth}
        prediction_graph_unique     = {models.Prediction, models.Model}
        model_graph_unique          = {models.Prediction}
        joint_graph_unique          = {}

        Descriptions:
        groundtruth_graph : All prediction or model related information removed.
        model_graph       : All groundtruth related information removed.
        prediction_graph  : Subgraph of model_graph. All groundtruth and model information removed.
        joint_graph       : Predictions and groundtruths combined under a full outer join.
        """

        groundtruth_graph_unique = {models.GroundTruth}
        model_graph_unique = {
            models.Model
        }  # exclude prediction as it is include in the graph definition.
        prediction_graph_unique = {models.Prediction}

        # edge case - only one table specified in args and filters
        if self._selected == self._filtered and len(self._selected) == 1:
            query = select(*self._args)
            expression = self._expression(self._selected)
            if expression is not None:
                query = query.where(expression)
            return query, None

        # edge case - catch intersection that resolves into an empty return.
        if self._selected.intersection(
            groundtruth_graph_unique
        ) and self._selected.intersection(model_graph_unique):
            raise RuntimeError(
                f"Cannot evaluate graph as invalid connection between queried tables. `{self._selected}`"
            )

        # create joint (selected + filtered) table set
        joint_set = self._selected.union(self._filtered)

        # create set of tables to select graph with
        selector_set = (
            {pivot_table}.union(joint_set)
            if isinstance(pivot_table, DeclarativeMeta)
            else joint_set
        )

        subquery_solver = None
        unique_set = None

        # query - graph groundtruth_graph
        if (
            groundtruth_graph_unique.issubset(selector_set)
            and not model_graph_unique.issubset(self._selected)
            and not prediction_graph_unique.issubset(self._selected)
        ):
            query_solver = self._solve_groundtruth_graph
            if model_graph_unique.issubset(joint_set):
                subquery_solver = self._solve_model_graph
                unique_set = model_graph_unique
            elif prediction_graph_unique.issubset(joint_set):
                subquery_solver = self._solve_prediction_graph
                unique_set = prediction_graph_unique
        # query - graph model_graph
        elif model_graph_unique.issubset(selector_set):
            query_solver = self._solve_model_graph
            if groundtruth_graph_unique.issubset(joint_set):
                subquery_solver = self._solve_groundtruth_graph
                unique_set = groundtruth_graph_unique
        # query - graph prediction_graph
        elif prediction_graph_unique.issubset(selector_set):
            query_solver = self._solve_prediction_graph
            if groundtruth_graph_unique.issubset(joint_set):
                subquery_solver = self._solve_groundtruth_graph
                unique_set = groundtruth_graph_unique
        # query - graph joint_graph
        else:
            query_solver = self._solve_joint_graph

        # generate statement
        if subquery_solver is not None:
            return self._solve_nested_graphs(
                query_solver=query_solver,
                subquery_solver=subquery_solver,
                unique_set=unique_set,
                pivot_table=pivot_table,
            )
        else:
            query = query_solver(
                self._args,
                self._selected,
                self._filtered,
            )
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

    def any(
        self,
        name: str = "generated_subquery",
        *,
        _pivot: DeclarativeMeta | None = None,
    ):
        """
        Generates a sqlalchemy subquery. Graph is chosen automatically as best fit.
        """
        query, subquery = self._select_graph(_pivot)
        if subquery is not None:
            query = query.where(models.Datum.id.in_(subquery))
        return query.subquery(name)

    def groundtruths(self, name: str = "generated_subquery"):
        """
        Generates a sqlalchemy subquery using a groundtruths-focused graph.
        """
        return self.any(name, _pivot=models.GroundTruth)

    def predictions(self, name: str = "generated_subquery"):
        """
        Generates a sqlalchemy subquery using a predictions-focused graph.
        """
        return self.any(name, _pivot=models.Prediction)

    def filter(self, filters: Filter):
        """Parses `schemas.Filter`"""
        if filters is None:
            return self
        if not isinstance(filters, Filter):
            raise TypeError(
                "filters should be of type `schemas.Filter` or `None`"
            )

        # datasets
        if filters.dataset_names:
            self._add_expressions(
                models.Dataset,
                [
                    models.Dataset.name == name
                    for name in filters.dataset_names
                    if isinstance(name, str)
                ],
            )
        if filters.dataset_metadata:
            self._add_expressions(
                models.Dataset,
                self.filter_by_metadata(
                    filters.dataset_metadata, models.Dataset
                ),
            )
        if filters.dataset_geospatial:
            raise NotImplementedError

        # models
        if filters.models_names:
            self._add_expressions(
                models.Model,
                [
                    models.Model.name == name
                    for name in filters.models_names
                    if isinstance(name, str)
                ],
            )
        if filters.models_metadata:
            self._add_expressions(
                models.Model,
                self.filter_by_metadata(filters.models_metadata, models.Model),
            )
        if filters.models_geospatial:
            raise NotImplementedError

        # datums
        if filters.datum_uids:
            self._add_expressions(
                models.Datum,
                [
                    models.Datum.uid == uid
                    for uid in filters.datum_uids
                    if isinstance(uid, str)
                ],
            )
        if filters.datum_metadata:
            self._add_expressions(
                models.Datum,
                self.filter_by_metadata(filters.datum_metadata, models.Datum),
            )
        if filters.datum_geospatial:
            raise NotImplementedError

        # annotations
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
        if filters.annotation_geometric_area:
            types = (
                filters.annotation_types
                if filters.annotation_types
                else [
                    enums.AnnotationType.BOX,
                    enums.AnnotationType.POLYGON,
                    enums.AnnotationType.MULTIPOLYGON,
                    enums.AnnotationType.RASTER,
                ]
            )
            for area_filter in filters.annotation_geometric_area:
                area_expr = []
                for atype in types:
                    match atype:
                        case enums.AnnotationType.BOX:
                            geom = models.Annotation.box
                            afunc = func.ST_Area
                        case enums.AnnotationType.POLYGON:
                            geom = models.Annotation.polygon
                            afunc = func.ST_Area
                        case enums.AnnotationType.MULTIPOLYGON:
                            geom = models.Annotation.multipolygon
                            afunc = func.ST_Area
                        case enums.AnnotationType.RASTER:
                            geom = models.Annotation.raster
                            afunc = func.ST_Count
                        case _:
                            raise RuntimeError
                    op = self._get_numeric_op(area_filter.operator)
                    area_expr.append(op(afunc(geom), area_filter.value))
                self._add_expressions(models.Annotation, area_expr)
        if filters.annotation_metadata:
            self._add_expressions(
                models.Annotation,
                self.filter_by_metadata(
                    filters.annotation_metadata, models.Annotation
                ),
            )
        if filters.annotation_geospatial:
            raise NotImplementedError

        # prediction
        if filters.prediction_scores:
            for score_filter in filters.prediction_scores:
                op = self._get_numeric_op(score_filter.operator)
                self._add_expressions(
                    models.Prediction,
                    [op(models.Prediction.score, score_filter.value)],
                )

        # labels
        if filters.label_ids:
            self._add_expressions(
                models.Label,
                [
                    models.Label.id == id
                    for id in filters.label_ids
                    if isinstance(id, int)
                ],
            )
        if filters.labels:
            self._add_expressions(
                models.Label,
                [
                    and_(
                        models.Label.key == key,
                        models.Label.value == value,
                    )
                    for label in filters.labels
                    if isinstance(label, dict) and len(label) == 1
                    for key, value in label.items()
                ],
            )
        if filters.label_keys:
            self._add_expressions(
                models.Label,
                [
                    models.Label.key == key
                    for key in filters.label_keys
                    if isinstance(key, str)
                ],
            )

        return self

    def _filter_by_metadatum(
        self,
        key: str,
        value_filter: NumericFilter | StringFilter,
        table: DeclarativeMeta,
    ) -> BinaryExpression:
        if isinstance(value_filter, NumericFilter):
            op = self._get_numeric_op(value_filter.operator)
            lhs = table.meta[key].astext.cast(Float)
        elif isinstance(value_filter, StringFilter):
            op = self._get_string_op(value_filter.operator)
            lhs = table.meta[key].astext
        else:
            raise NotImplementedError(
                f"metadatum value of type `{type(value_filter.value)}` is currently not supported"
            )
        return op(lhs, value_filter.value)

    def filter_by_metadata(
        self,
        metadata: dict[str, list[NumericFilter] | StringFilter],
        table: DeclarativeMeta,
    ) -> list[BinaryExpression]:
        expressions = [
            self._filter_by_metadatum(key, value, table)
            for key, value in metadata.items()
            if isinstance(value, StringFilter)
        ] + [
            self._filter_by_metadatum(key, value, table)
            for key, vlist in metadata.items()
            if isinstance(vlist, list)
            for value in metadata[key]
            if isinstance(value, NumericFilter)
        ]
        return [and_(*expressions)]
