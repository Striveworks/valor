# import operator

from sqlalchemy import and_, or_, select
from sqlalchemy.sql.elements import BinaryExpression

from velour_api import enums, schemas
from velour_api.backend import models
from velour_api.backend.graph import Graph, Node

# Global instance of Graph, prevents recomputation
graph = Graph()


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

    def __init__(self, table):
        self.table = table
        self.groundtruth_target = graph.select_groundtruth_graph_node(table)
        self.prediction_target = graph.select_prediction_graph_node(table)
        self.filter_by: dict[Node, list[BinaryExpression]] = {}

    def add_expressions(self, node: Node, expressions: list[BinaryExpression]):
        if node not in self.filter_by:
            self.filter_by[node] = []
        if len(expressions) == 1:
            self.filter_by[node].extend(expressions)
        elif len(expressions) > 1:
            self.filter_by[node].append(or_(*expressions))

    """ User methods """

    def _ids(self):
        """Returns id select statement for target that meets filter criteria."""

        # generate queries by groundtruth and prediction
        groundtruth_id_query = None
        prediction_id_query = None
        if self.groundtruth_target:
            groundtruth_id_query = graph.generate_query(
                self.groundtruth_target, self.filter_by
            )
        if self.prediction_target:
            prediction_id_query = graph.generate_query(
                self.prediction_target, self.filter_by
            )

        # resolve groundtruth and prediction split
        if self.groundtruth_target and self.prediction_target:
            q_ids = groundtruth_id_query.union(prediction_id_query)
        else:
            q_ids = (
                groundtruth_id_query
                if groundtruth_id_query is not None
                else prediction_id_query
            )
        return q_ids

    def ids(self):
        """Returns id subquery for target that meets filter criteria."""
        return self._ids().subquery("id")

    def query(self):
        """Returns sqlalchemy table rows"""
        q_ids = self._ids()
        return (
            select(self.table)
            .where(self.table.id.in_(q_ids))
            .subquery("labels")
        )

    def filter(self, filters: schemas.Filter):
        """Parses `schemas.Filter`"""

        # Check if groundtruth or predictions are discarded
        if filters.groundtruth_labels and not filters.prediction_labels:
            self.prediction_target = None
        elif not filters.groundtruth_labels and filters.prediction_labels:
            self.groundtruth_target = None

        # dataset filters
        if filters.datasets:
            self.filter_by_datasets(filters.datasets)

        # models
        if filters.models:
            self.filter_by_models(filters.models)

        # datums
        if filters.datums:
            self.filter_by_datums(filters.datums)

        # annotations
        if filters.annotations:
            self.filter_by_annotations(filters.annotations)
        if filters.groundtruth_annotations:
            self.filter_by_groundtruth_annotations(
                filters.groundtruth_annotations
            )
        if filters.prediction_annotations:
            self.filter_by_prediction_annotations(
                filters.prediction_annotations
            )

        # groundtruth
        if filters.groundtruths:
            raise NotImplementedError("groundtruth filters are WIP")

        # prediction
        if filters.predictions:
            raise NotImplementedError("prediction filters are WIP")

        # labels
        if filters.labels:
            self.filter_by_labels(filters.labels)
        if filters.groundtruth_labels:
            self.filter_by_groundtruth_labels(filters.groundtruth_labels)
        if filters.prediction_labels:
            self.filter_by_prediction_labels(filters.prediction_labels)

        return self

    """ dataset filter """

    def filter_by_datasets(self, filters: schemas.DatasetFilter):
        if filters.ids:
            expressions = [
                models.Dataset.id == id
                for id in filters.ids
                if isinstance(id, int)
            ]
            self.add_expressions(graph.dataset, expressions)
        if filters.names:
            expressions = [
                models.Dataset.name == name
                for name in filters.names
                if isinstance(name, str)
            ]
            self.add_expressions(graph.dataset, expressions)
        if filters.metadata:
            pass
        return self

    """ model filter """

    def filter_by_models(self, filters: schemas.ModelFilter):
        if filters.ids:
            expressions = [
                models.Model.id == id
                for id in filters.ids
                if isinstance(id, int)
            ]
            self.add_expressions(graph.model, expressions)
        if filters.names:
            expressions = [
                models.Model.name == name
                for name in filters.names
                if isinstance(name, str)
            ]
            self.add_expressions(graph.model, expressions)
        if filters.metadata:
            pass
        return self

    """ datum filter """

    def filter_by_datums(self, filters: schemas.DatumFilter):
        if filters.uids:
            expressions = [
                models.Datum.uid == uid
                for uid in filters.uids
                if isinstance(uid, str)
            ]
            self.add_expressions(graph.model, expressions)
        if filters.metadata:
            pass
        return self

    """ filter by label """

    def filter_by_labels(
        self,
        filters: schemas.LabelFilter,
        add_to_groundtruths: bool = True,
        add_to_predictions: bool = True,
    ):
        if filters.labels:
            expressions = [
                and_(
                    models.Label.key == label.key,
                    models.Label.value == label.value,
                )
                for label in filters.labels
                if isinstance(label, schemas.Label)
            ]
            if add_to_groundtruths:
                self.add_expressions(graph.groundtruth_label, expressions)
            if add_to_predictions:
                self.add_expressions(graph.prediction_label, expressions)
        if filters.keys:
            expressions = [
                models.Label.key == key
                for key in filters.keys
                if isinstance(key, str)
            ]
            if add_to_groundtruths:
                self.add_expressions(graph.groundtruth_label, expressions)
            if add_to_predictions:
                self.add_expressions(graph.prediction_label, expressions)
        return self

    def filter_by_groundtruth_labels(
        self,
        filters: schemas.LabelFilter,
    ):
        """Only applies filter to groundtruth labels."""
        return self.filter_by_labels(filters, add_to_predictions=False)

    def filter_by_prediction_labels(
        self,
        filters: schemas.LabelFilter,
    ):
        """Only applies filter to prediction labels."""
        return self.filter_by_labels(filters, add_to_groundtruths=False)

    """ filter by groundtruth """

    def filter_by_groundtruth():
        pass

    """ filter by prediction """

    def filter_by_prediction():
        pass

    """ filter by metadata """

    # def _filter_by_metadatum(
    #     self,
    #     metadatum: schemas.MetaDatum | models.MetaDatum,
    #     op: str,
    #     node: Node,
    # ):

    #     ops = {
    #         ">": operator.gt,
    #         "<": operator.lt,
    #         ">=": operator.ge,
    #         "<=": operator.le,
    #         "==": operator.eq,
    #         "!=": operator.ne,
    #     }
    #     if op not in ops:
    #         raise ValueError(f"invalid comparison operator `{operator}`")

    #     # compare name
    #     expression = [models.MetaDatum.name == metadatum.name]

    #     # sqlalchemy handler
    #     if isinstance(metadatum, models.MetaDatum):
    #         if metadatum.string_value is not None:
    #             expression.append(
    #                 ops[op](
    #                     models.MetaDatum.string_value, metadatum.string_value
    #                 )
    #             )
    #         elif metadatum.numeric_value is not None:
    #             expression.append(
    #                 ops[op](
    #                     models.MetaDatum.numeric_value, metadatum.numeric_value
    #                 )
    #             )
    #         elif metadatum.geo is not None:
    #             raise NotImplementedError("GeoJSON currently unsupported.")

    #     # schema handler
    #     elif isinstance(metadatum, schemas.MetaDatum):
    #         # compare value
    #         if isinstance(metadatum.value, str):
    #             expression.append(
    #                 ops[op](models.MetaDatum.string_value, metadatum.value)
    #             )
    #         if isinstance(metadatum.value, float):
    #             expression.append(
    #                 ops[op](models.MetaDatum.numeric_value, metadatum.value)
    #             )
    #         if isinstance(metadatum.value, schemas.GeoJSON):
    #             raise NotImplementedError("GeoJSON currently unsupported.")

    #     # unknown type
    #     else:
    #         return None

    #     return and_(*expression)

    # def filter_by_metadata(
    #     self,
    #     metadata: list[schemas.MetadataFilter],
    #     node: str,
    # ):
    #     # generate binary expressions
    #     expressions = [
    #         self._filter_by_metadatum(filt.metadatum, filt.operator, node)
    #         for filt in metadata
    #     ]

    #     # generate filter
    #     if len(expressions) == 1:
    #         self.filter_by.add(node)
    #         self._filters.extend(expressions)
    #     elif len(expressions) > 1:
    #         self.filter_by.add(node)
    #         self._filters.append(or_(*expressions))

    #     return self

    # def filter_by_metadatum_names(self, names: list[str]):
    #     # generate binary expressions
    #     expressions = [
    #         models.MetaDatum.name == name
    #         for name in names
    #         if isinstance(name, str)
    #     ]

    #     # generate filter
    #     if len(expressions) == 1:
    #         self.filter_by.add(graph.metadatum)
    #         self._filters.extend(expressions)
    #     elif len(expressions) > 1:
    #         self.filter_by.add(graph.metadatum)
    #         self._filters.append(or_(*expressions))

    #     return self

    """ filter by annotation """

    def filter_by_annotations(
        self,
        filters: schemas.AnnotationFilter,
        add_to_groundtruths: bool = True,
        add_to_predictions: bool = True,
    ):
        if filters.task_types:
            expressions = [
                models.Annotation.task_type == task_type.value
                for task_type in filters.task_types
                if isinstance(task_type, enums.TaskType)
            ]
            if add_to_groundtruths:
                self.add_expressions(self.groundtruth_target, expressions)
            if add_to_predictions:
                self.add_expressions(self.prediction_target, expressions)
        if filters.annotation_types:
            if enums.AnnotationType.NONE in filters.annotation_types:
                expressions = [
                    and_(
                        models.Annotation.box.is_(None),
                        models.Annotation.polygon.is_(None),
                        models.Annotation.multipolygon.is_(None),
                        models.Annotation.raster.is_(None),
                    )
                ]
                if add_to_groundtruths:
                    self.add_expressions(self.groundtruth_target, expressions)
                if add_to_predictions:
                    self.add_expressions(self.prediction_target, expressions)
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

                if add_to_groundtruths:
                    self.add_expressions(self.groundtruth_target, expressions)
                if add_to_predictions:
                    self.add_expressions(self.prediction_target, expressions)
        return self

    def filter_by_groundtruth_annotations(
        self, filters: schemas.AnnotationFilter
    ):
        return self.filter_by_annotations(filters, add_to_predictions=False)

    def filter_by_prediction_annotations(
        self, filters: schemas.AnnotationFilter
    ):
        return self.filter_by_annotations(filters, add_to_groundtruths=False)


if __name__ == "__main__":
    db = None
    query = (
        Query(graph.label_groundtruth)
        .filter_by_dataset_names(["dataset1"])
        .filter_by_model_names(["model1"])
        .filter_by_datum_uids(["uid1"])
    )
    print(str(query.ids()))

    # filter_by = ["annotation", "model"]
    # target = "label"

    # output = generate_query(target, filter_by)
    # print(f"SELECT FROM {target}")
    # for o in output:
    # print(o)
