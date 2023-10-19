# import operator

from sqlalchemy import or_, select
from sqlalchemy.sql.elements import BinaryExpression

from velour_api import schemas
from velour_api.backend import graph, models


class Query:
    def __init__(self, target: graph.Node | None = None):
        self.target = target
        self.filter_by: dict[graph.Node, list[BinaryExpression]] = {}
        self.constraints = set()

    @classmethod
    def select_from(cls, table):
        match table:
            case models.Dataset:
                return cls(graph.dataset)
            case models.Model:
                return cls(graph.model)
            case models.Datum:
                return cls(graph.datum)
            case _:
                raise ValueError("select_from ")

    @classmethod
    def select_groundtruth_from(cls, table):
        pass

    def add_expressions(
        self, node: graph.Node, expressions: list[BinaryExpression]
    ):
        if node not in self.filter_by:
            self.filter_by[node] = []
        if len(expressions) == 1:
            self.filter_by[graph.dataset].extend(expressions)
        elif len(expressions) > 1:
            self.filter_by[graph.dataset].append(or_(*expressions))

    """ User methods """

    def ids(self):
        """Returns ids from target that meet filter criteria."""
        id_query = graph.generate_query(self.target, self.filter_by)
        return id_query

    def query(self):
        """Returns sqlalchemy table rows"""
        q_ids = self.ids()
        return select(self.target.model).where(self.target.model.id.in_(q_ids))

    """ filtering member functions, always return self so that they can be chained """

    def filter(self, filt: schemas.Filter):
        """Parses `schemas.Filter` and operates all filters."""
        return self

    """ dataset filter """

    def filter_by_datasets(
        self, datasets: list[schemas.Dataset | models.Dataset]
    ):
        expressions = [
            models.Dataset.name == dataset.name
            for dataset in datasets
            if isinstance(dataset, schemas.Dataset | models.Dataset)
        ]
        self.add_expressions(graph.dataset, expressions)
        return self

    def filter_by_dataset_names(self, names: list[str]):
        expressions = [
            models.Dataset.name == name
            for name in names
            if isinstance(name, str)
        ]
        self.add_expressions(graph.dataset, expressions)
        return self

    """ model filter """

    def filter_by_models(self, models_: list[schemas.Model | models.Model]):
        expressions = [
            models.Model.name == model.name
            for model in models_
            if isinstance(model, schemas.Model | models.Model)
        ]
        self.add_expressions(graph.model, expressions)
        return self

    def filter_by_model_names(self, names: list[str]):
        expressions = [
            models.Model.name == name
            for name in names
            if isinstance(name, str)
        ]
        self.add_expressions(graph.model, expressions)
        return self

    """ datum filter """

    def filter_by_datums(self, datums: list[schemas.Datum | models.Datum]):
        expressions = [
            models.Datum.uid == datum.uid
            for datum in datums
            if isinstance(datum, schemas.Datum | models.Datum)
        ]
        self.add_expressions(graph.datum, expressions)
        return self

    def filter_by_datum_uids(self, uids: list[str]):
        expressions = [
            models.Datum.uid == uid for uid in uids if isinstance(uid, str)
        ]
        self.add_expressions(graph.datum, expressions)
        return self

    """ filter by label """

    # def filter_by_labels(
    #     self,
    #     labels: list[schemas.Label | models.Label],
    #     include_groundtruths: bool = True,
    #     include_predictions: bool = True,
    # ):
    #     # generate binary expressions
    #     expressions = [
    #         and_(
    #             models.Label.key == label.key,
    #             models.Label.value == label.value,
    #         )
    #         for label in labels
    #         if isinstance(label, schemas.Label | models.Label)
    #     ]
    #     self.add_expressions(???, expressions)
    #     return self

    # def filter_by_label_keys(
    #     self,
    #     label_keys: list[str],
    #     include_groundtruths: bool = True,
    #     include_predictions: bool = True,
    # ):
    #     # generate binary expressions
    #     expressions = [
    #         models.Label.key == label_key
    #         for label_key in label_keys
    #         if isinstance(label_key, str)
    #     ]
    #     if len(expressions) > 0:
    #         if include_groundtruths:
    #             self.filter_by.add(graph.label_groundtruth)
    #         elif include_predictions:
    #             self.filter_by.add(graph.label_prediction)
    #         else:
    #             raise ValueError(
    #                 "expected inclusion of groundtruths and/or prediction labels."
    #             )
    #         if len(expressions) == 1:
    #             self._filters.extend(expressions)
    #         else:
    #             self._filters.append(or_(*expressions))

    #     return self

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
    #     node: graph.Node,
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

    # def filter_by_task_types(self, task_types: list[enums.TaskType]):
    #     # generate binary expressions
    #     expressions = [
    #         models.Annotation.task_type == task_type.value
    #         for task_type in task_types
    #         if isinstance(task_type, enums.TaskType)
    #     ]
    #     self.add_expressions(gra)
    #     return self

    # def filter_by_annotation_types(
    #     self, annotation_types: list[enums.AnnotationType]
    # ):
    #     if enums.AnnotationType.NONE in annotation_types:
    #         self.filter_by.add(graph.annotation)
    #         self._filters.append(
    #             and_(
    #                 models.Annotation.box.is_(None),
    #                 models.Annotation.polygon.is_(None),
    #                 models.Annotation.multipolygon.is_(None),
    #                 models.Annotation.raster.is_(None),
    #             )
    #         )
    #     else:
    #         # collect binary expressions
    #         expressions = []
    #         if enums.AnnotationType.BOX in annotation_types:
    #             expressions.append(models.Annotation.box.isnot(None))
    #         if enums.AnnotationType.POLYGON in annotation_types:
    #             expressions.append(models.Annotation.polygon.isnot(None))
    #         if enums.AnnotationType.MULTIPOLYGON in annotation_types:
    #             expressions.append(models.Annotation.multipolygon.isnot(None))
    #         if enums.AnnotationType.RASTER in annotation_types:
    #             expressions.append(models.Annotation.raster.isnot(None))

    #         # generate joint filter
    #         if len(expressions) == 1:
    #             self.filter_by.add(graph.annotation)
    #             self._filters.extend(expressions)
    #         elif len(expressions) > 1:
    #             self.filter_by.add(graph.annotation)
    #             self._filters.append(or_(*expressions))

    #     return self


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
