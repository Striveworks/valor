# import operator

from sqlalchemy import and_, or_, select
from sqlalchemy.sql.elements import BinaryExpression

from velour_api import enums, schemas
from velour_api.backend import graph, models


class Query:
    def __init__(self, target: graph.Node | None = None):
        self._filters: list[BinaryExpression] = []
        self.target = target
        self.filter_by = set()
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

    """ User methods """

    def filter(self, filt: schemas.Filter):
        """Parses `schemas.Filter` and operates all filters."""

        # datasets
        if filt.datasets is not None:
            if filt.datasets.names:
                self.filter_by_dataset_names(filt.datasets.names)
            if filt.datasets.metadata:
                self.filter_by_metadata(
                    filt.datasets.metadata, graph.metadatum_dataset
                )

        # models
        if filt.models is not None:
            if filt.models.names:
                self.filter_by_model_names(filt.models.names)
            if filt.models.metadata:
                self.filter_by_metadata(
                    filt.models.metadata, graph.metadatum_model
                )

        # datums
        if filt.datums is not None:
            if filt.datums.uids:
                self.filter_by_datum_uids(filt.datums.uids)
            if filt.datums.metadata:
                self.filter_by_metadata(
                    filt.datums.metadata, graph.metadatum_datum
                )

        # annotations
        if filt.annotations is not None:
            if filt.annotations.task_types:
                self.filter_by_task_types(filt.annotations.task_types)
            if filt.annotations.annotation_types:
                self.filter_by_annotation_types(
                    filt.annotations.annotation_types
                )
            if filt.annotations.min_area:
                pass
            if filt.annotations.max_area:
                pass
            if filt.annotations.metadata:
                self.filter_by_metadata(
                    filt.annotations.metadata, graph.metadatum_annotation
                )

        # groundtruths
        if filt.groundtruths is not None:
            pass

        # predictions
        if filt.predictions is not None:
            pass

        # toggles
        # @TODO
        # unsure of what to do with this one
        # maybe just make it a evaluate param
        # filt.annotations.allow_conversion

        # labels
        if filt.labels is not None:
            if filt.labels.labels:
                self.filter_by_labels(filt.labels.labels)
            if filt.labels.keys:
                self.filter_by_label_keys(filt.labels.label_keys)

        # toggles
        if not filt.labels.include_groundtruths:
            self.constraints.add("groundtruth")
        if not filt.labels.include_predictions:
            self.constraints.add("prediction")

        return self

    def ids(self):
        """Returns ids from target that meet filter criteria."""
        id_query = graph.generate_query(self.target, self.filter_by)
        id_query = id_query.where(and_(*self._filters))
        return id_query

    def query(self):
        """Returns sqlalchemy table rows"""
        q_ids = self.ids()
        return select(self.target.model).where(self.target.model.id.in_(q_ids))

    """ dataset filter """

    def filter_by_datasets(
        self, datasets: list[schemas.Dataset | models.Dataset]
    ):
        # generate binary expressions
        expressions = [
            models.Dataset.name == dataset.name
            for dataset in datasets
            if isinstance(dataset, schemas.Dataset | models.Dataset)
        ]

        # generate filter
        if len(expressions) == 1:
            self.filter_by.add(graph.dataset)
            self._filters.extend(expressions)
        elif len(expressions) > 1:
            self.filter_by.add(graph.dataset)
            self._filters.append(or_(*expressions))

        return self

    def filter_by_dataset_names(self, names: list[str]):
        # generate binary expressions
        expressions = [
            models.Dataset.name == name
            for name in names
            if isinstance(name, str)
        ]

        # generate filter
        if len(expressions) == 1:
            self.filter_by.add(graph.dataset)
            self._filters.extend(expressions)
        elif len(expressions) > 1:
            self.filter_by.add(graph.dataset)
            self._filters.append(or_(*expressions))

        return self

    """ model filter """

    def filter_by_models(self, models_: list[schemas.Model | models.Model]):
        # generate binary expressions
        expressions = [
            models.Model.name == model.name
            for model in models_
            if isinstance(model, schemas.Model | models.Model)
        ]

        # generate filter
        if len(expressions) == 1:
            self.filter_by.add(graph.model)
            self._filters.extend(expressions)
        elif len(expressions) > 1:
            self.filter_by.add(graph.model)
            self._filters.append(or_(*expressions))

        return self

    def filter_by_model_names(self, names: list[str]):
        # generate binary expressions
        expressions = [
            models.Model.name == name
            for name in names
            if isinstance(name, str)
        ]

        # generate filter
        if len(expressions) == 1:
            self.filter_by.add(graph.model)
            self._filters.extend(expressions)
        elif len(expressions) > 1:
            self.filter_by.add(graph.model)
            self._filters.append(or_(*expressions))

        return self

    """ datum filter """

    def filter_by_datums(self, datums: list[schemas.Datum | models.Datum]):
        # generate binary expressions
        expressions = [
            models.Datum.uid == datum.uid
            for datum in datums
            if isinstance(datum, schemas.Datum | models.Datum)
        ]

        # generate filter
        if len(expressions) == 1:
            self.filter_by.add(graph.datum)
            self._filters.extend(expressions)
        elif len(expressions) > 1:
            self.filter_by.add(graph.datum)
            self._filters.append(or_(*expressions))

        return self

    def filter_by_datum_uids(self, uids: list[str]):
        # generate binary expressions
        expressions = [
            models.Datum.uid == uid for uid in uids if isinstance(uid, str)
        ]

        # generate filter
        if len(expressions) == 1:
            self.filter_by.add(graph.datum)
            self._filters.extend(expressions)
        elif len(expressions) > 1:
            self.filter_by.add(graph.datum)
            self._filters.append(or_(*expressions))

        return self

    """ filter by label """

    def filter_by_labels(
        self,
        labels: list[schemas.Label | models.Label],
        include_groundtruths: bool = True,
        include_predictions: bool = True,
    ):
        # generate binary expressions
        expressions = [
            and_(
                models.Label.key == label.key,
                models.Label.value == label.value,
            )
            for label in labels
            if isinstance(label, schemas.Label | models.Label)
        ]
        if len(expressions) > 0:
            if include_groundtruths:
                self.filter_by.add(graph.label_groundtruth)
            elif include_predictions:
                self.filter_by.add(graph.label_prediction)
            else:
                raise ValueError(
                    "expected inclusion of groundtruths and/or prediction labels."
                )
            if len(expressions) == 1:
                self._filters.extend(expressions)
            else:
                self._filters.append(or_(*expressions))

        return self

    def filter_by_label_keys(
        self,
        label_keys: list[str],
        include_groundtruths: bool = True,
        include_predictions: bool = True,
    ):
        # generate binary expressions
        expressions = [
            models.Label.key == label_key
            for label_key in label_keys
            if isinstance(label_key, str)
        ]
        if len(expressions) > 0:
            if include_groundtruths:
                self.filter_by.add(graph.label_groundtruth)
            elif include_predictions:
                self.filter_by.add(graph.label_prediction)
            else:
                raise ValueError(
                    "expected inclusion of groundtruths and/or prediction labels."
                )
            if len(expressions) == 1:
                self._filters.extend(expressions)
            else:
                self._filters.append(or_(*expressions))

        return self

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

    def filter_by_task_types(self, task_types: list[enums.TaskType]):
        # generate binary expressions
        expressions = [
            models.Annotation.task_type == task_type.value
            for task_type in task_types
            if isinstance(task_type, enums.TaskType)
        ]

        # generate filter
        if len(expressions) == 1:
            self.filter_by.add(graph.annotation)
            self._filters.extend(expressions)
        elif len(expressions) > 1:
            self.filter_by.add(graph.annotation)
            self._filters.append(or_(*expressions))

        return self

    def filter_by_annotation_types(
        self, annotation_types: list[enums.AnnotationType]
    ):
        if enums.AnnotationType.NONE in annotation_types:
            self.filter_by.add(graph.annotation)
            self._filters.append(
                and_(
                    models.Annotation.box.is_(None),
                    models.Annotation.polygon.is_(None),
                    models.Annotation.multipolygon.is_(None),
                    models.Annotation.raster.is_(None),
                )
            )
        else:
            # collect binary expressions
            expressions = []
            if enums.AnnotationType.BOX in annotation_types:
                expressions.append(models.Annotation.box.isnot(None))
            if enums.AnnotationType.POLYGON in annotation_types:
                expressions.append(models.Annotation.polygon.isnot(None))
            if enums.AnnotationType.MULTIPOLYGON in annotation_types:
                expressions.append(models.Annotation.multipolygon.isnot(None))
            if enums.AnnotationType.RASTER in annotation_types:
                expressions.append(models.Annotation.raster.isnot(None))

            # generate joint filter
            if len(expressions) == 1:
                self.filter_by.add(graph.annotation)
                self._filters.extend(expressions)
            elif len(expressions) > 1:
                self.filter_by.add(graph.annotation)
                self._filters.append(or_(*expressions))

        return self


if __name__ == "__main__":
    db = None
    query = (
        Query(graph.datum)
        .filter_by_dataset_names(["dataset1"])
        .filter_by_model_names(["model1"])
        .filter_by_datum_uids(["uid1"])
        .filter_by_labels([schemas.Label(key="k1", value="v1")])
        .filter_by_labels(
            [
                schemas.Label(key="k2", value="v2"),
                schemas.Label(key="k2", value="v3"),
                schemas.Label(key="k2", value="v4"),
            ]
        )
        .filter_by_label_keys(["k4"])
        .filter_by_task_types([enums.TaskType.CLASSIFICATION])
        .query()
    )
    print(str(query))

    # filter_by = ["annotation", "model"]
    # target = "label"

    # output = generate_query(target, filter_by)
    # print(f"SELECT FROM {target}")
    # for o in output:
    # print(o)
