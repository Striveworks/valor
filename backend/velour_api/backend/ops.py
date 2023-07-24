# from sqlalchemy.orm import Session
# from sqlalchemy import and_, or_
# from sqlalchemy.sql.elements import BinaryExpression

# from velour_api import schemas, enums
# from velour_api.backend import models


model_graph = {
    "dataset": {"datum", "metadatum"},
    "model": {"annotation", "metadatum"},
    "datum": {"annotation", "dataset", "metadatum"},
    "annotation": {"datum", "model", "prediction", "groundtruth", "metadatum"},
    "groundtruth": {"annotation", "label"},
    "prediction": {"annotation", "label"},
    "label": {"prediction", "groundtruth"},
    "metadatum": {"dataset", "model", "datum", "annotation"},
}

# model_mapping = {
#     "dataset": models.Dataset,
#     "model": models.Model,
#     "datum": models.Datum,
#     "annotation": models.Annotation,
#     "groundtruth": models.GroundTruth,
#     "prediction": models.Prediction,
#     "label": models.Label,
#     "metadatum": models.MetaDatum,
# }

# model_relationships = {
#     "dataset": {
#         "datum": models.Dataset.id == models.Datum.dataset_id,
#         "metadatum": models.Dataset.id == models.MetaDatum.dataset_id,
#     },
#     "model": {
#         "annotation": models.Model.id == models.Annotation.model_id,
#         "metadatum": models.Model.id == models.MetaDatum.model_id,
#     },
#     "datum": {
#         "annotation": models.Datum.id == models.Annotation.datum_id,
#         "dataset": models.Datum.dataset_id == models.Dataset.id,
#         "metadatum": models.Datum.id == models.MetaDatum.datum_id,
#     },
#     "annotation": {
#         "datum": models.Annotation.datum_id == models.Datum.id,
#         "model": models.Annotation.model_id == models.Model.id,
#         "prediction": models.Annotation.id == models.Prediction.annotation_id,
#         "groundtruth": models.Annotation.id == models.GroundTruth.annotation_id,
#         "metadatum": models.Annotation.id == models.MetaDatum.annotation_id,
#     },
#     "groundtruth": {
#         "annotation": models.GroundTruth.annotation_id == models.Annotation.id,
#         "label": models.GroundTruth.label_id == models.Label.id,
#     },
#     "prediction": {
#         "annotation": models.Prediction.annotation_id == models.Annotation.id,
#         "label": models.Prediction.label_id == models.Label.id,
#     },
#     "metadatum": {
#         "dataset": models.MetaDatum.dataset_id == models.Dataset.id,
#         "model": models.MetaDatum.model_id == models.Model.id,
#         "datum": models.MetaDatum.datum_id == models.Datum.id,
#         "annotation": models.MetaDatum.annotation_id == models.Annotation.id,
#     }
# }


def graph_search(
    source: str,
    target: str,
    blacklist: set = set(),
):
    if source == target:
        return target

    remaining_options = model_graph[source] - blacklist
    blacklist.update(remaining_options, source)
    if target in blacklist:
        blacklist.remove(target)

    path = {}
    for option in list(remaining_options):
        if retval := graph_search(
            source=option, target=target, blacklist=blacklist.copy()
        ):
            path[option] = retval

    if path:
        return path
    return None


def create_graph(source, target):
    blacklist = set()

    # Block metatypes if they are not referenced
    if "metadatum" not in [source, target]:
        blacklist.add("metadatum")
    if "label" not in [source, target]:
        blacklist.add("label")

    # Adjust graph if model is source/target
    if "model" in [source, target]:
        blacklist.add("groundtruth")

    # Check input validity
    if source in blacklist or target in blacklist:
        return None

    # Metadatum edge cases
    if source == "metadatum":
        if target in ["dataset", "model", "datum", "annotation"]:
            return {target: target}
        return {
            "dataset": graph_search("dataset", target, blacklist=blacklist),
            "model": graph_search("model", target, blacklist=blacklist),
            "datum": graph_search("datum", target, blacklist=blacklist),
            "annotation": graph_search(
                "annotation", target, blacklist=blacklist
            ),
        }
    elif target == "metadatum":
        if source in ["dataset", "model", "datum", "annotation"]:
            return {target: target}

    return graph_search(source, target, blacklist=blacklist)


class Join:
    def __init__(self, table, expression):
        self.table = table
        self.expressions = [expression]

    def or_(self, expression):
        if expression not in self.expressions:
            self.expressions.append(expression)

    def __str__(self):
        if len(self.expressions) == 1:
            return f"JOIN {self.table} ON {self.expressions[0]}"
        ret = f"JOIN {self.table} ON \nOR(\n"
        for e in self.expressions:
            ret += e
            ret += "\n"
        ret += ")"
        return ret


def create_join_structure(
    graph: dict, source: str, target: str, output={}, key_order=[]
) -> list[Join]:
    for key in graph:
        key_order.append(key)
        if key in output:
            output[key].or_(f"{key} == {source}")
        else:
            output[key] = Join(table=key, expression=f"{key} == {source}")
    for key in graph:
        if key != target:
            output, key_order = create_join_structure(
                graph[key],
                source=key,
                target=target,
                output=output,
                key_order=key_order,
            )
    return (output, key_order)


target = "label"
source = "datum"
graph = create_graph(source=source, target=target)
(output, key_order) = create_join_structure(
    graph, source=source, target=target
)


print(f"select from {source}")
for j in output:
    print(output[j])

# class QueryFilter:

#     def __init__(self, target):
#         self._target = target
#         self._filters = []
#         self._relationships = {
#             "dataset": {"datum", "metadatum"},
#             "model": {"prediction", "metadatum"},
#             "datum": {"dataset", }
#         }


#     @property
#     def filters(self) -> list[BinaryExpression]:
#         return self._filters

#     def filter_by_label(
#         label: schemas.Label | models.Label,
#     ):
#         if
#         return and_(
#             models.Label.key == label.key,
#             models.Label.value == label.value,
#         )

#     def filter_by_label_key(
#         label_key: str,
#     ):
#         return models.Label.key == label_key

#     def filter_by_metadatum(
#         metadatum: schemas.MetaDatum | models.MetaDatum
#     ):
#         if isinstance(schemas.MetaDatum):
#             pass
#         elif isinstance(models.MetaDatum):
#             pass

#     def filter(self, expressions: BinaryExpression):
#         if not isinstance(expressions, list) and isinstance(expressions, BinaryExpression):
#             self._filters.append(expressions)
#         else:
#             self._filters.extend(
#                 [
#                     expression
#                     for expression in expressions
#                     if isinstance(expression, BinaryExpression)
#                 ]
#             )
#         return self

#     """ `str` identifiers """

#     def filter_by_str(self, target_property, strings: str | list[str]):
#         if isinstance(strings, str):
#             self._filters.append(target_property == strings)
#         self._filters.extend(
#             [
#                 target_property == string
#                 for string in strings
#                 if isinstance(string, str)
#             ]
#         )
#         return self

#     """ `velour_api.backend.models` identifiers """

#     def _filter_by_id(self, target: Base, source: Base):
#         if type(target) is type(source):
#             self._filter(target.id == source.id)
#         elif isinstance(source, models.Dataset):
#             self._filters.append(target.dataset_id == source.id)
#         elif isinstance(source, models.Model):
#             self._filters.append(target.model_id == source.id)
#         elif isinstance(source, models.Datum):
#             self._filters.append(target.datum_id == source.id)
#         elif isinstance(source, models.Annotation):
#             self._filters.append(target.annotation_id == source.id)
#         elif isinstance(source, models.Label):
#             self._filters.append(target.label_id == source.id)
#         else:
#             raise NotImplementedError

#     def filter_by_id(
#         self,
#         target: Base,
#         sources: Base | list[Base],
#     ):
#         if not isinstance(sources, list) and issubclass(sources, Base):
#             self._filter_by_id(target, sources)
#         else:
#             self._filters.extend(
#                 [
#                     self._filter_by_id(target, source)
#                     for source in sources
#                     if not issubclass(source, Base)
#                 ]
#             )
#         return self

#     """ `velour_api.schemas` identifiers """

#     def filter_by_labels(self, labels: list[schemas.Label]):
#         self._filters.extend(
#             [
#                 and_(
#                     models.Label.key == label.key,
#                     models.Label.value == label.value,
#                 )
#                 for label in labels
#                 if isinstance(label, schemas.Label)
#             ]
#         )
#         return self

#     def _filter_by_metadatum(self, metadatum: schemas.MetaDatum):

#         # Compare name
#         comparison = [models.MetaDatum.name == metadatum.name]

#         # Compare value
#         if isinstance(metadatum.value, str):
#             comparison.append(models.MetaDatum.value == metadatum.value)
#         if isinstance(metadatum.value, float):
#             comparison.append(models.MetaDatum.value == metadatum.value)
#         if isinstance(metadatum.value, schemas.GeoJSON):
#             raise NotImplementedError("Havent implemented GeoJSON support.")

#         return comparison

#     def filter_by_metadata(self, metadata: list[schemas.MetaDatum]):
#         self._filters.extend(
#             [
#                 self._filter_by_metadatum(metadatum)
#                 for metadatum in metadata
#                 if isinstance(metadatum, schemas.MetaDatum)
#             ]
#         )
#         return self

#     """ `velour_api.enums` identifiers """

#     def filter_by_task_type(self, task_type: enums.TaskType):
#         self._filters.extend(
#             [
#                 models.Annotation.task_type == task_type.value
#                 for task_type in task_types
#                 if isinstance(task_type, enums.TaskType)
#             ]
#         )
#         return self

#     def filter_by_annotation_types(self, annotation_types: list[enums.AnnotationType]):
#         if enums.AnnotationType.NONE in annotation_types:
#             self._filters.extend(
#                 [
#                     models.Annotation.box.is_(None),
#                     models.Annotation.polygon.is_(None),
#                     models.Annotation.multipolygon.is_(None),
#                     models.Annotation.raster.is_(None),
#                 ]
#             )
#         else:
#             if enums.AnnotationType.BOX in annotation_types:
#                 self._filters.append(models.Annotation.box.isnot(None))
#             if enums.AnnotationType.POLYGON in annotation_types:
#                 self._filters.append(models.Annotation.polygon.isnot(None))
#             if enums.AnnotationType.MULTIPOLYGON in annotation_types:
#                 self._filters.append(models.Annotation.multipolygon.isnot(None))
#             if enums.AnnotationType.RASTER in annotation_types:
#                 self._filters.append(models.Annotation.raster.isnot(None))
#         return self
