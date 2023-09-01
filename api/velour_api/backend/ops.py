import operator

from sqlalchemy import and_, or_, select
from sqlalchemy.orm import Session
from sqlalchemy.sql.elements import BinaryExpression

from velour_api import enums, schemas
from velour_api.backend import models

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


model_mapping = {
    "dataset": models.Dataset,
    "model": models.Model,
    "datum": models.Datum,
    "annotation": models.Annotation,
    "groundtruth": models.GroundTruth,
    "prediction": models.Prediction,
    "label": models.Label,
    "metadatum": models.MetaDatum,
}


model_relationships = {
    "dataset": {
        "datum": models.Dataset.id == models.Datum.dataset_id,
        "metadatum": models.Dataset.id == models.MetaDatum.dataset_id,
    },
    "model": {
        "annotation": models.Model.id == models.Annotation.model_id,
        "metadatum": models.Model.id == models.MetaDatum.model_id,
    },
    "datum": {
        "annotation": models.Datum.id == models.Annotation.datum_id,
        "dataset": models.Datum.dataset_id == models.Dataset.id,
        "metadatum": models.Datum.id == models.MetaDatum.datum_id,
    },
    "annotation": {
        "datum": models.Annotation.datum_id == models.Datum.id,
        "model": models.Annotation.model_id == models.Model.id,
        "prediction": models.Annotation.id == models.Prediction.annotation_id,
        "groundtruth": models.Annotation.id
        == models.GroundTruth.annotation_id,
        "metadatum": models.Annotation.id == models.MetaDatum.annotation_id,
    },
    "groundtruth": {
        "annotation": models.GroundTruth.annotation_id == models.Annotation.id,
        "label": models.GroundTruth.label_id == models.Label.id,
    },
    "prediction": {
        "annotation": models.Prediction.annotation_id == models.Annotation.id,
        "label": models.Prediction.label_id == models.Label.id,
    },
    "metadatum": {
        "dataset": models.MetaDatum.dataset_id == models.Dataset.id,
        "model": models.MetaDatum.model_id == models.Model.id,
        "datum": models.MetaDatum.datum_id == models.Datum.id,
        "annotation": models.MetaDatum.annotation_id == models.Annotation.id,
    },
}


schemas_mapping = {
    "dataset": schemas.Dataset,
    "model": schemas.Model,
    "datum": schemas.Datum,
    "annotation": schemas.Annotation,
    "groundtruth": schemas.GroundTruth,
    "prediction": schemas.Prediction,
    "label": schemas.Label,
    "metadatum": schemas.MetaDatum,
}


class Join:
    def __init__(self, root: str, link: str):
        self.root = root
        self.links = set([link])

    def or_(self, root: str, link: str):
        if self.root != root:
            raise ValueError
        if isinstance(link, str):
            self.links.add(link)
        else:
            self.links.update(link)

    def __str__(self):
        if len(self.links) == 1:
            return f"JOIN {self.root} ON {self.root} == {list(self.links)[0]}"
        ret = f"JOIN {self.root} ON \nOR(\n"
        for e in self.links:
            ret += f"  {self.root} == {e}"
            ret += "\n"
        ret += ")"
        return ret

    def relation(self) -> tuple:
        # generate binary expressions
        expressions = [
            model_relationships[self.root][link] for link in self.links
        ]

        if len(expressions) == 1:
            return (
                model_mapping[self.root],
                expressions[0],
            )
        else:
            return (
                model_mapping[self.root],
                or_(*expressions),
            )


def _graph_generator(
    start: str,
    end: str,
    invalid: set = None,
):
    """Recursive function,"""

    # initialize
    if invalid is None:
        invalid = set()

    # check if source or target invalid
    if start in invalid or end in invalid:
        return None

    # check if source is target
    if start == end:
        return end

    # get options for next step
    remaining_options = model_graph[start] - invalid

    # add current position to list of traveled nodes
    invalid.add(start)

    # check & remote
    # if end in invalid:
    # invalid.remove(end)

    # iterate through branch options, construct path recursively
    path = {}
    for option in list(remaining_options):
        if retval := _graph_generator(start=option, end=end, invalid=invalid):
            path[option] = retval
    return path if path else None


def _flatten_graph(
    graph: dict,
    start: str,
    end: str,
    join_list: dict,
) -> list[Join]:
    """Recursive function"""

    # update from current layer
    for key in graph:
        if key in join_list:
            join_list[key].or_(root=key, link=start)
        else:
            join_list[key] = Join(root=key, link=start)

    # recurse to next layer
    for key in graph:
        if key not in end:
            join_list = _flatten_graph(
                graph=graph[key],
                start=key,
                end=end,
                join_list=join_list,
            )

    return join_list


def _generate_joins(source, targets, prune):
    # generate graphs
    graphs_with_target = [
        (_graph_generator(source, target, prune=prune.copy()), target)
        if target not in model_graph[source]  # check for direct connection
        else ({target: target}, target)
        for target in targets
    ]

    # Generate object relationships
    return [
        _flatten_graph(
            graph=graph,
            source=source,
            target=target,
            join_list=dict(),
        )
        for graph, target in graphs_with_target
        if graph is not None
    ]


def generate_query(source: str, targets: list[str], prune: set[str]):

    # Generate graphs
    graphs = _generate_joins(source=source, targets=targets, prune=prune)

    # edge case
    if not graphs:
        return None

    # Merge graphs
    master_graph = {}
    existing_keys = set()
    for graph in graphs:
        for key in graph:
            existing_keys.add(key)
            if key not in master_graph:
                master_graph[key] = graph[key]
            else:
                master_graph[key].or_(graph[key].root, graph[key].links)

    # Validate order-of-operations
    retlist = []
    sources = set([source])
    while len(existing_keys) > 0:
        for key in existing_keys:
            if master_graph[key].links.issubset(sources):
                retlist.append(master_graph[key])
                sources.add(key)
        existing_keys = existing_keys - sources

    return retlist


class BackendQuery:
    def __init__(self, source: str):
        self._filters = []
        self.source = source
        self.targets = set()
        self.constraints = set()

    @classmethod
    def model(cls):
        return cls("model")

    @classmethod
    def dataset(cls):
        return cls("dataset")

    @classmethod
    def datum(cls):
        return cls("datum")

    @classmethod
    def annotation(cls):
        return cls("annotation")

    @classmethod
    def groundtruth(cls):
        return cls("groundtruth")

    @classmethod
    def prediction(cls):
        return cls("prediction")

    @classmethod
    def label(cls):
        return cls("label")

    @classmethod
    def metadatum(cls):
        return cls("metadatum")

    @property
    def filters(self) -> list[BinaryExpression]:
        return self._filters

    def prune_graph(self) -> set[str]:
        # Set of id's to prune
        invalid = self.constraints.copy()

        # invalidate metadatum node if not referenced
        if "metadatum" not in [self.source, *self.targets]:
            invalid.add("metadatum")

        # invalidate label node if not referenced
        if "label" not in [self.source, *self.targets]:
            invalid.add("label")

        # invalidate groundtruth node if model is source/target
        if "model" in [self.source, *self.targets]:
            invalid.add("groundtruth")

        # validate source
        if self.source in invalid:
            return None

        # validate targets
        self.targets = list(set(self.targets) - invalid)

        return invalid

    def __str__(self):

        # Get all rows of source table
        if len(self.targets) == 0:
            return f"SELECT FROM {self.source}"

        # sanity check
        if self.source in self.targets:
            self.targets.remove(self.source)

        qstruct = generate_query(
            source=self.source, targets=self.targets, prune=self.prune_graph()
        )

        ret = f"SELECT FROM {self.source}\n"
        for join in qstruct:
            ret += f"{str(join)}\n"
        ret += "WHERE\n"
        for filt in self._filters:
            ret += f"  {filt},\n"
        return ret

    def query_ids(self, db: Session):
        """Returns list of rows from source that meet filter criteria."""

        # sanity check
        if self.source in self.targets:
            self.targets.remove(self.source)

        # Get all rows of source table
        if len(self.targets) == 0:
            return select(model_mapping[self.source].id)

        # serialize request from graph
        qstruct = generate_query(
            source=self.source, targets=self.targets, prune=self.prune_graph()
        )

        # select source
        src = model_mapping[self.source]
        q_ids = select(src.id)

        # join intermediate tables
        for join in qstruct:
            m, r = join.relation()
            q_ids = q_ids.join(m, r)

        # add filter conditions
        q_ids = q_ids.where(and_(*self._filters))

        # return select statement of valid row ids
        return q_ids

    def query(self, db: Session):
        """Returns sqlalchemy table rows"""

        # get source object
        src = model_mapping[self.source]

        # get valid row ids
        q_ids = self.ids(db)

        # return rows from source table
        return (
            db.query(model_mapping[self.source]).where(src.id.in_(q_ids)).all()
        )

    def filter(self, filt: schemas.Filter):
        """Parses `schemas.Filter` and operates all filters."""

        # datasets
        if filt.datasets is not None:
            if filt.datasets.names:
                self.filter_by_dataset_names(filt.datasets.names)
            if filt.datasets.metadata:
                self.filter_by_metadata(filt.datasets.metadata)

        # models
        if filt.models is not None:
            if filt.models.names:
                self.filter_by_model_names(filt.models.names)
            if filt.models.metadata:
                self.filter_by_metadata(filt.models.metadata)

        # datums
        if filt.datums is not None:
            if filt.datums.uids:
                self.filter_by_datum_uids(filt.datums.uids)
            if filt.datums.metadata:
                self.filter_by_metadata(filt.datums.metadata)

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
                self.filter_by_metadata(filt.annotations.metadata)

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
            self.targets.add("dataset")
            self._filters.extend(expressions)
        elif len(expressions) > 1:
            self.targets.add("dataset")
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
            self.targets.add("dataset")
            self._filters.extend(expressions)
        elif len(expressions) > 1:
            self.targets.add("dataset")
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
            self.targets.add("model")
            self._filters.extend(expressions)
        elif len(expressions) > 1:
            self.targets.add("model")
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
            self.targets.add("model")
            self._filters.extend(expressions)
        elif len(expressions) > 1:
            self.targets.add("model")
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
            self.targets.add("datum")
            self._filters.extend(expressions)
        elif len(expressions) > 1:
            self.targets.add("datum")
            self._filters.append(or_(*expressions))

        return self

    def filter_by_datum_uids(self, uids: list[str]):
        # generate binary expressions
        expressions = [
            models.Datum.uid == uid for uid in uids if isinstance(uid, str)
        ]

        # generate filter
        if len(expressions) == 1:
            self.targets.add("datum")
            self._filters.extend(expressions)
        elif len(expressions) > 1:
            self.targets.add("datum")
            self._filters.append(or_(*expressions))

        return self

    """ filter by label """

    def filter_by_labels(self, labels: list[schemas.Label | models.Label]):
        # generate binary expressions
        expressions = [
            and_(
                models.Label.key == label.key,
                models.Label.value == label.value,
            )
            for label in labels
            if isinstance(label, schemas.Label | models.Label)
        ]

        # generate filter
        if len(expressions) == 1:
            self.targets.add("label")
            self._filters.extend(expressions)
        elif len(expressions) > 1:
            self.targets.add("label")
            self._filters.append(or_(*expressions))

        return self

    def filter_by_label_keys(
        self,
        label_keys: list[str],
    ):
        # generate binary expressions
        expressions = [
            models.Label.key == label_key
            for label_key in label_keys
            if isinstance(label_key, str)
        ]

        # generate filter
        if len(expressions) == 1:
            self.targets.add("label")
            self._filters.extend(expressions)
        elif len(expressions) > 1:
            self.targets.add("label")
            self._filters.append(or_(*expressions))

        return self

    """ filter by metadata """

    def filter_by_metadatum(
        self,
        metadatum: schemas.MetaDatum | models.MetaDatum,
        operator: str,
    ):

        ops = {
            ">": operator.gt,
            "<": operator.lt,
            ">=": operator.ge,
            "<=": operator.le,
            "==": operator.eq,
            "!=": operator.ne,
        }
        if operator not in ops:
            raise ValueError(f"invalid comparison operator `{operator}`")

        # Compare name
        expression = [models.MetaDatum.name == metadatum.name]

        # sqlalchemy handler
        if isinstance(metadatum, models.MetaDatum):
            expression.append(
                ops[operator](
                    models.MetaDatum.string_value, metadatum.string_value
                )
            )
            expression.append(
                ops[operator](
                    models.MetaDatum.numeric_value, metadatum.numeric_value
                )
            )
            expression.append(
                ops[operator](models.MetaDatum.geo, metadatum.geo)
            )

        # schema handler
        elif isinstance(metadatum, schemas.MetaDatum):
            # Compare value
            if isinstance(metadatum.value, str):
                expression.append(
                    ops[operator](
                        models.MetaDatum.string_value, metadatum.value
                    )
                )
            if isinstance(metadatum.value, float):
                expression.append(
                    ops[operator](
                        models.MetaDatum.numeric_value, metadatum.value
                    )
                )
            if isinstance(metadatum.value, schemas.GeoJSON):
                raise NotImplementedError("GeoJSON currently unsupported.")

        # unknown type
        else:
            return None

        return or_(*expression)

    def filter_by_metadata(self, metadata: list[schemas.MetadataFilter]):
        # generate binary expressions
        expressions = [
            self.filter_by_metadatum(filt.metadatum, filt.operator)
            for filt in metadata
        ]

        # generate filter
        if len(expressions) == 1:
            self.targets.add("metadatum")
            self._filters.extend(expressions)
        elif len(expressions) > 1:
            self.targets.add("metadatum")
            self._filters.append(or_(*expressions))

        return self

    def filter_by_metadatum_names(self, names: list[str]):
        # generate binary expressions
        expressions = [
            models.MetaDatum.name == name
            for name in names
            if isinstance(name, str)
        ]

        # generate filter
        if len(expressions) == 1:
            self.targets.add("metadatum")
            self._filters.extend(expressions)
        elif len(expressions) > 1:
            self.targets.add("metadatum")
            self._filters.append(or_(*expressions))

        return self

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
            self.targets.add("annotation")
            self._filters.extend(expressions)
        elif len(expressions) > 1:
            self.targets.add("annotation")
            self._filters.append(or_(*expressions))

        return self

    def filter_by_annotation_types(
        self, annotation_types: list[enums.AnnotationType]
    ):
        if enums.AnnotationType.NONE in annotation_types:
            self.targets.add("annotation")
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
                self.targets.add("annotation")
                self._filters.extend(expressions)
            elif len(expressions) > 1:
                self.targets.add("annotation")
                self._filters.append(or_(*expressions))

        return self


# db = None
# query = (
#     BackendQuery.datum()
#     .filter_by_dataset_name("dataset1")
#     .filter_by_model_name("model1")
#     .filter_by_datum_uid("uid1")
#     .filter_by_label(schemas.Label(key="k1", value="v1"))
#     .filter_by_labels(
#         [
#             schemas.Label(key="k2", value="v2"),
#             schemas.Label(key="k2", value="v3"),
#             schemas.Label(key="k2", value="v4"),
#         ]
#     )
#     .filter_by_label_key("k4")
#     .filter_by_metadata(
#         [
#             schemas.MetaDatum(name="n1", value=0.5),
#             schemas.MetaDatum(name="n2", value=0.1),
#         ]
#     )
#     .filter_by_metadatum(schemas.MetaDatum(name="n3", value="v1"))
#     .filter_by_metadatum_name("n4")
#     .filter_by_task_type(enums.TaskType.CLASSIFICATION)
#     .query(db)
# )
# print(str(query))


# targets = ["annotation", "model"]
# source = "label"

# output = generate_query(source, targets)
# print(f"SELECT FROM {source}")
# for o in output:
#     print(o)
