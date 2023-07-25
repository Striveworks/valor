from sqlalchemy.orm import Session
from sqlalchemy import and_, or_, select
from sqlalchemy.sql.elements import BinaryExpression

from velour_api import schemas, enums
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
        "groundtruth": models.Annotation.id == models.GroundTruth.annotation_id,
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
    }
}


class Join:
    def __init__(self, lhs, rhs):
        self.lhs = lhs
        self.rhs = set([rhs])

    def or_(self, lhs: str, rhs: str):
        if self.lhs != lhs:
            raise ValueError
        if isinstance(rhs, str):
            self.rhs.add(rhs)
        else:
            self.rhs.update(rhs)

    def __str__(self):
        if len(self.rhs) == 1:
            return f"JOIN {self.lhs} ON {self.lhs} == {list(self.rhs)[0]}"
        ret = f"JOIN {self.lhs} ON \nOR(\n"
        for e in self.rhs:
            ret += f"  {self.lhs} == {e}"
            ret += "\n"
        ret += ")"
        return ret
    
    def relation(self) -> tuple:
        if len(self.rhs) == 1:
            pass
        else:
            return (
                model_mapping[self.lhs],
                or_(
                    [
                        model_relationships[self.lhs][rhs]
                        for rhs in self.rhs
                    ]
                )
            )


def _graph_generator(
    source: str,
    target: str,
    prune: set = set(),
):
    if source == target:
        return target

    remaining_options = model_graph[source] - prune
    prune.update(remaining_options, source)
    if target in prune:
        prune.remove(target)

    path = {}
    for option in list(remaining_options):
        if retval := _graph_generator(
            source=option, target=target, prune=prune.copy()
        ):
            path[option] = retval

    if path:
        return path
    return None


def _graph_interpreter(
    graph: dict, 
    source: str, 
    target: str, 
    output={}, 
    key_order=[]
) -> list[Join]:
    """ Recursive function """
    for key in graph:
        key_order.insert(0, key)
        if key in output:
            output[key].or_(lhs=key, rhs=source)
        else:
            output[key] = Join(lhs=key, rhs=source)
    for key in graph:
        if key != target:
            output, key_order = _graph_interpreter(
                graph[key],
                source=key,
                target=target,
                output=output,
                key_order=key_order,
            )
    return (output, key_order)


def _generate_joins(source, target):

        # Check for direct connection
        if target in model_graph[source]:
            graph = {target: target}
        else:   
            # Set of id's to prune
            prune = set()

            # Block metatypes if they are not referenced
            if "metadatum" not in [source, target]:
                prune.add("metadatum")
            if "label" not in [source, target]:
                prune.add("label")

            # Prune graph if model is source/target
            if "model" in [source, target]:
                prune.add("groundtruth")

            # Check input validity
            if source in prune or target in prune:
                return None

            graph = _graph_generator(source, target, prune=prune)

        # Generate object relationships
        return _graph_interpreter(
            graph, source=source, target=target
        )


def generate_query(source: str, targets: list[str]):
        graphs = [
            _generate_joins(source=source, target=target)
            for target in targets
        ]

        if not graphs:
            return None

        # Merge graphs
        master_graph = {}
        master_key_order = []
        existing_keys = set()
        for graph, key_order in graphs:
            for key in graph:
                if key not in master_graph:
                    master_graph[key] = graph[key]
                else:
                    master_graph[key].or_(graph[key].lhs, graph[key].rhs)
                
                if key not in existing_keys:
                    master_key_order.append(key)
                    existing_keys.add(key)

        # Validate order-of-operations
        retlist = []
        for key in master_key_order:
            retlist.append(master_graph[key])

        return (retlist)
    

class BackendQuery:

    def __init__(self, source: str):
        self._filters = []
        self.source = source
        self.targets = set()

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
    
    def __str__(self):

        # Get all rows of source table
        if len(self.targets) == 0:
            return f"SELECT FROM {self.source}"

        # sanity check
        if self.source in self.targets:
            self.targets.remove(self.source)

        qstruct = generate_query(source=self.source, targets=self.targets)

        ret = f"SELECT FROM {self.source}\n"
        for join in qstruct:
            ret += f"{str(join)}\n"
        ret += "WHERE\n"
        for filt in self._filters:
            ret += f"  {filt},\n"
        return ret 

    def ids(self, db: Session):
        """ Returns list of rows from source that meet filter criteria. """
        
        # sanity check
        if self.source in self.targets:
            self.targets.remove(self.source)
        
        # Get all rows of source table
        if len(self.targets) == 0:
            return select(model_mapping[self.source].id)

        # serialize request from graph
        qstruct = generate_query(self.source, self.targets)

        # select source
        src = model_mapping[self.source]
        q_ids = select(src.id)

        # join intermediate tables
        for join in qstruct:
            m, r = join.relation
            q_ids = q_ids.join(m, r)

        # add filter conditions
        q_ids.where(
            and_(*self._filters)
        )
        
        # return select statement of valid row ids
        return q_ids
    
    def all(self, db: Session):
        """ Returns sqlalchemy table rows """

        # get source object 
        src = model_mapping[self.source]

        # get valid row ids
        q_ids = self.ids(db)

        # return rows from source table
        return (
            db.query(model_mapping[self.source])
            .where(src.id.in_(q_ids))
            .all()
        )

    def filter(self, req: schemas.Filter):
        """ Parses `schemas.Filter` and operates all filters. """

        # generate filter expressions
        self.filter_by_dataset_names(req.filter_by_dataset_names)
        self.filter_by_model_names(req.filter_by_model_names)
        self.filter_by_datum_uids(req.filter_by_datum_uids)
        self.filter_by_task_types(req.filter_by_task_types)
        self.filter_by_annotation_types(req.filter_by_annotation_types)
        self.filter_by_label_keys(req.filter_by_label_keys)
        self.filter_by_labels(req.filter_by_labels)
        self.filter_by_metadata(req.filter_by_metadata)

        return self

    """ dataset filter """

    def filter_by_datasets(self, datasets: list[schemas.Dataset | models.Dataset]):
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
            models.Datum.uid == uid
            for uid in uids
            if isinstance(uid, str)
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

    def _create_metadatum_expression(self, metadatum: schemas.MetaDatum | models.MetaDatum):

        # Compare name
        expression = [models.MetaDatum.name == metadatum.name]

        # sqlalchemy handler
        if isinstance(metadatum, models.MetaDatum):
            expression.append(models.MetaDatum.string_value == metadatum.string_value)
            expression.append(models.MetaDatum.numeric_value == metadatum.numeric_value)
            expression.append(models.MetaDatum.geo == metadatum.geo)

        # schema handler
        elif isinstance(metadatum, schemas.MetaDatum):
            # Compare value
            if isinstance(metadatum.value, str):
                expression.append(models.MetaDatum.string_value == metadatum.value)
            if isinstance(metadatum.value, float):
                expression.append(models.MetaDatum.numeric_value == metadatum.value)
            if isinstance(metadatum.value, schemas.GeoJSON):
                raise NotImplementedError("Havent implemented GeoJSON support.")
        
        # unknown type
        else:
            return None

        return or_(*expression)

    def filter_by_metadata(self, metadata: list[schemas.MetaDatum | models.MetaDatum]):
        # generate binary expressions
        expressions = [
            self._create_metadatum_expression(metadatum)
            for metadatum in metadata
            if metadatum is not None
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

    def filter_by_annotation_types(self, annotation_types: list[enums.AnnotationType]):
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


# target = "metadatum"
# source = "annotation"
# output, keyorder = _generate_joins(source=source, target=target)

# print(f"select from {source}")
# for item in output:
#     print(item)