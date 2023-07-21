from sqlalchemy.orm import Session
from sqlalchemy import and_, or_
from sqlalchemy.sql.elements import BinaryExpression

from velour_api import schemas, enums
from velour_api.backend import models, core
from velour_api.backend.database import Base


class QueryFilter:

    def __init__(self, target: Base = None):
        self._target = target
        self._filters = []
        self._relationships = [
            {sets}           
        ]

    @property
    def filters(self) -> list[BinaryExpression]:
        return self._filters
    
    def join(self, subtarget: Base):

    def filter(self, expressions: BinaryExpression):
        if not isinstance(expressions, list) and isinstance(expressions, BinaryExpression):
            self._filters.append(expressions)
        else:
            self._filters.extend(
                [
                    expression
                    for expression in expressions
                    if isinstance(expression, BinaryExpression)
                ]
            )
        return self

    """ `str` identifiers """

    def filter_by_str(self, target_property, strings: str | list[str]):
        if isinstance(strings, str):
            self._filters.append(target_property == strings)
        self._filters.extend(
            [
                target_property == string
                for string in strings
                if isinstance(string, str)
            ]
        )
        return self
    
    """ `velour_api.backend.models` identifiers """

    def _filter_by_id(self, target: Base, source: Base):
        if type(target) is type(source):
            self._filter(target.id == source.id)
        elif isinstance(source, models.Dataset):
            self._filters.append(target.dataset_id == source.id)
        elif isinstance(source, models.Model):
            self._filters.append(target.model_id == source.id)
        elif isinstance(source, models.Datum):
            self._filters.append(target.datum_id == source.id)
        elif isinstance(source, models.Annotation):
            self._filters.append(target.annotation_id == source.id)
        elif isinstance(source, models.Label):
            self._filters.append(target.label_id == source.id)
        else:
            raise NotImplementedError

    def filter_by_id(
        self, 
        target: Base,
        sources: Base | list[Base],
    ):
        if not isinstance(sources, list) and issubclass(sources, Base):
            self._filter_by_id(target, sources)
        else:
            self._filters.extend(
                [
                    self._filter_by_id(target, source)
                    for source in sources
                    if not issubclass(source, Base)
                ]
            )
        return self

    """ `velour_api.schemas` identifiers """

    def filter_by_labels(self, labels: list[schemas.Label]):
        self._filters.extend(
            [
                and_(
                    models.Label.key == label.key,
                    models.Label.value == label.value,
                )
                for label in labels
                if isinstance(label, schemas.Label)
            ]
        )
        return self

    def _filter_by_metadatum(self, metadatum: schemas.MetaDatum):

        # Compare name
        comparison = [models.MetaDatum.name == metadatum.name]

        # Compare value
        if isinstance(metadatum.value, str):
            comparison.append(models.MetaDatum.value == metadatum.value)
        if isinstance(metadatum.value, float):
            comparison.append(models.MetaDatum.value == metadatum.value)
        if isinstance(metadatum.value, schemas.GeoJSON):
            raise NotImplementedError("Havent implemented GeoJSON support.")
        
        return comparison

    def filter_by_metadata(self, metadata: list[schemas.MetaDatum]):
        self._filters.extend(
            [
                self._filter_by_metadatum(metadatum)
                for metadatum in metadata
                if isinstance(metadatum, schemas.MetaDatum)
            ]
        )
        return self

    """ `velour_api.enums` identifiers """
    
    def filter_by_task_types(self, task_types: list[enums.TaskType]):
        self._filters.extend(
            [
                models.Annotation.task_type == task_type.value
                for task_type in task_types
                if isinstance(task_type, enums.TaskType)
            ]
        )
        return self

    def filter_by_annotation_types(self, annotation_types: list[enums.AnnotationType]):
        if enums.AnnotationType.NONE in annotation_types:
            self._filters.extend(
                [
                    models.Annotation.box.is_(None),
                    models.Annotation.polygon.is_(None),
                    models.Annotation.multipolygon.is_(None),
                    models.Annotation.raster.is_(None),
                ]
            )
        else:
            if enums.AnnotationType.BOX in annotation_types:
                self._filters.append(models.Annotation.box.isnot(None))
            if enums.AnnotationType.POLYGON in annotation_types:
                self._filters.append(models.Annotation.polygon.isnot(None))
            if enums.AnnotationType.MULTIPOLYGON in annotation_types:
                self._filters.append(models.Annotation.multipolygon.isnot(None))
            if enums.AnnotationType.RASTER in annotation_types:
                self._filters.append(models.Annotation.raster.isnot(None))
        return self
