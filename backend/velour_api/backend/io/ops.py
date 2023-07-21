from sqlalchemy.orm import Session

from velour_api import schemas, enums
from velour_api.backend import models, core

def create_filter(
    filter_by_dataset: models.Dataset = None,
    filter_by_model: models.Model = None,
    filter_by_datum: models.Datum = None,
    filter_by_annotation: models.Annotation = None,
    filter_by_key: list[str] = [],
    filter_by_task_type: list[enums.TaskType] = [],
    filter_by_annotation_type: list[enums.AnnotationType] = [],
    filter_by_metadata: list[schemas.MetaDatum] = [],
):
    pass


class QueryFilter:

    def __init__(self, db: Session):
        self.db = db
        self.filters = []

    def filter_by_dataset_

    def filter_by_datasets(self, datasets: list[schemas.Dataset]):
        rows = [
            core.get_dataset(self.db, dataset.name)
            for dataset in datasets
        ]
        self.filters.extend(
            [
                models.Dataset.id == row.id
                for row in rows
            ]
        )

    def filter_by_models(self, models: list[schemas.Model]):
        pass

    def filter_by_datums(self, datums: list[schemas.Datum]):
        pass

    def filter_by_annotations(self, annotations: list[schemas.Annotation]):
        pass

    def filter_by_label_key(self, keys = list[str]):
        pass

    def filter_by_task_type(self, task_types: list[enums.TaskType]):
        pass

    def filter_by_annotation_type(self, annotation_types: list[enums.AnnotationType]):
        pass

    def filter_by_metadata(self, metadata: list[schemas.MetaDatum]):
        pass