from sqlalchemy import func, insert, select
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session

from . import models, schemas


class DatasetAlreadyExistsError(Exception):
    def __init__(self, name: str):
        return super().__init__(f"Dataset with name '{name}' already exists.")


class ModelAlreadyExistsError(Exception):
    def __init__(self, name: str):
        return super().__init__(f"Model with name '{name}' already exists.")


class DatasetDoesNotExistError(Exception):
    def __init__(self, name: str):
        return super().__init__(f"Dataset with name '{name}' does not exist.")


class DatasetIsFinalizedError(Exception):
    def __init__(self, name: str):
        return super().__init__(
            f"Cannot add images or annotations to dataset '{name}' since it is finalized."
        )


class ModelDoesNotExistError(Exception):
    def __init__(self, name: str):
        return super().__init__(f"Model with name '{name}' does not exist.")


class ImageDoesNotExistError(Exception):
    def __init__(self, uri: str):
        return super().__init__(f"Image with uri '{uri}' does not exist.")


def _wkt_polygon_from_detection(det: schemas.DetectionBase) -> str:
    """Returns the "Well-known text" format of a detection"""
    pts = det.boundary
    # in PostGIS polygon has to begin and end at the same point
    if pts[0] != pts[-1]:
        pts = pts + [pts[0]]
    return (
        "POLYGON (("
        + ", ".join([" ".join([str(pt[0]), str(pt[1])]) for pt in pts])
        + "))"
    )


def bulk_insert_and_return_ids(
    db: Session, model: type, mappings: list[dict]
) -> list[int]:
    """Bulk adds to the database

    model
        the class that represents the database table
    mappings
        dictionaries mapping column names to values
    """
    added_ids = db.scalars(insert(model).returning(model.id), mappings)
    db.commit()
    return added_ids.all()


def _create_detection_mappings(
    detections: list[schemas.DetectionBase], image_ids: list[str]
) -> list[dict[str, str]]:
    return [
        {
            "boundary": _wkt_polygon_from_detection(detection),
            "image_id": image_id,
        }
        for detection, image_id in zip(detections, image_ids)
    ]


def _create_label_tuple_to_id_dict(
    db, detections: list[schemas.DetectionBase]
) -> dict[tuple, str]:
    """Goes through detections and adds a label if it doesn't exist. Return is a mapping from
    `tuple(label)` (since `label` is not hashable) to label id
    """
    label_tuple_to_id = {}
    for detection in detections:
        for label in detection.labels:
            label_tuple = tuple(label)
            if label_tuple not in label_tuple_to_id:
                label_tuple_to_id[label_tuple] = get_or_create_row(
                    db, models.Label, {"key": label.key, "value": label.value}
                )
    return label_tuple_to_id


def create_groundtruth_detections(
    db: Session,
    data: schemas.GroundTruthDetectionsCreate,
) -> list[int]:
    # create (if they don't exist) the image rows and get the ids
    dset = get_dataset(db, dataset_name=data.dataset_name)
    if not dset.draft:
        raise DatasetIsFinalizedError(data.dataset_name)
    dset_id = dset.id
    image_ids = []
    for detection in data.detections:
        image_ids.append(
            get_or_create_row(
                db=db,
                model_class=models.Image,
                mapping={"dataset_id": dset_id, "uri": detection.image.uri},
            )
        )

    # create gt detections
    det_mappings = _create_detection_mappings(
        detections=data.detections, image_ids=image_ids
    )
    det_ids = bulk_insert_and_return_ids(
        db, models.GroundTruthDetection, det_mappings
    )

    label_tuple_to_id = _create_label_tuple_to_id_dict(db, data.detections)

    labeled_gt_mappings = [
        {
            "detection_id": gt_det_id,
            "label_id": label_tuple_to_id[tuple(label)],
        }
        for gt_det_id, detection in zip(det_ids, data.detections)
        for label in detection.labels
    ]

    return bulk_insert_and_return_ids(
        db, models.LabeledGroundTruthDetection, labeled_gt_mappings
    )


def create_predicted_detections(
    db: Session, data: schemas.PredictedDetectionsCreate
) -> list[int]:
    """
    Raises
    ------
    ModelDoesNotExistError
        if the model with name `data.model_name` does not exist
    """
    model_id = get_model(db, model_name=data.model_name).id
    # get image ids from uris (these images should already exist)
    image_ids = [
        get_image(db, uri=detection.image.uri).id
        for detection in data.detections
    ]

    det_mappings = _create_detection_mappings(
        detections=data.detections, image_ids=image_ids
    )
    for m in det_mappings:
        m["model_id"] = model_id
    det_ids = bulk_insert_and_return_ids(
        db, models.PredictedDetection, det_mappings
    )

    label_tuple_to_id = _create_label_tuple_to_id_dict(db, data.detections)

    labeled_pred_mappings = [
        {
            "detection_id": gt_det_id,
            "label_id": label_tuple_to_id[tuple(label)],
            "score": detection.score,
        }
        for gt_det_id, detection in zip(det_ids, data.detections)
        for label in detection.labels
    ]

    return bulk_insert_and_return_ids(
        db, models.LabeledPredictedDetection, labeled_pred_mappings
    )


def get_or_create_row(
    db: Session,
    model_class: type,
    mapping: dict,
) -> int:
    """Tries to get the row defined by mapping. If that exists then
    its id is returned. Otherwise a row is created by `mapping` and the newly created
    row's id is returned
    """
    # create the query from the mapping
    where_expressions = [
        (getattr(model_class, k) == v) for k, v in mapping.items()
    ]
    where_expression = where_expressions[0]
    for exp in where_expressions[1:]:
        where_expression = where_expression & exp

    db_element = db.scalar(select(model_class).where(where_expression))

    if not db_element:
        db_element = model_class(**mapping)
        db.add(db_element)
        db.flush()
        db.commit()

    return db_element.id


def create_dataset(db: Session, dataset: schemas.DatasetCreate):
    """Creates a dataset

    Raises
    ------
    DatasetAlreadyExistsError
        if the dataset name already exists
    """
    try:
        db.add(models.Dataset(name=dataset.name, draft=True))
        db.commit()
    except IntegrityError:
        db.rollback()
        raise DatasetAlreadyExistsError(dataset.name)


def get_datasets(db: Session) -> list[schemas.Dataset]:
    return [
        schemas.Dataset(name=d.name, draft=d.draft)
        for d in db.scalars(select(models.Dataset))
    ]


def get_dataset(db: Session, dataset_name: str) -> models.Dataset:
    ret = db.scalar(
        select(models.Dataset).where(models.Dataset.name == dataset_name)
    )
    if ret is None:
        raise DatasetDoesNotExistError(dataset_name)

    return ret


def get_models(db: Session) -> list[schemas.Model]:
    return [
        schemas.Model(name=m.name) for m in db.scalars(select(models.Model))
    ]


def create_model(db: Session, model: schemas.Model):
    """Creates a dataset

    Raises
    ------
    ModelAlreadyExistsError
        if the model uri already exists
    """
    try:
        db.add(models.Model(name=model.name))
        db.commit()
    except IntegrityError:
        db.rollback()
        raise ModelAlreadyExistsError(model.name)


def finalize_dataset(db: Session, dataset_name: str) -> None:
    dset = get_dataset(db, dataset_name)
    dset.draft = False
    db.commit()


def get_model(db: Session, model_name: str) -> models.Model:
    ret = db.scalar(
        select(models.Model).where(models.Model.name == model_name)
    )
    if ret is None:
        raise ModelDoesNotExistError(model_name)

    return ret


def get_image(db: Session, uri: str) -> models.Image:
    ret = db.scalar(select(models.Image).where(models.Image.uri == uri))
    if ret is None:
        raise ImageDoesNotExistError(uri)

    return ret


def get_labels_in_dataset(
    db: Session, dataset_name: str
) -> list[models.Label]:
    # TODO must be a better and more SQLy way of doing this
    dset = get_dataset(db, dataset_name)
    unique_ids = set()
    for image in dset.images:
        unique_ids.update(get_unique_label_ids_in_image(image))

    return db.scalars(
        select(models.Label).where(models.Label.id.in_(unique_ids))
    ).all()


def get_all_labels(db: Session) -> list[schemas.Label]:
    return [
        schemas.Label(key=label.key, value=label.value)
        for label in db.scalars(select(models.Label))
    ]


def get_images_in_dataset(
    db: Session, dataset_name: str
) -> list[models.Label]:
    # TODO must be a better and more SQLy way of doing this
    dset = get_dataset(db, dataset_name)
    return dset.images


def get_unique_label_ids_in_image(image: models.Image) -> set[int]:
    ret = set()
    for det in image.ground_truth_detections:
        for labeled_det in det.labeled_ground_truth_detections:
            ret.add(labeled_det.label.id)

    return ret


def delete_dataset(db: Session, dataset_name: str):
    dset = get_dataset(db, dataset_name)

    db.delete(dset)
    db.commit()


def delete_model(db: Session, model_name: str):
    model = get_model(db, model_name)

    db.delete(model)
    db.commit()


def number_of_rows(db: Session, model_cls: type) -> int:
    return db.scalar(select(func.count(model_cls.id)))
