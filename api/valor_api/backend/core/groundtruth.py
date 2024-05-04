import time

from sqlalchemy import delete, select
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session

from valor_api import enums, exceptions, schemas
from valor_api.backend import core, models


def create_groundtruths(db: Session, groundtruths: list[schemas.GroundTruth]):
    print("INSIDE create_groundtruths")
    # check dataset status
    start = time.perf_counter()
    dataset_names = set(
        [groundtruth.dataset_name for groundtruth in groundtruths]
    )

    dataset_name_to_dataset = {
        dataset_name: core.fetch_dataset(db=db, name=dataset_name)
        for dataset_name in dataset_names
    }
    for dataset in dataset_name_to_dataset.values():
        if dataset.status != enums.TableStatus.CREATING:
            raise exceptions.DatasetFinalizedError(dataset.name)

    print(f"get_dataset_status: {time.perf_counter() - start:.4f}")

    start = time.perf_counter()
    datums = core.create_datums(
        db,
        [groundtruth.datum for groundtruth in groundtruths],
        [
            dataset_name_to_dataset[groundtruth.dataset_name]
            for groundtruth in groundtruths
        ],
    )
    print(f"create_datums: {time.perf_counter() - start:.4f}")

    start = time.perf_counter()
    all_labels = [
        label
        for groundtruth in groundtruths
        for annotation in groundtruth.annotations
        for label in annotation.labels
    ]
    label_dict = core.create_labels(db=db, labels=all_labels)
    print(f"create_labels: {time.perf_counter() - start:.4f}")

    start = time.perf_counter()
    # create annotations
    annotation_ids = core.create_annotations(
        db=db,
        annotations=[groundtruth.annotations for groundtruth in groundtruths],
        datums=datums,
        model=None,
    )
    print(f"create_annotations: {time.perf_counter() - start:.4f}")

    print("starting creating groundtruth_mappings")
    start = time.perf_counter()
    groundtruth_mappings = []
    for groundtruth, annotation_ids_per_groundtruth in zip(
        groundtruths, annotation_ids
    ):
        for i, annotation in enumerate(groundtruth.annotations):
            for label in annotation.labels:
                groundtruth_mappings.append(
                    {
                        "annotation_id": annotation_ids_per_groundtruth[i],
                        "label_id": label_dict[(label.key, label.value)],
                    }
                )
    print(f"create_groundtruth_mappings: {time.perf_counter() - start:.4f}")

    start = time.perf_counter()
    print("starting bulk insert")
    try:
        db.bulk_insert_mappings(models.GroundTruth, groundtruth_mappings)
        db.commit()
    except IntegrityError as e:
        db.rollback()
        import pdb

        pdb.set_trace()
        raise Exception()
    print(f"bulk_insert_mappings: {time.perf_counter() - start:.4f}")
    print("EXITING create_groundtruth")


def create_groundtruth(
    db: Session,
    groundtruth: schemas.GroundTruth,
):
    """
    Creates a ground truth.

    Parameters
    ----------
    db : Session
        The database Session to query against.
    groundtruth: schemas.GroundTruth
        The ground truth to create.
    """
    print("INSIDE create_groundtruth")
    # fetch dataset row
    start = time.perf_counter()
    dataset = core.fetch_dataset(db=db, name=groundtruth.dataset_name)
    print(f"fetch_dataset: {time.perf_counter() - start:.4f}")

    # check dataset status
    start = time.perf_counter()
    if (
        core.get_dataset_status(db=db, name=groundtruth.dataset_name)
        != enums.TableStatus.CREATING
    ):
        raise exceptions.DatasetFinalizedError(groundtruth.dataset_name)
    print(f"get_dataset_status: {time.perf_counter() - start:.4f}")

    # create datum
    start = time.perf_counter()
    datum = core.create_datum(db=db, datum=groundtruth.datum, dataset=dataset)
    print(f"create_datum: {time.perf_counter() - start:.4f}")

    # create labels
    start = time.perf_counter()
    all_labels = [
        label
        for annotation in groundtruth.annotations
        for label in annotation.labels
    ]
    label_list = core.create_labels(db=db, labels=all_labels)
    print(f"create_labels: {time.perf_counter() - start:.4f}")

    start = time.perf_counter()
    # create annotations
    annotation_list = core.create_annotations(
        db=db,
        annotations=groundtruth.annotations,
        datum=datum,
        model=None,
    )
    print(f"create_annotations: {time.perf_counter() - start:.4f}")

    # create groundtruths
    label_idx = 0
    groundtruth_list = []
    for i, annotation in enumerate(groundtruth.annotations):
        for label in label_list[
            label_idx : label_idx + len(annotation.labels)
        ]:
            groundtruth_list.append(
                models.GroundTruth(
                    annotation_id=annotation_list[i].id,
                    label_id=label.id,
                )
            )
        label_idx += len(annotation.labels)

    start = time.perf_counter()
    try:
        db.add_all(groundtruth_list)
        db.commit()
    except IntegrityError:
        db.rollback()
        raise Exception
    print(f"add_all: {time.perf_counter() - start:.4f}")


def get_groundtruth(
    db: Session,
    dataset_name: str,
    datum_uid: str,
) -> schemas.GroundTruth:
    """
    Fetch a ground truth.

    Parameters
    ----------
    db : Session
        The database Session to query against.
    dataset_name : str
        The name of the dataset.
    datum_uid: str
        The UID of the datum to fetch.


    Returns
    ----------
    schemas.GroundTruth
        The requested ground truth.
    """
    # retrieve from table
    dataset = core.fetch_dataset(db, name=dataset_name)
    datum = core.fetch_datum(db, dataset_id=dataset.id, uid=datum_uid)
    return schemas.GroundTruth(
        dataset_name=dataset.name,
        datum=schemas.Datum(
            uid=datum.uid,
            metadata=datum.meta,
        ),
        annotations=core.get_annotations(db, datum),
    )


def delete_groundtruths(
    db: Session,
    dataset: models.Dataset,
):
    """
    Delete all groundtruths from a dataset.

    Parameters
    ----------
    db : Session
        The database session.
    dataset : models.Dataset
        The dataset row that is being deleted.

    Raises
    ------
    RuntimeError
        If dataset is not in deletion state.
    """

    if dataset.status != enums.TableStatus.DELETING:
        raise RuntimeError(
            f"Attempted to delete groundtruths from dataset `{dataset.name}` which has status `{dataset.status}`"
        )

    subquery = (
        select(models.GroundTruth.id.label("id"))
        .join(
            models.Annotation,
            models.Annotation.id == models.GroundTruth.annotation_id,
        )
        .join(models.Datum, models.Datum.id == models.Annotation.datum_id)
        .where(models.Datum.dataset_id == dataset.id)
        .subquery()
    )
    delete_stmt = delete(models.GroundTruth).where(
        models.GroundTruth.id == subquery.c.id
    )

    try:
        db.execute(delete_stmt)
        db.commit()
    except IntegrityError as e:
        db.rollback()
        raise e
