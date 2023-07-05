from sqlalchemy.orm import Session
from sqlalchemy import select, insert

from velour_api import schemas, exceptions
from velour_api.backend import models

from velour_api.backend.core.metadata import create_metadata
from velour_api.backend.core.label import create_label
from velour_api.backend.core.geometry import create_geometry


def get_dataset_info(
    db: Session,
    dataset_id: int,
) -> schemas.DatasetInfo:
    # return dataset info with all assoiated metadata
    pass


def get_dataset_id(
    db: Session,
    dataset: schemas.DatasetInfo,
):
    return db.scalar(
        select(models.Dataset.id)
        .where(models.Dataset.name == dataset.name)
    )


def create_dataset_info(
    db: Session,
    info: schemas.DatasetInfo,
):
    mapping = {
        "name": info.name,
    }
    dataset_id = db.scalar(
        insert(models.Dataset)
        .values(mapping)
        .returning(models.Dataset.id)
    )
    db.commit()
    create_metadata(db, info.metadata, dataset_id=dataset_id)
    return dataset_id


def create_datum(
    db: Session, 
    datum: schemas.Datum,
    dataset_id: int,
):
    mapping = {
        "uid": datum.uid,
        "dataset_id": dataset_id,
    }
    datum_id = db.scalar(
        insert(models.Datum)
        .values(mapping)
        .returning(models.Datum.id)
    )
    db.commit()
    create_metadata(db, datum.metadata, dataset_id=dataset_id, datum_id=datum_id)
    return datum_id


def create_groundtruths(
    db: Session,
    gts: schemas.GroundTruth,
    dataset_id: int,
    datum_id: int,
):
    
    # @TODO: Need to iterate through each label and assign same annotation, datum, dataset to it.
    mapping = [
        {
        "dataset_id": dataset_id,
        "datum_id": datum_id,
        "geometry_id": create_geometry(db, gt.annotation),
        "label_id": create_label(db, gt.label)
        }
        for gt in gts
    ]


def create_dataset(
    db: Session,
    dataset: schemas.Dataset,
) -> int:
    
    # Check if dataset already exists.
    if get_dataset_id(db, dataset.info):
        raise exceptions.DatasetAlreadyExistsError(dataset.info.name)
    
    # Create dataset
    dataset_id = create_dataset_info(db, dataset.info)
    for datum in dataset.datums:
        datum_id = create_datum(db, datum, dataset_id)
        create_groundtruths(db, datum.gts, dataset_id=dataset_id, datum_id=datum_id)



def _add_datums_to_dataset(
    db: Session, dataset_name, datums: list[schemas.Datum]
) -> list[models.Datum]:
    """Adds images defined by URIs to a dataset (creating the Image rows if they don't exist),
    returning the list of image ids"""
    dset = get_dataset(db, dataset_name=dataset_name)
    if not dset.draft:
        raise exceptions.DatasetIsFinalizedError(dataset_name)
    dset_id = dset.id

    db_datums = [
        _get_or_create_row(
            db=db,
            model_class=models.Datum,
            mapping={
                "dataset_id": dset_id,
                **datum.dict(exclude={"metadata"}),
            },
        )
        for datum in datums
    ]

    for datum, db_datum in zip(datums, db_datums):
        for metadatum in datum.metadata:
            metadatum_id = _get_or_create_row(
                db=db,
                model_class=models.Metadatum,
                mapping=_metadatum_mapping(metadatum=metadatum),
            ).id

            db.add(
                models.DatumMetadatumLink(
                    datum_id=db_datum.id, metadatum_id=metadatum_id
                )
            )

    db.commit()
    return db_datums
