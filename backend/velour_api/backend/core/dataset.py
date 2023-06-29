from sqlalchemy.orm import Session

from velour_api import schemas
from velour_api.backend import models


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
