import pytest
from sqlalchemy.orm import Session

from valor_api import crud, exceptions, schemas


def test_dataset_finalization(db: Session):

    crud.create_dataset(db=db, dataset=schemas.Dataset(name="dataset"))
    with pytest.raises(exceptions.DatasetEmptyError):
        crud.finalize(db=db, dataset_name="dataset")

    crud.create_groundtruths(
        db=db,
        groundtruths=[
            schemas.GroundTruth(
                dataset_name="dataset",
                datum=schemas.Datum(uid="123"),
                annotations=[
                    schemas.Annotation(
                        labels=[schemas.Label(key="class", value="dog")],
                    )
                ],
            )
        ],
    )
    crud.finalize(db=db, dataset_name="dataset")
