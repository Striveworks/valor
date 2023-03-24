from sqlalchemy.orm import Session

from ._read import get_dataset


def finalize_dataset(db: Session, dataset_name: str) -> None:
    dset = get_dataset(db, dataset_name)
    dset.draft = False
    db.commit()
