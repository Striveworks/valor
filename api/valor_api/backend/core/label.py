from sqlalchemy.dialects.postgresql import insert
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session

from valor_api.backend import models


def insert_labels(db: Session, labels: list[str]) -> list[int]:
    try:
        label_ids = db.execute(
            insert(models.Label)
            .values([{"value": label} for label in labels])
            .on_conflict_do_nothing(index_elements=["value"])
            .returning(models.Label.id)
        ).all()
        db.commit()
        return [label_id[0] for label_id in label_ids]

    except IntegrityError as e:
        db.rollback()
        raise e
