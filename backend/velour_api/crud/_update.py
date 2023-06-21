from sqlalchemy.orm import Session

import velour_api.stateflow as stateflow
from velour_api import exceptions, models

from ._read import get_dataset, get_model


@stateflow.finalize
def finalize_dataset(db: Session, dataset_name: str) -> None:
    dataset = get_dataset(db, dataset_name)
    dataset.finalized = True
    db.commit()


# def _check_finalized_inferences(
#     db: Session, model_name: str, dataset_name: str
# ) -> bool:
#     """Checks if inferences of model given by `model_name` on dataset given by `dataset_name`
#     are finalized
#     """
#     model_id = get_model(db, model_name).id
#     dataset_id = get_dataset(db, dataset_name).id
#     entries = db.scalars(
#         select(models.FinalizedInferences).where(
#             and_(
#                 models.FinalizedInferences.model_id == model_id,
#                 models.FinalizedInferences.dataset_id == dataset_id,
#             )
#         )
#     ).all()
#     # this should never happen because of uniqueness constraint
#     if len(entries) > 1:
#         raise RuntimeError(
#             f"got multiple entries for finalized inferences with model id {model_id} "
#             f"and dataset id {dataset_id}, which should never happen"
#         )

#     return len(entries) != 0


@stateflow.finalize
def finalize_inferences(
    db: Session, model_name: str, dataset_name: str
) -> None:
    dataset = get_dataset(db, dataset_name)
    if not dataset.finalized:
        raise exceptions.DatasetIsNotFinalizedError(dataset_name)

    model_id = get_model(db, model_name).id
    dataset_id = dataset.id

    db.add(
        models.FinalizedInferences(dataset_id=dataset_id, model_id=model_id)
    )
    db.commit()
