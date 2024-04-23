from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session

from valor import Client, Dataset, Model
from valor_api import crud, enums, schemas
from valor_api.backend import core


def test_restart_failed_evaluation(db: Session, client: Client):
    crud.create_dataset(db=db, dataset=schemas.Dataset(name="dataset"))
    crud.create_model(db=db, model=schemas.Model(name="model"))
    crud.finalize(db=db, dataset_name="dataset")

    # retrieve dataset and model on the client-side
    dataset = Dataset.get("dataset")
    model = Model.get("model")
    assert dataset
    assert model

    # create evaluation and overwrite status to failed
    eval1 = model.evaluate_classification(dataset, allow_retries=False)
    assert eval1.status == enums.EvaluationStatus.DONE
    try:
        evaluation = core.fetch_evaluation_from_id(
            db=db, evaluation_id=eval1.id
        )
        evaluation.status = enums.EvaluationStatus.FAILED
        db.commit()
    except IntegrityError as e:
        db.rollback()
        raise e

    # get evaluation and verify it is failed
    eval2 = model.evaluate_classification(dataset, allow_retries=False)
    assert eval2.id == eval1.id
    assert eval2.status == enums.EvaluationStatus.FAILED

    # get evaluation and allow retries, this should result in a finished eval
    eval3 = model.evaluate_classification(dataset, allow_retries=True)
    assert eval3.id == eval1.id
    assert eval3.status == enums.EvaluationStatus.DONE
