import pytest
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session

from valor import Client
from valor_api.backend.models import Evaluation
from valor_api.enums import EvaluationStatus, TaskType
from valor_api.schemas import EvaluationParameters
from valor_api.schemas.migrations import DeprecatedFilter


@pytest.fixture
def deprecated_filter() -> DeprecatedFilter:
    return DeprecatedFilter(
        model_names=["1", "2"],
        model_metadata={
            "geospatial": [
                {
                    "operator": "inside",
                    "value": {
                        "type": "polygon",
                        "coordinates": [
                            [
                                [124.0, 37.0],
                                [128.0, 37.0],
                                [128.0, 40.0],
                                [124.0, 40.0],
                            ]
                        ],
                    },
                }
            ],
        },
        bounding_box_area=[
            {
                "operator": ">=",
                "value": 10.0,
            },
            {
                "operator": "<=",
                "value": 2000.0,
            },
        ],
        label_keys=["k1"],
    )


@pytest.fixture
def evaluation_with_deprecated_filter(
    db: Session, deprecated_filter: DeprecatedFilter
):

    # manually add to database
    row_id = 0
    try:
        row = Evaluation(
            id=row_id,
            dataset_names=["1", "2"],
            model_name="3",
            filters=deprecated_filter.model_dump(),
            parameters=EvaluationParameters(
                task_type=TaskType.CLASSIFICATION
            ).model_dump(),
            status=EvaluationStatus.DONE,
            meta=dict(),
        )
        db.add(row)
        db.commit()
    except IntegrityError as e:
        db.rollback()
        raise e

    yield row_id

    # delete evaluation
    try:
        db.delete(row)
        db.commit()
    except IntegrityError as e:
        db.rollback()
        raise e


def test_filter_migration(
    client: Client,
    evaluation_with_deprecated_filter: Evaluation,
    deprecated_filter: DeprecatedFilter,
):
    # get row id
    row_id = evaluation_with_deprecated_filter

    # verify deprecated format is accessible to client
    evaluations = client.get_evaluations(evaluation_ids=[row_id])
    assert len(evaluations) == 1
    assert evaluations[0].filters == deprecated_filter.model_dump(
        exclude_none=True
    )
