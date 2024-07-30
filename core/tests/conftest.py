import pytest
from valor_core import schemas


@pytest.fixture
def evaluate_image_clf_groundtruths():
    return [
        schemas.GroundTruth(
            datum=schemas.Datum(
                uid="uid5",
                metadata={
                    "height": 900,
                    "width": 300,
                },
            ),
            annotations=[
                schemas.Annotation(
                    labels=[
                        schemas.Label(key="k4", value="v4"),
                        schemas.Label(key="k5", value="v5"),
                    ],
                ),
            ],
        ),
        schemas.GroundTruth(
            datum=schemas.Datum(
                uid="uid6",
                metadata={
                    "height": 900,
                    "width": 300,
                },
            ),
            annotations=[
                schemas.Annotation(
                    labels=[schemas.Label(key="k4", value="v4")],
                )
            ],
        ),
        schemas.GroundTruth(
            datum=schemas.Datum(
                uid="uid8",
                metadata={
                    "height": 900,
                    "width": 300,
                },
            ),
            annotations=[
                schemas.Annotation(
                    labels=[schemas.Label(key="k3", value="v3")],
                )
            ],
        ),
    ]


@pytest.fixture
def evaluate_image_clf_predictions():
    return [
        schemas.Prediction(
            datum=schemas.Datum(
                uid="uid5",
                metadata={
                    "height": 900,
                    "width": 300,
                },
            ),
            annotations=[
                schemas.Annotation(
                    labels=[
                        schemas.Label(key="k4", value="v1", score=0.47),
                        schemas.Label(key="k4", value="v8", score=0.53),
                        schemas.Label(key="k5", value="v1", score=1.0),
                    ],
                )
            ],
        ),
        schemas.Prediction(
            datum=schemas.Datum(
                uid="uid6",
                metadata={
                    "height": 900,
                    "width": 300,
                },
            ),
            annotations=[
                schemas.Annotation(
                    labels=[
                        schemas.Label(key="k4", value="v4", score=0.71),
                        schemas.Label(key="k4", value="v5", score=0.29),
                    ],
                )
            ],
        ),
        schemas.Prediction(
            datum=schemas.Datum(
                uid="uid8",
                metadata={
                    "height": 900,
                    "width": 300,
                },
            ),
            annotations=[
                schemas.Annotation(
                    labels=[
                        schemas.Label(key="k3", value="v1", score=1.0),
                    ],
                )
            ],
        ),
    ]
