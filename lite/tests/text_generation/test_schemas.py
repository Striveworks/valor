import pytest
from valor_lite.text_generation import metric, schemas


@pytest.fixture
def metadata() -> dict[str, dict[str, str | float]]:
    return {
        "m1": {"type": "string", "value": "v1"},
        "m2": {"type": "float", "value": 0.1},
    }


@pytest.fixture
def text() -> str:
    return "Test string."


@pytest.fixture
def context_list() -> list[str]:
    return ["context1", "context2", "context3"]


def test_datum(
    text: str,
):
    schemas.Datum(uid="123")
    schemas.Datum(uid="123", metadata={})
    schemas.Datum(uid="123", metadata={"name": 1})
    schemas.Datum(uid="123", text=text)

    with pytest.raises(TypeError):
        schemas.Datum(uid=123)  # type: ignore
    with pytest.raises(TypeError):
        schemas.Datum(uid="123", metadata=1)  # type: ignore
    with pytest.raises(TypeError):
        schemas.Datum(uid="123", metadata=[1])  # type: ignore
    with pytest.raises(TypeError):
        schemas.Datum(uid="123", text=55)  # type: ignore


def test_annotation(
    text: str,
    context_list: list[str],
    metadata: dict[str, dict[str, str | float]],
):
    # valid
    schemas.Annotation(
        text=text,
        metadata=metadata,
    )
    schemas.Annotation(
        text=text,
        context_list=context_list,
    )
    schemas.Annotation(
        context_list=context_list,
    )  # Some text generation metrics only use the prediction context list and not the prediction text.

    # test `__post_init__`
    with pytest.raises(TypeError):
        schemas.Annotation(
            text=["This should not be a list."],  # type: ignore - testing
        )
    with pytest.raises(TypeError):
        schemas.Annotation(
            context_list="This should be a list of strings.",  # type: ignore - testing
        )
    with pytest.raises(TypeError):
        schemas.Annotation(
            context_list=[1, 2],  # type: ignore - testing
        )


def test_groundtruth():
    datum = schemas.Datum(uid="uid0")
    gts = [
        schemas.Annotation(text="Answer 1."),
        schemas.Annotation(text="Answer 2."),
    ]

    # valid
    schemas.GroundTruth(
        datum=datum,
        annotations=gts,
    )

    # test `__post_init__`
    with pytest.raises(TypeError):
        schemas.GroundTruth(
            datum="schemas.Datum",  # type: ignore - testing
            annotations=gts,
        )
    with pytest.raises(TypeError):
        schemas.GroundTruth(
            datum=datum,
            annotations=gts[0],  # type: ignore - testing
        )

    with pytest.raises(TypeError):
        schemas.GroundTruth(
            datum=datum,
            annotations=[gts[0], gts[1], "schemas.Annotation"],  # type: ignore - testing
        )

    assert schemas.GroundTruth(
        datum=datum,
        annotations=gts,
    ) == schemas.GroundTruth(
        datum=datum,
        annotations=gts,
    )


def test_prediction():
    datums = [
        schemas.Datum(uid="uid0"),
        schemas.Datum(uid="uid1"),
    ]
    preds = [
        schemas.Annotation(text="Generated answer 1."),
        schemas.Annotation(context_list=["context 1", "context 2"]),
    ]

    schemas.Prediction(datum=datums[0], annotations=[preds[0]])
    schemas.Prediction(datum=datums[1], annotations=[preds[1]])

    # test `__post_init__`
    with pytest.raises(TypeError):
        schemas.Prediction(datum="schemas.Datum", annotations=[preds[0]])  # type: ignore - testing
    with pytest.raises(TypeError):
        schemas.Prediction(
            datum=datums[0],
            annotations=preds[0],  # type: ignore - testing
        )

    with pytest.raises(TypeError):
        schemas.Prediction(
            datum=datums[0],
            annotations=[preds[0], preds[1], "schemas.Annotation"],  # type: ignore - testing
        )

    assert schemas.Prediction(
        datum=datums[0], annotations=[preds[0]]
    ) == schemas.Prediction(datum=datums[0], annotations=[preds[0]])


def test_EvaluationParameters():
    schemas.EvaluationParameters()

    # Typical evaluation parameters for a text generation task
    schemas.EvaluationParameters(
        metrics_to_return=[
            metric.MetricType.AnswerCorrectness,
            metric.MetricType.BLEU,
            metric.MetricType.ContextPrecision,
            metric.MetricType.ContextRecall,
        ],
        llm_api_params={
            "client": "openai",
            "api_key": "test_key",
            "data": {
                "seed": 2024,
                "model": "gpt-4o",
            },
        },
        metric_params={
            "BLEU": {
                "weights": [0.5, 0.3, 0.1, 0.1],
            },
        },
    )

    with pytest.raises(TypeError):
        schemas.EvaluationParameters(metrics_to_return={"bad": "inputs"})  # type: ignore
