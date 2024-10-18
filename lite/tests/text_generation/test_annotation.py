import pytest
from valor_lite.text_generation import annotation


@pytest.fixture
def text() -> str:
    return "Test string."


@pytest.fixture
def context_list() -> list[str]:
    return ["context1", "context2", "context3"]


def test_datum(
    text: str,
):
    annotation.Datum(uid="123")
    annotation.Datum(uid="123", text=text)

    with pytest.raises(TypeError):
        annotation.Datum(uid=123)  # type: ignore
    with pytest.raises(TypeError):
        annotation.Datum(uid="123", text=55)  # type: ignore


def test_annotation(
    text: str,
    context_list: list[str],
):
    # valid
    annotation.Annotation(
        text=text,
    )
    annotation.Annotation(
        context_list=context_list,
    )
    annotation.Annotation(
        text=text,
        context_list=context_list,
    )

    # test `__post_init__`
    with pytest.raises(TypeError):
        annotation.Annotation(
            text=["This should not be a list."],  # type: ignore - testing
        )
    with pytest.raises(TypeError):
        annotation.Annotation(
            context_list="This should be a list of strings.",  # type: ignore - testing
        )
    with pytest.raises(TypeError):
        annotation.Annotation(
            context_list=[1, 2],  # type: ignore - testing
        )


def test_groundtruth():
    datum = annotation.Datum(uid="uid0")
    gts = [
        annotation.Annotation(text="Answer 1."),
        annotation.Annotation(text="Answer 2."),
    ]

    # valid
    annotation.GroundTruth(
        datum=datum,
        annotations=gts,
    )

    # test `__post_init__`
    with pytest.raises(TypeError):
        annotation.GroundTruth(
            datum="schemas.Datum",  # type: ignore - testing
            annotations=gts,
        )
    with pytest.raises(TypeError):
        annotation.GroundTruth(
            datum=datum,
            annotations=gts[0],  # type: ignore - testing
        )

    with pytest.raises(TypeError):
        annotation.GroundTruth(
            datum=datum,
            annotations=[gts[0], gts[1], "schemas.Annotation"],  # type: ignore - testing
        )

    assert annotation.GroundTruth(
        datum=datum,
        annotations=gts,
    ) == annotation.GroundTruth(
        datum=datum,
        annotations=gts,
    )


def test_prediction():
    datums = [
        annotation.Datum(uid="uid0"),
        annotation.Datum(uid="uid1"),
    ]
    preds = [
        annotation.Annotation(text="Generated answer 1."),
        annotation.Annotation(context_list=["context 1", "context 2"]),
    ]

    annotation.Prediction(datum=datums[0], annotations=[preds[0]])
    annotation.Prediction(datum=datums[1], annotations=[preds[1]])

    # test `__post_init__`
    with pytest.raises(TypeError):
        annotation.Prediction(datum="schemas.Datum", annotations=[preds[0]])  # type: ignore - testing
    with pytest.raises(TypeError):
        annotation.Prediction(
            datum=datums[0],
            annotations=preds[0],  # type: ignore - testing
        )

    with pytest.raises(TypeError):
        annotation.Prediction(
            datum=datums[0],
            annotations=[preds[0], preds[1], "schemas.Annotation"],  # type: ignore - testing
        )

    assert annotation.Prediction(
        datum=datums[0], annotations=[preds[0]]
    ) == annotation.Prediction(datum=datums[0], annotations=[preds[0]])
