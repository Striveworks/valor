import copy

import pytest
from valor_lite.nlp.generation import enums, schemas


@pytest.fixture
def metadata() -> dict[str, dict[str, str | float]]:
    return {
        "m1": {"type": "string", "value": "v1"},
        "m2": {"type": "float", "value": 0.1},
    }


@pytest.fixture
def labels() -> list[schemas.Label]:
    return [
        schemas.Label(key="k1", value="v1"),
        schemas.Label(key="k1", value="v2"),
        schemas.Label(key="k2", value="v3"),
        schemas.Label(key="k3", value="v4"),
    ]


@pytest.fixture
def text() -> str:
    return "Test string."


@pytest.fixture
def context_list() -> list[str]:
    return ["context1", "context2", "context3"]


def test_label():
    # valid
    l1 = schemas.Label(key="test", value="value")

    # test validation
    with pytest.raises(TypeError):
        assert schemas.Label(key=123, value="123")  # type: ignore - testing
    with pytest.raises(TypeError):
        assert schemas.Label(key="123", value=123)  # type: ignore - testing

    # test member fn `__eq__`
    l2 = schemas.Label(key="test", value="value")
    assert l1 == l2

    # test member fn `__ne__`
    l3 = schemas.Label(key="test", value="other")
    assert l1 != l3

    # test member fn `__hash__`
    assert l1.__hash__() == l2.__hash__()


def test_scored_label():
    l1 = schemas.Label(key="test", value="value")

    # valid
    s1 = schemas.Label(key="test", value="value", score=0.5)
    s2 = schemas.Label(key="test", value="value", score=0.5)
    s3 = schemas.Label(key="test", value="value", score=0.1)
    s4 = schemas.Label(key="test", value="other", score=0.5)
    s5 = schemas.Label(key="other", value="value", score=0.5)

    # test validation
    with pytest.raises(TypeError):
        assert schemas.Label(key="k", value="v", score="boo")  # type: ignore - testing

    # test property `key`
    assert l1.key == "test"

    # test property `value`
    assert l1.value == "value"

    # test member fn `__eq__`
    assert s1 == s2
    assert not (s1 == s3)
    assert not (s1 == s4)
    assert not (s1 == s5)

    # test member fn `__ne__`
    assert not (s1 != s2)
    assert s1 != s3
    assert s1 != s4
    assert s1 != s5

    # test member fn `__hash__`
    assert s1.__hash__() == s2.__hash__()
    assert s1.__hash__() != s3.__hash__()
    assert s1.__hash__() != s4.__hash__()
    assert s1.__hash__() != s5.__hash__()


def test_label_equality():
    label1 = schemas.Label(key="test", value="value")
    label2 = schemas.Label(key="test", value="value")
    label3 = schemas.Label(key="test", value="other")
    label4 = schemas.Label(key="other", value="value")

    eq1 = label1 == label2
    assert eq1

    eq2 = label1 == label3
    assert not eq2

    eq3 = label1 == label4
    assert not eq3


def test_label_score():
    label1 = schemas.Label(key="test", value="value", score=0.5)
    label2 = schemas.Label(key="test", value="value", score=0.5)
    label3 = schemas.Label(key="test", value="value", score=0.1)
    assert label1.score
    assert label2.score
    assert label3.score

    b1 = label1.score == label2.score
    assert b1

    b2 = label1.score > label3.score
    assert b2

    b3 = label1.score < label3.score
    assert not b3

    b4 = label1.score >= label2.score
    assert b4

    b5 = label1.score != label3.score
    assert b5

    b6 = label1.score != label2.score
    assert not b6


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
    labels: list[schemas.Label],
    text: str,
    context_list: list[str],
    metadata: dict[str, dict[str, str | float]],
):
    # valid
    schemas.Annotation(labels=labels)
    schemas.Annotation(labels=labels, metadata={})
    schemas.Annotation(
        labels=labels,
        metadata=metadata,
    )
    schemas.Annotation(
        text=text,
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
            labels=labels,
            metadata=[1234],  # type: ignore - testing
        )
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


def test_groundtruth_annotation():
    l1 = schemas.Label(key="test", value="value")
    l2 = schemas.Label(key="test", value="other")
    l3 = schemas.Label(key="other", value="value")

    # valid
    schemas.Annotation(
        labels=[l1, l2, l3],
    )

    # test `__post_init__`
    with pytest.raises(TypeError):
        schemas.Annotation(labels=l1)  # type: ignore - testing
    with pytest.raises(TypeError):
        schemas.Annotation(labels=[l1, l2, "label"])  # type: ignore - testing


def test_prediction_annotation():
    l1 = schemas.Label(key="test", value="value")
    l2 = schemas.Label(key="test", value="other")
    l3 = schemas.Label(key="other", value="value")

    s1 = copy.deepcopy(l1)
    s1.score = 0.5
    s2 = copy.deepcopy(l2)
    s2.score = 0.5
    s3 = copy.deepcopy(l3)
    s3.score = 1

    # valid
    schemas.Annotation(labels=[s1, s2, s3])

    # test `__post_init__`
    with pytest.raises(TypeError):
        schemas.Annotation(labels=s1)  # type: ignore - testing
    with pytest.raises(TypeError):
        schemas.Annotation(labels=[s1, s2, "label"])  # type: ignore - testing


def test_groundtruth():
    label = schemas.Label(key="test", value="value")
    datum = schemas.Datum(uid="somefile")
    gts = [
        schemas.Annotation(labels=[label]),
        schemas.Annotation(labels=[label]),
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
    scored_label = schemas.Label(key="test", value="value", score=1.0)
    datum = schemas.Datum(uid="somefile")
    pds = [
        schemas.Annotation(
            labels=[scored_label],
        ),
        schemas.Annotation(
            labels=[scored_label],
        ),
    ]

    schemas.Prediction(datum=datum, annotations=pds)

    # test `__post_init__`
    with pytest.raises(TypeError):
        schemas.Prediction(datum="schemas.Datum", annotations=pds)  # type: ignore - testing
    with pytest.raises(TypeError):
        schemas.Prediction(
            datum=datum,
            annotations=pds[0],  # type: ignore - testing
        )

    with pytest.raises(TypeError):
        schemas.Prediction(
            datum=datum,
            annotations=[pds[0], pds[1], "schemas.Annotation"],  # type: ignore - testing
        )

    assert schemas.Prediction(
        datum=datum, annotations=pds
    ) == schemas.Prediction(datum=datum, annotations=pds)


def test_EvaluationParameters():
    schemas.EvaluationParameters()

    schemas.EvaluationParameters(
        iou_thresholds_to_compute=[0.2, 0.6],
        iou_thresholds_to_return=[],
    )

    schemas.EvaluationParameters(
        iou_thresholds_to_compute=[],
        iou_thresholds_to_return=[],
    )

    schemas.EvaluationParameters(
        metrics_to_return=[
            enums.MetricType.AP,
            enums.MetricType.AR,
        ],
    )

    # Typical evaluation parameters for a text generation task
    schemas.EvaluationParameters(
        metrics_to_return=[
            enums.MetricType.AnswerCorrectness,
            enums.MetricType.BLEU,
            enums.MetricType.ContextPrecision,
            enums.MetricType.ContextRecall,
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

    schemas.EvaluationParameters(
        convert_annotations_to_type=enums.AnnotationType.BOX,
    )

    with pytest.raises(TypeError):
        schemas.EvaluationParameters(
            label_map=[
                [["class_name", "maine coon cat"], ["class", "cat"]],
                [["class", "siamese cat"], ["class", "cat"]],
                [["class", "british shorthair"], ["class", "cat"]],
            ],  # type: ignore
        )

    with pytest.raises(TypeError):
        schemas.EvaluationParameters(label_map={"bad": "inputs"})  # type: ignore

    with pytest.raises(TypeError):
        schemas.EvaluationParameters(metrics_to_return={"bad": "inputs"})  # type: ignore
