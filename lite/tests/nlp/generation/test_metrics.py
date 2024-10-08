import pytest
from valor_lite.nlp.generation import metrics
from valor_lite.nlp.generation.enums import ROUGEType


def test_AnswerCorrectnessMetric():
    parameters = {
        "datum_uid": "uid1",
        "prediction": "text",
    }

    m = metrics.AnswerCorrectnessMetric(
        value=0.5,
        parameters=parameters,
    )
    assert m.to_dict() == {
        "type": "AnswerCorrectness",
        "value": 0.5,
        "parameters": parameters,
    }

    with pytest.raises(TypeError):
        metrics.AnswerCorrectnessMetric(value="value", parameters=parameters)  # type: ignore - testing

    with pytest.raises(ValueError):
        metrics.AnswerCorrectnessMetric(value=1.3, parameters=parameters)


def test_AnswerRelevanceMetric():
    parameters = {
        "datum_uid": "uid1",
        "prediction": "text",
    }

    m = metrics.AnswerRelevanceMetric(
        value=0.5,
        parameters=parameters,
    )
    assert m.to_dict() == {
        "type": "AnswerRelevance",
        "value": 0.5,
        "parameters": parameters,
    }

    with pytest.raises(TypeError):
        metrics.AnswerRelevanceMetric(value="value", parameters=parameters)  # type: ignore - testing

    with pytest.raises(ValueError):
        metrics.AnswerRelevanceMetric(value=1.3, parameters=parameters)


def test_BiasMetric():
    parameters = {
        "datum_uid": "uid1",
        "prediction": "text",
    }

    m = metrics.BiasMetric(
        value=0.5,
        parameters=parameters,
    )
    assert m.to_dict() == {
        "type": "Bias",
        "value": 0.5,
        "parameters": parameters,
    }

    with pytest.raises(TypeError):
        metrics.BiasMetric(value="value", parameters=parameters)  # type: ignore - testing

    with pytest.raises(ValueError):
        metrics.BiasMetric(value=-0.2, parameters=parameters)


def test_BLEUMetric():
    parameters = {
        "datum_uid": "uid1",
        "prediction": "text",
        "weights": [0.25, 0.25, 0.25, 0.25],
    }

    m = metrics.BLEUMetric(
        value=0.5,
        parameters=parameters,
    )
    assert m.to_dict() == {
        "type": "BLEU",
        "value": 0.5,
        "parameters": parameters,
    }

    with pytest.raises(TypeError):
        metrics.BLEUMetric(value="value", parameters=parameters)  # type: ignore - testing

    with pytest.raises(ValueError):
        metrics.BLEUMetric(value=1.3, parameters=parameters)


def test_ContextPrecisionMetric():
    parameters = {
        "datum_uid": "uid1",
        "context_list": ["context1", "context2"],
    }

    m = metrics.ContextPrecisionMetric(
        value=0.5,
        parameters=parameters,
    )
    assert m.to_dict() == {
        "type": "ContextPrecision",
        "value": 0.5,
        "parameters": parameters,
    }

    with pytest.raises(TypeError):
        metrics.ContextPrecisionMetric(value="value", parameters=parameters)  # type: ignore - testing

    with pytest.raises(ValueError):
        metrics.ContextPrecisionMetric(value=1.3, parameters=parameters)


def test_ContextRecallMetric():
    parameters = {
        "datum_uid": "uid1",
        "context_list": ["context1", "context2"],
    }

    m = metrics.ContextRecallMetric(
        value=0.5,
        parameters=parameters,
    )
    assert m.to_dict() == {
        "type": "ContextRecall",
        "value": 0.5,
        "parameters": parameters,
    }

    with pytest.raises(TypeError):
        metrics.ContextRecallMetric(value="value", parameters=parameters)  # type: ignore - testing

    with pytest.raises(ValueError):
        metrics.ContextRecallMetric(value=1.3, parameters=parameters)


def test_ContextRelevanceMetric():
    parameters = {
        "datum_uid": "uid1",
        "context_list": ["context1", "context2"],
    }

    m = metrics.ContextRelevanceMetric(
        value=0.5,
        parameters=parameters,
    )
    assert m.to_dict() == {
        "type": "ContextRelevance",
        "value": 0.5,
        "parameters": parameters,
    }

    with pytest.raises(TypeError):
        metrics.ContextRelevanceMetric(value="value", parameters=parameters)  # type: ignore - testing

    with pytest.raises(ValueError):
        metrics.ContextRelevanceMetric(value=1.3, parameters=parameters)


def test_FaithfulnessMetric():
    parameters = {
        "datum_uid": "uid1",
        "prediction": "text",
        "context_list": ["context1", "context2"],
    }

    m = metrics.FaithfulnessMetric(
        value=0.5,
        parameters=parameters,
    )
    assert m.to_dict() == {
        "type": "Faithfulness",
        "value": 0.5,
        "parameters": parameters,
    }

    with pytest.raises(TypeError):
        metrics.FaithfulnessMetric(value="value", parameters=parameters)  # type: ignore - testing

    with pytest.raises(ValueError):
        metrics.FaithfulnessMetric(value=1.3, parameters=parameters)


def test_HallucinationMetric():
    parameters = {
        "datum_uid": "uid1",
        "prediction": "text",
        "context_list": ["context1", "context2"],
    }

    m = metrics.HallucinationMetric(
        value=0.5,
        parameters=parameters,
    )
    assert m.to_dict() == {
        "type": "Hallucination",
        "value": 0.5,
        "parameters": parameters,
    }

    with pytest.raises(TypeError):
        metrics.HallucinationMetric(value="value", parameters=parameters)  # type: ignore - testing

    with pytest.raises(ValueError):
        metrics.HallucinationMetric(value=1.3, parameters=parameters)


def test_ROUGEMetric():
    parameters = {
        "datum_uid": "uid1",
        "prediction": "text",
        "rouge_types": [
            ROUGEType.ROUGE1,
            ROUGEType.ROUGE2,
            ROUGEType.ROUGEL,
            ROUGEType.ROUGELSUM,
        ],
        "use_stemmer": False,
    }

    m = metrics.ROUGEMetric(
        value={
            "rouge1": 0.8,
            "rouge2": 0.6,
            "rougeL": 0.5,
            "rougeLsum": 0.7,
        },
        parameters=parameters,
    )
    assert m.to_dict() == {
        "type": "ROUGE",
        "value": {
            "rouge1": 0.8,
            "rouge2": 0.6,
            "rougeL": 0.5,
            "rougeLsum": 0.7,
        },
        "parameters": parameters,
    }

    with pytest.raises(TypeError):
        metrics.ROUGEMetric(value="value", parameters=parameters)  # type: ignore - testing

    with pytest.raises(ValueError):
        metrics.ROUGEMetric(
            value={
                "rouge1": 0.8,
                "rouge2": 1.2,
                "rougeL": 0.5,
                "rougeLsum": 0.7,
            },
            parameters=parameters,
        )


def test_SummaryCoherenceMetric():
    parameters = {
        "datum_uid": "uid1",
        "prediction": "text",
    }

    m = metrics.SummaryCoherenceMetric(
        value=2,
        parameters=parameters,
    )
    assert m.to_dict() == {
        "type": "SummaryCoherence",
        "value": 2,
        "parameters": parameters,
    }

    with pytest.raises(TypeError):
        metrics.SummaryCoherenceMetric(value=0.7, parameters=parameters)  # type: ignore - testing

    with pytest.raises(TypeError):
        metrics.SummaryCoherenceMetric(value=2.5, parameters=parameters)  # type: ignore - testing

    with pytest.raises(TypeError):
        metrics.SummaryCoherenceMetric(value="value", parameters=parameters)  # type: ignore - testing

    with pytest.raises(ValueError):
        metrics.SummaryCoherenceMetric(value=0, parameters=parameters)


def test_ToxicityMetric():
    parameters = {
        "datum_uid": "uid1",
        "prediction": "text",
    }

    m = metrics.ToxicityMetric(
        value=0.5,
        parameters=parameters,
    )
    assert m.to_dict() == {
        "type": "Toxicity",
        "value": 0.5,
        "parameters": parameters,
    }

    with pytest.raises(TypeError):
        metrics.ToxicityMetric(value="value", parameters=parameters)  # type: ignore - testing

    with pytest.raises(ValueError):
        metrics.ToxicityMetric(value=1.3, parameters=parameters)
