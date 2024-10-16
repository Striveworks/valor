import pytest
from valor_lite.text_generation import metrics
from valor_lite.text_generation.enums import ROUGEType


def test_text_gen_metric_status():
    parameters = {
        "datum_uid": "uid1",
        "prediction": "text",
    }

    m = metrics.AnswerCorrectnessMetric(
        status="success",
        value=0.5,
        parameters=parameters,
    )
    assert m.to_dict() == {
        "type": "AnswerCorrectness",
        "status": "success",
        "value": 0.5,
        "parameters": parameters,
    }

    m = metrics.AnswerCorrectnessMetric(
        status="error",
        value=None,
        parameters=parameters,
    )
    assert m.to_dict() == {
        "type": "AnswerCorrectness",
        "status": "error",
        "value": None,
        "parameters": parameters,
    }

    with pytest.raises(ValueError):
        metrics.AnswerCorrectnessMetric(
            status="error",
            value=0.5,
            parameters=parameters,
        )


def test_AnswerCorrectnessMetric():
    parameters = {
        "datum_uid": "uid1",
        "prediction": "text",
    }

    m = metrics.AnswerCorrectnessMetric(
        status="success",
        value=0.5,
        parameters=parameters,
    )
    assert m.to_dict() == {
        "type": "AnswerCorrectness",
        "status": "success",
        "value": 0.5,
        "parameters": parameters,
    }

    with pytest.raises(TypeError):
        metrics.AnswerCorrectnessMetric(status="success", value="value", parameters=parameters)  # type: ignore - testing

    with pytest.raises(ValueError):
        metrics.AnswerCorrectnessMetric(
            status="success", value=1.3, parameters=parameters
        )


def test_AnswerRelevanceMetric():
    parameters = {
        "datum_uid": "uid1",
        "prediction": "text",
    }

    m = metrics.AnswerRelevanceMetric(
        status="success",
        value=0.5,
        parameters=parameters,
    )
    assert m.to_dict() == {
        "type": "AnswerRelevance",
        "status": "success",
        "value": 0.5,
        "parameters": parameters,
    }

    with pytest.raises(TypeError):
        metrics.AnswerRelevanceMetric(status="success", value="value", parameters=parameters)  # type: ignore - testing

    with pytest.raises(ValueError):
        metrics.AnswerRelevanceMetric(
            status="success", value=1.3, parameters=parameters
        )


def test_BiasMetric():
    parameters = {
        "datum_uid": "uid1",
        "prediction": "text",
    }

    m = metrics.BiasMetric(
        status="success",
        value=0.5,
        parameters=parameters,
    )
    assert m.to_dict() == {
        "type": "Bias",
        "status": "success",
        "value": 0.5,
        "parameters": parameters,
    }

    with pytest.raises(TypeError):
        metrics.BiasMetric(status="success", value="value", parameters=parameters)  # type: ignore - testing

    with pytest.raises(ValueError):
        metrics.BiasMetric(status="success", value=-0.2, parameters=parameters)


def test_BLEUMetric():
    parameters = {
        "datum_uid": "uid1",
        "prediction": "text",
        "weights": [0.25, 0.25, 0.25, 0.25],
    }

    m = metrics.BLEUMetric(
        status="success",
        value=0.5,
        parameters=parameters,
    )
    assert m.to_dict() == {
        "type": "BLEU",
        "status": "success",
        "value": 0.5,
        "parameters": parameters,
    }

    with pytest.raises(TypeError):
        metrics.BLEUMetric(status="success", value="value", parameters=parameters)  # type: ignore - testing

    with pytest.raises(ValueError):
        metrics.BLEUMetric(status="success", value=1.3, parameters=parameters)


def test_ContextPrecisionMetric():
    parameters = {
        "datum_uid": "uid1",
        "context_list": ["context1", "context2"],
    }

    m = metrics.ContextPrecisionMetric(
        status="success",
        value=0.5,
        parameters=parameters,
    )
    assert m.to_dict() == {
        "type": "ContextPrecision",
        "status": "success",
        "value": 0.5,
        "parameters": parameters,
    }

    with pytest.raises(TypeError):
        metrics.ContextPrecisionMetric(status="success", value="value", parameters=parameters)  # type: ignore - testing

    with pytest.raises(ValueError):
        metrics.ContextPrecisionMetric(
            status="success", value=1.3, parameters=parameters
        )


def test_ContextRecallMetric():
    parameters = {
        "datum_uid": "uid1",
        "context_list": ["context1", "context2"],
    }

    m = metrics.ContextRecallMetric(
        status="success",
        value=0.5,
        parameters=parameters,
    )
    assert m.to_dict() == {
        "type": "ContextRecall",
        "status": "success",
        "value": 0.5,
        "parameters": parameters,
    }

    with pytest.raises(TypeError):
        metrics.ContextRecallMetric(status="success", value="value", parameters=parameters)  # type: ignore - testing

    with pytest.raises(ValueError):
        metrics.ContextRecallMetric(
            status="success", value=1.3, parameters=parameters
        )


def test_ContextRelevanceMetric():
    parameters = {
        "datum_uid": "uid1",
        "context_list": ["context1", "context2"],
    }

    m = metrics.ContextRelevanceMetric(
        status="success",
        value=0.5,
        parameters=parameters,
    )
    assert m.to_dict() == {
        "type": "ContextRelevance",
        "status": "success",
        "value": 0.5,
        "parameters": parameters,
    }

    with pytest.raises(TypeError):
        metrics.ContextRelevanceMetric(status="success", value="value", parameters=parameters)  # type: ignore - testing

    with pytest.raises(ValueError):
        metrics.ContextRelevanceMetric(
            status="success", value=1.3, parameters=parameters
        )


def test_FaithfulnessMetric():
    parameters = {
        "datum_uid": "uid1",
        "prediction": "text",
        "context_list": ["context1", "context2"],
    }

    m = metrics.FaithfulnessMetric(
        status="success",
        value=0.5,
        parameters=parameters,
    )
    assert m.to_dict() == {
        "type": "Faithfulness",
        "status": "success",
        "value": 0.5,
        "parameters": parameters,
    }

    with pytest.raises(TypeError):
        metrics.FaithfulnessMetric(status="success", value="value", parameters=parameters)  # type: ignore - testing

    with pytest.raises(ValueError):
        metrics.FaithfulnessMetric(
            status="success", value=1.3, parameters=parameters
        )


def test_HallucinationMetric():
    parameters = {
        "datum_uid": "uid1",
        "prediction": "text",
        "context_list": ["context1", "context2"],
    }

    m = metrics.HallucinationMetric(
        status="success",
        value=0.5,
        parameters=parameters,
    )
    assert m.to_dict() == {
        "type": "Hallucination",
        "status": "success",
        "value": 0.5,
        "parameters": parameters,
    }

    with pytest.raises(TypeError):
        metrics.HallucinationMetric(status="success", value="value", parameters=parameters)  # type: ignore - testing

    with pytest.raises(ValueError):
        metrics.HallucinationMetric(
            status="success", value=1.3, parameters=parameters
        )


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
        status="success",
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
        "status": "success",
        "value": {
            "rouge1": 0.8,
            "rouge2": 0.6,
            "rougeL": 0.5,
            "rougeLsum": 0.7,
        },
        "parameters": parameters,
    }

    with pytest.raises(TypeError):
        metrics.ROUGEMetric(status="success", value="value", parameters=parameters)  # type: ignore - testing

    with pytest.raises(ValueError):
        metrics.ROUGEMetric(
            status="success",
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
        status="success",
        value=2,
        parameters=parameters,
    )
    assert m.to_dict() == {
        "type": "SummaryCoherence",
        "status": "success",
        "value": 2,
        "parameters": parameters,
    }

    with pytest.raises(TypeError):
        metrics.SummaryCoherenceMetric(status="success", value=0.7, parameters=parameters)  # type: ignore - testing

    with pytest.raises(TypeError):
        metrics.SummaryCoherenceMetric(status="success", value=2.5, parameters=parameters)  # type: ignore - testing

    with pytest.raises(TypeError):
        metrics.SummaryCoherenceMetric(status="success", value="value", parameters=parameters)  # type: ignore - testing

    with pytest.raises(ValueError):
        metrics.SummaryCoherenceMetric(
            status="success", value=0, parameters=parameters
        )


def test_ToxicityMetric():
    parameters = {
        "datum_uid": "uid1",
        "prediction": "text",
    }

    m = metrics.ToxicityMetric(
        status="success",
        value=0.5,
        parameters=parameters,
    )
    assert m.to_dict() == {
        "type": "Toxicity",
        "status": "success",
        "value": 0.5,
        "parameters": parameters,
    }

    with pytest.raises(TypeError):
        metrics.ToxicityMetric(status="success", value="value", parameters=parameters)  # type: ignore - testing

    with pytest.raises(ValueError):
        metrics.ToxicityMetric(
            status="success", value=1.3, parameters=parameters
        )
