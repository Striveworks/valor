import pytest
from valor_lite.text_generation import metric
from valor_lite.text_generation.metric import ROUGEType


def test_text_gen_metric_status():
    parameters = {
        "datum_uid": "uid1",
        "prediction": "text",
    }

    m = metric.AnswerCorrectnessMetric(
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

    m = metric.AnswerCorrectnessMetric(
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

    # Status should be "success" or "error"
    with pytest.raises(TypeError):
        metric.AnswerCorrectnessMetric(
            status=0,  # type: ignore - testing
            value=0.5,
            parameters=parameters,
        )

    # Status should be "success" or "error"
    with pytest.raises(ValueError):
        metric.AnswerCorrectnessMetric(
            status="failure",
            value=0.5,
            parameters=parameters,
        )

    # Value should be None if status is "error"
    with pytest.raises(ValueError):
        metric.AnswerCorrectnessMetric(
            status="error",
            value=0.5,
            parameters=parameters,
        )

    # Parameters should be a dictionary
    with pytest.raises(TypeError):
        metric.AnswerCorrectnessMetric(
            status="success",
            value=0.2,
            parameters=4,  # type: ignore - testing
        )


def test_AnswerCorrectnessMetric():
    parameters = {
        "datum_uid": "uid1",
        "prediction": "text",
    }

    m = metric.AnswerCorrectnessMetric(
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
        metric.AnswerCorrectnessMetric(status="success", value="value", parameters=parameters)  # type: ignore - testing

    with pytest.raises(ValueError):
        metric.AnswerCorrectnessMetric(
            status="success", value=1.3, parameters=parameters
        )


def test_AnswerRelevanceMetric():
    parameters = {
        "datum_uid": "uid1",
        "prediction": "text",
    }

    m = metric.AnswerRelevanceMetric(
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
        metric.AnswerRelevanceMetric(status="success", value="value", parameters=parameters)  # type: ignore - testing

    with pytest.raises(ValueError):
        metric.AnswerRelevanceMetric(
            status="success", value=1.3, parameters=parameters
        )


def test_BiasMetric():
    parameters = {
        "datum_uid": "uid1",
        "prediction": "text",
    }

    m = metric.BiasMetric(
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
        metric.BiasMetric(status="success", value="value", parameters=parameters)  # type: ignore - testing

    with pytest.raises(ValueError):
        metric.BiasMetric(status="success", value=-0.2, parameters=parameters)


def test_BLEUMetric():
    parameters = {
        "datum_uid": "uid1",
        "prediction": "text",
        "weights": [0.25, 0.25, 0.25, 0.25],
    }

    m = metric.BLEUMetric(
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
        metric.BLEUMetric(status="success", value="value", parameters=parameters)  # type: ignore - testing

    with pytest.raises(ValueError):
        metric.BLEUMetric(status="success", value=1.3, parameters=parameters)


def test_ContextPrecisionMetric():
    parameters = {
        "datum_uid": "uid1",
        "context_list": ["context1", "context2"],
    }

    m = metric.ContextPrecisionMetric(
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
        metric.ContextPrecisionMetric(status="success", value="value", parameters=parameters)  # type: ignore - testing

    with pytest.raises(ValueError):
        metric.ContextPrecisionMetric(
            status="success", value=1.3, parameters=parameters
        )


def test_ContextRecallMetric():
    parameters = {
        "datum_uid": "uid1",
        "context_list": ["context1", "context2"],
    }

    m = metric.ContextRecallMetric(
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
        metric.ContextRecallMetric(status="success", value="value", parameters=parameters)  # type: ignore - testing

    with pytest.raises(ValueError):
        metric.ContextRecallMetric(
            status="success", value=1.3, parameters=parameters
        )


def test_ContextRelevanceMetric():
    parameters = {
        "datum_uid": "uid1",
        "context_list": ["context1", "context2"],
    }

    m = metric.ContextRelevanceMetric(
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
        metric.ContextRelevanceMetric(status="success", value="value", parameters=parameters)  # type: ignore - testing

    with pytest.raises(ValueError):
        metric.ContextRelevanceMetric(
            status="success", value=1.3, parameters=parameters
        )


def test_FaithfulnessMetric():
    parameters = {
        "datum_uid": "uid1",
        "prediction": "text",
        "context_list": ["context1", "context2"],
    }

    m = metric.FaithfulnessMetric(
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
        metric.FaithfulnessMetric(status="success", value="value", parameters=parameters)  # type: ignore - testing

    with pytest.raises(ValueError):
        metric.FaithfulnessMetric(
            status="success", value=1.3, parameters=parameters
        )


def test_HallucinationMetric():
    parameters = {
        "datum_uid": "uid1",
        "prediction": "text",
        "context_list": ["context1", "context2"],
    }

    m = metric.HallucinationMetric(
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
        metric.HallucinationMetric(status="success", value="value", parameters=parameters)  # type: ignore - testing

    with pytest.raises(ValueError):
        metric.HallucinationMetric(
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

    m = metric.ROUGEMetric(
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

    # Value should be a dictionary of ROUGE keys and values
    with pytest.raises(TypeError):
        metric.ROUGEMetric(status="success", value="value", parameters=parameters)  # type: ignore - testing

    # Value keys should be strings
    with pytest.raises(TypeError):
        metric.ROUGEMetric(
            status="success",
            value={
                1: 0.8,  # type: ignore - testing
                2: 1.2,  # type: ignore - testing
            },
            parameters=parameters,
        )

    # Values should be int or float
    with pytest.raises(TypeError):
        metric.ROUGEMetric(
            status="success",
            value={
                "rouge1": "0.3",  # type: ignore - testing
            },
            parameters=parameters,
        )

    # Invalid value for rouge2
    with pytest.raises(ValueError):
        metric.ROUGEMetric(
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

    m = metric.SummaryCoherenceMetric(
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
        metric.SummaryCoherenceMetric(status="success", value=0.7, parameters=parameters)  # type: ignore - testing

    with pytest.raises(TypeError):
        metric.SummaryCoherenceMetric(status="success", value=2.5, parameters=parameters)  # type: ignore - testing

    with pytest.raises(TypeError):
        metric.SummaryCoherenceMetric(status="success", value="value", parameters=parameters)  # type: ignore - testing

    with pytest.raises(ValueError):
        metric.SummaryCoherenceMetric(
            status="success", value=0, parameters=parameters
        )


def test_ToxicityMetric():
    parameters = {
        "datum_uid": "uid1",
        "prediction": "text",
    }

    m = metric.ToxicityMetric(
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
        metric.ToxicityMetric(status="success", value="value", parameters=parameters)  # type: ignore - testing

    with pytest.raises(ValueError):
        metric.ToxicityMetric(
            status="success", value=1.3, parameters=parameters
        )
