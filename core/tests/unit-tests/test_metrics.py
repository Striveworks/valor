import pytest
from valor_core import metrics, schemas
from valor_core.enums import ROUGEType


def test_APMetric():
    ap_metric = metrics.APMetric(
        iou=0.2, value=0.5, label=schemas.Label(key="k1", value="v1")
    )

    with pytest.raises(TypeError):
        metrics.APMetric(
            iou=None, value=0.5, label=schemas.Label(key="k1", value="v1")  # type: ignore - purposefully throwing error
        )

    with pytest.raises(TypeError):
        metrics.APMetric(iou=0.1, value=0.5, label="k1")  # type: ignore - purposefully throwing error

    assert all(
        [
            key in ["label", "parameters", "value", "type"]
            for key in ap_metric.to_dict().keys()
        ]
    )


def test_APMetricAveragedOverIOUs():
    ap_averaged_metric = metrics.APMetricAveragedOverIOUs(
        ious=set([0.1, 0.2]),
        value=0.5,
        label=schemas.Label(key="k1", value="v1"),
    )

    with pytest.raises(TypeError):
        metrics.APMetricAveragedOverIOUs(
            ious=None, value=0.5, label=schemas.Label(key="k1", value="v1")  # type: ignore - purposefully throwing error
        )

    with pytest.raises(TypeError):
        metrics.APMetricAveragedOverIOUs(
            ious=set([0.1, 0.2]), value=0.5, label="k1"  # type: ignore - purposefully throwing error
        )

    assert all(
        [
            key in ["label", "parameters", "value", "type"]
            for key in ap_averaged_metric.to_dict().keys()
        ]
    )


def test_mAPMetric():
    map_metric = metrics.mAPMetric(iou=0.2, value=0.5, label_key="key")

    with pytest.raises(TypeError):
        metrics.mAPMetric(iou=None, value=0.5, label_key="key")  # type: ignore - purposefully throwing error

    with pytest.raises(TypeError):
        metrics.mAPMetric(iou=0.1, value="value", label_key="key")  # type: ignore - purposefully throwing error

    with pytest.raises(TypeError):
        metrics.mAPMetric(iou=0.1, value=0.5, label_key=None)  # type: ignore - purposefully throwing error

    assert all(
        [
            key in ["label", "parameters", "value", "type"]
            for key in map_metric.to_dict()
        ]
    )


def test_mAPMetricAveragedOverIOUs():
    map_averaged_metric = metrics.mAPMetricAveragedOverIOUs(
        ious=set([0.1, 0.2]), value=0.5, label_key="key"
    )

    with pytest.raises(TypeError):
        metrics.mAPMetricAveragedOverIOUs(ious=None, value=0.5, label_key="key")  # type: ignore - purposefully throwing error

    with pytest.raises(TypeError):
        metrics.mAPMetricAveragedOverIOUs(ious=set([0.1, 0.2]), value="value", label_key="key")  # type: ignore - purposefully throwing error

    with pytest.raises(TypeError):
        map_averaged_metric = metrics.mAPMetricAveragedOverIOUs(
            ious=set([0.1, 0.2]), value=0.5, label_key=None  # type: ignore - purposefully throwing error
        )

    assert all(
        [
            key in ["label", "parameters", "value", "type"]
            for key in map_averaged_metric.to_dict()
        ]
    )


def test_ConfusionMatrixEntry():
    metrics.ConfusionMatrixEntry(
        prediction="pred", groundtruth="gt", count=123
    )

    with pytest.raises(TypeError):
        metrics.ConfusionMatrixEntry(
            prediction=None, groundtruth="gt", count=123  # type: ignore - purposefully throwing error
        )

    with pytest.raises(TypeError):
        metrics.ConfusionMatrixEntry(
            prediction="pred", groundtruth=123, count=123  # type: ignore - purposefully throwing error
        )

    with pytest.raises(TypeError):
        metrics.ConfusionMatrixEntry(
            prediction="pred", groundtruth="gt", count="not an int"  # type: ignore - purposefully throwing error
        )


def test__BaseConfusionMatrix():
    metrics._BaseConfusionMatrix(
        label_key="label",
        entries=[
            metrics.ConfusionMatrixEntry(
                prediction="pred1", groundtruth="gt1", count=123
            ),
            metrics.ConfusionMatrixEntry(
                prediction="pred2", groundtruth="gt2", count=234
            ),
        ],
    )

    with pytest.raises(TypeError):
        metrics._BaseConfusionMatrix(
            label_key=123,  # type: ignore - purposefully throwing error
            entries=[
                metrics.ConfusionMatrixEntry(
                    prediction="pred1", groundtruth="gt1", count=123
                ),
                metrics.ConfusionMatrixEntry(
                    prediction="pred2", groundtruth="gt2", count=234
                ),
            ],
        )

    with pytest.raises(TypeError):
        metrics._BaseConfusionMatrix(label_key="label", entries=None)  # type: ignore - purposefully throwing error

    with pytest.raises(TypeError):
        metrics._BaseConfusionMatrix(
            label_key="label", entries=["not an entry"]  # type: ignore - purposefully throwing error
        )


def test_ConfusionMatrix():
    confusion_matrix = metrics.ConfusionMatrix(
        label_key="label",
        entries=[
            metrics.ConfusionMatrixEntry(
                prediction="pred1", groundtruth="gt1", count=123
            ),
            metrics.ConfusionMatrixEntry(
                prediction="pred2", groundtruth="gt2", count=234
            ),
        ],
    )

    with pytest.raises(TypeError):
        metrics.ConfusionMatrix(
            label_key=123,
            entries=[
                metrics.ConfusionMatrixEntry(
                    prediction="pred1", groundtruth="gt1", count=123
                ),
                metrics.ConfusionMatrixEntry(
                    prediction="pred2", groundtruth="gt2", count=234
                ),
            ],
        )

    with pytest.raises(TypeError):
        metrics.ConfusionMatrix(label_key="label", entries=None)

    with pytest.raises(TypeError):
        metrics.ConfusionMatrix(label_key="label", entries=["not an entry"])

    assert all(
        [key in ["label_key", "entries"] for key in confusion_matrix.to_dict()]
    )


def test_AccuracyMetric():
    acc_metric = metrics.AccuracyMetric(label_key="key", value=0.5)

    with pytest.raises(TypeError):
        metrics.AccuracyMetric(label_key=None, value=0.5)  # type: ignore - purposefully throwing error

    with pytest.raises(TypeError):
        metrics.AccuracyMetric(label_key="key", value="value")  # type: ignore - purposefully throwing error

    assert all(
        [
            key in ["label", "parameters", "value", "type"]
            for key in acc_metric.to_dict()
        ]
    )


def test_PrecisionMetric():
    precision_recall_metric = metrics.PrecisionMetric(
        label=schemas.Label(key="key", value="value"), value=0.5
    )
    mapping = precision_recall_metric.to_dict()

    assert all([key in ["value", "type", "label"] for key in mapping])

    assert mapping["type"] == "Precision"


def test_RecallMetric():
    precision_recall_metric = metrics.RecallMetric(
        label=schemas.Label(key="key", value="value"), value=0.5
    )
    mapping = precision_recall_metric.to_dict()

    assert all(
        [key in ["label", "parameters", "value", "type"] for key in mapping]
    )

    assert mapping["type"] == "Recall"


def test_F1Metric():
    precision_recall_metric = metrics.F1Metric(
        label=schemas.Label(key="key", value="value"), value=0.5
    )
    mapping = precision_recall_metric.to_dict()

    assert all(
        [key in ["label", "parameters", "value", "type"] for key in mapping]
    )

    assert mapping["type"] == "F1"


def test_ROCAUCMetric():
    roc_auc_metric = metrics.ROCAUCMetric(label_key="key", value=0.2)

    with pytest.raises(TypeError):
        metrics.ROCAUCMetric(label_key=None, value=0.2)  # type: ignore - purposefully throwing error

    with pytest.raises(TypeError):
        metrics.ROCAUCMetric(label_key=123, value=0.2)  # type: ignore - purposefully throwing error

    with pytest.raises(TypeError):
        metrics.ROCAUCMetric(label_key="key", value="not a number")  # type: ignore - purposefully throwing error

    assert all(
        [
            key in ["value", "type", "evaluation_id", "parameters"]
            for key in roc_auc_metric.to_dict()
        ]
    )


def test_PrecisionRecallCurve():

    m = metrics.PrecisionRecallCurve(
        label_key="k1",
        pr_curve_iou_threshold=0.5,
        value={"v1": {0.25: {"tp": 1}}},
    )
    assert m.to_dict() == {
        "parameters": {"label_key": "k1"},
        "value": {"v1": {0.25: {"tp": 1}}},
        "type": "PrecisionRecallCurve",
    }


def test_DetailedPrecisionRecallCurve():

    m = metrics.DetailedPrecisionRecallCurve(
        label_key="k1",
        pr_curve_iou_threshold=0.5,
        value={"v1": {0.25: {"tp": {"total": 3}}}},
    )
    assert m.to_dict() == {
        "parameters": {"label_key": "k1"},
        "value": {"v1": {0.25: {"tp": {"total": 3}}}},
        "type": "DetailedPrecisionRecallCurve",
    }


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
