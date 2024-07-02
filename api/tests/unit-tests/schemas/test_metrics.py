import pytest
from pydantic import ValidationError

from valor_api import schemas


def test_Metric():
    schemas.Metric(
        type="detection",
        parameters={},
        value=0.2,
        label=schemas.Label(key="k1", value="v1"),
    )

    schemas.Metric(type="detection")

    with pytest.raises(ValidationError):
        schemas.Metric(
            type="detection",
            parameters=123,  # type: ignore - purposefully throwing error
            value=0.2,
            label=schemas.Label(key="k1", value="v1"),
        )


def test_APMetric():
    ap_metric = schemas.APMetric(
        iou=0.2, value=0.5, label=schemas.Label(key="k1", value="v1")
    )

    with pytest.raises(ValidationError):
        schemas.APMetric(
            iou=None, value=0.5, label=schemas.Label(key="k1", value="v1")  # type: ignore - purposefully throwing error
        )

    with pytest.raises(ValidationError):
        schemas.APMetric(
            iou=0.1, value=None, label=schemas.Label(key="k1", value="v1")  # type: ignore - purposefully throwing error
        )

    with pytest.raises(ValidationError):
        schemas.APMetric(iou=0.1, value=0.5, label="k1")  # type: ignore - purposefully throwing error

    assert all(
        [
            key in ["value", "label_id", "type", "evaluation_id", "parameters"]
            for key in ap_metric.db_mapping(label_id=1, evaluation_id=1)
        ]
    )


def test_APMetricAveragedOverIOUs():
    ap_averaged_metric = schemas.APMetricAveragedOverIOUs(
        ious=set([0.1, 0.2]),
        value=0.5,
        label=schemas.Label(key="k1", value="v1"),
    )

    with pytest.raises(ValidationError):
        schemas.APMetricAveragedOverIOUs(
            ious=None, value=0.5, label=schemas.Label(key="k1", value="v1")  # type: ignore - purposefully throwing error
        )

    with pytest.raises(ValidationError):
        schemas.APMetricAveragedOverIOUs(
            ious=set([0.1, 0.2]),
            value=None,  # type: ignore - purposefully throwing error
            label=schemas.Label(key="k1", value="v1"),
        )

    with pytest.raises(ValidationError):
        schemas.APMetricAveragedOverIOUs(
            ious=set([0.1, 0.2]), value=0.5, label="k1"  # type: ignore - purposefully throwing error
        )

    assert all(
        [
            key in ["value", "label_id", "type", "evaluation_id", "parameters"]
            for key in ap_averaged_metric.db_mapping(
                label_id=1, evaluation_id=1
            )
        ]
    )


def test_mAPMetric():
    map_metric = schemas.mAPMetric(iou=0.2, value=0.5, label_key="key")

    with pytest.raises(ValidationError):
        schemas.mAPMetric(iou=None, value=0.5, label_key="key")  # type: ignore - purposefully throwing error

    with pytest.raises(ValidationError):
        schemas.mAPMetric(iou=0.1, value=None, label_key="key")  # type: ignore - purposefully throwing error

    with pytest.raises(ValidationError):
        schemas.mAPMetric(iou=0.1, value="value", label_key="key")  # type: ignore - purposefully throwing error

    with pytest.raises(ValidationError):
        schemas.mAPMetric(iou=0.1, value=0.5, label_key=None)  # type: ignore - purposefully throwing error

    assert all(
        [
            key in ["value", "type", "evaluation_id", "parameters"]
            for key in map_metric.db_mapping(evaluation_id=1)
        ]
    )


def test_mAPMetricAveragedOverIOUs():
    map_averaged_metric = schemas.mAPMetricAveragedOverIOUs(
        ious=set([0.1, 0.2]), value=0.5, label_key="key"
    )

    with pytest.raises(ValidationError):
        schemas.mAPMetricAveragedOverIOUs(ious=None, value=0.5, label_key="key")  # type: ignore - purposefully throwing error

    with pytest.raises(ValidationError):
        schemas.mAPMetricAveragedOverIOUs(ious=set([0.1, 0.2]), value=None, label_key="key")  # type: ignore - purposefully throwing error

    with pytest.raises(ValidationError):
        schemas.mAPMetricAveragedOverIOUs(ious=set([0.1, 0.2]), value="value", label_key="key")  # type: ignore - purposefully throwing error

    with pytest.raises(ValidationError):
        map_averaged_metric = schemas.mAPMetricAveragedOverIOUs(
            ious=set([0.1, 0.2]), value=0.5, label_key=None  # type: ignore - purposefully throwing error
        )

    assert all(
        [
            key in ["value", "type", "evaluation_id", "parameters"]
            for key in map_averaged_metric.db_mapping(evaluation_id=1)
        ]
    )


def test_ConfusionMatrixEntry():
    schemas.ConfusionMatrixEntry(
        prediction="pred", groundtruth="gt", count=123
    )

    with pytest.raises(ValidationError):
        schemas.ConfusionMatrixEntry(
            prediction=None, groundtruth="gt", count=123  # type: ignore - purposefully throwing error
        )

    with pytest.raises(ValidationError):
        schemas.ConfusionMatrixEntry(
            prediction="pred", groundtruth=123, count=123  # type: ignore - purposefully throwing error
        )

    with pytest.raises(ValidationError):
        schemas.ConfusionMatrixEntry(
            prediction="pred", groundtruth="gt", count="not an int"  # type: ignore - purposefully throwing error
        )


def test__BaseConfusionMatrix():
    schemas.metrics._BaseConfusionMatrix(
        label_key="label",
        entries=[
            schemas.ConfusionMatrixEntry(
                prediction="pred1", groundtruth="gt1", count=123
            ),
            schemas.ConfusionMatrixEntry(
                prediction="pred2", groundtruth="gt2", count=234
            ),
        ],
    )

    with pytest.raises(ValidationError):
        schemas.metrics._BaseConfusionMatrix(
            label_key=123,  # type: ignore - purposefully throwing error
            entries=[
                schemas.ConfusionMatrixEntry(
                    prediction="pred1", groundtruth="gt1", count=123
                ),
                schemas.ConfusionMatrixEntry(
                    prediction="pred2", groundtruth="gt2", count=234
                ),
            ],
        )

    with pytest.raises(ValidationError):
        schemas.metrics._BaseConfusionMatrix(label_key="label", entries=None)  # type: ignore - purposefully throwing error

    with pytest.raises(ValidationError):
        schemas.metrics._BaseConfusionMatrix(
            label_key="label", entries=["not an entry"]  # type: ignore - purposefully throwing error
        )


def test_ConfusionMatrix():
    confusion_matrix = schemas.metrics.ConfusionMatrix(
        label_key="label",
        entries=[
            schemas.ConfusionMatrixEntry(
                prediction="pred1", groundtruth="gt1", count=123
            ),
            schemas.ConfusionMatrixEntry(
                prediction="pred2", groundtruth="gt2", count=234
            ),
        ],
    )

    with pytest.raises(ValidationError):
        schemas.metrics.ConfusionMatrix(
            label_key=123,
            entries=[
                schemas.ConfusionMatrixEntry(
                    prediction="pred1", groundtruth="gt1", count=123
                ),
                schemas.ConfusionMatrixEntry(
                    prediction="pred2", groundtruth="gt2", count=234
                ),
            ],
        )

    with pytest.raises(ValidationError):
        schemas.metrics.ConfusionMatrix(label_key="label", entries=None)

    with pytest.raises(ValidationError):
        schemas.metrics.ConfusionMatrix(
            label_key="label", entries=["not an entry"]
        )

    assert all(
        [
            key in ["label_key", "value", "evaluation_id"]
            for key in confusion_matrix.db_mapping(evaluation_id=1)
        ]
    )


def test_AccuracyMetric():
    acc_metric = schemas.AccuracyMetric(label_key="key", value=0.5)

    with pytest.raises(ValidationError):
        schemas.AccuracyMetric(label_key=None, value=0.5)  # type: ignore - purposefully throwing error

    with pytest.raises(ValidationError):
        schemas.AccuracyMetric(label_key="key", value="value")  # type: ignore - purposefully throwing error

    assert all(
        [
            key in ["value", "type", "evaluation_id", "parameters"]
            for key in acc_metric.db_mapping(evaluation_id=1)
        ]
    )


def test__PrecisionRecallF1Base():
    schemas.metrics._PrecisionRecallF1Base(
        label=schemas.Label(key="key", value="value"), value=0.5
    )

    null_value = schemas.metrics._PrecisionRecallF1Base(
        label=schemas.Label(key="key", value="value"), value=None
    )

    assert null_value.value == -1

    with pytest.raises(ValidationError):
        schemas.metrics._PrecisionRecallF1Base(label=None, value=0.5)  # type: ignore - purposefully throwing error

    with pytest.raises(ValidationError):
        schemas.metrics._PrecisionRecallF1Base(
            label=schemas.Label(key="key", value="value"), value="value"  # type: ignore - purposefully throwing error
        )


def test_PrecisionMetric():
    precision_recall_metric = schemas.metrics.PrecisionMetric(
        label=schemas.Label(key="key", value="value"), value=0.5
    )
    mapping = precision_recall_metric.db_mapping(label_id=1, evaluation_id=2)

    assert all(
        [
            key in ["value", "type", "evaluation_id", "label_id"]
            for key in mapping
        ]
    )

    assert mapping["type"] == "Precision"


def test_RecallMetric():
    precision_recall_metric = schemas.metrics.RecallMetric(
        label=schemas.Label(key="key", value="value"), value=0.5
    )
    mapping = precision_recall_metric.db_mapping(label_id=1, evaluation_id=2)

    assert all(
        [
            key in ["value", "type", "evaluation_id", "label_id"]
            for key in mapping
        ]
    )

    assert mapping["type"] == "Recall"


def test_F1Metric():
    precision_recall_metric = schemas.metrics.F1Metric(
        label=schemas.Label(key="key", value="value"), value=0.5
    )
    mapping = precision_recall_metric.db_mapping(label_id=1, evaluation_id=2)

    assert all(
        [
            key in ["value", "type", "evaluation_id", "label_id"]
            for key in mapping
        ]
    )

    assert mapping["type"] == "F1"


def test_ROCAUCMetric():
    roc_auc_metric = schemas.ROCAUCMetric(label_key="key", value=0.2)

    with pytest.raises(ValidationError):
        schemas.ROCAUCMetric(label_key=None, value=0.2)  # type: ignore - purposefully throwing error

    with pytest.raises(ValidationError):
        schemas.ROCAUCMetric(label_key=123, value=0.2)  # type: ignore - purposefully throwing error

    with pytest.raises(ValidationError):
        schemas.ROCAUCMetric(label_key="key", value="not a number")  # type: ignore - purposefully throwing error

    assert all(
        [
            key in ["value", "type", "evaluation_id", "parameters"]
            for key in roc_auc_metric.db_mapping(evaluation_id=1)
        ]
    )


def test_IOUMetric():
    iou_metric = schemas.IOUMetric(
        label=schemas.Label(key="key", value="value"), value=0.2
    )

    with pytest.raises(ValidationError):
        schemas.IOUMetric(label=None, value=0.2)  # type: ignore - purposefully throwing error

    with pytest.raises(ValidationError):
        schemas.IOUMetric(label="not a label", value=0.2)  # type: ignore - purposefully throwing error

    with pytest.raises(ValidationError):
        schemas.IOUMetric(
            label=schemas.Label(key="key", value="value"), value=None  # type: ignore - purposefully throwing error
        )

    with pytest.raises(ValidationError):
        schemas.IOUMetric(
            label=schemas.Label(key="key", value="value"), value="not a value"  # type: ignore - purposefully throwing error
        )
    assert all(
        [
            key in ["value", "type", "evaluation_id", "label_id"]
            for key in iou_metric.db_mapping(evaluation_id=1, label_id=2)
        ]
    )


def test_mIOUMetric():
    iou_metric = schemas.mIOUMetric(value=0.2, label_key="key")

    with pytest.raises(ValidationError):
        schemas.mIOUMetric(value=None, label_key="key")  # type: ignore - purposefully throwing error

    with pytest.raises(ValidationError):
        schemas.mIOUMetric(value="not a value", label_key="key")  # type: ignore - purposefully throwing error

    with pytest.raises(ValidationError):
        schemas.mIOUMetric(value="not a value")  # type: ignore - purposefully throwing error

    assert all(
        [
            key in ["value", "type", "evaluation_id", "parameters"]
            for key in iou_metric.db_mapping(evaluation_id=1)
        ]
    )


def test_AnswerRelevanceMetric():
    metric = schemas.AnswerRelevanceMetric(
        value=0.421,
        parameters={
            "dataset_uid": "01",
            "dataset_name": "test_dataset",
            "prediction": "some prediction",
        },
    )

    with pytest.raises(ValidationError):
        schemas.AnswerRelevanceMetric(
            value=None,  # type: ignore
            parameters={
                "dataset_uid": "01",
                "dataset_name": "test_dataset",
                "prediction": "some prediction",
            },
        )

    with pytest.raises(ValidationError):
        schemas.AnswerRelevanceMetric(
            value={"key": 0.6},  # type: ignore
            parameters={
                "dataset_uid": "01",
                "dataset_name": "test_dataset",
                "prediction": "some prediction",
            },
        )

    with pytest.raises(ValidationError):
        schemas.AnswerRelevanceMetric(
            value=0.42,  # type: ignore
            parameters="not a valid parameter",  # type: ignore
        )

    assert all(
        [
            key in ["value", "type", "evaluation_id", "parameters"]
            for key in metric.db_mapping(evaluation_id=1)
        ]
    )


def test_BLEUMetric():
    metric = schemas.BLEUMetric(
        value=0.421,
        parameters={
            "dataset_uid": "01",
            "dataset_name": "test_dataset",
            "prediction": "some prediction",
        },
    )

    with pytest.raises(ValidationError):
        schemas.BLEUMetric(
            value=None,  # type: ignore
            parameters={
                "dataset_uid": "01",
                "dataset_name": "test_dataset",
                "prediction": "some prediction",
            },
        )

    with pytest.raises(ValidationError):
        schemas.BLEUMetric(
            value={"some": "dict"},  # type: ignore
            parameters={
                "dataset_uid": "01",
                "dataset_name": "test_dataset",
                "prediction": "some prediction",
            },
        )

    with pytest.raises(ValidationError):
        schemas.BLEUMetric(
            value={
                "rouge1": 1.0,
                "rouge2": 1.0,
                "rougeL": 1.0,
                "rougeLsum": 1.0,
            },  # type: ignore
            parameters={
                "dataset_uid": "01",
                "dataset_name": "test_dataset",
                "prediction": "some prediction",
            },
        )

    with pytest.raises(ValidationError):
        schemas.BLEUMetric(
            value=0.42,  # type: ignore
            parameters="not a valid parameter",  # type: ignore
        )

    assert all(
        [
            key in ["value", "type", "evaluation_id", "parameters"]
            for key in metric.db_mapping(evaluation_id=1)
        ]
    )


def test_CoherenceMetric():
    metric = schemas.CoherenceMetric(
        value=3,
        parameters={
            "dataset_uid": "01",
            "dataset_name": "test_dataset",
            "prediction": "some prediction",
        },
    )

    with pytest.raises(ValidationError):
        schemas.CoherenceMetric(
            value=None,  # type: ignore
            parameters={
                "dataset_uid": "01",
                "dataset_name": "test_dataset",
                "prediction": "some prediction",
            },
        )

    with pytest.raises(ValidationError):
        schemas.CoherenceMetric(
            value=2.5,  # type: ignore
            parameters={
                "dataset_uid": "01",
                "dataset_name": "test_dataset",
                "prediction": "some prediction",
            },
        )

    with pytest.raises(ValidationError):
        schemas.CoherenceMetric(
            value={"key": 4},  # type: ignore
            parameters={
                "dataset_uid": "01",
                "dataset_name": "test_dataset",
                "prediction": "some prediction",
            },
        )

    with pytest.raises(ValidationError):
        schemas.CoherenceMetric(
            value=5,  # type: ignore
            parameters="not a valid parameter",  # type: ignore
        )

    assert all(
        [
            key in ["value", "type", "evaluation_id", "parameters"]
            for key in metric.db_mapping(evaluation_id=1)
        ]
    )


def test_ROUGEMetric():
    metric = schemas.ROUGEMetric(
        value={
            "rouge1": 1.0,
            "rouge2": 1.0,
            "rougeL": 1.0,
            "rougeLsum": 1.0,
        },
        parameters={
            "dataset_uid": "01",
            "dataset_name": "test_dataset",
            "prediction": "some prediction",
        },
    )

    with pytest.raises(ValidationError):
        schemas.ROUGEMetric(
            value=None,  # type: ignore
            parameters={
                "dataset_uid": "01",
                "dataset_name": "test_dataset",
                "prediction": "some prediction",
            },
        )

    with pytest.raises(ValidationError):
        schemas.ROUGEMetric(
            value=0.24,  # type: ignore
            parameters={
                "dataset_uid": "01",
                "dataset_name": "test_dataset",
                "prediction": "some prediction",
            },
        )

    with pytest.raises(ValidationError):
        schemas.ROUGEMetric(
            value={
                "rouge1": 1.0,
                "rouge2": 1.0,
                "rougeL": 1.0,
                "rougeLsum": 1.0,
            },
            parameters=None,  # type: ignore
        )

    with pytest.raises(ValidationError):
        schemas.ROUGEMetric(
            value={
                "rouge1": 1.0,
                "rouge2": 1.0,
                "rougeL": 1.0,
                "rougeLsum": 1.0,
            },
            parameters="not a valid parameter",  # type: ignore
        )

    assert all(
        [
            key in ["value", "type", "evaluation_id", "parameters"]
            for key in metric.db_mapping(evaluation_id=1)
        ]
    )
