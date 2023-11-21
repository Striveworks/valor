import pytest
from pydantic import ValidationError

from velour_api import enums, schemas
from velour_api.schemas.core import _format_name, _format_uid


def test__format_name():
    assert _format_name("dataset1") == "dataset1"
    assert _format_name("dataset-1") == "dataset-1"
    assert _format_name("dataset_1") == "dataset_1"
    with pytest.raises(ValueError) as e:
        _format_name("data!@#$%^&*()'set_1")
    assert "illegal characters" in str(e)


def test__format_uid():
    assert _format_uid("uid1") == "uid1"
    assert _format_uid("uid-1") == "uid-1"
    assert _format_uid("uid_1") == "uid_1"
    assert _format_uid("uid1.png") == "uid1.png"
    assert _format_uid("folder/uid1.png") == "folder/uid1.png"
    with pytest.raises(ValueError) as e:
        _format_uid("uid!@#$%^&*()'_1")
    assert "illegal characters" in str(e)


def test_metadata_Metadatum():
    # valid
    schemas.Metadatum(key="name", value="value")
    schemas.Metadatum(
        key="name",
        value=123,
    )
    schemas.Metadatum(
        key="name",
        value=123.0,
    )

    # test property `name`
    with pytest.raises(ValidationError):
        schemas.Metadatum(
            key=("name",),
            value=123,
        )

    # test property `value`
    with pytest.raises(ValidationError):
        schemas.Metadatum(
            key="name",
            value=[1, 2, 3],
        )
    with pytest.raises(ValidationError):
        schemas.Metadatum(
            key="name",
            value=(1, 2, 3),
        )
    with pytest.raises(ValidationError):
        schemas.Metadatum(
            key="name",
            value=schemas.geometry.Point(x=1, y=1),
        )


def test_dataset(metadata):
    # valid
    schemas.Dataset(name="dataset1")
    schemas.Dataset(
        name="dataset1",
        metadata={},
    )
    schemas.Dataset(
        name="dataset1",
        metadata=metadata,
    )
    schemas.Dataset(
        id=1,
        name="dataset1",
        metadata=metadata,
    )

    # test property `name`
    with pytest.raises(ValidationError):
        schemas.Dataset(
            name=(12,),
        )
    with pytest.raises(ValidationError):
        schemas.Dataset(name="dataset@")

    # test property `metadata`
    with pytest.raises(ValidationError):
        schemas.Dataset(
            name="123",
            metadata={123: 12434},
        )
    with pytest.raises(ValidationError):
        schemas.Dataset(
            name="123",
            metadata=[{123: 12434}, "123"],
        )

    # test property `id`
    with pytest.raises(ValidationError):
        schemas.Dataset(
            id="value",
            name="123",
            metadata=[{123: 12434}, "123"],
        )


def test_model(metadata):
    # valid
    schemas.Model(name="model1")
    schemas.Model(
        name="model1",
        metadata={},
    )
    schemas.Model(
        name="model1",
        metadata=metadata,
    )
    schemas.Model(
        id=1,
        name="model1",
        metadata=metadata,
    )

    # test property `name`
    with pytest.raises(ValidationError):
        schemas.Model(
            name=(12,),
        )
    with pytest.raises(ValidationError):
        schemas.Dataset(name="model@")

    # test property `metadata`
    with pytest.raises(ValidationError):
        schemas.Model(
            name="123",
            metadata={123: 123},
        )
    with pytest.raises(ValidationError):
        schemas.Model(
            name="123",
            metadata=[{123: 12434}, "123"],
        )

    # test property `id`
    with pytest.raises(ValidationError):
        schemas.Model(
            id="value",
            name="123",
            metadata=[{123: 12434}, "123"],
        )


def test_datum(metadata):
    # valid
    schemas.Datum(
        uid="123",
        dataset="name",
    )

    # test property `uid`
    with pytest.raises(ValidationError):
        schemas.Datum(
            uid=("uid",),
            dataset="name",
        )
    with pytest.raises(ValidationError):
        schemas.Datum(
            uid="uid@",
            dataset="name",
        )
    with pytest.raises(ValidationError):
        schemas.Datum(
            uid=123,
            dataset="name",
        )

    # test property `metadata`
    with pytest.raises(ValidationError):
        schemas.Datum(
            uid="123",
            dataset="name",
            metadata={123: 123},
        )


def test_annotation_without_scores(metadata, bbox, polygon, raster, labels):
    # valid
    gt = schemas.Annotation(
        task_type=enums.TaskType.CLASSIFICATION,
        labels=labels,
    )
    schemas.Annotation(
        task_type=enums.TaskType.DETECTION,
        labels=labels,
        metadata={},
    )
    schemas.Annotation(
        task_type=enums.TaskType.SEGMENTATION,
        labels=labels,
        metadata={},
        bounding_box=bbox,
        polygon=polygon,
        raster=raster,
    )
    schemas.Annotation(
        task_type=enums.TaskType.CLASSIFICATION.value,
        labels=labels,
    )
    schemas.Annotation(
        task_type=enums.TaskType.DETECTION.value,
        labels=labels,
    )
    schemas.Annotation(
        task_type=enums.TaskType.SEGMENTATION.value,
        labels=labels,
    )

    # test property `task_type`
    with pytest.raises(ValidationError):
        schemas.Annotation(task_type="custom")

    # test property `labels`
    with pytest.raises(ValidationError):
        schemas.Annotation(
            labels=labels[0],
            task_type=enums.TaskType.CLASSIFICATION,
        )
    with pytest.raises(ValidationError):
        schemas.Annotation(
            labels=[labels[0], 123],
            task_type=enums.TaskType.CLASSIFICATION,
        )
    with pytest.raises(ValidationError):
        schemas.Annotation(
            labels=[],
            task_type=enums.TaskType.CLASSIFICATION,
        )
    assert gt.labels == labels

    # test property `metadata`
    with pytest.raises(ValidationError):
        schemas.Annotation(
            task_type=enums.TaskType.CLASSIFICATION.value,
            labels=labels,
            metadata={123: 123},
        )

    # test geometric properties
    with pytest.raises(ValidationError):
        schemas.Annotation(
            task_type=enums.TaskType.DETECTION,
            labels=labels,
            bounding_box=polygon,
        )
    with pytest.raises(ValidationError):
        schemas.Annotation(
            task_type=enums.TaskType.DETECTION,
            labels=labels,
            polygon=bbox,
        )
    with pytest.raises(ValidationError):
        schemas.Annotation(
            task_type=enums.TaskType.DETECTION,
            labels=labels,
            multipolygon=bbox,
        )
    with pytest.raises(ValidationError):
        schemas.Annotation(
            task_type=enums.TaskType.DETECTION,
            labels=labels,
            raster=bbox,
        )


def test_annotation_with_scores(
    metadata, bbox, polygon, raster, scored_labels
):
    # valid
    pd = schemas.Annotation(
        task_type=enums.TaskType.CLASSIFICATION, labels=scored_labels
    )
    schemas.Annotation(
        task_type=enums.TaskType.DETECTION, labels=scored_labels, metadata={}
    )
    schemas.Annotation(
        task_type=enums.TaskType.SEGMENTATION,
        labels=scored_labels,
        metadata={},
        bounding_box=bbox,
        polygon=polygon,
        raster=raster,
    )
    schemas.Annotation(
        task_type=enums.TaskType.CLASSIFICATION.value, labels=scored_labels
    )
    schemas.Annotation(
        task_type=enums.TaskType.DETECTION.value, labels=scored_labels
    )
    schemas.Annotation(
        task_type=enums.TaskType.SEGMENTATION.value, labels=scored_labels
    )

    # test property `task_type`
    with pytest.raises(ValidationError):
        schemas.Annotation(
            task_type="custom",
        )

    # test property `scored_labels`
    with pytest.raises(ValidationError) as e:
        schemas.Annotation(
            labels=scored_labels[0], task_type=enums.TaskType.CLASSIFICATION
        )
    assert "should be a valid dictionary or instance of Label" in str(
        e.value.errors()[0]["msg"]
    )

    assert set(pd.labels) == set(scored_labels)

    # test property `metadata`
    with pytest.raises(ValidationError):
        schemas.Annotation(
            task_type=enums.TaskType.CLASSIFICATION.value,
            labels=scored_labels,
            metadata=123,
        )
    with pytest.raises(ValidationError):
        schemas.Annotation(
            task_type=enums.TaskType.CLASSIFICATION.value,
            labels=scored_labels,
            metadata={123: "123"},
        )

    # test geometric properties
    with pytest.raises(ValidationError):
        schemas.Annotation(
            task_type=enums.TaskType.DETECTION,
            labels=scored_labels,
            bounding_box=polygon,
        )
    with pytest.raises(ValidationError):
        schemas.Annotation(
            task_type=enums.TaskType.DETECTION,
            labels=scored_labels,
            polygon=bbox,
        )
    with pytest.raises(ValidationError):
        schemas.Annotation(
            task_type=enums.TaskType.DETECTION,
            labels=scored_labels,
            multipolygon=bbox,
        )
    with pytest.raises(ValidationError) as e:
        schemas.Annotation(
            task_type=enums.TaskType.DETECTION,
            labels=scored_labels,
            raster=bbox,
        )


def test_groundtruth(metadata, groundtruth_annotations):
    # valid
    gt = schemas.GroundTruth(
        datum=schemas.Datum(
            uid="uid",
            dataset="name",
        ),
        annotations=groundtruth_annotations,
    )

    # test property `datum`
    assert gt.datum == schemas.Datum(
        uid="uid",
        dataset="name",
    )
    with pytest.raises(ValidationError):
        schemas.GroundTruth(
            datum=schemas.Datum(
                uid="uid",
            ),
            annotations=groundtruth_annotations,
        )

    # test property `annotations`
    assert gt.annotations == groundtruth_annotations
    with pytest.raises(ValidationError):
        schemas.GroundTruth(
            datum=schemas.Datum(
                uid="uid",
                dataset="name",
            ),
            annotations="annotation",
        )
    with pytest.raises(ValidationError):
        schemas.GroundTruth(
            datum=schemas.Datum(
                uid="uid",
                dataset="name",
            ),
            annotations=[],
        )
    with pytest.raises(ValidationError):
        schemas.GroundTruth(
            datum=schemas.Datum(
                uid="uid",
                dataset="name",
            ),
            annotations=[groundtruth_annotations[0], 1234],
        )


def test_prediction(metadata, predicted_annotations, labels, scored_labels):
    # valid
    md = schemas.Prediction(
        model="name1",
        datum=schemas.Datum(uid="uid", dataset="name"),
        annotations=predicted_annotations,
    )

    # test property `model`
    assert md.model == "name1"
    with pytest.raises(ValidationError):
        schemas.Prediction(
            model=("name",),
            datum=schemas.Datum(uid="uid"),
            annotations=predicted_annotations,
        )
    with pytest.raises(ValidationError):
        schemas.Prediction(
            model="name@#$#@",
            datum=schemas.Datum(uid="uid"),
            annotations=predicted_annotations,
        )

    # test property `datum`
    assert md.datum == schemas.Datum(
        uid="uid",
        dataset="name",
    )
    with pytest.raises(ValidationError):
        schemas.Prediction(
            model="name",
            datum="datum_uid",
            annotations=predicted_annotations,
        )

    # test property `annotations`
    assert md.annotations == predicted_annotations
    with pytest.raises(ValidationError):
        schemas.Prediction(
            model="name",
            datum=schemas.Datum(
                uid="uid",
                dataset="name",
            ),
            annotations="annotation",
        )
    with pytest.raises(ValidationError):
        schemas.Prediction(
            model="name",
            datum=schemas.Datum(
                uid="uid",
                dataset="name",
            ),
            annotations=[],
        )
    with pytest.raises(ValidationError):
        schemas.Prediction(
            model="name",
            datum=schemas.Datum(
                uid="uid",
                dataset="name",
            ),
            annotations=[predicted_annotations[0], 1234],
        )

    # check sum to 1
    with pytest.raises(ValidationError) as e:
        schemas.Prediction(
            model="name",
            datum=schemas.Datum(
                uid="uid",
                dataset="name",
            ),
            annotations=[
                schemas.Annotation(
                    labels=scored_labels[1:],
                    task_type=enums.TaskType.CLASSIFICATION,
                )
            ],
        )
    assert "prediction scores must sum to 1" in str(e.value.errors()[0]["msg"])

    # check score is provided
    for task_type in [
        enums.TaskType.CLASSIFICATION,
        enums.TaskType.DETECTION,
    ]:
        with pytest.raises(ValueError) as e:
            schemas.Prediction(
                model="name",
                datum=schemas.Datum(
                    uid="uid",
                    dataset="name",
                ),
                annotations=[
                    schemas.Annotation(labels=labels, task_type=task_type)
                ],
            )
        assert "Missing score for label" in str(e)

    with pytest.raises(ValueError) as e:
        schemas.Prediction(
            model="name",
            datum=schemas.Datum(
                uid="uid",
                dataset="name",
            ),
            annotations=[
                schemas.Annotation(
                    labels=scored_labels,
                    task_type=enums.TaskType.SEGMENTATION,
                )
            ],
        )
    assert "Semantic segmentation tasks cannot have scores" in str(e)


def test_semantic_segmentation_validation():
    # this is valid
    gt = schemas.GroundTruth(
        datum=schemas.Datum(
            uid="uid",
            dataset="name",
        ),
        annotations=[
            schemas.Annotation(
                task_type=enums.TaskType.SEGMENTATION,
                labels=[
                    schemas.Label(key="k1", value="v1"),
                    schemas.Label(key="k2", value="v2"),
                ],
            ),
            schemas.Annotation(
                task_type=enums.TaskType.SEGMENTATION,
                labels=[schemas.Label(key="k1", value="v3")],
            ),
        ],
    )

    assert len(gt.annotations) == 2

    with pytest.raises(ValidationError) as e:
        schemas.GroundTruth(
            datum=schemas.Datum(
                uid="uid",
                dataset="name",
            ),
            annotations=[
                schemas.Annotation(
                    task_type=enums.TaskType.SEGMENTATION,
                    labels=[
                        schemas.Label(key="k1", value="v1"),
                        schemas.Label(key="k1", value="v1"),
                    ],
                ),
                schemas.Annotation(
                    task_type=enums.TaskType.SEGMENTATION,
                    labels=[schemas.Label(key="k3", value="v3")],
                ),
            ],
        )
    assert "one annotation per label" in str(e.value)

    with pytest.raises(ValidationError) as e:
        schemas.GroundTruth(
            datum=schemas.Datum(
                uid="uid",
                dataset="name",
            ),
            annotations=[
                schemas.Annotation(
                    task_type=enums.TaskType.SEGMENTATION,
                    labels=[
                        schemas.Label(key="k1", value="v1"),
                        schemas.Label(key="k1", value="v2"),
                    ],
                ),
                schemas.Annotation(
                    task_type=enums.TaskType.SEGMENTATION,
                    labels=[schemas.Label(key="k1", value="v1")],
                ),
            ],
        )

    assert "one annotation per label" in str(e.value)

    # this is valid
    schemas.Prediction(
        model="model",
        datum=schemas.Datum(
            uid="uid",
            dataset="name",
        ),
        annotations=[
            schemas.Annotation(
                task_type=enums.TaskType.SEGMENTATION,
                labels=[
                    schemas.Label(key="k1", value="v1"),
                    schemas.Label(key="k2", value="v2"),
                ],
            ),
            schemas.Annotation(
                task_type=enums.TaskType.SEGMENTATION,
                labels=[schemas.Label(key="k1", value="v3")],
            ),
        ],
    )

    with pytest.raises(ValueError) as e:
        schemas.Prediction(
            model="model",
            datum=schemas.Datum(
                uid="uid",
                dataset="name",
            ),
            annotations=[
                schemas.Annotation(
                    task_type=enums.TaskType.SEGMENTATION,
                    labels=[
                        schemas.Label(key="k1", value="v1"),
                        schemas.Label(key="k1", value="v1"),
                    ],
                ),
                schemas.Annotation(
                    task_type=enums.TaskType.SEGMENTATION,
                    labels=[schemas.Label(key="k3", value="v3")],
                ),
            ],
        )

    assert "one annotation per label" in str(e.value)
