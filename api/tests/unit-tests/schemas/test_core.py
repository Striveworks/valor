import numpy as np
import pytest
from pydantic import ValidationError

from valor_api import schemas
from valor_api.schemas.validators import validate_type_string


def test_validate_type_string():
    validate_type_string("dataset1")
    validate_type_string("dataset-1")
    validate_type_string("dataset_1")
    validate_type_string("data!@#$%^&*()'set_1")


def test__format_uid():
    validate_type_string("uid1")
    validate_type_string("uid-1")
    validate_type_string("uid_1")
    validate_type_string("uid1.png")
    validate_type_string("folder/uid1.png")
    validate_type_string("uid!@#$%^&*()'_1")


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
        name="dataset1",
        metadata=metadata,
    )

    # test property `name`
    with pytest.raises(ValidationError):
        schemas.Dataset(
            name=(12,),  # type: ignore - purposefully throwing error
        )
    with pytest.raises(ValidationError):
        schemas.Dataset(name=None)  # type: ignore - purposefully throwing error

    # test property `metadata`
    with pytest.raises(ValidationError):
        schemas.Dataset(
            name="123",
            metadata={123: 12434},  # type: ignore - purposefully throwing error
        )
    with pytest.raises(ValidationError):
        schemas.Dataset(
            name="123",
            metadata=[{123: 12434}, "123"],  # type: ignore - purposefully throwing error
        )

    # test property `id`
    with pytest.raises(ValidationError):
        schemas.Dataset(
            id="value",  # type: ignore - purposefully throwing error
            name="123",
            metadata=[{123: 12434}, "123"],  # type: ignore - purposefully throwing error
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
        name="model1",
        metadata=metadata,
    )

    # test property `name`
    with pytest.raises(ValidationError):
        schemas.Model(
            name=(12,),  # type: ignore - purposefully throwing error
        )
    with pytest.raises(ValidationError):
        schemas.Dataset(name=None)  # type: ignore - purposefully throwing error

    # test property `metadata`
    with pytest.raises(ValidationError):
        schemas.Model(
            name="123",
            metadata={123: 123},  # type: ignore - purposefully throwing error
        )
    with pytest.raises(ValidationError):
        schemas.Model(
            name="123",
            metadata=[{123: 12434}, "123"],  # type: ignore - purposefully throwing error
        )

    # test property `id`
    with pytest.raises(ValidationError):
        schemas.Model(
            id="value",  # type: ignore - purposefully throwing error
            name="123",
            metadata=[{123: 12434}, "123"],  # type: ignore - purposefully throwing error
        )


def test_datum(metadata):
    # valid
    valid_datum = schemas.Datum(
        uid="123",
    )

    # test property `uid`
    with pytest.raises(ValidationError):
        schemas.Datum(
            uid=("uid",),  # type: ignore - purposefully throwing error
        )
    with pytest.raises(ValidationError):
        schemas.Datum(
            uid=123,  # type: ignore - purposefully throwing error
        )
    with pytest.raises(ValidationError):
        schemas.Datum(
            uid=None,  # type: ignore - purposefully throwing error
        )

    # test property `metadata`
    with pytest.raises(ValidationError):
        schemas.Datum(
            uid="123",
            metadata={123: 123},  # type: ignore - purposefully throwing error
        )

    # test `__eq__`
    other_datum = schemas.Datum(
        uid="123",
    )
    assert valid_datum == other_datum

    other_datum = schemas.Datum(uid="123", metadata={"fake": "metadata"})
    assert not valid_datum == other_datum


def test_annotation_without_scores(metadata, bbox, polygon, raster, labels):
    # valid
    gt = schemas.Annotation(
        labels=labels,
    )
    schemas.Annotation(
        labels=labels,
        metadata={},
        bounding_box=bbox,
    )
    schemas.Annotation(
        labels=labels,
        metadata={},
        raster=raster,
    )
    schemas.Annotation(
        labels=labels,
    )
    schemas.Annotation(
        labels=labels,
        bounding_box=bbox,
    )
    schemas.Annotation(
        labels=labels,
        raster=raster,
    )
    schemas.Annotation(
        labels=[],
    )

    # test property `labels`
    with pytest.raises(ValidationError):
        schemas.Annotation(
            labels=labels[0],
        )
    with pytest.raises(ValidationError):
        schemas.Annotation(
            labels=[labels[0], 123],  # type: ignore - purposefully throwing error
        )
    assert gt.labels == labels

    # test property `metadata`
    with pytest.raises(ValidationError):
        schemas.Annotation(
            labels=labels,
            metadata={123: 123},  # type: ignore - purposefully throwing error
        )

    # test geometric properties
    with pytest.raises(ValidationError):
        schemas.Annotation(
            labels=labels,
            bounding_box=polygon,
        )
    with pytest.raises(ValidationError):
        schemas.Annotation(
            labels=labels,
            polygon=bbox,
        )
    with pytest.raises(ValidationError):
        schemas.Annotation(
            labels=labels,
            multipolygon=bbox,  # type: ignore - purposefully throwing error
        )
    with pytest.raises(ValidationError):
        schemas.Annotation(
            labels=labels,
            raster=bbox,
        )


def test_annotation_with_scores(
    metadata, bbox, polygon, raster, scored_labels
):
    # valid
    pd = schemas.Annotation(labels=scored_labels)
    schemas.Annotation(
        labels=scored_labels,
        metadata={},
        bounding_box=bbox,
    )
    schemas.Annotation(
        labels=scored_labels,
        metadata={},
        raster=raster,
    )
    schemas.Annotation(
        labels=scored_labels,
    )
    schemas.Annotation(
        labels=scored_labels,
        bounding_box=bbox,
    )
    schemas.Annotation(
        labels=scored_labels,
        raster=raster,
    )

    # test property `scored_labels`
    with pytest.raises(ValidationError) as e:
        schemas.Annotation(
            labels=scored_labels[0],
        )
    assert "should be a valid dictionary or instance of Label" in str(
        e.value.errors()[0]["msg"]
    )

    assert set(pd.labels) == set(scored_labels)

    # test property `metadata`
    with pytest.raises(ValidationError):
        schemas.Annotation(
            labels=scored_labels,
            metadata=123,  # type: ignore - purposefully throwing error
        )
    with pytest.raises(ValidationError):
        schemas.Annotation(
            labels=scored_labels,
            metadata={123: "123"},  # type: ignore - purposefully throwing error
        )

    # test geometric properties
    with pytest.raises(ValidationError):
        schemas.Annotation(
            labels=scored_labels,
            bounding_box=polygon,
        )
    with pytest.raises(ValidationError):
        schemas.Annotation(
            labels=scored_labels,
            polygon=bbox,
        )
    with pytest.raises(ValidationError):
        schemas.Annotation(
            labels=scored_labels,
            multipolygon=bbox,  # type: ignore - purposefully throwing error
        )
    with pytest.raises(ValidationError) as e:
        schemas.Annotation(
            labels=scored_labels,
            raster=bbox,
        )


def test_groundtruth(metadata, groundtruth_annotations, raster):
    # valid
    schemas.GroundTruth(
        dataset_name="name",
        datum=schemas.Datum(
            uid="uid",
        ),
        annotations=[
            schemas.Annotation(
                labels=[schemas.Label(key="k1", value="v1")],
                raster=raster,
            )
        ],
    )
    gt = schemas.GroundTruth(
        dataset_name="name",
        datum=schemas.Datum(
            uid="uid",
        ),
        annotations=groundtruth_annotations,
    )

    # test property `datum`
    assert gt.datum == schemas.Datum(
        uid="uid",
    )
    with pytest.raises(ValidationError):
        schemas.GroundTruth(
            datum=schemas.Datum(  # type: ignore - purposefully throwing error
                uid="uid",
            ),
            annotations=groundtruth_annotations,
        )

    # test property `annotations`
    assert gt.annotations == groundtruth_annotations
    schemas.GroundTruth(
        dataset_name="name",
        datum=schemas.Datum(
            uid="uid",
        ),
        annotations=[],
    )
    with pytest.raises(ValidationError):
        schemas.GroundTruth(
            dataset_name="name",
            datum=schemas.Datum(
                uid="uid",
            ),
            annotations="annotation",  # type: ignore - purposefully throwing error
        )
    with pytest.raises(ValidationError):
        schemas.GroundTruth(
            dataset_name="name",
            datum=schemas.Datum(
                uid="uid",
            ),
            annotations=[groundtruth_annotations[0], 1234],  # type: ignore - purposefully throwing error
        )


def test_prediction(metadata, predicted_annotations, labels, scored_labels):
    # valid
    md = schemas.Prediction(
        dataset_name="name",
        model_name="name1",
        datum=schemas.Datum(uid="uid"),
        annotations=predicted_annotations,
    )

    # test property `model`
    assert md.model_name == "name1"
    with pytest.raises(ValidationError):
        schemas.Prediction(
            model_name=("name",),
            datum=schemas.Datum(uid="uid"),  # type: ignore - purposefully throwing error
            annotations=predicted_annotations,
        )
    with pytest.raises(ValidationError):
        schemas.Prediction(
            model_name="name@#$#@",
            datum=schemas.Datum(uid="uid"),  # type: ignore - purposefully throwing error
            annotations=predicted_annotations,
        )

    # test property `datum`
    assert md.datum == schemas.Datum(
        uid="uid",
    )
    with pytest.raises(ValidationError):
        schemas.Prediction(
            dataset_name="name",
            model_name="name",
            datum="datum_uid",  # type: ignore - purposefully throwing error
            annotations=predicted_annotations,
        )

    # test property `annotations`
    assert md.annotations == predicted_annotations
    schemas.Prediction(
        dataset_name="name",
        model_name="name",
        datum=schemas.Datum(
            uid="uid",
        ),
        annotations=[],
    )
    with pytest.raises(ValidationError):
        schemas.Prediction(
            dataset_name="name",
            model_name="name",
            datum=schemas.Datum(
                uid="uid",
            ),
            annotations="annotation",  # type: ignore - purposefully throwing error
        )
    with pytest.raises(ValidationError):
        schemas.Prediction(
            dataset_name="name",
            model_name="name",
            datum=schemas.Datum(
                uid="uid",
            ),
            annotations=[predicted_annotations[0], 1234],  # type: ignore - purposefully throwing error
        )

    # check sum to 1
    with pytest.raises(ValidationError) as e:
        schemas.Prediction(
            dataset_name="name",
            model_name="name",
            datum=schemas.Datum(
                uid="uid",
            ),
            annotations=[
                schemas.Annotation(
                    labels=scored_labels[1:],
                )
            ],
        )
    assert "prediction scores must sum to 1" in str(e.value.errors()[0]["msg"])

    # check score is provided
    with pytest.raises(ValueError) as e:
        schemas.Prediction(
            dataset_name="name",
            model_name="name",
            datum=schemas.Datum(
                uid="uid",
            ),
            annotations=[
                schemas.Annotation(
                    labels=labels,
                )
            ],
        )
    assert "Prediction labels must have scores for classification" in str(e)

    with pytest.raises(ValueError) as e:
        schemas.Prediction(
            dataset_name="name",
            model_name="name",
            datum=schemas.Datum(
                uid="uid",
            ),
            annotations=[
                schemas.Annotation(
                    labels=labels,
                    bounding_box=schemas.Box.from_extrema(0, 1, 0, 1),
                )
            ],
        )
    assert "Prediction labels must have scores for object detection" in str(e)

    with pytest.raises(ValueError) as e:
        schemas.Prediction(
            dataset_name="name",
            model_name="name",
            datum=schemas.Datum(
                uid="uid",
                metadata={
                    "height": 10,
                    "width": 10,
                },
            ),
            annotations=[
                schemas.Annotation(
                    labels=scored_labels,
                    raster=schemas.Raster.from_numpy(np.zeros((10, 10)) == 0),
                    is_instance_segmentation=False,
                )
            ],
        )
    assert "Semantic segmentation tasks cannot have scores" in str(e)

    # check inappropriate usage of is_instance_segmentation
    with pytest.raises(ValidationError) as e:
        schemas.Prediction(
            dataset_name="name",
            model_name="name",
            datum=schemas.Datum(
                uid="uid",
            ),
            annotations=[
                schemas.Annotation(
                    labels=scored_labels[1:], is_instance_segmentation=True
                )
            ],
        )
    assert (
        "is_instance_segmentation should only be used when passing a Raster"
        in str(e.value.errors()[0]["msg"])
    )


def test_semantic_segmentation_validation():
    # this is valid
    gt = schemas.GroundTruth(
        dataset_name="name",
        datum=schemas.Datum(
            uid="uid",
        ),
        annotations=[
            schemas.Annotation(
                labels=[
                    schemas.Label(key="k1", value="v1"),
                    schemas.Label(key="k2", value="v2"),
                ],
                raster=schemas.Raster.from_numpy(np.zeros((10, 10)) == 1),
            ),
            schemas.Annotation(
                labels=[schemas.Label(key="k1", value="v3")],
                raster=schemas.Raster.from_numpy(np.zeros((10, 10)) == 1),
            ),
        ],
    )

    assert len(gt.annotations) == 2

    with pytest.raises(ValidationError) as e:
        schemas.GroundTruth(
            dataset_name="name",
            datum=schemas.Datum(
                uid="uid",
            ),
            annotations=[
                schemas.Annotation(
                    labels=[
                        schemas.Label(key="k1", value="v1"),
                        schemas.Label(key="k1", value="v1"),
                    ],
                    raster=schemas.Raster.from_numpy(np.zeros((10, 10)) == 1),
                ),
                schemas.Annotation(
                    labels=[schemas.Label(key="k3", value="v3")],
                    raster=schemas.Raster.from_numpy(np.zeros((10, 10)) == 1),
                ),
            ],
        )
    assert "one annotation per label" in str(e.value)

    with pytest.raises(ValidationError) as e:
        schemas.GroundTruth(
            dataset_name="name",
            datum=schemas.Datum(
                uid="uid",
            ),
            annotations=[
                schemas.Annotation(
                    labels=[
                        schemas.Label(key="k1", value="v1"),
                        schemas.Label(key="k1", value="v2"),
                    ],
                    raster=schemas.Raster.from_numpy(np.zeros((10, 10)) == 1),
                ),
                schemas.Annotation(
                    labels=[schemas.Label(key="k1", value="v1")],
                    raster=schemas.Raster.from_numpy(np.zeros((10, 10)) == 1),
                ),
            ],
        )

    assert "one annotation per label" in str(e.value)

    # this is valid
    schemas.Prediction(
        dataset_name="name",
        model_name="model",
        datum=schemas.Datum(
            uid="uid",
        ),
        annotations=[
            schemas.Annotation(
                labels=[
                    schemas.Label(key="k1", value="v1"),
                    schemas.Label(key="k2", value="v2"),
                ],
                raster=schemas.Raster.from_numpy(np.zeros((10, 10)) == 1),
            ),
            schemas.Annotation(
                labels=[schemas.Label(key="k1", value="v3")],
                raster=schemas.Raster.from_numpy(np.zeros((10, 10)) == 1),
            ),
        ],
    )

    with pytest.raises(ValueError) as e:
        schemas.Prediction(
            dataset_name="name",
            model_name="model",
            datum=schemas.Datum(
                uid="uid",
            ),
            annotations=[
                schemas.Annotation(
                    labels=[
                        schemas.Label(key="k1", value="v1"),
                        schemas.Label(key="k1", value="v1"),
                    ],
                    raster=schemas.Raster.from_numpy(np.zeros((10, 10)) == 1),
                ),
                schemas.Annotation(
                    labels=[schemas.Label(key="k3", value="v3")],
                    raster=schemas.Raster.from_numpy(np.zeros((10, 10)) == 1),
                ),
            ],
        )

    assert "one annotation per label" in str(e.value)
