from base64 import b64encode

import pytest
from sqlalchemy.orm import Session

from velour_api import crud, enums, schemas
from velour_api.backend.query.label import (
    _get_dataset_label_keys,
    _get_dataset_labels,
    _get_model_label_keys,
    _get_model_labels,
)

dset_name = "test_dataset"
model_name = "test_model"


@pytest.fixture
def semantic_seg_gt_anns1(
    img1_gt_mask_bytes1: bytes,
) -> schemas.Annotation:
    return schemas.Annotation(
        task_type=enums.TaskType.SEG,
        raster=schemas.Raster(mask=b64encode(img1_gt_mask_bytes1).decode()),
        labels=[
            schemas.Label(key="semsegk1", value="semsegv1"),
            schemas.Label(key="semsegk2", value="semsegv2"),
        ],
    )


@pytest.fixture
def semantic_seg_gt_anns2(
    img2_gt_mask_bytes1: bytes,
) -> schemas.Annotation:
    return schemas.Annotation(
        task_type=enums.TaskType.SEG,
        raster=schemas.Raster(mask=b64encode(img2_gt_mask_bytes1).decode()),
        labels=[
            schemas.Label(key="semsegk2", value="semsegv2"),
            schemas.Label(key="semsegk3", value="semsegv3"),
        ],
    )


@pytest.fixture
def semantic_seg_pred_anns1(img1_gt_mask_bytes1: bytes) -> schemas.Annotation:
    return schemas.Annotation(
        task_type=enums.TaskType.SEG,
        raster=schemas.Raster(mask=b64encode(img1_gt_mask_bytes1).decode()),
        labels=[
            schemas.Label(key="semsegk1", value="semsegv1"),
            schemas.Label(key="semsegk2", value="semsegv2"),
        ],
    )


@pytest.fixture
def semantic_seg_pred_anns2(img2_gt_mask_bytes1: bytes) -> schemas.Annotation:
    return schemas.Annotation(
        task_type=enums.TaskType.SEG,
        raster=schemas.Raster(mask=b64encode(img2_gt_mask_bytes1).decode()),
        labels=[
            schemas.Label(key="semsegk2", value="semsegv2"),
            schemas.Label(key="semsegk3_pred", value="semsegv3_pred"),
        ],
    )


@pytest.fixture
def instance_seg_gt_anns1(
    img1_gt_mask_bytes1: bytes,
) -> schemas.Annotation:
    return schemas.Annotation(
        task_type=enums.TaskType.DET,
        raster=schemas.Raster(mask=b64encode(img1_gt_mask_bytes1).decode()),
        labels=[
            schemas.Label(key="inssegk1", value="inssegv1"),
            schemas.Label(key="inssegk2", value="inssegv2"),
        ],
    )


@pytest.fixture
def instance_seg_gt_anns2(
    img2_gt_mask_bytes1: bytes,
) -> schemas.Annotation:
    return schemas.Annotation(
        task_type=enums.TaskType.DET,
        raster=schemas.Raster(mask=b64encode(img2_gt_mask_bytes1).decode()),
        labels=[
            schemas.Label(key="inssegk2", value="inssegv2"),
            schemas.Label(key="inssegk3", value="inssegv3"),
        ],
    )


def test_label(
    db: Session,
    img1: schemas.ImageMetadata,
    img2: schemas.ImageMetadata,
    semantic_seg_gt_anns1: schemas.Annotation,
    semantic_seg_gt_anns2: schemas.Annotation,
    semantic_seg_pred_anns1: schemas.Annotation,
    semantic_seg_pred_anns2: schemas.Annotation,
    instance_seg_gt_anns1: schemas.Annotation,
    instance_seg_gt_anns2: schemas.Annotation,
):
    """Tests the label query methods"""
    crud.create_dataset(db=db, dataset=schemas.Dataset(name=dset_name))
    crud.create_model(db=db, model=schemas.Model(name=model_name))

    datum1 = img1.toDatum()
    datum2 = img2.toDatum()

    gts = [
        schemas.GroundTruth(
            datum=datum1,
            annotations=[
                semantic_seg_gt_anns1,
                instance_seg_gt_anns1,
            ],
        ),
        schemas.GroundTruth(
            datum=datum2,
            annotations=[
                semantic_seg_gt_anns2,
                instance_seg_gt_anns2,
            ],
        ),
    ]
    pds = [
        schemas.Prediction(
            model=model_name,
            datum=datum1,
            annotations=[
                semantic_seg_pred_anns1,
            ],
        ),
        schemas.Prediction(
            model=model_name,
            datum=datum2,
            annotations=[
                semantic_seg_pred_anns2,
            ],
        ),
    ]

    for gt in gts:
        crud.create_groundtruth(db=db, groundtruth=gt)

    for pred in pds:
        crud.create_prediction(db=db, prediction=pred)

    assert _get_dataset_label_keys(
        db,
        dataset_name=dset_name,
        task_type=enums.TaskType.SEG,
    ) == {"semsegk1", "semsegk2", "semsegk3"}

    assert _get_dataset_labels(
        db,
        dataset_name=dset_name,
        annotation_type=enums.AnnotationType.RASTER,
        task_types=[enums.TaskType.SEG],
    ) == {
        schemas.Label(key="semsegk1", value="semsegv1"),
        schemas.Label(key="semsegk2", value="semsegv2"),
        schemas.Label(key="semsegk3", value="semsegv3"),
    }
    assert (
        _get_dataset_labels(
            db,
            dataset_name=dset_name,
            annotation_type=enums.AnnotationType.POLYGON,
            task_types=[enums.TaskType.SEG],
        )
        == set()
    )

    assert _get_model_label_keys(
        db,
        dataset_name=dset_name,
        model_name=model_name,
        task_type=enums.TaskType.SEG,
    ) == {"semsegk1", "semsegk2", "semsegk3_pred"}

    assert _get_model_labels(
        db,
        dataset_name=dset_name,
        model_name=model_name,
        annotation_type=enums.AnnotationType.RASTER,
        task_types=[enums.TaskType.SEG],
    ) == {
        schemas.Label(key="semsegk1", value="semsegv1"),
        schemas.Label(key="semsegk2", value="semsegv2"),
        schemas.Label(key="semsegk3_pred", value="semsegv3_pred"),
    }

    assert (
        _get_model_labels(
            db,
            dataset_name=dset_name,
            model_name=model_name,
            annotation_type=enums.AnnotationType.POLYGON,
            task_types=[enums.TaskType.SEG],
        )
        == set()
    )

    assert (
        _get_dataset_label_keys(
            db, dataset_name=dset_name, task_type=enums.TaskType.CLF
        )
        == set()
    )
    assert (
        _get_model_label_keys(
            db,
            dataset_name=dset_name,
            model_name=model_name,
            task_type=enums.TaskType.CLF,
        )
        == set()
    )

    assert _get_dataset_label_keys(
        db,
        dataset_name=dset_name,
        task_type=enums.TaskType.DET,
    ) == {"inssegk1", "inssegk2", "inssegk3"}

    assert _get_dataset_labels(
        db,
        dataset_name=dset_name,
        annotation_type=enums.AnnotationType.RASTER,
        task_types=[enums.TaskType.DET],
    ) == {
        schemas.Label(key="inssegk1", value="inssegv1"),
        schemas.Label(key="inssegk2", value="inssegv2"),
        schemas.Label(key="inssegk3", value="inssegv3"),
    }
    assert _get_dataset_labels(
        db,
        dataset_name=dset_name,
        annotation_type=enums.AnnotationType.RASTER,
        task_types=[
            enums.TaskType.DET,
            enums.TaskType.SEG,
        ],
    ) == {
        schemas.Label(key="inssegk1", value="inssegv1"),
        schemas.Label(key="inssegk2", value="inssegv2"),
        schemas.Label(key="inssegk3", value="inssegv3"),
        schemas.Label(key="semsegk1", value="semsegv1"),
        schemas.Label(key="semsegk2", value="semsegv2"),
        schemas.Label(key="semsegk3", value="semsegv3"),
    }

    assert _get_dataset_label_keys(
        db,
        dataset_name=dset_name,
        task_type=enums.TaskType.SEG,
    ) == {"semsegk1", "semsegk2", "semsegk3"}

    assert _get_model_label_keys(
        db,
        dataset_name=dset_name,
        model_name=model_name,
        task_type=enums.TaskType.SEG,
    ) == {"semsegk1", "semsegk2", "semsegk3_pred"}

    assert (
        _get_model_label_keys(db, dset_name, model_name, enums.TaskType.DET)
        == set()
    )

    assert _get_model_labels(
        db,
        dset_name,
        model_name,
        annotation_type=enums.AnnotationType.RASTER,
        task_types=[
            enums.TaskType.DET,
            enums.TaskType.SEG,
        ],
    ) == {
        schemas.Label(key="semsegk1", value="semsegv1"),
        schemas.Label(key="semsegk2", value="semsegv2"),
        schemas.Label(key="semsegk3_pred", value="semsegv3_pred"),
    }
    assert (
        _get_model_labels(
            db,
            dset_name,
            model_name,
            annotation_type=enums.AnnotationType.RASTER,
            task_types=[enums.TaskType.DET],
        )
        == set()
    )
