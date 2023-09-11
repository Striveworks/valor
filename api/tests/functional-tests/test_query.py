from base64 import b64encode

import pytest
from sqlalchemy.orm import Session

from velour_api import crud, enums, schemas
from velour_api.backend.query.label import (
    _get_dataset_label_keys,
    _get_model_label_keys,
)

dset_name = "test_dataset"
model_name = "test_model"


@pytest.fixture
def semantic_seg_gts(
    img1: schemas.Image,
    img2: schemas.Image,
    img1_gt_mask_bytes1: bytes,
    img2_gt_mask_bytes1: bytes,
) -> list[schemas.GroundTruth]:
    return [
        schemas.GroundTruth(
            datum=img1.to_datum(),
            annotations=[
                schemas.Annotation(
                    task_type=enums.TaskType.SEMANTIC_SEGMENTATION,
                    raster=schemas.Raster(
                        mask=b64encode(img1_gt_mask_bytes1).decode()
                    ),
                    labels=[
                        schemas.Label(key="semsegk1", value="semsegv1"),
                        schemas.Label(key="semsegk2", value="semsegv2"),
                    ],
                )
            ],
        ),
        schemas.GroundTruth(
            datum=img2.to_datum(),
            annotations=[
                schemas.Annotation(
                    task_type=enums.TaskType.SEMANTIC_SEGMENTATION,
                    raster=schemas.Raster(
                        mask=b64encode(img2_gt_mask_bytes1).decode()
                    ),
                    labels=[
                        schemas.Label(key="semsegk2", value="semsegv2"),
                        schemas.Label(key="semsegk3", value="semsegv3"),
                    ],
                )
            ],
        ),
    ]


@pytest.fixture
def semantic_seg_preds(
    img1: schemas.Image,
    img2: schemas.Image,
    img1_gt_mask_bytes1: bytes,
    img2_gt_mask_bytes1: bytes,
) -> list[schemas.Prediction]:
    return [
        schemas.Prediction(
            model=model_name,
            datum=img1.to_datum(),
            annotations=[
                schemas.Annotation(
                    task_type=enums.TaskType.SEMANTIC_SEGMENTATION,
                    raster=schemas.Raster(
                        mask=b64encode(img1_gt_mask_bytes1).decode()
                    ),
                    labels=[
                        schemas.Label(key="semsegk1", value="semsegv1"),
                        schemas.Label(key="semsegk2", value="semsegv2"),
                    ],
                )
            ],
        ),
        schemas.Prediction(
            model=model_name,
            datum=img2.to_datum(),
            annotations=[
                schemas.Annotation(
                    task_type=enums.TaskType.SEMANTIC_SEGMENTATION,
                    raster=schemas.Raster(
                        mask=b64encode(img2_gt_mask_bytes1).decode()
                    ),
                    labels=[
                        schemas.Label(key="semsegk2", value="semsegv2"),
                        schemas.Label(
                            key="semsegk3_pred", value="semsegv3_pred"
                        ),
                    ],
                )
            ],
        ),
    ]


@pytest.fixture
def instance_segs_gt(
    img1: schemas.Image,
    img2: schemas.Image,
    img1_gt_mask_bytes1: bytes,
    img2_gt_mask_bytes1: bytes,
) -> list[schemas.GroundTruth]:
    return [
        schemas.GroundTruth(
            datum=img1.to_datum(),
            annotations=[
                schemas.Annotation(
                    task_type=enums.TaskType.INSTANCE_SEGMENTATION,
                    raster=schemas.Raster(
                        mask=b64encode(img1_gt_mask_bytes1).decode()
                    ),
                    labels=[
                        schemas.Label(key="inssegk1", value="inssegv1"),
                        schemas.Label(key="inssegk2", value="inssegv2"),
                    ],
                )
            ],
        ),
        schemas.GroundTruth(
            datum=img2.to_datum(),
            annotations=[
                schemas.Annotation(
                    task_type=enums.TaskType.INSTANCE_SEGMENTATION,
                    raster=schemas.Raster(
                        mask=b64encode(img2_gt_mask_bytes1).decode()
                    ),
                    labels=[
                        schemas.Label(key="inssegk2", value="inssegv2"),
                        schemas.Label(key="inssegk3", value="inssegv3"),
                    ],
                )
            ],
        ),
    ]


def test_label(
    db: Session,
    semantic_seg_gts: list[schemas.GroundTruth],
    semantic_seg_preds: list[schemas.Prediction],
    instance_segs_gt: list[schemas.GroundTruth],
):
    """Tests the label query methods"""
    crud.create_dataset(db=db, dataset=schemas.Dataset(name=dset_name))
    crud.create_model(db=db, model=schemas.Model(name=model_name))

    for gt in semantic_seg_gts:
        crud.create_groundtruth(db=db, groundtruth=gt)

    for pred in semantic_seg_preds:
        crud.create_prediction(db=db, prediction=pred)

    assert _get_dataset_label_keys(
        db,
        dataset_name=dset_name,
        task_type=enums.TaskType.SEMANTIC_SEGMENTATION,
    ) == {"semsegk1", "semsegk2", "semsegk3"}

    assert _get_model_label_keys(
        db,
        dataset_name=dset_name,
        model_name=model_name,
        task_type=enums.TaskType.SEMANTIC_SEGMENTATION,
    ) == {"semsegk1", "semsegk2", "semsegk3_pred"}

    for task_type in [
        enums.TaskType.INSTANCE_SEGMENTATION,
        enums.TaskType.DETECTION,
        enums.TaskType.CLASSIFICATION,
    ]:
        assert (
            _get_dataset_label_keys(
                db, dataset_name=dset_name, task_type=task_type
            )
            == set()
        )
        assert (
            _get_model_label_keys(
                db,
                dataset_name=dset_name,
                model_name=model_name,
                task_type=task_type,
            )
            == set()
        )

    for gt in instance_segs_gt:
        crud.create_groundtruth(db=db, groundtruth=gt)

    assert _get_dataset_label_keys(
        db,
        dataset_name=dset_name,
        task_type=enums.TaskType.INSTANCE_SEGMENTATION,
    ) == {"inssegk1", "inssegk2", "inssegk3"}

    assert _get_dataset_label_keys(
        db,
        dataset_name=dset_name,
        task_type=enums.TaskType.SEMANTIC_SEGMENTATION,
    ) == {"semsegk1", "semsegk2", "semsegk3"}

    assert _get_model_label_keys(
        db,
        dataset_name=dset_name,
        model_name=model_name,
        task_type=enums.TaskType.SEMANTIC_SEGMENTATION,
    ) == {"semsegk1", "semsegk2", "semsegk3_pred"}

    assert (
        _get_model_label_keys(
            db, dset_name, model_name, enums.TaskType.INSTANCE_SEGMENTATION
        )
        == set()
    )
