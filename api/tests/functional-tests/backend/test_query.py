from base64 import b64encode

import pytest
from sqlalchemy.orm import Session

from velour_api import crud, enums, schemas
from velour_api.backend.core.label import get_label_keys, get_labels


@pytest.fixture
def semantic_seg_gt_anns1(
    img1_gt_mask_bytes1: bytes,
) -> schemas.Annotation:
    return schemas.Annotation(
        task_type=enums.TaskType.SEGMENTATION,
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
        task_type=enums.TaskType.SEGMENTATION,
        raster=schemas.Raster(mask=b64encode(img2_gt_mask_bytes1).decode()),
        labels=[
            schemas.Label(key="semsegk2", value="semsegv2"),
            schemas.Label(key="semsegk3", value="semsegv3"),
        ],
    )


@pytest.fixture
def semantic_seg_pred_anns1(img1_gt_mask_bytes1: bytes) -> schemas.Annotation:
    return schemas.Annotation(
        task_type=enums.TaskType.SEGMENTATION,
        raster=schemas.Raster(mask=b64encode(img1_gt_mask_bytes1).decode()),
        labels=[
            schemas.Label(key="semsegk1", value="semsegv1"),
            schemas.Label(key="semsegk2", value="semsegv2"),
        ],
    )


@pytest.fixture
def semantic_seg_pred_anns2(img2_gt_mask_bytes1: bytes) -> schemas.Annotation:
    return schemas.Annotation(
        task_type=enums.TaskType.SEGMENTATION,
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
        task_type=enums.TaskType.DETECTION,
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
        task_type=enums.TaskType.DETECTION,
        raster=schemas.Raster(mask=b64encode(img2_gt_mask_bytes1).decode()),
        labels=[
            schemas.Label(key="inssegk2", value="inssegv2"),
            schemas.Label(key="inssegk3", value="inssegv3"),
        ],
    )


def test_label(
    db: Session,
    dataset_name: str,
    model_name: str,
    img1: schemas.Datum,
    img2: schemas.Datum,
    semantic_seg_gt_anns1: schemas.Annotation,
    semantic_seg_gt_anns2: schemas.Annotation,
    semantic_seg_pred_anns1: schemas.Annotation,
    semantic_seg_pred_anns2: schemas.Annotation,
    instance_seg_gt_anns1: schemas.Annotation,
    instance_seg_gt_anns2: schemas.Annotation,
):
    """Tests the label query methods"""
    crud.create_dataset(db=db, dataset=schemas.Dataset(name=dataset_name))
    crud.create_model(db=db, model=schemas.Model(name=model_name))

    datum1 = img1
    datum2 = img2

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

    assert get_label_keys(
        db,
        schemas.Filter(
            dataset_names=[dataset_name],
            task_types=[enums.TaskType.SEGMENTATION],
        ),
        ignore_predictions=True,
    ) == {"semsegk1", "semsegk2", "semsegk3"}

    assert get_labels(
        db,
        schemas.Filter(
            dataset_names=[dataset_name],
            task_types=[enums.TaskType.SEGMENTATION],
            annotation_types=[enums.AnnotationType.RASTER],
        ),
        ignore_predictions=True,
    ) == {
        schemas.Label(key="semsegk1", value="semsegv1"),
        schemas.Label(key="semsegk2", value="semsegv2"),
        schemas.Label(key="semsegk3", value="semsegv3"),
    }
    assert (
        get_labels(
            db,
            schemas.Filter(
                dataset_names=[dataset_name],
                task_types=[enums.TaskType.SEGMENTATION],
                annotation_types=[enums.AnnotationType.POLYGON],
            ),
            ignore_predictions=True,
        )
        == set()
    )

    assert get_label_keys(
        db,
        schemas.Filter(
            models_names=[model_name],
            dataset_names=[dataset_name],
            task_types=[enums.TaskType.SEGMENTATION],
        ),
        ignore_groundtruths=True,
    ) == {"semsegk1", "semsegk2", "semsegk3_pred"}

    assert get_labels(
        db,
        schemas.Filter(
            models_names=[model_name],
            dataset_names=[dataset_name],
            annotation_types=[enums.AnnotationType.RASTER],
            task_types=[enums.TaskType.SEGMENTATION],
        ),
        ignore_groundtruths=True,
    ) == {
        schemas.Label(key="semsegk1", value="semsegv1"),
        schemas.Label(key="semsegk2", value="semsegv2"),
        schemas.Label(key="semsegk3_pred", value="semsegv3_pred"),
    }

    assert (
        get_labels(
            db,
            schemas.Filter(
                models_names=[model_name],
                dataset_names=[dataset_name],
                annotation_types=[enums.AnnotationType.POLYGON],
                task_types=[enums.TaskType.SEGMENTATION],
            ),
            ignore_groundtruths=True,
        )
        == set()
    )

    assert (
        get_label_keys(
            db,
            schemas.Filter(
                dataset_names=[dataset_name],
                task_types=[enums.TaskType.CLASSIFICATION],
            ),
            ignore_predictions=True,
        )
        == set()
    )
    assert (
        get_labels(
            db,
            schemas.Filter(
                models_names=[model_name],
                dataset_names=[dataset_name],
                task_types=[enums.TaskType.CLASSIFICATION],
            ),
            ignore_groundtruths=True,
        )
        == set()
    )

    assert get_label_keys(
        db,
        schemas.Filter(
            dataset_names=[dataset_name],
            task_types=[enums.TaskType.DETECTION],
        ),
        ignore_predictions=True,
    ) == {"inssegk1", "inssegk2", "inssegk3"}

    assert get_labels(
        db,
        schemas.Filter(
            dataset_names=[dataset_name],
            annotation_types=[enums.AnnotationType.RASTER],
            task_types=[enums.TaskType.DETECTION],
        ),
        ignore_predictions=True,
    ) == {
        schemas.Label(key="inssegk1", value="inssegv1"),
        schemas.Label(key="inssegk2", value="inssegv2"),
        schemas.Label(key="inssegk3", value="inssegv3"),
    }
    assert get_labels(
        db,
        schemas.Filter(
            dataset_names=[dataset_name],
            annotation_types=[enums.AnnotationType.RASTER],
            task_types=[
                enums.TaskType.DETECTION,
                enums.TaskType.SEGMENTATION,
            ],
        ),
        ignore_predictions=True,
    ) == {
        schemas.Label(key="inssegk1", value="inssegv1"),
        schemas.Label(key="inssegk2", value="inssegv2"),
        schemas.Label(key="inssegk3", value="inssegv3"),
        schemas.Label(key="semsegk1", value="semsegv1"),
        schemas.Label(key="semsegk2", value="semsegv2"),
        schemas.Label(key="semsegk3", value="semsegv3"),
    }

    assert get_label_keys(
        db,
        schemas.Filter(
            dataset_names=[dataset_name],
            task_types=[enums.TaskType.SEGMENTATION],
        ),
        ignore_predictions=True,
    ) == {"semsegk1", "semsegk2", "semsegk3"}

    assert get_label_keys(
        db,
        schemas.Filter(
            models_names=[model_name],
            dataset_names=[dataset_name],
            task_types=[enums.TaskType.SEGMENTATION],
        ),
        ignore_groundtruths=True,
    ) == {"semsegk1", "semsegk2", "semsegk3_pred"}

    assert (
        get_labels(
            db,
            schemas.Filter(
                models_names=[model_name],
                dataset_names=[dataset_name],
                task_types=[enums.TaskType.DETECTION],
            ),
            ignore_groundtruths=True,
        )
        == set()
    )

    assert get_labels(
        db,
        schemas.Filter(
            models_names=[model_name],
            dataset_names=[dataset_name],
            annotation_types=[enums.AnnotationType.RASTER],
            task_types=[
                enums.TaskType.SEGMENTATION,
                enums.TaskType.DETECTION,
            ],
            ignore_groundtruths=True,
        ),
    ) == {
        schemas.Label(key="semsegk1", value="semsegv1"),
        schemas.Label(key="semsegk2", value="semsegv2"),
        schemas.Label(key="semsegk3_pred", value="semsegv3_pred"),
    }
    assert (
        get_labels(
            db,
            schemas.Filter(
                models_names=[model_name],
                dataset_names=[dataset_name],
                annotation_types=[enums.AnnotationType.RASTER],
                task_types=[enums.TaskType.DETECTION],
            ),
            ignore_groundtruths=True,
        )
        == set()
    )
