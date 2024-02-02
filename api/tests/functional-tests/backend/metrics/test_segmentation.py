from sqlalchemy import and_, select
from sqlalchemy.orm import Session

from velour_api import crud, enums, schemas
from velour_api.backend.metrics.segmentation import (
    _compute_segmentation_metrics,
    _count_groundtruths,
    _count_predictions,
    _count_true_positives,
    _generate_groundtruth_query,
    _generate_prediction_query,
)
from velour_api.backend.models import Label


def _create_gt_data(
    db: Session,
    dataset_name: str,
    gt_semantic_segs_create: list[schemas.GroundTruth],
):
    crud.create_dataset(
        db=db,
        dataset=schemas.Dataset(name=dataset_name),
    )
    for gt in gt_semantic_segs_create:
        crud.create_groundtruth(db=db, groundtruth=gt)
    crud.finalize(db=db, dataset_name=dataset_name)


def _create_data(
    db: Session,
    dataset_name: str,
    model_name: str,
    gt_semantic_segs_create: list[schemas.GroundTruth],
    pred_semantic_segs_img1_create: schemas.Prediction,
    pred_semantic_segs_img2_create: schemas.Prediction,
):
    _create_gt_data(
        db=db,
        dataset_name=dataset_name,
        gt_semantic_segs_create=gt_semantic_segs_create,
    )

    crud.create_model(db=db, model=schemas.Model(name=model_name))

    crud.create_prediction(db=db, prediction=pred_semantic_segs_img1_create)
    crud.create_prediction(db=db, prediction=pred_semantic_segs_img2_create)


def test_query_generators(
    db: Session,
    dataset_name: str,
    model_name: str,
    gt_semantic_segs_create: list[schemas.GroundTruth],
    pred_semantic_segs_img1_create: schemas.Prediction,
    pred_semantic_segs_img2_create: schemas.Prediction,
):
    _create_data(
        db=db,
        dataset_name=dataset_name,
        model_name=model_name,
        gt_semantic_segs_create=gt_semantic_segs_create,
        pred_semantic_segs_img1_create=pred_semantic_segs_img1_create,
        pred_semantic_segs_img2_create=pred_semantic_segs_img2_create,
    )

    groundtruth_filter = schemas.Filter(
        dataset_names=[dataset_name],
        task_types=[enums.TaskType.SEGMENTATION],
        raster=True,
        label_ids=None,
    )

    prediction_filter = schemas.Filter(
        dataset_names=[dataset_name],
        model_names=[model_name],
        task_types=[enums.TaskType.SEGMENTATION],
        raster=True,
        label_ids=None,
    )

    for label_key, label_value, expected_number in [
        ("k1", "v1", 2),
        ("k1", "v2", 1),
        ("k2", "v2", 1),
        ("k3", "v3", 1),
    ]:
        label_id = db.scalar(
            select(Label.id).where(
                and_(Label.key == label_key, Label.value == label_value)
            )
        )

        assert label_id is not None

        groundtruth_filter.label_ids = [label_id]
        q = _generate_groundtruth_query(groundtruth_filter)
        data = db.query(q).all()
        assert len(data) == expected_number

    groundtruth_filter.label_ids = [10000000]
    q = _generate_groundtruth_query(groundtruth_filter)
    data = db.query(q).all()
    assert len(data) == 0

    for label_key, label_value, expected_number in [
        ("k1", "v1", 2),
        ("k1", "v2", 0),
        ("k2", "v2", 1),
        ("k2", "v3", 2),
    ]:
        label_id = db.scalar(
            select(Label.id).where(
                and_(Label.key == label_key, Label.value == label_value)
            )
        )

        assert label_id is not None

        prediction_filter.label_ids = [label_id]
        q = _generate_prediction_query(prediction_filter)
        data = db.query(q).all()
        assert len(data) == expected_number

    prediction_filter.label_ids = [10000000]
    q = _generate_prediction_query(prediction_filter)
    data = db.query(q).all()
    assert len(data) == 0


def _create_groundtruth_tuples(
    gts: list[schemas.GroundTruth], label: schemas.Label
):
    return [
        (gt.datum.uid, ann.raster.array)
        for gt in gts
        for ann in gt.annotations
        if label in ann.labels
    ]


def _create_prediction_tuples(
    preds: list[schemas.Prediction], label: schemas.Label
):
    return [
        (pred.datum.uid, ann.raster.array)
        for pred in preds
        for ann in pred.annotations
        if label in ann.labels
    ]


def _help_count_true_positives(
    gts: list[schemas.GroundTruth],
    preds: list[schemas.Prediction],
    label: schemas.Label,
) -> int:
    gts = _create_groundtruth_tuples(gts, label)
    preds = _create_prediction_tuples(preds, label)

    datum_ids = set([gt[0] for gt in gts]).intersection(
        [pred[0] for pred in preds]
    )

    ret = 0
    for datum_id in datum_ids:
        gt_mask = [gt[1] for gt in gts if gt[0] == datum_id][0]
        pred_mask = [pred[1] for pred in preds if pred[0] == datum_id][0]

        ret += (gt_mask * pred_mask).sum()

    return ret


def test_count_true_positives(
    db: Session,
    dataset_name: str,
    model_name: str,
    gt_semantic_segs_create: list[schemas.GroundTruth],
    pred_semantic_segs_img1_create: schemas.Prediction,
    pred_semantic_segs_img2_create: schemas.Prediction,
):
    _create_data(
        db=db,
        dataset_name=dataset_name,
        model_name=model_name,
        gt_semantic_segs_create=gt_semantic_segs_create,
        pred_semantic_segs_img1_create=pred_semantic_segs_img1_create,
        pred_semantic_segs_img2_create=pred_semantic_segs_img2_create,
    )

    groundtruth_filter = schemas.Filter(
        dataset_names=[dataset_name],
        task_types=[enums.TaskType.SEGMENTATION],
        raster=True,
        label_ids=None,
    )

    prediction_filter = schemas.Filter(
        dataset_names=[dataset_name],
        model_names=[model_name],
        task_types=[enums.TaskType.SEGMENTATION],
        raster=True,
        label_ids=None,
    )

    for k, v in [("k1", "v1"), ("k2", "v2")]:
        label_id = db.scalar(
            select(Label).where(and_(Label.key == k, Label.value == v))
        ).id

        expected = _help_count_true_positives(
            gt_semantic_segs_create,
            [pred_semantic_segs_img1_create, pred_semantic_segs_img2_create],
            schemas.Label(key=k, value=v),
        )

        groundtruth_filter.label_ids = [label_id]
        prediction_filter.label_ids = [label_id]
        tps = _count_true_positives(
            db=db,
            groundtruth_subquery=_generate_groundtruth_query(
                groundtruth_filter
            ),
            prediction_subquery=_generate_prediction_query(prediction_filter),
        )

        assert expected == tps


def _help_count_groundtruths(
    gts: list[schemas.GroundTruth], label: schemas.Label
) -> int:
    gts = _create_groundtruth_tuples(gts, label)

    ret = 0
    for gt in gts:
        ret += gt[1].sum()

    return ret


def test_count_groundtruths(
    db: Session,
    dataset_name: str,
    gt_semantic_segs_create: list[schemas.GroundTruth],
):
    _create_gt_data(
        db=db,
        dataset_name=dataset_name,
        gt_semantic_segs_create=gt_semantic_segs_create,
    )

    groundtruth_filter = schemas.Filter(
        dataset_names=[dataset_name],
        task_types=[enums.TaskType.SEGMENTATION],
        raster=True,
        label_ids=None,
    )

    for k, v in [("k1", "v1"), ("k1", "v2"), ("k3", "v3"), ("k2", "v2")]:
        label_id = db.scalar(
            select(Label).where(and_(Label.key == k, Label.value == v))
        ).id

        expected = _help_count_groundtruths(
            gt_semantic_segs_create, schemas.Label(key=k, value=v)
        )

        groundtruth_filter.label_ids = [label_id]
        assert (
            _count_groundtruths(
                db,
                _generate_groundtruth_query(groundtruth_filter),
            )
            == expected
        )

    groundtruth_filter.label_ids = [1000000]
    assert (
        _count_groundtruths(
            db,
            _generate_groundtruth_query(groundtruth_filter),
        )
        is None
    )


def _help_count_predictions(
    preds: list[schemas.Prediction], label: schemas.Label
) -> int:
    preds = _create_prediction_tuples(preds, label)

    ret = 0
    for pred in preds:
        ret += pred[1].sum()

    return ret


def test_count_predictions(
    db: Session,
    dataset_name: str,
    model_name: str,
    gt_semantic_segs_create: list[schemas.GroundTruth],
    pred_semantic_segs_img1_create: schemas.Prediction,
    pred_semantic_segs_img2_create: schemas.Prediction,
):
    _create_data(
        db=db,
        dataset_name=dataset_name,
        model_name=model_name,
        gt_semantic_segs_create=gt_semantic_segs_create,
        pred_semantic_segs_img1_create=pred_semantic_segs_img1_create,
        pred_semantic_segs_img2_create=pred_semantic_segs_img2_create,
    )

    prediction_filter = schemas.Filter(
        dataset_names=[dataset_name],
        model_names=[model_name],
        task_types=[enums.TaskType.SEGMENTATION],
        raster=True,
        label_ids=None,
    )

    for k, v in [("k1", "v1"), ("k1", "v2"), ("k2", "v3"), ("k2", "v2")]:
        label_id = db.scalar(
            select(Label).where(and_(Label.key == k, Label.value == v))
        ).id

        expected = _help_count_predictions(
            [pred_semantic_segs_img1_create, pred_semantic_segs_img2_create],
            schemas.Label(key=k, value=v),
        )

        prediction_filter.label_ids = [label_id]
        assert (
            _count_predictions(
                db,
                _generate_prediction_query(prediction_filter),
            )
            == expected
        )

    prediction_filter.label_ids = [1000000]
    assert (
        _count_predictions(db, _generate_prediction_query(prediction_filter))
        == 0
    )


def test_compute_segmentation_metrics(
    db: Session,
    dataset_name: str,
    model_name: str,
    gt_semantic_segs_create: list[schemas.GroundTruth],
    pred_semantic_segs_img1_create: schemas.Prediction,
    pred_semantic_segs_img2_create: schemas.Prediction,
):
    _create_data(
        db=db,
        dataset_name=dataset_name,
        model_name=model_name,
        gt_semantic_segs_create=gt_semantic_segs_create,
        pred_semantic_segs_img1_create=pred_semantic_segs_img1_create,
        pred_semantic_segs_img2_create=pred_semantic_segs_img2_create,
    )

    prediction_filter = schemas.Filter(
        model_names=[model_name],
        dataset_names=[dataset_name],
    )
    groundtruth_filter = schemas.Filter(
        model_names=[model_name],
        dataset_names=[dataset_name],
        task_types=[enums.TaskType.SEGMENTATION],
        raster=True,
    )

    metrics = _compute_segmentation_metrics(
        db,
        parameters=schemas.EvaluationParameters(
            task_type=enums.TaskType.CLASSIFICATION, label_map=None
        ),
        prediction_filter=prediction_filter,
        groundtruth_filter=groundtruth_filter,
    )
    # should have five metrics (one IOU for each of the four labels, and one mIOU)
    assert len(metrics) == 5
    for metric in metrics[:-1]:
        assert isinstance(metric, schemas.IOUMetric)
        assert metric.value < 1.0
    assert isinstance(metrics[-1], schemas.mIOUMetric)
    assert metrics[-1].value < 1.0
