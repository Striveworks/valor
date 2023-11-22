import pytest
from sqlalchemy import and_, select
from sqlalchemy.orm import Session

from velour_api import crud, schemas
from velour_api.backend.metrics.segmentation import (
    _gt_query,
    _pred_query,
    compute_segmentation_metrics,
    get_groundtruth_labels,
    gt_count,
    pred_count,
    tp_count,
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


def test__gt_query_and_pred_query(
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

        q = _gt_query(dataset_name, label_id=label_id)
        data = db.execute(q).all()
        assert len(data) == expected_number

    q = _gt_query(dataset_name, label_id=10000000)
    data = db.execute(q).all()
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

        q = _pred_query(dataset_name, model_name=model_name, label_id=label_id)
        data = db.execute(q).all()
        assert len(data) == expected_number

    q = _pred_query(dataset_name, model_name=model_name, label_id=10000000)
    data = db.execute(q).all()
    assert len(data) == 0


def _gt_tuples(gts: list[schemas.GroundTruth], label: schemas.Label):
    return [
        (gt.datum.uid, ann.raster.array)
        for gt in gts
        for ann in gt.annotations
        if label in ann.labels
    ]


def _pred_tuples(preds: list[schemas.Prediction], label: schemas.Label):
    return [
        (pred.datum.uid, ann.raster.array)
        for pred in preds
        for ann in pred.annotations
        if label in ann.labels
    ]


def _tp_count(
    gts: list[schemas.GroundTruth],
    preds: list[schemas.Prediction],
    label: schemas.Label,
) -> int:
    gts = _gt_tuples(gts, label)
    preds = _pred_tuples(preds, label)

    datum_ids = set([gt[0] for gt in gts]).intersection(
        [pred[0] for pred in preds]
    )

    ret = 0
    for datum_id in datum_ids:
        gt_mask = [gt[1] for gt in gts if gt[0] == datum_id][0]
        pred_mask = [pred[1] for pred in preds if pred[0] == datum_id][0]

        ret += (gt_mask * pred_mask).sum()

    return ret


def test_tp_count(
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

    for k, v in [("k1", "v1"), ("k2", "v2")]:
        label_id = db.scalar(
            select(Label).where(and_(Label.key == k, Label.value == v))
        ).id

        expected = _tp_count(
            gt_semantic_segs_create,
            [pred_semantic_segs_img1_create, pred_semantic_segs_img2_create],
            schemas.Label(key=k, value=v),
        )

        tps = tp_count(
            db=db,
            dataset_name=dataset_name,
            model_name=model_name,
            label_id=label_id,
        )

        assert expected == tps


def _gt_count(gts: list[schemas.GroundTruth], label: schemas.Label) -> int:
    gts = _gt_tuples(gts, label)

    ret = 0
    for gt in gts:
        ret += gt[1].sum()

    return ret


def test_gt_count(
    db: Session,
    dataset_name: str,
    gt_semantic_segs_create: list[schemas.GroundTruth],
):
    _create_gt_data(
        db=db,
        dataset_name=dataset_name,
        gt_semantic_segs_create=gt_semantic_segs_create,
    )

    for k, v in [("k1", "v1"), ("k1", "v2"), ("k3", "v3"), ("k2", "v2")]:
        label_id = db.scalar(
            select(Label).where(and_(Label.key == k, Label.value == v))
        ).id

        expected = _gt_count(
            gt_semantic_segs_create, schemas.Label(key=k, value=v)
        )

        assert (
            gt_count(db=db, dataset_name=dataset_name, label_id=label_id)
            == expected
        )

    with pytest.raises(RuntimeError) as exc_info:
        gt_count(db=db, dataset_name=dataset_name, label_id=1000000)

    assert "No groundtruth pixels for label" in str(exc_info)


def _pred_count(preds: list[schemas.Prediction], label: schemas.Label) -> int:
    preds = _pred_tuples(preds, label)

    ret = 0
    for pred in preds:
        ret += pred[1].sum()

    return ret


def test_pred_count(
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

    for k, v in [("k1", "v1"), ("k1", "v2"), ("k2", "v3"), ("k2", "v2")]:
        label_id = db.scalar(
            select(Label).where(and_(Label.key == k, Label.value == v))
        ).id

        expected = _pred_count(
            [pred_semantic_segs_img1_create, pred_semantic_segs_img2_create],
            schemas.Label(key=k, value=v),
        )

        assert (
            pred_count(
                db=db,
                model_name=model_name,
                dataset_name=dataset_name,
                label_id=label_id,
            )
            == expected
        )

    assert (
        pred_count(
            db=db,
            dataset_name=dataset_name,
            model_name=model_name,
            label_id=1000000,
        )
        == 0
    )


def test_get_groundtruth_labels(
    db: Session,
    dataset_name: str,
    gt_semantic_segs_create: list[schemas.GroundTruth],
):
    _create_gt_data(
        db=db,
        dataset_name=dataset_name,
        gt_semantic_segs_create=gt_semantic_segs_create,
    )
    labels = get_groundtruth_labels(db, dataset_name)

    assert len(labels) == 4

    assert set([label[:2] for label in labels]) == {
        ("k1", "v1"),
        ("k1", "v2"),
        ("k2", "v2"),
        ("k3", "v3"),
    }

    assert len(set([label[-1] for label in labels])) == 4


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

    metrics = compute_segmentation_metrics(db, dataset_name, model_name)
    # should have five metrics (one IOU for each of the four labels, and one mIOU)
    assert len(metrics) == 5
    for metric in metrics[:-1]:
        assert isinstance(metric, schemas.IOUMetric)
        assert metric.value < 1.0
    assert isinstance(metrics[-1], schemas.mIOUMetric)
    assert metrics[-1].value < 1.0
