from sqlalchemy import and_, select
from sqlalchemy.orm import Session

from valor_api import crud, enums, schemas
from valor_api.backend.core import create_or_get_evaluations
from valor_api.backend.metrics.segmentation import (
    _compute_segmentation_metrics,
    _count_groundtruths,
    _count_predictions,
    _count_true_positives,
    _generate_groundtruth_query,
    _generate_prediction_query,
    compute_semantic_segmentation_metrics,
)
from valor_api.backend.models import Label


def _create_gt_data(
    db: Session,
    dataset_name: str,
    gt_semantic_segs_create: list[schemas.GroundTruth],
):
    crud.create_dataset(
        db=db,
        dataset=schemas.Dataset(name=dataset_name),
    )
    crud.create_groundtruths(db=db, groundtruths=gt_semantic_segs_create)
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

    crud.create_predictions(
        db=db,
        predictions=[
            pred_semantic_segs_img1_create,
            pred_semantic_segs_img2_create,
        ],
    )


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

    prediction_filter = schemas.Filter(
        predictions=schemas.LogicalFunction(
            args=[
                schemas.Condition(
                    lhs=schemas.Symbol(
                        name=schemas.SupportedSymbol.MODEL_NAME
                    ),
                    rhs=schemas.Value.infer(model_name),
                    op=schemas.FilterOperator.EQ,
                ),
                schemas.Condition(
                    lhs=schemas.Symbol(
                        name=schemas.SupportedSymbol.DATASET_NAME
                    ),
                    rhs=schemas.Value.infer(dataset_name),
                    op=schemas.FilterOperator.EQ,
                ),
                schemas.Condition(
                    lhs=schemas.Symbol(name=schemas.SupportedSymbol.TASK_TYPE),
                    rhs=schemas.Value.infer(
                        enums.TaskType.SEMANTIC_SEGMENTATION
                    ),
                    op=schemas.FilterOperator.CONTAINS,
                ),
                schemas.Condition(
                    lhs=schemas.Symbol(name=schemas.SupportedSymbol.RASTER),
                    op=schemas.FilterOperator.ISNOTNULL,
                ),
            ],
            op=schemas.LogicalOperator.AND,
        )
    )
    groundtruth_filter = schemas.Filter(
        groundtruths=schemas.LogicalFunction(
            args=[
                schemas.Condition(
                    lhs=schemas.Symbol(
                        name=schemas.SupportedSymbol.DATASET_NAME
                    ),
                    rhs=schemas.Value.infer(dataset_name),
                    op=schemas.FilterOperator.EQ,
                ),
                schemas.Condition(
                    lhs=schemas.Symbol(name=schemas.SupportedSymbol.TASK_TYPE),
                    rhs=schemas.Value.infer(
                        enums.TaskType.SEMANTIC_SEGMENTATION
                    ),
                    op=schemas.FilterOperator.CONTAINS,
                ),
                schemas.Condition(
                    lhs=schemas.Symbol(name=schemas.SupportedSymbol.RASTER),
                    op=schemas.FilterOperator.ISNOTNULL,
                ),
            ],
            op=schemas.LogicalOperator.AND,
        )
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
        groundtruth_filter.labels = schemas.Condition(
            lhs=schemas.Symbol(name=schemas.SupportedSymbol.LABEL_ID),
            rhs=schemas.Value.infer(label_id),
            op=schemas.FilterOperator.EQ,
        )

        q = _generate_groundtruth_query(groundtruth_filter)
        data = db.query(q).all()
        assert len(data) == expected_number

    groundtruth_filter.labels = schemas.Condition(
        lhs=schemas.Symbol(name=schemas.SupportedSymbol.LABEL_ID),
        rhs=schemas.Value.infer(10000000),
        op=schemas.FilterOperator.EQ,
    )
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

        prediction_filter.labels = schemas.Condition(
            lhs=schemas.Symbol(name=schemas.SupportedSymbol.LABEL_ID),
            rhs=schemas.Value.infer(label_id),
            op=schemas.FilterOperator.EQ,
        )

        q = _generate_prediction_query(prediction_filter)
        data = db.query(q).all()
        assert len(data) == expected_number

    prediction_filter.labels = schemas.Condition(
        lhs=schemas.Symbol(name=schemas.SupportedSymbol.LABEL_ID),
        rhs=schemas.Value.infer(10000000),
        op=schemas.FilterOperator.EQ,
    )
    q = _generate_prediction_query(prediction_filter)
    data = db.query(q).all()
    assert len(data) == 0


def _create_groundtruth_tuples(
    gts: list[schemas.GroundTruth], label: schemas.Label
):
    assert all(
        [
            ann.raster is not None
            for gt in gts
            for ann in gt.annotations
            if label in ann.labels
        ]
    )

    return [
        (gt.datum.uid, ann.raster.array)  # type: ignore - handled by the assertion above
        for gt in gts
        for ann in gt.annotations
        if label in ann.labels
    ]


def _create_prediction_tuples(
    preds: list[schemas.Prediction], label: schemas.Label
):
    assert all(
        [
            isinstance(ann.raster, schemas.Raster)
            for pred in preds
            for ann in pred.annotations
            if label in ann.labels
        ]
    )
    return [
        (pred.datum.uid, ann.raster.array)  # type: ignore - handled by the assertion above
        for pred in preds
        for ann in pred.annotations
        if label in ann.labels
    ]


def _help_count_true_positives(
    gts: list[schemas.GroundTruth],
    preds: list[schemas.Prediction],
    label: schemas.Label,
) -> int:
    groundtruths = _create_groundtruth_tuples(gts, label)
    predictions = _create_prediction_tuples(preds, label)

    datum_ids = set([gt[0] for gt in groundtruths]).intersection(
        [pred[0] for pred in predictions]
    )

    ret = 0
    for datum_id in datum_ids:
        gt_mask = [gt[1] for gt in groundtruths if gt[0] == datum_id][0]
        pred_mask = [pred[1] for pred in predictions if pred[0] == datum_id][0]

        ret += (gt_mask * pred_mask).sum()

    return ret


def test__count_true_positives(
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
        predictions=schemas.LogicalFunction(
            args=[
                schemas.Condition(
                    lhs=schemas.Symbol(
                        name=schemas.SupportedSymbol.MODEL_NAME
                    ),
                    rhs=schemas.Value.infer(model_name),
                    op=schemas.FilterOperator.EQ,
                ),
                schemas.Condition(
                    lhs=schemas.Symbol(
                        name=schemas.SupportedSymbol.DATASET_NAME
                    ),
                    rhs=schemas.Value.infer(dataset_name),
                    op=schemas.FilterOperator.EQ,
                ),
                schemas.Condition(
                    lhs=schemas.Symbol(name=schemas.SupportedSymbol.TASK_TYPE),
                    rhs=schemas.Value.infer(
                        enums.TaskType.SEMANTIC_SEGMENTATION
                    ),
                    op=schemas.FilterOperator.CONTAINS,
                ),
                schemas.Condition(
                    lhs=schemas.Symbol(name=schemas.SupportedSymbol.RASTER),
                    op=schemas.FilterOperator.ISNOTNULL,
                ),
            ],
            op=schemas.LogicalOperator.AND,
        )
    )
    groundtruth_filter = schemas.Filter(
        groundtruths=schemas.LogicalFunction(
            args=[
                schemas.Condition(
                    lhs=schemas.Symbol(
                        name=schemas.SupportedSymbol.DATASET_NAME
                    ),
                    rhs=schemas.Value.infer(dataset_name),
                    op=schemas.FilterOperator.EQ,
                ),
                schemas.Condition(
                    lhs=schemas.Symbol(name=schemas.SupportedSymbol.TASK_TYPE),
                    rhs=schemas.Value.infer(
                        enums.TaskType.SEMANTIC_SEGMENTATION
                    ),
                    op=schemas.FilterOperator.CONTAINS,
                ),
                schemas.Condition(
                    lhs=schemas.Symbol(name=schemas.SupportedSymbol.RASTER),
                    op=schemas.FilterOperator.ISNOTNULL,
                ),
            ],
            op=schemas.LogicalOperator.AND,
        )
    )

    for k, v in [("k1", "v1"), ("k2", "v2")]:
        label = db.scalar(
            select(Label).where(and_(Label.key == k, Label.value == v))
        )

        assert label is not None
        label_id = label.id

        expected = _help_count_true_positives(
            gt_semantic_segs_create,
            [pred_semantic_segs_img1_create, pred_semantic_segs_img2_create],
            schemas.Label(key=k, value=v),
        )

        groundtruth_filter.labels = schemas.Condition(
            lhs=schemas.Symbol(name=schemas.SupportedSymbol.LABEL_ID),
            rhs=schemas.Value.infer(label_id),
            op=schemas.FilterOperator.EQ,
        )
        prediction_filter.labels = schemas.Condition(
            lhs=schemas.Symbol(name=schemas.SupportedSymbol.LABEL_ID),
            rhs=schemas.Value.infer(label_id),
            op=schemas.FilterOperator.EQ,
        )

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
    groundtruths = _create_groundtruth_tuples(gts, label)

    ret = 0
    for gt in groundtruths:
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
        groundtruths=schemas.LogicalFunction(
            args=[
                schemas.Condition(
                    lhs=schemas.Symbol(
                        name=schemas.SupportedSymbol.DATASET_NAME
                    ),
                    rhs=schemas.Value.infer(dataset_name),
                    op=schemas.FilterOperator.EQ,
                ),
                schemas.Condition(
                    lhs=schemas.Symbol(name=schemas.SupportedSymbol.TASK_TYPE),
                    rhs=schemas.Value.infer(
                        enums.TaskType.SEMANTIC_SEGMENTATION
                    ),
                    op=schemas.FilterOperator.CONTAINS,
                ),
                schemas.Condition(
                    lhs=schemas.Symbol(name=schemas.SupportedSymbol.RASTER),
                    op=schemas.FilterOperator.ISNOTNULL,
                ),
            ],
            op=schemas.LogicalOperator.AND,
        )
    )

    for k, v in [("k1", "v1"), ("k1", "v2"), ("k3", "v3"), ("k2", "v2")]:
        label = db.scalar(
            select(Label).where(and_(Label.key == k, Label.value == v))
        )

        assert label is not None
        label_id = label.id

        expected = _help_count_groundtruths(
            gt_semantic_segs_create, schemas.Label(key=k, value=v)
        )

        groundtruth_filter.labels = schemas.Condition(
            lhs=schemas.Symbol(name=schemas.SupportedSymbol.LABEL_ID),
            rhs=schemas.Value.infer(label_id),
            op=schemas.FilterOperator.EQ,
        )
        assert (
            _count_groundtruths(
                db,
                _generate_groundtruth_query(groundtruth_filter),
            )
            == expected
        )

    groundtruth_filter.labels = schemas.Condition(
        lhs=schemas.Symbol(name=schemas.SupportedSymbol.LABEL_ID),
        rhs=schemas.Value.infer(1000000),
        op=schemas.FilterOperator.EQ,
    )
    assert (
        _count_groundtruths(
            db,
            _generate_groundtruth_query(groundtruth_filter),
        )
        == 0
    )


def _help_count_predictions(
    preds: list[schemas.Prediction], label: schemas.Label
) -> int:
    predictions = _create_prediction_tuples(preds, label)

    ret = 0
    for pred in predictions:
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
        predictions=schemas.LogicalFunction(
            args=[
                schemas.Condition(
                    lhs=schemas.Symbol(
                        name=schemas.SupportedSymbol.MODEL_NAME
                    ),
                    rhs=schemas.Value.infer(model_name),
                    op=schemas.FilterOperator.EQ,
                ),
                schemas.Condition(
                    lhs=schemas.Symbol(
                        name=schemas.SupportedSymbol.DATASET_NAME
                    ),
                    rhs=schemas.Value.infer(dataset_name),
                    op=schemas.FilterOperator.EQ,
                ),
                schemas.Condition(
                    lhs=schemas.Symbol(name=schemas.SupportedSymbol.TASK_TYPE),
                    rhs=schemas.Value.infer(
                        enums.TaskType.SEMANTIC_SEGMENTATION
                    ),
                    op=schemas.FilterOperator.CONTAINS,
                ),
                schemas.Condition(
                    lhs=schemas.Symbol(name=schemas.SupportedSymbol.RASTER),
                    op=schemas.FilterOperator.ISNOTNULL,
                ),
            ],
            op=schemas.LogicalOperator.AND,
        )
    )

    for k, v in [("k1", "v1"), ("k1", "v2"), ("k2", "v3"), ("k2", "v2")]:
        label = db.scalar(
            select(Label).where(and_(Label.key == k, Label.value == v))
        )
        assert label
        label_id = label.id

        expected = _help_count_predictions(
            [pred_semantic_segs_img1_create, pred_semantic_segs_img2_create],
            schemas.Label(key=k, value=v),
        )

        prediction_filter.labels = schemas.Condition(
            lhs=schemas.Symbol(name=schemas.SupportedSymbol.LABEL_ID),
            rhs=schemas.Value.infer(label_id),
            op=schemas.FilterOperator.EQ,
        )
        assert (
            _count_predictions(
                db,
                _generate_prediction_query(prediction_filter),
            )
            == expected
        )

    prediction_filter.labels = schemas.Condition(
        lhs=schemas.Symbol(name=schemas.SupportedSymbol.LABEL_ID),
        rhs=schemas.Value.infer(1000000),
        op=schemas.FilterOperator.EQ,
    )
    assert (
        _count_predictions(db, _generate_prediction_query(prediction_filter))
        == 0
    )


def test__compute_segmentation_metrics(
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
        predictions=schemas.LogicalFunction(
            args=[
                schemas.Condition(
                    lhs=schemas.Symbol(
                        name=schemas.SupportedSymbol.MODEL_NAME
                    ),
                    rhs=schemas.Value.infer(model_name),
                    op=schemas.FilterOperator.EQ,
                ),
                schemas.Condition(
                    lhs=schemas.Symbol(
                        name=schemas.SupportedSymbol.DATASET_NAME
                    ),
                    rhs=schemas.Value.infer(dataset_name),
                    op=schemas.FilterOperator.EQ,
                ),
                schemas.Condition(
                    lhs=schemas.Symbol(name=schemas.SupportedSymbol.TASK_TYPE),
                    rhs=schemas.Value.infer(
                        enums.TaskType.SEMANTIC_SEGMENTATION
                    ),
                    op=schemas.FilterOperator.CONTAINS,
                ),
                schemas.Condition(
                    lhs=schemas.Symbol(name=schemas.SupportedSymbol.RASTER),
                    op=schemas.FilterOperator.ISNOTNULL,
                ),
            ],
            op=schemas.LogicalOperator.AND,
        )
    )
    groundtruth_filter = schemas.Filter(
        groundtruths=schemas.LogicalFunction(
            args=[
                schemas.Condition(
                    lhs=schemas.Symbol(
                        name=schemas.SupportedSymbol.DATASET_NAME
                    ),
                    rhs=schemas.Value.infer(dataset_name),
                    op=schemas.FilterOperator.EQ,
                ),
                schemas.Condition(
                    lhs=schemas.Symbol(name=schemas.SupportedSymbol.TASK_TYPE),
                    rhs=schemas.Value.infer(
                        enums.TaskType.SEMANTIC_SEGMENTATION
                    ),
                    op=schemas.FilterOperator.CONTAINS,
                ),
                schemas.Condition(
                    lhs=schemas.Symbol(name=schemas.SupportedSymbol.RASTER),
                    op=schemas.FilterOperator.ISNOTNULL,
                ),
            ],
            op=schemas.LogicalOperator.AND,
        )
    )

    metrics = _compute_segmentation_metrics(
        db,
        parameters=schemas.EvaluationParameters(
            task_type=enums.TaskType.SEMANTIC_SEGMENTATION, label_map=None
        ),
        prediction_filter=prediction_filter,
        groundtruth_filter=groundtruth_filter,
    )
    # should have seven metrics
    # (one IOU for each of the four labels from the groundtruth set, and three mIOUs for each included label key)
    assert len(metrics) == 7
    for metric in metrics[:-3]:
        assert isinstance(metric, schemas.IOUMetric)
        assert metric.value < 1.0
    assert all([isinstance(m, schemas.mIOUMetric) for m in metrics[-3:]])
    assert all([m.value < 1.0 for m in metrics[-3:]])


def test_compute_semantic_segmentation_metrics(
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

    job_request = schemas.EvaluationRequest(
        dataset_names=[dataset_name],
        model_names=[model_name],
        parameters=schemas.EvaluationParameters(
            task_type=enums.TaskType.SEMANTIC_SEGMENTATION,
        ),
    )

    evaluations = create_or_get_evaluations(db=db, job_request=job_request)
    assert len(evaluations) == 1
    assert evaluations[0].status == enums.EvaluationStatus.PENDING

    _ = compute_semantic_segmentation_metrics(
        db=db, evaluation_id=evaluations[0].id
    )

    evaluations = create_or_get_evaluations(db=db, job_request=job_request)
    assert len(evaluations) == 1
    assert evaluations[0].status in {
        enums.EvaluationStatus.RUNNING,
        enums.EvaluationStatus.DONE,
    }

    metrics = evaluations[0].metrics

    expected_metrics = {
        # none of these three labels have a predicted label
        schemas.Label(key="k1", value="v2", score=None): 0,
        schemas.Label(key="k2", value="v2", score=None): 0,
        schemas.Label(key="k3", value="v3", score=None): 0,
        # this last metric value should round to .33
        schemas.Label(key="k1", value="v1", score=None): 0.33,
    }

    expected_mIOU_metrics = {"k1": (0.33 + 0) / 2, "k2": 0, "k3": 0}

    assert metrics
    for metric in metrics:
        assert isinstance(metric.value, float)
        if metric.type == "mIOU":
            assert metric.parameters
            assert metric.parameters["label_key"]
            assert (
                metric.value
                - expected_mIOU_metrics[metric.parameters["label_key"]]
            ) <= 0.01
        else:
            # the IOU value for (k1, v1) is bound between .327 and .336
            assert metric.label
            assert (metric.value - expected_metrics[metric.label]) <= 0.01
