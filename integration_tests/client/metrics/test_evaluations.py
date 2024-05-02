import pytest
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session

from valor import (
    Annotation,
    Client,
    Dataset,
    Datum,
    GroundTruth,
    Label,
    Model,
    Prediction,
)
from valor.enums import AnnotationType, EvaluationStatus, TaskType
from valor.exceptions import ClientException
from valor_api import crud, enums, schemas
from valor_api.backend import core


def test_restart_failed_evaluation(db: Session, client: Client):
    crud.create_dataset(db=db, dataset=schemas.Dataset(name="dataset"))
    crud.create_model(db=db, model=schemas.Model(name="model"))
    crud.finalize(db=db, dataset_name="dataset")

    # retrieve dataset and model on the client-side
    dataset = Dataset.get("dataset")
    model = Model.get("model")
    assert dataset
    assert model

    # create evaluation and overwrite status to failed
    eval1 = model.evaluate_classification(dataset, allow_retries=False)
    assert eval1.status == enums.EvaluationStatus.DONE
    try:
        evaluation = core.fetch_evaluation_from_id(
            db=db, evaluation_id=eval1.id
        )
        evaluation.status = enums.EvaluationStatus.FAILED
        db.commit()
    except IntegrityError as e:
        db.rollback()
        raise e

    # get evaluation and verify it is failed
    eval2 = model.evaluate_classification(dataset, allow_retries=False)
    assert eval2.id == eval1.id
    assert eval2.status == enums.EvaluationStatus.FAILED

    # get evaluation and allow retries, this should result in a finished eval
    eval3 = model.evaluate_classification(dataset, allow_retries=True)
    assert eval3.id == eval1.id
    assert eval3.status == enums.EvaluationStatus.DONE


def test_get_sorted_evaluations(
    client: Client,
    gt_clfs_tabular: list[int],
    pred_clfs_tabular: list[list[float]],
    gt_semantic_segs1: list[GroundTruth],
    gt_semantic_segs2: list[GroundTruth],
    pred_semantic_segs: list[Prediction],
    gt_dets1: list[GroundTruth],
    pred_dets: list[Prediction],
):
    """Fill psql with evaluations, then make sure the metrics_to_sort_by parameter on get_evaluations works correctly."""
    # evaluate classification
    dataset = Dataset.create(name="clf_dataset")
    for i, label_value in enumerate(gt_clfs_tabular):
        gt = GroundTruth(
            datum=Datum(
                uid=f"uid{i}",
                metadata={
                    "md1": f"md1-val{i % 3}",
                    "md2": f"md2-val{i % 4}",
                    "md3": i % 3 == 2,
                },
            ),
            annotations=[
                Annotation(
                    task_type=TaskType.CLASSIFICATION,
                    labels=[Label(key="class", value=str(label_value))],
                )
            ],
        )
        dataset.add_groundtruth(gt)
    dataset.finalize()

    model = Model.create(name="clf_model")
    for i, pred in enumerate(pred_clfs_tabular):
        pd = Prediction(
            datum=Datum(
                uid=f"uid{i}",
                metadata={
                    "md1": f"md1-val{i % 3}",
                    "md2": f"md2-val{i % 4}",
                    "md3": i % 3 == 2,
                },
            ),
            annotations=[
                Annotation(
                    task_type=TaskType.CLASSIFICATION,
                    labels=[
                        Label(key="class", value=str(pidx), score=pred[pidx])
                        for pidx in range(len(pred))
                    ],
                )
            ],
        )
        model.add_prediction(dataset, pd)
    model.finalize_inferences(dataset)

    clf_eval_1 = model.evaluate_classification(
        dataset,
        filter_by=[
            Datum.metadata["md3"]
            == True  # noqa: E712 - 'is' keyword is not overloadable, so we have to use 'symbol == True'
        ],
    )
    assert clf_eval_1.wait_for_completion(timeout=30) == EvaluationStatus.DONE

    clf_eval_2 = model.evaluate_classification(
        dataset,
    )
    assert clf_eval_2.wait_for_completion(timeout=30) == EvaluationStatus.DONE

    clf_eval_3 = model.evaluate_classification(
        dataset,
        filter_by=[
            Datum.metadata["md1"] == "md1-val2",
        ],
    )
    assert clf_eval_3.wait_for_completion(timeout=30) == EvaluationStatus.DONE

    # evaluate semantic segmentation
    dataset = Dataset.create("seg_dataset")
    model = Model.create("seg_model")

    for gt in gt_semantic_segs1 + gt_semantic_segs2:
        dataset.add_groundtruth(gt)
    dataset.finalize()

    for pred in pred_semantic_segs:
        model.add_prediction(dataset, pred)
    model.finalize_inferences(dataset)

    seg_eval_1 = model.evaluate_segmentation(dataset)
    assert seg_eval_1.wait_for_completion(timeout=30) == EvaluationStatus.DONE

    # evaluate detection
    dataset = Dataset.create("det_dataset")
    for gt in gt_dets1:
        dataset.add_groundtruth(gt)
    dataset.finalize()

    model = Model.create("det_model")
    for pd in pred_dets:
        model.add_prediction(dataset, pd)
    model.finalize_inferences(dataset)

    det_eval_1 = model.evaluate_detection(
        dataset,
        iou_thresholds_to_compute=[0.1, 0.6],
        iou_thresholds_to_return=[0.1, 0.6],
        filter_by=[
            Label.key == "k1",
        ],
        convert_annotations_to_type=AnnotationType.BOX,
    )
    assert det_eval_1.wait_for_completion(timeout=30) == EvaluationStatus.DONE

    det_eval_2 = model.evaluate_detection(
        dataset,
        iou_thresholds_to_compute=[0.1, 0.6],
        iou_thresholds_to_return=[0.1, 0.6],
    )
    assert det_eval_2.wait_for_completion(timeout=30) == EvaluationStatus.DONE

    # start by getting the unsorted evaluations
    unsorted_evaluations = client.get_evaluations(
        datasets=["clf_dataset", "seg_dataset", "det_dataset"]
    )
    assert len(unsorted_evaluations) == 6
    assert [
        evaluation.parameters.task_type for evaluation in unsorted_evaluations
    ] == [
        "classification",
        "classification",
        "classification",
        "object-detection",
        "object-detection",
        "semantic-segmentation",
    ]

    assert [
        metric
        for eval in unsorted_evaluations[:3]
        for metric in eval.metrics
        if metric["type"] == "mAccuracy"
    ] == [
        {"type": "mAccuracy", "value": 0.3333333333333333},
        {"type": "mAccuracy", "value": 0.5},
        {"type": "mAccuracy", "value": 0.3333333333333333},
    ]

    # next, sort the classification metrics by mAccuracy
    evaluations = client.get_evaluations(
        datasets=["clf_dataset", "seg_dataset", "det_dataset"],
        metrics_to_sort_by=["mAccuracy"],
    )
    assert len(evaluations) == 6
    assert [evaluation.parameters.task_type for evaluation in evaluations] == [
        "classification",
        "classification",
        "classification",
        "object-detection",
        "object-detection",
        "semantic-segmentation",
    ]

    # check that clf metrics are now sorted
    assert [
        metric
        for eval in evaluations[:3]
        for metric in eval.metrics
        if metric["type"] == "mAccuracy"
    ] == [
        {"type": "mAccuracy", "value": 0.5},
        {"type": "mAccuracy", "value": 0.3333333333333333},
        {"type": "mAccuracy", "value": 0.3333333333333333},
    ]

    # check that det metrics are unsorted
    assert [
        metric
        for eval in evaluations[3:5]
        for metric in eval.metrics
        if metric["type"] == "mAPAveragedOverIOUs"
    ] == [
        {
            "type": "mAPAveragedOverIOUs",
            "parameters": {"ious": [0.1, 0.6]},
            "value": 0.2524752475247525,
        },
        {
            "type": "mAPAveragedOverIOUs",
            "parameters": {"ious": [0.1, 0.6]},
            "value": 0.504950495049505,
        },
    ]

    # sort all task types and check outputs
    evaluations = client.get_evaluations(
        datasets=["clf_dataset", "seg_dataset", "det_dataset"],
        metrics_to_sort_by=[
            "mAccuracy",
            "mIOU",
            "mAPAveragedOverIOUs",
        ],
    )
    assert len(evaluations) == 6
    assert [evaluation.parameters.task_type for evaluation in evaluations] == [
        "classification",
        "classification",
        "classification",
        "object-detection",
        "object-detection",
        "semantic-segmentation",
    ]
    assert [
        metric
        for eval in evaluations[:3]
        for metric in eval.metrics
        if metric["type"] == "mAccuracy"
    ] == [
        {"type": "mAccuracy", "value": 0.5},
        {"type": "mAccuracy", "value": 0.3333333333333333},
        {"type": "mAccuracy", "value": 0.3333333333333333},
    ]
    assert [
        metric
        for eval in evaluations[3:5]
        for metric in eval.metrics
        if metric["type"] == "mAPAveragedOverIOUs"
    ] == [
        {
            "type": "mAPAveragedOverIOUs",
            "parameters": {"ious": [0.1, 0.6]},
            "value": 0.504950495049505,
        },
        {
            "type": "mAPAveragedOverIOUs",
            "parameters": {"ious": [0.1, 0.6]},
            "value": 0.2524752475247525,
        },
    ]

    # check that the order of metrics_to_sort_by doesn't matter
    evaluations = client.get_evaluations(
        datasets=["clf_dataset", "seg_dataset", "det_dataset"],
        metrics_to_sort_by=[
            "mAPAveragedOverIOUs",
            "mAccuracy",
            "mIOU",
        ],
    )
    assert len(evaluations) == 6
    assert [evaluation.parameters.task_type for evaluation in evaluations] == [
        "classification",
        "classification",
        "classification",
        "object-detection",
        "object-detection",
        "semantic-segmentation",
    ]
    assert [
        metric
        for eval in evaluations[:3]
        for metric in eval.metrics
        if metric["type"] == "mAccuracy"
    ] == [
        {"type": "mAccuracy", "value": 0.5},
        {"type": "mAccuracy", "value": 0.3333333333333333},
        {"type": "mAccuracy", "value": 0.3333333333333333},
    ]
    assert [
        metric
        for eval in evaluations[3:5]
        for metric in eval.metrics
        if metric["type"] == "mAPAveragedOverIOUs"
    ] == [
        {
            "type": "mAPAveragedOverIOUs",
            "parameters": {"ious": [0.1, 0.6]},
            "value": 0.504950495049505,
        },
        {
            "type": "mAPAveragedOverIOUs",
            "parameters": {"ious": [0.1, 0.6]},
            "value": 0.2524752475247525,
        },
    ]

    # check that we get an error if we pass in a metric that is too granular to sort by
    with pytest.raises(ClientException):
        evaluations = client.get_evaluations(
            datasets=["clf_dataset", "seg_dataset", "det_dataset"],
            metrics_to_sort_by=[
                "mAPAveragedOverIOUs",
                "Accuracy",
                "mIOU",
            ],
        )
