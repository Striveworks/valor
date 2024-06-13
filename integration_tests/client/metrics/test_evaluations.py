import pytest
from sqlalchemy.orm import Session

from valor import (
    Annotation,
    Client,
    Dataset,
    Datum,
    Filter,
    GroundTruth,
    Label,
    Model,
    Prediction,
)
from valor.enums import EvaluationStatus
from valor.exceptions import ClientException
from valor_api import crud, enums, schemas
from valor_api.backend import core


def test_restart_failed_evaluation(db: Session, client: Client):
    crud.create_dataset(db=db, dataset=schemas.Dataset(name="dataset"))
    crud.create_groundtruths(
        db=db,
        groundtruths=[
            schemas.GroundTruth(
                dataset_name="dataset",
                datum=schemas.Datum(uid="123"),
                annotations=[
                    schemas.Annotation(
                        labels=[schemas.Label(key="class", value="dog")],
                    )
                ],
            )
        ],
    )
    crud.create_model(db=db, model=schemas.Model(name="model"))
    crud.create_predictions(
        db=db,
        predictions=[
            schemas.Prediction(
                dataset_name="dataset",
                model_name="model",
                datum=schemas.Datum(uid="123"),
                annotations=[
                    schemas.Annotation(
                        labels=[
                            schemas.Label(key="class", value="dog", score=1.0)
                        ],
                    )
                ],
            )
        ],
    )
    crud.finalize(db=db, dataset_name="dataset")

    # retrieve dataset and model on the client-side
    dataset = Dataset.get("dataset")
    model = Model.get("model")
    assert dataset
    assert model

    # create evaluation
    eval1 = model.evaluate_classification(dataset, allow_retries=False)
    eval1.wait_for_completion(
        timeout=30
    )  # the overwrite below doesn't work unless status is DONE
    assert eval1.status == enums.EvaluationStatus.DONE

    # overwrite status to failed
    evaluation = core.fetch_evaluation_from_id(db=db, evaluation_id=eval1.id)
    evaluation.status = enums.EvaluationStatus.FAILED
    db.commit()

    # get evaluation and verify it is failed
    eval2 = model.evaluate_classification(dataset, allow_retries=False)
    assert eval2.id == eval1.id
    assert eval2.status == enums.EvaluationStatus.FAILED

    # get evaluation and allow retries, this should result in a finished eval
    eval3 = model.evaluate_classification(dataset, allow_retries=True)
    eval3.wait_for_completion(timeout=30)
    assert eval3.id == eval1.id
    assert eval3.status == enums.EvaluationStatus.DONE


def test_get_sorted_evaluations(
    client: Client,
    gt_clfs_tabular: list[int],
    pred_clfs_tabular: list[list[float]],
    gt_semantic_segs1: list[GroundTruth],
    gt_semantic_segs2: list[GroundTruth],
    pred_semantic_segs: list[Prediction],
    gts_det_with_label_maps: list[GroundTruth],
    preds_det_with_label_maps: list[Prediction],
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
        filters=Filter(datums=(Datum.metadata["md3"] == True)),  # noqa: E712
    )
    assert clf_eval_1.wait_for_completion(timeout=30) == EvaluationStatus.DONE

    clf_eval_2 = model.evaluate_classification(
        dataset,
    )
    assert clf_eval_2.wait_for_completion(timeout=30) == EvaluationStatus.DONE

    clf_eval_3 = model.evaluate_classification(
        dataset,
        filters=Filter(datums=(Datum.metadata["md1"] == "md1-val2")),
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
    for gt in gts_det_with_label_maps:
        dataset.add_groundtruth(gt)
    dataset.finalize()

    model = Model.create("det_model")
    for pd in preds_det_with_label_maps:
        model.add_prediction(dataset, pd)
    model.finalize_inferences(dataset)

    label_mapping = {
        Label(key="class_name", value="maine coon cat"): Label(
            key="class", value="cat"
        ),
        Label(key="class", value="siamese cat"): Label(
            key="class", value="cat"
        ),
        Label(key="class", value="british shorthair"): Label(
            key="class", value="cat"
        ),
    }
    det_eval_1 = model.evaluate_detection(
        dataset,
        iou_thresholds_to_compute=[0.1, 0.6],
        iou_thresholds_to_return=[0.1, 0.6],
        label_map=label_mapping,
    )
    assert det_eval_1.wait_for_completion(timeout=30) == EvaluationStatus.DONE

    det_eval_2 = model.evaluate_detection(
        dataset,
        iou_thresholds_to_compute=[0.1, 0.6],
        iou_thresholds_to_return=[0.1, 0.6],
    )
    assert det_eval_2.wait_for_completion(timeout=30) == EvaluationStatus.DONE

    label_mapping = {
        # map the ground truths
        Label(key="class_name", value="maine coon cat"): Label(
            key="class", value="bar"
        ),
        Label(key="class", value="siamese cat"): Label(
            key="class", value="bar"
        ),
        Label(key="class", value="british shorthair"): Label(
            key="class", value="bar"
        ),
        # map the predictions
        Label(key="class", value="cat"): Label(key="class", value="bar"),
        Label(key="class_name", value="cat"): Label(key="class", value="bar"),
    }

    det_eval_3 = model.evaluate_detection(
        dataset,
        iou_thresholds_to_compute=[0.1, 0.6],
        iou_thresholds_to_return=[0.1, 0.6],
        label_map=label_mapping,
    )
    assert det_eval_3.wait_for_completion(timeout=30) == EvaluationStatus.DONE

    # start by getting the unsorted evaluations
    unsorted_evaluations = client.get_evaluations(
        datasets=["clf_dataset", "seg_dataset", "det_dataset"]
    )
    assert len(unsorted_evaluations) == 7
    assert [
        evaluation.parameters.task_type for evaluation in unsorted_evaluations
    ] == [
        "classification",
        "classification",
        "classification",
        "object-detection",
        "object-detection",
        "object-detection",
        "semantic-segmentation",
    ]

    # next, sort the classification metrics by Accuracy
    evaluations = client.get_evaluations(
        datasets=["clf_dataset", "seg_dataset", "det_dataset"],
        metrics_to_sort_by={
            "Accuracy": "class",
        },
    )
    assert len(evaluations) == 7
    assert [evaluation.parameters.task_type for evaluation in evaluations] == [
        "classification",
        "classification",
        "classification",
        "object-detection",
        "object-detection",
        "object-detection",
        "semantic-segmentation",
    ]

    # check that clf metrics are now sorted
    ordered_accuracy_metrics = [
        metric
        for eval in evaluations
        for metric in eval.metrics
        if metric["type"] == "Accuracy"
    ]
    assert ordered_accuracy_metrics == [
        {
            "type": "Accuracy",
            "value": 0.5,
            "parameters": {"label_key": "class"},
        },
        {
            "type": "Accuracy",
            "value": 0.3333333333333333,
            "parameters": {"label_key": "class"},
        },
        {
            "type": "Accuracy",
            "value": 0.3333333333333333,
            "parameters": {"label_key": "class"},
        },
    ]

    # repeat, but also sort by precision
    evaluations = client.get_evaluations(
        datasets=["clf_dataset", "seg_dataset", "det_dataset"],
        metrics_to_sort_by={
            "Accuracy": "class",
            "Precision": {"key": "class", "value": "1"},
        },
    )
    assert len(evaluations) == 7
    assert [evaluation.parameters.task_type for evaluation in evaluations] == [
        "classification",
        "classification",
        "classification",
        "object-detection",
        "object-detection",
        "object-detection",
        "semantic-segmentation",
    ]

    ordered_accuracy_metrics = [
        metric
        for eval in evaluations
        for metric in eval.metrics
        if metric["type"] == "Accuracy"
    ]
    assert ordered_accuracy_metrics == [
        {
            "type": "Accuracy",
            "value": 0.5,
            "parameters": {"label_key": "class"},
        },
        {
            "type": "Accuracy",
            "value": 0.3333333333333333,
            "parameters": {"label_key": "class"},
        },
        {
            "type": "Accuracy",
            "value": 0.3333333333333333,
            "parameters": {"label_key": "class"},
        },
    ]

    ordered_precision_metrics = [
        metric
        for eval in evaluations
        for metric in eval.metrics
        if metric["type"] == "Precision"
        and metric["label"] == {"key": "class", "value": "1"}
    ]
    assert ordered_precision_metrics == [
        {
            "type": "Precision",
            "value": 0.6666666666666666,
            "label": {"key": "class", "value": "1"},
        },
        {
            "type": "Precision",
            "value": 0.0,
            "label": {"key": "class", "value": "1"},
        },
        {
            "type": "Precision",
            "value": 0.0,
            "label": {"key": "class", "value": "1"},
        },
    ]

    # sort all task types and check outputs
    evaluations = client.get_evaluations(
        datasets=["clf_dataset", "seg_dataset", "det_dataset"],
        metrics_to_sort_by={
            "Accuracy": "class",
            "IOU": {"key": "k2", "value": "v2"},
            "mAPAveragedOverIOUs": "class",
        },
    )
    assert len(evaluations) == 7
    assert [evaluation.parameters.task_type for evaluation in evaluations] == [
        "classification",
        "classification",
        "classification",
        "object-detection",
        "object-detection",
        "object-detection",
        "semantic-segmentation",
    ]

    ordered_accuracy_metrics = [
        metric
        for eval in evaluations
        for metric in eval.metrics
        if metric["type"] == "Accuracy"
    ]
    assert ordered_accuracy_metrics == [
        {
            "type": "Accuracy",
            "value": 0.5,
            "parameters": {"label_key": "class"},
        },
        {
            "type": "Accuracy",
            "value": 0.3333333333333333,
            "parameters": {"label_key": "class"},
        },
        {
            "type": "Accuracy",
            "value": 0.3333333333333333,
            "parameters": {"label_key": "class"},
        },
    ]

    ordered_map_metrics = [
        metric
        for eval in evaluations
        for metric in eval.metrics
        if metric["type"] == "mAPAveragedOverIOUs"
        and metric["parameters"]["label_key"] == "class"
    ]
    assert ordered_map_metrics == [
        {
            "type": "mAPAveragedOverIOUs",
            "parameters": {"ious": [0.1, 0.6], "label_key": "class"},
            "value": 0.6633663366336634,
        },
        {
            "type": "mAPAveragedOverIOUs",
            "parameters": {"ious": [0.1, 0.6], "label_key": "class"},
            "value": 0.33663366336633666,
        },
        {
            "type": "mAPAveragedOverIOUs",
            "parameters": {"ious": [0.1, 0.6], "label_key": "class"},
            "value": 0.0,
        },
    ]

    # note: we don't check IOU because there's only one segmentation evaluation

    # check that we get an error if we pass an incorrect dict
    with pytest.raises(ClientException):
        evaluations = client.get_evaluations(
            datasets=["clf_dataset", "seg_dataset", "det_dataset"],
            metrics_to_sort_by=[
                "mAPAveragedOverIOUs",
                "Accuracy",
                "mIOU",
            ],
        )

    with pytest.raises(ClientException):
        evaluations = client.get_evaluations(
            datasets=["clf_dataset", "seg_dataset", "det_dataset"],
            metrics_to_sort_by={"Accuracy": {"fake": "dictionary"}},
        )

    # assert that nonsensical sort items basically don't do anything
    evaluations = client.get_evaluations(
        datasets=["clf_dataset", "seg_dataset", "det_dataset"],
        metrics_to_sort_by={
            "Accuracy": "not a real class",
        },
    )
    assert len(evaluations) == 7
    assert [evaluation.parameters.task_type for evaluation in evaluations] == [
        "classification",
        "classification",
        "classification",
        "object-detection",
        "object-detection",
        "object-detection",
        "semantic-segmentation",
    ]

    evaluations = client.get_evaluations(
        datasets=["clf_dataset", "seg_dataset", "det_dataset"],
        metrics_to_sort_by={
            "not a real metric": "k1",
        },
    )
    assert len(evaluations) == 7
    assert [evaluation.parameters.task_type for evaluation in evaluations] == [
        "classification",
        "classification",
        "classification",
        "object-detection",
        "object-detection",
        "object-detection",
        "semantic-segmentation",
    ]
