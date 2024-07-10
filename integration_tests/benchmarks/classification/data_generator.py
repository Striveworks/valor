import time
import json
from typing import List, Tuple

import numpy as np


from valor import (
    connect,
    Client,
    Dataset,
    Model,
    Datum,
    Annotation,
    GroundTruth,
    Prediction,
    Label,
)
from valor.enums import MetricType
from valor.schemas import (
    Box,
    Raster,
)
from valor.exceptions import DatasetAlreadyExistsError, ModelAlreadyExistsError


def create_score(idx, n_values):

    if n_values < 3:
        raise ValueError

    if idx == 0:
        return 0.6
    elif idx == 1:
        return 0.25
    elif idx == 2:
        return 0.15
    else:
        return 0
    

def create_datums(count: int) -> List[Datum]:
    """

    TODO Add metadata fuzzing.
    
    """
    return [
        Datum(uid=f"uid{idx}", metadata=None)
        for idx in range(count)
    ]


def create_classifications(
    n_datums: int = 10,
    n_keys: int = 3,
    n_values: int = 10,
) -> Tuple[List[GroundTruth], List[Prediction]]:
    
    ground_truths = []
    predictions = []
    for datum in create_datums(n_datums):

        ground_truth_labels = [
            Label(key=f"k{key}", value=f"v{value}")
            for key, value in enumerate(np.random.randint(0, n_values, size=(n_keys,)))
        ]
        ground_truth = GroundTruth(
            datum=datum,
            annotations=[
                Annotation(
                    labels=ground_truth_labels,
                    metadata=None,
                )
            ]
        )
        ground_truths.append(ground_truth)

        prediction_labels = [
            Label(key=f"k{key}", value=f"v{value}", score=create_score(idx, n_values))
            for key in range(n_keys)
            for idx, value in enumerate(np.random.permutation(n_values))
        ]
        prediction = Prediction(
            datum=datum,
            annotations=[
                Annotation(
                    labels=prediction_labels,
                    metadata=None,
                )
            ]
        )
        predictions.append(prediction)

    return (ground_truths, predictions)


def run_base_evaluation(dset: Dataset, model: Model):
    """Run a base evaluation (with no PR curves)."""
    evaluation = model.evaluate_classification(dset)
    evaluation.wait_for_completion()
    return evaluation


def run_pr_curve_evaluation(dset: Dataset, model: Model):
    """Run a base evaluation with PrecisionRecallCurve included."""
    evaluation = model.evaluate_classification(
        dset,
        metrics_to_return=[
            MetricType.Accuracy,
            MetricType.Precision,
            MetricType.Recall,
            MetricType.F1,
            MetricType.ROCAUC,
            MetricType.PrecisionRecallCurve,
        ],
    )
    evaluation.wait_for_completion()
    return evaluation


def run_detailed_pr_curve_evaluation(dset: Dataset, model: Model):
    """Run a base evaluation with PrecisionRecallCurve and DetailedPrecisionRecallCurve included."""

    evaluation = model.evaluate_classification(
        dset,
        metrics_to_return=[
            MetricType.Accuracy,
            MetricType.Precision,
            MetricType.Recall,
            MetricType.F1,
            MetricType.ROCAUC,
            MetricType.PrecisionRecallCurve,
            MetricType.DetailedPrecisionRecallCurve,
        ],
    )
    evaluation.wait_for_completion()
    return evaluation


if __name__ == "__main__":
    
    connect("http://0.0.0.0:8000")
    client = Client()

    print()
    print("=== Data Generation Benchmark ===")

    n_datums = 10000
    n_keys = 3
    n_values = 20

    gts, pds = create_classifications(n_datums=n_datums, n_keys=n_keys, n_values=n_values)

    dataset_name = "dataset"
    model_name = "model"

    try:
        dataset = Dataset.create(dataset_name, metadata=None)
        model = Model.create(model_name, metadata=None)


        start = time.time()
        dataset.add_groundtruths(gts, timeout=-1)
        gt_ingestion = time.time() - start

        start = time.time()
        model.add_predictions(dataset, pds, timeout=-1)
        pd_ingestion = time.time() - start

        dataset.finalize()

        eval_base = run_base_evaluation(dataset, model)
        eval_base_pr = run_pr_curve_evaluation(dataset, model)
        eval_base_pr_detailed = run_detailed_pr_curve_evaluation(dataset, model)

        start = time.time()
        client.delete_dataset(dataset_name)
        ds_deletion_time = time.time() - start

        start = time.time()
        client.delete_model(model_name)
        md_deletion_time = time.time() - start

        results = {
            "info": {
                "number_of_datums": n_datums,
                "number_of_unique_labels": eval_base.meta["labels"],
                "number_of_annotations": eval_base.meta["annotations"],
            },
            "ingestion": {
                "groundtruths": f"{(gt_ingestion):.1f} seconds",
                "predictions": f"{(pd_ingestion):.1f} seconds",
            },
            "evaluation": {
                "base": f"{(eval_base.meta['duration']):.1f} seconds",
                "base+pr": f"{(eval_base_pr.meta['duration']):.1f} seconds",
                "base+pr+detailed": f"{(eval_base_pr_detailed.meta['duration']):.1f} seconds",
            },
            "deletion": {
                "dataset": f"{(ds_deletion_time):.1f} seconds",
                "model": f"{(md_deletion_time):.1f} seconds",
            }
        }
        print(json.dumps(results, indent=2))

    except DatasetAlreadyExistsError as e:
        client.delete_dataset(dataset_name)
        raise e
    except ModelAlreadyExistsError as e:
        client.delete_model(model_name)
        raise e
