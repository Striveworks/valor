import json
import time
from typing import List, Tuple

import numpy as np
from pydantic import BaseModel

from valor import (
    Annotation,
    Client,
    Dataset,
    Datum,
    GroundTruth,
    Label,
    Model,
    Prediction,
    connect,
)
from valor.enums import MetricType
from valor.exceptions import DatasetAlreadyExistsError, ModelAlreadyExistsError


class Benchmark(BaseModel):
    number_of_datums: int = -1
    number_of_annotations: int = -1
    number_of_unique_labels: int = -1

    groundtruth_ingestion: list[float] = list()
    groundtruth_deletion: list[float] = list()

    prediction_ingestion: list[float] = list()
    prediction_deletion: list[float] = list()

    eval_base: list[float] = list()
    eval_base_pr: list[float] = list()
    eval_base_pr_detailed: list[float] = list()

    @property
    def results(self) -> str:
        results_dict = {
            "info": {
                "number_of_datums": self.number_of_datums,
                "number_of_annotations": self.number_of_annotations,
                "number_of_unique_labels": self.number_of_unique_labels,
            },
            "ingestion": {
                "groundtruths": f"{(np.mean(self.groundtruth_ingestion)):.1f} seconds",
                "predictions": f"{(np.mean(self.prediction_ingestion)):.1f} seconds",
            },
            "evaluation": {
                "base": f"{(np.mean(self.eval_base)):.1f} seconds",
                "base+pr": f"{(np.mean(self.eval_base_pr)):.1f} seconds",
                "base+pr+detailed": f"{(np.mean(self.eval_base_pr_detailed)):.1f} seconds",
            },
            "deletion": {
                "dataset": f"{(np.mean(self.groundtruth_deletion)):.1f} seconds",
                "model": f"{(np.mean(self.prediction_deletion)):.1f} seconds",
            },
        }
        return json.dumps(results_dict, indent=2)


def create_score(idx, n_values):

    if n_values < 2:
        raise ValueError
    elif n_values == 2:
        if idx == 0:
            return 0.9
        else:
            return 0.1

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
    return [Datum(uid=f"uid{idx}", metadata=None) for idx in range(count)]


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
            for key, value in enumerate(
                np.random.randint(0, n_values, size=(n_keys,))
            )
        ]
        ground_truth = GroundTruth(
            datum=datum,
            annotations=[
                Annotation(
                    labels=ground_truth_labels,
                    metadata=None,
                )
            ],
        )
        ground_truths.append(ground_truth)

        prediction_labels = [
            Label(
                key=f"k{key}",
                value=f"v{value}",
                score=create_score(idx, n_values),
            )
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
            ],
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


def run(
    client: Client,
    benchmark: Benchmark,
    n_datums: int,
    n_label_keys: int,
    n_label_values: int,
    *_,
    label_mapping: None = None,
):
    dataset_name = f"benchmark_dataset_{time.time()}"
    model_name = f"benchmark_model_{time.time()}"

    gts, pds = create_classifications(
        n_datums=n_datums, n_keys=n_label_keys, n_values=n_label_values
    )

    try:
        dataset = Dataset.create(dataset_name, metadata=None)
        model = Model.create(model_name, metadata=None)
    except (DatasetAlreadyExistsError, ModelAlreadyExistsError) as e:
        client.delete_dataset(dataset_name)
        client.delete_model(model_name)
        raise e

    start = time.time()
    dataset.add_groundtruths(gts, timeout=-1)
    groundtruth_ingestion = time.time() - start

    start = time.time()
    model.add_predictions(dataset, pds, timeout=-1)
    prediction_ingestion = time.time() - start

    dataset.finalize()

    eval_base = run_base_evaluation(dataset, model)
    eval_base_pr = run_pr_curve_evaluation(dataset, model)
    eval_base_pr_detailed = run_detailed_pr_curve_evaluation(dataset, model)

    start = time.time()
    client.delete_dataset(dataset_name)
    groundtruth_deletion_time = time.time() - start

    start = time.time()
    client.delete_model(model_name)
    prediction_deletion_time = time.time() - start

    benchmark.number_of_datums = n_datums
    benchmark.number_of_annotations = eval_base.meta["annotations"]
    benchmark.number_of_unique_labels = eval_base.meta["labels"]

    benchmark.groundtruth_ingestion.append(groundtruth_ingestion)
    benchmark.prediction_ingestion.append(prediction_ingestion)

    benchmark.eval_base.append(eval_base.meta["duration"])
    benchmark.eval_base_pr.append(eval_base_pr.meta["duration"])
    benchmark.eval_base_pr_detailed.append(
        eval_base_pr_detailed.meta["duration"]
    )

    benchmark.groundtruth_deletion.append(groundtruth_deletion_time)
    benchmark.prediction_deletion.append(prediction_deletion_time)

    return benchmark


if __name__ == "__main__":

    connect("http://0.0.0.0:8000")
    client = Client()

    print()
    print("=== Data Generation Benchmark ===")
    print()

    number_of_datums_trials = [1, 10, 100, 1000, 10000]
    number_of_label_keys_trials = [1, 10]
    number_of_label_values_trials = [10, 100]
    number_of_trials = 1

    for n_datums in number_of_datums_trials:
        for n_label_keys in number_of_label_keys_trials:
            for n_label_values in number_of_label_values_trials:

                print(
                    f"=== {n_datums} Datums, {n_label_keys*n_label_values} Labels ==="
                )

                benchmark = Benchmark()
                for trial in range(number_of_trials):
                    benchmark = run(
                        client=client,
                        benchmark=benchmark,
                        n_datums=n_datums,
                        n_label_keys=n_label_keys,
                        n_label_values=n_label_values,
                    )

                print(benchmark.results)
