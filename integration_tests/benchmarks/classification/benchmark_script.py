import json
import os
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import requests

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

connect("http://0.0.0.0:8000")
client = Client()


def download_data_if_not_exists(file_path: Path, file_url: str):
    """Download the data from a public bucket if it doesn't exist in the repo."""
    if os.path.exists(file_path):
        return

    response = json.loads(requests.get(file_url).text)
    with open(file_path, "w+") as file:
        json.dump(response, file, indent=4)


def write_results_to_file(write_path: Path, results: list[dict]):
    """Write results to results.json"""
    current_datetime = datetime.now().strftime("%d/%m/%Y %H:%M:%S")
    if os.path.isfile(write_path):
        with open(write_path, "r") as file:
            file.seek(0)
            data = json.load(file)
    else:
        data = {}

    data[current_datetime] = results

    with open(write_path, "w+") as file:
        json.dump(data, file, indent=4)


def ingest_groundtruths(
    dset: Dataset,
    raw: dict,
    pair_limit: int,
    timeout: int | None,
):
    """Ingest groundtruths into Valor."""

    groundtruths = []
    slice_ = (
        raw["groundtruth_prediction_pairs"][:pair_limit]
        if pair_limit != -1
        else raw["groundtruth_prediction_pairs"]
    )
    for groundtruth, _ in slice_:
        groundtruths.append(
            GroundTruth(
                datum=Datum(
                    uid=groundtruth["value"]["datum"]["uid"],
                    metadata={"width": 224, "height": 224},
                ),
                annotations=[
                    Annotation(
                        labels=[
                            Label(
                                key=label["key"],
                                value=label["value"],
                                score=label["score"],
                            )
                            for label in annotation["labels"]
                        ],
                    )
                    for annotation in groundtruth["value"]["annotations"]
                ],
            )
        )

    dset.add_groundtruths(groundtruths, timeout=timeout)


def ingest_predictions(
    dset: Dataset,
    model: Model,
    raw: dict,
    pair_limit: int,
    timeout: int | None,
):
    """Ingest the data into Valor."""

    predictions = []
    slice_ = (
        raw["groundtruth_prediction_pairs"][:pair_limit]
        if pair_limit != -1
        else raw["groundtruth_prediction_pairs"]
    )
    for _, prediction in slice_:

        predictions.append(
            Prediction(
                datum=Datum(
                    uid=prediction["value"]["datum"]["uid"],
                    metadata={"width": 224, "height": 224},
                ),
                annotations=[
                    Annotation(
                        labels=[
                            Label(
                                key=label["key"],
                                value=label["value"],
                                score=label["score"],
                            )
                            for label in annotation["labels"]
                        ],
                    )
                    for annotation in prediction["value"]["annotations"]
                ],
            )
        )

    model.add_predictions(dset, predictions, timeout=timeout)


def run_base_evaluation(dset: Dataset, model: Model, timeout: int | None):
    """Run a base evaluation (with no PR curves)."""
    try:
        evaluation = model.evaluate_classification(dset)
        evaluation.wait_for_completion(timeout=timeout)
    except TimeoutError:
        raise TimeoutError(
            f"Base evaluation timed out when processing {evaluation.meta['datums']} datums."  # type: ignore
        )
    return evaluation


def run_pr_curve_evaluation(dset: Dataset, model: Model, timeout: int | None):
    """Run a base evaluation with PrecisionRecallCurve included."""
    try:
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
        evaluation.wait_for_completion(timeout=timeout)
    except TimeoutError:
        raise TimeoutError(
            f"PR evaluation timed out when processing {evaluation.meta['datums']} datums."  # type: ignore
        )
    return evaluation


def run_detailed_pr_curve_evaluation(
    dset: Dataset, model: Model, timeout: int | None
):
    """Run a base evaluation with PrecisionRecallCurve and DetailedPrecisionRecallCurve included."""

    try:
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
        evaluation.wait_for_completion(timeout=timeout)
    except TimeoutError:
        raise TimeoutError(
            f"Detailed evaluation timed out when processing {evaluation.meta['datums']} datums."  # type: ignore
        )
    return evaluation


@dataclass
class DataBenchmark:
    ingestion: float
    finalization: float
    deletion: float

    def result(self) -> dict[str, float | str]:
        return {
            "ingestion": round(self.ingestion, 2),
            "finalization": round(self.finalization, 2),
            "deletion": round(self.deletion, 2),
        }


@dataclass
class EvaluationBenchmark:
    limit: int
    gt_stats: DataBenchmark
    pd_stats: DataBenchmark
    n_datums: int
    n_annotations: int
    n_labels: int
    eval_base: float
    eval_base_pr: float
    eval_base_pr_detail: float

    def result(self) -> dict[str, float | str | dict[str, str | float]]:
        return {
            "limit": self.limit,
            "groundtruths": self.gt_stats.result(),
            "predictions": self.pd_stats.result(),
            "evaluation": {
                "number_of_datums": self.n_datums,
                "number_of_annotations": self.n_annotations,
                "number_of_labels": self.n_labels,
                "base": round(self.eval_base, 2),
                "base+pr": round(self.eval_base_pr, 2),
                "base+pr+detailed": round(self.eval_base_pr_detail, 2),
            },
        }


def run_benchmarking_analysis(
    limits: list[int],
    results_file: str = "results.json",
    data_file: str = "data.json",
    ingestion_timeout: int | None = 150,
    evaluation_timeout: int | None = 40,
):
    """Time various function calls and export the results."""
    current_directory = Path(os.path.dirname(os.path.realpath(__file__)))
    write_path = current_directory / Path(results_file)
    data_path = current_directory / Path(data_file)

    download_data_if_not_exists(
        file_path=data_path,
        file_url="https://pub-fae71003f78140bdaedf32a7c8d331d2.r2.dev/classification_data.json",
    )

    with open(data_path) as file:
        file.seek(0)
        raw_data = json.load(file)

    results = list()
    for limit in limits:

        dset = Dataset.create(name=f"bird-identification-{time.time()}")
        model = Model.create(name=f"some_model-{time.time()}")

        # ingest groundtruths
        start_time = time.time()
        ingest_groundtruths(
            dset=dset,
            raw=raw_data,
            pair_limit=limit,
            timeout=ingestion_timeout,
        )
        gt_ingest_time = time.time() - start_time

        # finalize groundtruths
        start_time = time.time()
        dset.finalize()
        gt_finalization_time = time.time() - start_time

        # ingest predictions
        start_time = time.time()
        ingest_predictions(
            dset=dset,
            model=model,
            raw=raw_data,
            pair_limit=limit,
            timeout=ingestion_timeout,
        )
        pd_ingest_time = time.time() - start_time

        # finalize predictions
        start_time = time.time()
        model.finalize_inferences(dset)
        pd_finalization_time = time.time() - start_time

        # run evaluations
        eval_base = run_base_evaluation(
            dset=dset, model=model, timeout=evaluation_timeout
        )
        eval_pr = run_pr_curve_evaluation(
            dset=dset, model=model, timeout=evaluation_timeout
        )
        # NOTE: turned this off due to long runtimes causing TimeoutError
        # eval_detail = run_detailed_pr_curve_evaluation(
        #     dset=dset, model=model, timeout=evaluation_timeout
        # )

        # delete model
        start = time.time()
        client.delete_model(model.name, timeout=30)
        pd_deletion_time = time.time() - start

        # delete dataset
        start = time.time()
        client.delete_dataset(dset.name, timeout=30)
        gt_deletion_time = time.time() - start

        results.append(
            EvaluationBenchmark(
                limit=limit,
                gt_stats=DataBenchmark(
                    ingestion=gt_ingest_time,
                    finalization=gt_finalization_time,
                    deletion=gt_deletion_time,
                ),
                pd_stats=DataBenchmark(
                    ingestion=pd_ingest_time,
                    finalization=pd_finalization_time,
                    deletion=pd_deletion_time,
                ),
                n_datums=eval_base.meta["datums"],
                n_annotations=eval_base.meta["annotations"],
                n_labels=eval_base.meta["labels"],
                eval_base=eval_base.meta["duration"],
                eval_base_pr=eval_pr.meta["duration"],
                eval_base_pr_detail=-1,
            ).result()
        )

    write_results_to_file(write_path=write_path, results=results)


if __name__ == "__main__":
    run_benchmarking_analysis(limits=[5000, 5000])
