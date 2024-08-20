import json
import os
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import requests
from valor_core import (
    Annotation,
    Datum,
    GroundTruth,
    Label,
    Prediction,
    enums,
    evaluate_classification,
)


def time_it(fn, *args, **kwargs) -> tuple[float, Any]:
    start = time.time()
    results = fn(*args, **kwargs)
    return (time.time() - start, results)


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


def ingest_groundtruths(raw: dict, pair_limit: int) -> list[GroundTruth]:
    """Ingest the data into Valor."""

    groundtruths = []
    slice_ = (
        raw["groundtruth_prediction_pairs"][:pair_limit]
        if pair_limit != -1
        else raw["groundtruth_prediction_pairs"]
    )
    for groundtruth, prediction in slice_:
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

    return groundtruths


def ingest_predictions(raw: dict, pair_limit: int) -> list[Prediction]:
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

    return predictions


def run_base_evaluation(groundtruths, predictions):
    """Run a base evaluation (with no PR curves)."""
    evaluation = evaluate_classification(groundtruths, predictions)
    return evaluation


def run_pr_curve_evaluation(groundtruths, predictions):
    """Run a base evaluation with PrecisionRecallCurve included."""
    evaluation = evaluate_classification(
        groundtruths=groundtruths,
        predictions=predictions,
        metrics_to_return=[
            enums.MetricType.Accuracy,
            enums.MetricType.Precision,
            enums.MetricType.Recall,
            enums.MetricType.F1,
            enums.MetricType.ROCAUC,
            enums.MetricType.PrecisionRecallCurve,
        ],
    )
    return evaluation


def run_detailed_pr_curve_evaluation(groundtruths, predictions):
    """Run a base evaluation with PrecisionRecallCurve and DetailedPrecisionRecallCurve included."""
    evaluation = evaluate_classification(
        groundtruths=groundtruths,
        predictions=predictions,
        metrics_to_return=[
            enums.MetricType.Accuracy,
            enums.MetricType.Precision,
            enums.MetricType.Recall,
            enums.MetricType.F1,
            enums.MetricType.ROCAUC,
            enums.MetricType.PrecisionRecallCurve,
            enums.MetricType.DetailedPrecisionRecallCurve,
        ],
    )
    return evaluation


@dataclass
class DataBenchmark:
    ingestion: float

    def result(self) -> dict[str, float | str]:
        return {
            "ingestion": round(self.ingestion, 2),
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

        # ingest groundtruths
        gt_ingest_time, groundtruths = time_it(
            ingest_groundtruths,
            raw=raw_data,
            pair_limit=limit,
        )

        # ingest predictions
        pd_ingest_time, predictions = time_it(
            ingest_predictions,
            raw=raw_data,
            pair_limit=limit,
        )

        # run evaluations
        eval_base = run_base_evaluation(groundtruths, predictions)
        eval_pr = run_pr_curve_evaluation(groundtruths, predictions)
        eval_detail = run_detailed_pr_curve_evaluation(
            groundtruths, predictions
        )

        assert eval_base.meta
        assert eval_pr.meta
        assert eval_detail.meta

        results.append(
            EvaluationBenchmark(
                limit=limit,
                gt_stats=DataBenchmark(
                    ingestion=gt_ingest_time,
                ),
                pd_stats=DataBenchmark(
                    ingestion=pd_ingest_time,
                ),
                n_datums=eval_base.meta["datums"],
                n_annotations=eval_base.meta["annotations"],
                n_labels=eval_base.meta["labels"],
                eval_base=eval_base.meta["duration"],
                eval_base_pr=eval_pr.meta["duration"],
                eval_base_pr_detail=eval_detail.meta["duration"],
            ).result()
        )

    write_results_to_file(write_path=write_path, results=results)


if __name__ == "__main__":
    run_benchmarking_analysis(limits=[5000, 5000])
