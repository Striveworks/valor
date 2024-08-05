import json
import os
import time
from datetime import datetime

import requests
from valor_core import (
    Annotation,
    Datum,
    EvaluationParameters,
    GroundTruth,
    Label,
    Prediction,
    enums,
    evaluate_classification,
)


def download_data_if_not_exists(file_path: str, file_url: str):
    """Download the data from a public bucket if it doesn't exist in the repo."""
    if os.path.exists(file_path):
        return

    response = json.loads(requests.get(file_url).text)
    with open(file_path, "w+") as file:
        json.dump(response, file, indent=4)


def write_results_to_file(write_path: str, result_dict: dict):
    """Write results to results.json"""
    current_datetime = datetime.now().strftime("%d/%m/%Y %H:%M:%S")

    if os.path.isfile(write_path):
        with open(write_path, "r") as file:
            file.seek(0)
            data = json.load(file)
    else:
        data = {}

    data[current_datetime] = result_dict

    with open(write_path, "w+") as file:
        json.dump(data, file, indent=4)


def create_groundtruths_and_predictions(raw: dict, pair_limit: int):
    """Ingest the data into Valor."""

    groundtruths = []
    predictions = []
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

    return groundtruths, predictions


def run_base_evaluation(groundtruths, predictions):
    """Run a base evaluation (with no PR curves)."""
    evaluation = evaluate_classification(groundtruths, predictions)
    return evaluation


def run_pr_curve_evaluation(groundtruths, predictions):
    """Run a base evaluation with PrecisionRecallCurve included."""
    evaluation = evaluate_classification(
        groundtruths=groundtruths,
        predictions=predictions,
        parameters=EvaluationParameters(
            metrics_to_return=[
                enums.MetricType.Accuracy,
                enums.MetricType.Precision,
                enums.MetricType.Recall,
                enums.MetricType.F1,
                enums.MetricType.ROCAUC,
                enums.MetricType.PrecisionRecallCurve,
            ],
        ),
    )
    return evaluation


def run_detailed_pr_curve_evaluation(groundtruths, predictions):
    """Run a base evaluation with PrecisionRecallCurve and DetailedPrecisionRecallCurve included."""
    evaluation = evaluate_classification(
        groundtruths=groundtruths,
        predictions=predictions,
        parameters=EvaluationParameters(
            metrics_to_return=[
                enums.MetricType.Accuracy,
                enums.MetricType.Precision,
                enums.MetricType.Recall,
                enums.MetricType.F1,
                enums.MetricType.ROCAUC,
                enums.MetricType.PrecisionRecallCurve,
                enums.MetricType.DetailedPrecisionRecallCurve,
            ],
        ),
    )
    return evaluation


def run_benchmarking_analysis(
    limits_to_test: list[int] = [5000, 5000],
    results_file: str = "results.json",
    data_file: str = "data.json",
):
    """Time various function calls and export the results."""
    current_directory = os.path.dirname(os.path.realpath(__file__))
    write_path = f"{current_directory}/{results_file}"
    data_path = f"{current_directory}/{data_file}"

    download_data_if_not_exists(
        file_path=data_path,
        file_url="https://pub-fae71003f78140bdaedf32a7c8d331d2.r2.dev/classification_data.json",
    )

    with open(data_path) as file:
        file.seek(0)
        raw_data = json.load(file)

    for limit in limits_to_test:

        start_time = time.time()

        groundtruths, predictions = create_groundtruths_and_predictions(
            raw=raw_data, pair_limit=limit
        )
        creation_time = time.time() - start_time

        # run evaluations
        base_eval = run_base_evaluation(
            groundtruths=groundtruths, predictions=predictions
        )
        pr_eval = run_pr_curve_evaluation(
            groundtruths=groundtruths, predictions=predictions
        )
        detailed_pr_eval = run_detailed_pr_curve_evaluation(
            groundtruths=groundtruths, predictions=predictions
        )

        # handle type errors
        assert base_eval.meta
        assert pr_eval.meta
        assert detailed_pr_eval.meta

        results = {
            "number_of_datums": limit,
            "number_of_unique_labels": base_eval.meta["labels"],
            "number_of_annotations": base_eval.meta["annotations"],
            "creation_runtime": f"{(creation_time):.1f} seconds",
            "eval_runtime": f"{(base_eval.meta['duration']):.1f} seconds",
            "pr_eval_runtime": f"{(pr_eval.meta['duration']):.1f} seconds",
            "detailed_pr_eval_runtime": f"{(detailed_pr_eval.meta['duration']):.1f} seconds",
        }
        write_results_to_file(write_path=write_path, result_dict=results)


if __name__ == "__main__":
    run_benchmarking_analysis()
