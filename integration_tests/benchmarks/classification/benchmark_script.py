import json
import os
import time
from datetime import datetime

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

connect("http://0.0.0.0:8000")
client = Client()


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


def ingest_groundtruths_and_predictions(
    dset: Dataset, model: Model, raw: dict, pair_limit: int
):
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

    dset.add_groundtruths(groundtruths, timeout=150)
    model.add_predictions(dset, predictions, timeout=150)

    dset.finalize()
    model.finalize_inferences(dataset=dset)


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
            "Accuracy",
            "Precision",
            "Recall",
            "F1",
            "ROCAUC",
            "PrecisionRecallCurve",
        ],
    )
    evaluation.wait_for_completion()
    return evaluation


def run_detailed_pr_curve_evaluation(dset: Dataset, model: Model):
    """Run a base evaluation with PrecisionRecallCurve and DetailedPrecisionRecallCurve included."""

    evaluation = model.evaluate_classification(
        dset,
        metrics_to_return=[
            "Accuracy",
            "Precision",
            "Recall",
            "F1",
            "ROCAUC",
            "PrecisionRecallCurve",
            "DetailedPrecisionRecallCurve",
        ],
    )
    evaluation.wait_for_completion()
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

        dset = Dataset.create(name=f"bird-identification-{time.time()}")
        model = Model.create(name=f"some_model-{time.time()}")

        start_time = time.time()

        ingest_groundtruths_and_predictions(
            dset=dset, model=model, raw=raw_data, pair_limit=limit
        )
        ingest_time = time.time() - start_time

        try:
            eval_ = run_base_evaluation(dset=dset, model=model)
        except TimeoutError:
            raise TimeoutError(
                f"Evaluation timed out when processing {limit} datums."
            )

        start = time.time()
        client.delete_dataset(dset.name, timeout=30)
        client.delete_model(model.name, timeout=30)
        deletion_time = time.time() - start

        results = {
            "number_of_datums": limit,
            "number_of_unique_labels": eval_.meta["labels"],
            "number_of_annotations": eval_.meta["annotations"],
            "ingest_runtime": f"{(ingest_time):.1f} seconds",
            "eval_runtime": f"{(eval_.meta['duration']):.1f} seconds",
            "del_runtime": f"{(deletion_time):.1f} seconds",
        }
        write_results_to_file(write_path=write_path, result_dict=results)


if __name__ == "__main__":
    run_benchmarking_analysis()
