import json
import os
from datetime import datetime
from time import time

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
) -> int:
    """Ingest the data into Valor."""

    groundtruths = []
    predictions = []
    labels = set()
    slice_ = (
        raw["groundtruth_prediction_pairs"][:pair_limit]
        if pair_limit != -1
        else raw["groundtruth_prediction_pairs"]
    )
    for groundtruth, prediction in slice_:
        labels.update(
            (
                (label["key"], label["value"])
                for annotation in groundtruth["value"]["annotations"]
                for label in annotation["labels"]
            )
        )
        labels.update(
            (
                (label["key"], label["value"])
                for annotation in prediction["value"]["annotations"]
                for label in annotation["labels"]
            )
        )

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

    for gt in groundtruths:
        dset.add_groundtruth(gt)

    for pred in predictions:
        model.add_prediction(dset, pred)

    dset.finalize()
    model.finalize_inferences(dataset=dset)

    return len(labels)


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
    limits_to_test: list[int] = [100, 500],
    results_file: str = "results.json",
    data_file: str = "data.json",
):
    """Time various function calls and export the results."""
    current_directory = os.path.dirname(os.path.realpath(__file__))
    write_path = f"{current_directory}/{results_file}"
    read_path = f"{current_directory}/{data_file}"
    for limit in limits_to_test:
        dset = Dataset.create(name="bird-identification")
        model = Model.create(name="some_model")

        with open(read_path) as f:
            raw_data = json.load(f)

        start_time = time()

        number_of_labels = ingest_groundtruths_and_predictions(
            dset=dset, model=model, raw=raw_data, pair_limit=limit
        )
        ingest_time = time() - start_time

        run_base_evaluation(dset=dset, model=model)
        ingest_and_evaluation = time() - start_time

        results = {
            "number_of_datums": limit,
            "number_of_unique_labels": number_of_labels,
            "ingest_runtime": f"{(ingest_time):.1f} seconds",
            "ingest_and_evaluation_runtime": f"{(ingest_and_evaluation):.1f} seconds",
        }
        write_results_to_file(write_path=write_path, result_dict=results)

        client.delete_dataset(dset.name, timeout=30)
        client.delete_model(model.name, timeout=30)


if __name__ == "__main__":
    run_benchmarking_analysis()
