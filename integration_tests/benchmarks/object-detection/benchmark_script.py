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
from valor.schemas import MultiPolygon, Polygon, Raster

connect("http://0.0.0.0:8000")
client = Client()


def download_data_if_not_exists(file_path: str, file_url: str):
    """Download the data from a public bucket if it doesn't exist in the repo."""
    if os.path.exists(file_path):
        return

    response = json.loads(requests.get(file_url).text)
    with open(file_path, "w+") as file:
        json.dump(response, file, indent=4)


def _convert_wkt_to_coordinates(wkt: str) -> list[list[tuple]]:
    """Convert a WKT string into a nested list of coordinates."""
    return [
        [tuple(float(y) for y in x) for x in json.loads(wkt)["coordinates"][0]]
    ]


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
    dset: Dataset, model: Model, raw: list, pair_limit: int
):
    """Ingest the data into Valor."""
    groundtruths = []
    predictions = []

    for datum_id, data in raw[:pair_limit]:
        datum = Datum(
            uid=str(datum_id),
            metadata=data["datum_metadata"],
        )
        groundtruths.append(
            GroundTruth(
                datum=datum,
                annotations=list(
                    [
                        Annotation(
                            is_instance=ann["is_instance"],
                            labels=list(
                                [
                                    Label(
                                        key=label["key"],
                                        value=label["value"],
                                    )
                                    for label in ann["labels"]
                                ]
                            ),
                            bounding_box=(
                                _convert_wkt_to_coordinates(ann["box"])
                                if ann["box"]
                                else None
                            ),
                            raster=(
                                Raster.from_geometry(
                                    geometry=MultiPolygon(
                                        [
                                            _convert_wkt_to_coordinates(
                                                ann["raster"]
                                            )
                                        ]
                                    ),
                                    height=data["datum_metadata"]["height"],
                                    width=data["datum_metadata"]["width"],
                                )
                                if ann["raster"]
                                else None
                            ),
                            polygon=(
                                (
                                    Polygon(
                                        _convert_wkt_to_coordinates(
                                            ann["polygon"]
                                        )
                                    )
                                )
                                if ann["polygon"]
                                else None
                            ),
                        )
                        for ann in data["groundtruth_annotations"]
                    ]
                ),
            )
        )

        predictions.append(
            Prediction(
                datum=datum,
                annotations=list(
                    [
                        Annotation(
                            is_instance=ann["is_instance"],
                            labels=list(
                                [
                                    Label(
                                        key=label["key"],
                                        value=label["value"],
                                        score=label["score"],
                                    )
                                    for label in ann["labels"]
                                ]
                            ),
                            bounding_box=(
                                _convert_wkt_to_coordinates(ann["box"])
                                if ann["box"]
                                else None
                            ),
                            raster=(
                                Raster.from_geometry(
                                    geometry=MultiPolygon(
                                        [
                                            _convert_wkt_to_coordinates(
                                                ann["raster"]
                                            )
                                        ]
                                    ),
                                    height=data["datum_metadata"]["height"],
                                    width=data["datum_metadata"]["width"],
                                )
                                if ann["raster"]
                                else None
                            ),
                            polygon=(
                                (
                                    Polygon(
                                        _convert_wkt_to_coordinates(
                                            ann["polygon"]
                                        )
                                    )
                                )
                                if ann["polygon"]
                                else None
                            ),
                        )
                        for ann in data["prediction_annotations"]
                    ]
                ),
            )
        )

    for gt in groundtruths:
        dset.add_groundtruth(gt)

    for pred in predictions:
        model.add_prediction(dset, pred)

    dset.finalize()
    model.finalize_inferences(dataset=dset)


def run_base_evaluation(dset: Dataset, model: Model):
    """Run a base evaluation (with no PR curves)."""
    evaluation = model.evaluate_detection(dset)
    evaluation.wait_for_completion(timeout=60)
    return evaluation


def run_pr_curve_evaluation(dset: Dataset, model: Model):
    """Run a base evaluation with PrecisionRecallCurve included."""
    evaluation = model.evaluate_detection(
        dset,
        metrics_to_return=[
            "AP",
            "AR",
            "mAP",
            "APAveragedOverIOUs",
            "mAR",
            "mAPAveragedOverIOUs",
            "PrecisionRecallCurve",
        ],
    )
    evaluation.wait_for_completion()
    return evaluation


def run_detailed_pr_curve_evaluation(dset: Dataset, model: Model):
    """Run a base evaluation with PrecisionRecallCurve and DetailedPrecisionRecallCurve included."""

    evaluation = model.evaluate_detection(
        dset,
        metrics_to_return=[
            "AP",
            "AR",
            "mAP",
            "APAveragedOverIOUs",
            "mAR",
            "mAPAveragedOverIOUs",
            "PrecisionRecallCurve",
            "DetailedPrecisionRecallCurve",
        ],
    )
    evaluation.wait_for_completion()
    return evaluation


def run_benchmarking_analysis(
    limits_to_test: list[int] = [6, 6],
    results_file: str = "results.json",
    data_file: str = "data.json",
):
    """Time various function calls and export the results."""
    current_directory = os.path.dirname(os.path.realpath(__file__))
    write_path = f"{current_directory}/{results_file}"
    data_path = f"{current_directory}/{data_file}"

    download_data_if_not_exists(
        file_path=data_path,
        file_url="https://pub-fae71003f78140bdaedf32a7c8d331d2.r2.dev/detection_data.json",
    )

    with open(data_path) as file:
        file.seek(0)
        raw_data = json.load(file)

    for limit in limits_to_test:
        dset = Dataset.create(name="coco-dataset")
        model = Model.create(name="coco-model")

        # convert dict into list of tuples so we can slice it
        raw_data_tuple = [(key, value) for key, value in raw_data.items()]

        start_time = time.time()

        ingest_groundtruths_and_predictions(
            dset=dset, model=model, raw=raw_data_tuple, pair_limit=limit
        )
        ingest_time = time.time() - start_time

        time.sleep(30)

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
