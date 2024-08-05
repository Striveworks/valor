import json
import os
from datetime import datetime
from time import time

import requests
from valor_core import (
    Annotation,
    Box,
    Datum,
    EvaluationParameters,
    GroundTruth,
    Label,
    MultiPolygon,
    Polygon,
    Prediction,
    Raster,
    enums,
    evaluate_detection,
)


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


def create_groundtruths_and_predictions(raw: list, pair_limit: int):
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
                                Box(_convert_wkt_to_coordinates(ann["box"]))
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
                                Box(_convert_wkt_to_coordinates(ann["box"]))
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

    return groundtruths, predictions


def run_base_evaluation(groundtruths, predictions):
    """Run a base evaluation (with no PR curves)."""
    evaluation = evaluate_detection(groundtruths, predictions)
    return evaluation


def run_pr_curve_evaluation(groundtruths, predictions):
    """Run a base evaluation with PrecisionRecallCurve included."""
    evaluation = evaluate_detection(
        groundtruths=groundtruths,
        predictions=predictions,
        parameters=EvaluationParameters(
            metrics_to_return=[
                enums.MetricType.AP,
                enums.MetricType.AR,
                enums.MetricType.mAP,
                enums.MetricType.APAveragedOverIOUs,
                enums.MetricType.mAR,
                enums.MetricType.mAPAveragedOverIOUs,
                enums.MetricType.PrecisionRecallCurve,
            ],
        ),
    )
    return evaluation


def run_detailed_pr_curve_evaluation(groundtruths, predictions):
    """Run a base evaluation with PrecisionRecallCurve and DetailedPrecisionRecallCurve included."""
    evaluation = evaluate_detection(
        groundtruths=groundtruths,
        predictions=predictions,
        parameters=EvaluationParameters(
            metrics_to_return=[
                enums.MetricType.AP,
                enums.MetricType.AR,
                enums.MetricType.mAP,
                enums.MetricType.APAveragedOverIOUs,
                enums.MetricType.mAR,
                enums.MetricType.mAPAveragedOverIOUs,
                enums.MetricType.PrecisionRecallCurve,
            ],
        ),
    )
    return evaluation


def run_benchmarking_analysis(
    limits_to_test: list[int] = [100, 100],
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

        # convert dict into list of tuples so we can slice it
        raw_data_tuple = [(key, value) for key, value in raw_data.items()]

        start_time = time()

        groundtruths, predictions = create_groundtruths_and_predictions(
            raw=raw_data_tuple, pair_limit=limit
        )
        creation_time = time() - start_time

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

        # TODO check these values
        print(base_eval.metrics)

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
