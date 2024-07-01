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
from valor.schemas import MultiPolygon, Polygon, Raster

connect("http://0.0.0.0:8000")
client = Client()


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
) -> tuple[int, int]:
    """Ingest the data into Valor. Returns the number of groundtruths and predictions."""
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

    return (len(groundtruths), len(predictions))


def run_base_evaluation(dset: Dataset, model: Model):
    """Run a base evaluation (with no PR curves)."""
    evaluation = model.evaluate_detection(dset)
    evaluation.wait_for_completion()
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
    limits_to_test: list[int] = [10, 10, 10],
    results_file: str = "results.json",
    data_file: str = "data.json",
):
    """Time various function calls and export the results."""
    current_directory = os.path.dirname(os.path.realpath(__file__))
    write_path = f"{current_directory}/{results_file}"
    read_path = f"{current_directory}/{data_file}"
    for limit in limits_to_test:
        dset = Dataset.create(name="coco-dataset")
        model = Model.create(name="coco-model")

        with open(read_path) as file:
            file.seek(0)
            raw_data = json.load(file)

        # convert dict into list of tuples so we can slice it
        raw_data_tuple = [(key, value) for key, value in raw_data.items()]

        start_time = time()

        len_gt, len_pd = ingest_groundtruths_and_predictions(
            dset=dset, model=model, raw=raw_data_tuple, pair_limit=limit
        )
        ingest_time = time() - start_time

        run_base_evaluation(dset=dset, model=model)
        ingest_and_evaluation = time() - start_time

        results = {
            "number_of_datums": limit,
            "number_of_groundtruths": len_gt,
            "number_of_predictions": len_pd,
            "ingest_runtime": f"{(ingest_time):.1f} seconds",
            "ingest_and_evaluation_runtime": f"{(ingest_and_evaluation):.1f} seconds",
        }
        write_results_to_file(write_path=write_path, result_dict=results)

        client.delete_dataset(dset.name, timeout=30)
        client.delete_model(model.name, timeout=30)


if __name__ == "__main__":
    run_benchmarking_analysis()
