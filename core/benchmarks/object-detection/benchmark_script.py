import io
import json
import os
import re
from base64 import b64decode
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from time import time

import numpy as np
import PIL.Image
import requests
from tqdm import tqdm
from valor_core import (
    Annotation,
    Box,
    Datum,
    GroundTruth,
    Label,
    Polygon,
    Prediction,
    Raster,
    enums,
    evaluate_detection,
)
from valor_core.schemas import RasterData


def time_it(fn, *args, **kwargs):
    start = time()
    results = fn(*args, **kwargs)
    return (time() - start, results)


def download_data_if_not_exists(
    file_name: str,
    file_path: Path,
    url: str,
):
    """Download the data from a public bucket if it doesn't exist locally."""

    if not os.path.exists(file_path):
        response = requests.get(url, stream=True)
        if response.status_code == 200:
            total_size = int(response.headers.get("content-length", 0))
            with open(file_path, "wb") as f:
                with tqdm(
                    total=total_size,
                    unit="B",
                    unit_scale=True,
                    unit_divisor=1024,
                    desc=file_name,
                ) as pbar:
                    for chunk in response.iter_content(chunk_size=1024):
                        if chunk:
                            f.write(chunk)
                            pbar.update(1024)
        else:
            raise RuntimeError(response)
    else:
        print(f"{file_name} already exists locally.")


def write_results_to_file(write_path: Path, result_dict: dict):
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


def ingest_groundtruths(
    path: Path,
    limit: int,
) -> list[GroundTruth]:
    groundtruths = []
    with open(path, "r") as f:
        for line in f:
            gt_dict = json.loads(line)
            gt_dict["datum"].pop("text")
            gt_dict["datum"] = Datum(**gt_dict["datum"])

            annotations = []
            for ann in gt_dict["annotations"]:
                ann.pop("text")
                ann.pop("context_list")

                labels = []
                for label in ann["labels"]:
                    labels.append(Label(**label))
                ann["labels"] = labels

                ann["bounding_box"] = (
                    Box(ann["bounding_box"]) if ann["bounding_box"] else None
                )
                ann["polygon"] = (
                    Polygon(ann["polygon"]) if ann["polygon"] else None
                )

                if ann["raster"]:
                    if ann["raster"]["geometry"] is not None:
                        ann["raster"] = Raster(
                            RasterData(
                                mask=None, geometry=ann["raster"]["geometry"]
                            )
                        )
                    elif ann["raster"]["geometry"] is None:
                        # decode raster
                        mask_bytes = b64decode(ann["raster"]["mask"])
                        with io.BytesIO(mask_bytes) as f:
                            img = PIL.Image.open(f)
                            ann["raster"] = Raster(
                                RasterData(mask=np.array(img), geometry=None)
                            )
                else:
                    ann["raster"] = None

                annotations.append(Annotation(**ann))
            gt_dict["annotations"] = annotations

            gt = GroundTruth(**gt_dict)
            groundtruths.append(gt)
            if len(groundtruths) >= limit:
                return groundtruths
    return groundtruths


def ingest_predictions(
    datum_uids: list[str],
    path: Path,
    limit: int,
) -> list[Prediction]:

    pattern = re.compile(r'"uid":\s*"(\d+)"')

    predictions = []
    with open(path, "r") as f:
        count = 0
        for line in f:
            match = pattern.search(line)
            if not match:
                continue
            elif match.group(1) not in datum_uids:
                continue
            pd_dict = json.loads(line)

            pd_dict["datum"].pop("text")
            pd_dict["datum"] = Datum(**pd_dict["datum"])

            annotations = []
            for ann in pd_dict["annotations"]:
                ann.pop("text")
                ann.pop("context_list")

                labels = []
                for label in ann["labels"]:
                    labels.append(Label(**label))
                ann["labels"] = labels

                ann["bounding_box"] = (
                    Box(ann["bounding_box"]) if ann["bounding_box"] else None
                )
                ann["polygon"] = (
                    Polygon(ann["polygon"]) if ann["polygon"] else None
                )

                if ann["raster"]:
                    if ann["raster"]["geometry"] is not None:
                        ann["raster"] = Raster(
                            RasterData(
                                mask=None, geometry=ann["raster"]["geometry"]
                            )
                        )
                    elif ann["raster"]["geometry"] is None:
                        # decode raster
                        mask_bytes = b64decode(ann["raster"]["mask"])
                        with io.BytesIO(mask_bytes) as f:
                            img = PIL.Image.open(f)
                            ann["raster"] = Raster(
                                RasterData(mask=np.array(img), geometry=None)
                            )
                else:
                    ann["raster"] = None
            pd_dict["annotations"] = annotations

            pd = Prediction(**pd_dict)
            predictions.append(pd)
            count += 1
            if count >= limit:
                return predictions
    return predictions


def run_base_evaluation(groundtruths, predictions):
    """Run a base evaluation (with no PR curves)."""
    evaluation = evaluate_detection(groundtruths, predictions)
    return evaluation


def run_pr_curve_evaluation(groundtruths, predictions):
    """Run a base evaluation with PrecisionRecallCurve included."""
    evaluation = evaluate_detection(
        groundtruths=groundtruths,
        predictions=predictions,
        metrics_to_return=[
            enums.MetricType.AP,
            enums.MetricType.AR,
            enums.MetricType.mAP,
            enums.MetricType.APAveragedOverIOUs,
            enums.MetricType.mAR,
            enums.MetricType.mAPAveragedOverIOUs,
            enums.MetricType.PrecisionRecallCurve,
        ],
    )
    return evaluation


def run_detailed_pr_curve_evaluation(groundtruths, predictions):
    """Run a base evaluation with PrecisionRecallCurve and DetailedPrecisionRecallCurve included."""
    evaluation = evaluate_detection(
        groundtruths=groundtruths,
        predictions=predictions,
        metrics_to_return=[
            enums.MetricType.AP,
            enums.MetricType.AR,
            enums.MetricType.mAP,
            enums.MetricType.APAveragedOverIOUs,
            enums.MetricType.mAR,
            enums.MetricType.mAPAveragedOverIOUs,
            enums.MetricType.PrecisionRecallCurve,
        ],
    )
    return evaluation


def run_benchmarking_analysis(
    limits_to_test: list[int],
    results_file: str = "results.json",
    data_file: str = "data.json",
    compute_box: bool = True,
    compute_raster: bool = True,
):
    """Time various function calls and export the results."""
    current_directory = Path(os.path.dirname(os.path.realpath(__file__)))
    write_path = current_directory / Path(results_file)

    gt_box_filename = "gt_objdet_coco_bbox.jsonl"
    gt_raster_filename = "gt_objdet_coco_raster.jsonl"
    pd_box_filename = "pd_objdet_yolo_bbox.jsonl"
    pd_raster_filename = "pd_objdet_yolo_raster.jsonl"

    # cache data locally
    for filename in [
        gt_box_filename,
        gt_raster_filename,
        pd_box_filename,
        pd_raster_filename,
    ]:
        file_path = current_directory / Path(filename)
        url = f"https://pub-fae71003f78140bdaedf32a7c8d331d2.r2.dev/{filename}"
        download_data_if_not_exists(
            file_name=filename, file_path=file_path, url=url
        )

    @dataclass
    class Dummy:
        meta: dict

    dummy_value = Dummy(meta={"labels": -1, "annotations": -1, "duration": -1})

    for limit in limits_to_test:

        gt_boxes = []
        gt_rasters = []
        pd_boxes = []
        pd_rasters = []
        gt_box_ingest_time = -1
        pd_box_ingest_time = -1
        base_eval_box = dummy_value
        pr_eval_box = dummy_value
        detailed_pr_eval_box = dummy_value
        gt_raster_ingest_time = -1
        pd_raster_ingest_time = -1
        base_eval_raster = dummy_value
        pr_eval_raster = dummy_value
        detailed_pr_eval_raster = dummy_value

        if compute_box:

            # ingestion
            gt_box_ingest_time, gt_boxes = time_it(
                ingest_groundtruths,
                path=current_directory / Path(gt_box_filename),
                limit=limit,
            )
            box_datum_uids = [gt.datum.uid for gt in gt_boxes]
            pd_box_ingest_time, pd_boxes = time_it(
                ingest_predictions,
                datum_uids=box_datum_uids,
                path=current_directory / Path(pd_box_filename),
                limit=limit,
            )

            # evaluation
            base_eval_box = run_base_evaluation(
                groundtruths=gt_boxes, predictions=pd_boxes
            )
            pr_eval_box = run_pr_curve_evaluation(
                groundtruths=gt_boxes, predictions=pd_boxes
            )
            detailed_pr_eval_box = run_detailed_pr_curve_evaluation(
                groundtruths=gt_boxes, predictions=pd_boxes
            )

            # handle type errors
            if not (
                base_eval_box.meta
                and pr_eval_box.meta
                and detailed_pr_eval_box.meta
            ):
                raise ValueError("Metadata isn't defined for all objects.")

        if compute_raster:

            # ingestion
            gt_raster_ingest_time, gt_rasters = time_it(
                ingest_groundtruths,
                path=current_directory / Path(gt_raster_filename),
                limit=limit,
            )
            raster_datum_uids = [gt.datum.uid for gt in gt_rasters]
            pd_raster_ingest_time, pd_rasters = time_it(
                ingest_predictions,
                datum_uids=raster_datum_uids,
                path=current_directory / Path(pd_raster_filename),
                limit=limit,
            )

            # evaluation
            base_eval_raster = run_base_evaluation(
                groundtruths=gt_rasters, predictions=pd_rasters
            )
            pr_eval_raster = run_pr_curve_evaluation(
                groundtruths=gt_rasters, predictions=pd_rasters
            )
            detailed_pr_eval_raster = run_detailed_pr_curve_evaluation(
                groundtruths=gt_rasters, predictions=pd_rasters
            )

            if not (
                base_eval_raster.meta
                and pr_eval_raster.meta
                and detailed_pr_eval_raster.meta
            ):
                raise ValueError("Metadata isn't defined for all objects.")

        results = {
            "box": {
                "info": {
                    "number_of_datums": len(gt_boxes),
                    "number_of_unique_labels": base_eval_box.meta["labels"],
                    "number_of_annotations": base_eval_box.meta["annotations"],
                },
                "ingestion": {
                    "groundtruth": f"{(gt_box_ingest_time):.1f} seconds",
                    "prediction": f"{(pd_box_ingest_time):.1f} seconds",
                },
                "evaluation": {
                    "base": f"{(base_eval_box.meta['duration']):.1f} seconds",
                    "base+pr": f"{(pr_eval_box.meta['duration']):.1f} seconds",
                    "base+pr+detailed": f"{(detailed_pr_eval_box.meta['duration']):.1f} seconds",
                },
            },
            "raster": {
                "info": {
                    "number_of_datums": len(gt_rasters),
                    "number_of_unique_labels": base_eval_raster.meta["labels"],
                    "number_of_annotations": base_eval_raster.meta[
                        "annotations"
                    ],
                },
                "ingestion": {
                    "groundtruth": f"{(gt_raster_ingest_time):.1f} seconds",
                    "prediction": f"{(pd_raster_ingest_time):.1f} seconds",
                },
                "evaluation": {
                    "base": f"{(base_eval_raster.meta['duration']):.1f} seconds",
                    "base+pr": f"{(pr_eval_raster.meta['duration']):.1f} seconds",
                    "base+pr+detailed": f"{(detailed_pr_eval_raster.meta['duration']):.1f} seconds",
                },
            },
        }
        write_results_to_file(write_path=write_path, result_dict=results)

        if base_eval_box.meta["duration"] > 30:
            raise TimeoutError("Base evaluation took longer than 30 seconds.")


if __name__ == "__main__":
    run_benchmarking_analysis(
        limits_to_test=[5000, 5000, 5000],
        compute_box=True,
        compute_raster=False,
    )
