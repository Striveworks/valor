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
    ValorDetectionManager,
    enums,
    evaluate_detection,
)
from valor_core.enums import AnnotationType


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
    dtype: AnnotationType,
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

                if ann["bounding_box"] and dtype == AnnotationType.BOX:
                    ann["bounding_box"] = Box(ann["bounding_box"])
                    annotations.append(Annotation(**ann))

                if ann["polygon"] and dtype == AnnotationType.POLYGON:
                    ann["polygon"] = Polygon(ann["polygon"])
                    annotations.append(Annotation(**ann))

                if ann["raster"] and dtype == AnnotationType.RASTER:
                    mask_bytes = b64decode(ann["raster"]["mask"])
                    with io.BytesIO(mask_bytes) as f:
                        img = PIL.Image.open(f)
                        w, h = img.size
                        if ann["raster"]["geometry"] is not None:
                            ann["raster"] = Raster.from_geometry(
                                ann["raster"]["geometry"],
                                width=w,
                                height=h,
                            )
                        elif ann["raster"]["geometry"] is None:
                            # decode raster
                            ann["raster"] = Raster(mask=np.array(img))
                    annotations.append(Annotation(**ann))

            gt_dict["annotations"] = annotations
            gt = GroundTruth(**gt_dict)
            groundtruths.append(gt)
            if len(groundtruths) >= limit:
                return groundtruths
    return groundtruths


def ingest_predictions(
    dtype: AnnotationType,
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

                if ann["bounding_box"] and dtype == AnnotationType.BOX:
                    ann["bounding_box"] = Box(ann["bounding_box"])
                    annotations.append(Annotation(**ann))

                if ann["polygon"] and dtype == AnnotationType.POLYGON:
                    ann["polygon"] = Polygon(ann["polygon"])
                    annotations.append(Annotation(**ann))

                if ann["raster"] and dtype == AnnotationType.RASTER:
                    mask_bytes = b64decode(ann["raster"]["mask"])
                    with io.BytesIO(mask_bytes) as f:
                        img = PIL.Image.open(f)
                        w, h = img.size
                        if ann["raster"]["geometry"] is not None:
                            ann["raster"] = Raster.from_geometry(
                                ann["raster"]["geometry"],
                                width=w,
                                height=h,
                            )
                        elif ann["raster"]["geometry"] is None:
                            # decode raster
                            ann["raster"] = Raster(mask=np.array(img))
                    annotations.append(Annotation(**ann))

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
            enums.MetricType.DetailedPrecisionRecallCurve,
        ],
    )
    return evaluation


def run_base_evaluation_with_manager(groundtruths, predictions):
    """Run a base evaluation (with no PR curves) using ValorDetectionManager."""
    manager = ValorDetectionManager()
    manager.add_data(groundtruths=groundtruths, predictions=predictions)
    return manager.evaluate()


def run_pr_curve_evaluation_with_manager(groundtruths, predictions):
    """Run a base evaluation with PrecisionRecallCurve included using ValorDetectionManager."""
    manager = ValorDetectionManager(
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

    manager.add_data(groundtruths=groundtruths, predictions=predictions)

    return manager.evaluate()


def run_detailed_pr_curve_evaluation_with_manager(groundtruths, predictions):
    """Run a base evaluation with PrecisionRecallCurve and DetailedPrecisionRecallCurve included using ValorDetectionManager."""

    manager = ValorDetectionManager(
        metrics_to_return=[
            enums.MetricType.AP,
            enums.MetricType.AR,
            enums.MetricType.mAP,
            enums.MetricType.APAveragedOverIOUs,
            enums.MetricType.mAR,
            enums.MetricType.mAPAveragedOverIOUs,
            enums.MetricType.PrecisionRecallCurve,
            enums.MetricType.DetailedPrecisionRecallCurve,
        ],
    )

    manager.add_data(groundtruths=groundtruths, predictions=predictions)

    return manager.evaluate()


@dataclass
class DataBenchmark:
    dtype: str
    ingestion: float

    def result(self) -> dict[str, float | str]:
        return {
            "dtype": self.dtype,
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
    limits_to_test: list[int],
    combinations: list[tuple[AnnotationType, AnnotationType]] | None = None,
    results_file: str = "results.json",
    ingestion_chunk_timeout: int = 30,
    evaluation_timeout: int = 30,
    compute_pr: bool = True,
    compute_detailed: bool = True,
):
    """Time various function calls and export the results."""
    current_directory = Path(__file__).parent
    write_path = current_directory / Path(results_file)

    gt_box_filename = "gt_objdet_coco_bbox.jsonl"
    gt_polygon_filename = "gt_objdet_coco_polygon.jsonl"
    # gt_multipolygon_filename = "gt_objdet_coco_raster_multipolygon.jsonl"
    gt_raster_filename = "gt_objdet_coco_raster_bitmask.jsonl"
    pd_box_filename = "pd_objdet_yolo_bbox.jsonl"
    pd_polygon_filename = "pd_objdet_yolo_polygon.jsonl"
    # pd_multipolygon_filename = "pd_objdet_yolo_multipolygon.jsonl"
    pd_raster_filename = "pd_objdet_yolo_raster.jsonl"

    groundtruth_caches = {
        AnnotationType.BOX: gt_box_filename,
        AnnotationType.POLYGON: gt_polygon_filename,
        # AnnotationType.MULTIPOLYGON: gt_multipolygon_filename,
        AnnotationType.RASTER: gt_raster_filename,
    }
    prediction_caches = {
        AnnotationType.BOX: pd_box_filename,
        AnnotationType.POLYGON: pd_polygon_filename,
        # AnnotationType.MULTIPOLYGON: pd_multipolygon_filename,
        AnnotationType.RASTER: pd_raster_filename,
    }

    # default is to perform all combinations
    if combinations is None:
        combinations = [
            (gt_type, pd_type)
            for gt_type in groundtruth_caches
            for pd_type in prediction_caches
        ]

    # cache data locally
    filenames = [
        *list(groundtruth_caches.values()),
        *list(prediction_caches.values()),
    ]
    for filename in filenames:
        file_path = current_directory / Path(filename)
        url = f"https://pub-fae71003f78140bdaedf32a7c8d331d2.r2.dev/{filename}"
        download_data_if_not_exists(
            file_name=filename, file_path=file_path, url=url
        )

    # iterate through datum limits
    results = list()
    for limit in limits_to_test:
        for gt_type, pd_type in combinations:

            gt_filename = groundtruth_caches[gt_type]
            pd_filename = prediction_caches[pd_type]

            # gt ingestion
            gt_ingest_time, groundtruths = time_it(
                ingest_groundtruths,
                dtype=gt_type,
                path=current_directory / Path(gt_filename),
                limit=limit,
            )

            # pd ingestion
            datum_uids = [gt.datum.uid for gt in groundtruths]
            pd_ingest_time, predictions = time_it(
                ingest_predictions,
                dtype=pd_type,
                datum_uids=datum_uids,
                path=current_directory / Path(pd_filename),
                limit=limit,
            )

            # run evaluations
            eval_pr = None
            eval_detail = None
            eval_base = run_base_evaluation_with_manager(
                groundtruths, predictions
            )
            if compute_pr:
                eval_pr = run_pr_curve_evaluation_with_manager(
                    groundtruths, predictions
                )
            if compute_detailed:
                eval_detail = run_detailed_pr_curve_evaluation(
                    groundtruths, predictions
                )

            assert eval_base.meta

            results.append(
                EvaluationBenchmark(
                    limit=limit,
                    gt_stats=DataBenchmark(
                        dtype=gt_type,
                        ingestion=gt_ingest_time,
                    ),
                    pd_stats=DataBenchmark(
                        dtype=pd_type,
                        ingestion=pd_ingest_time,
                    ),
                    n_datums=eval_base.meta["datums"],
                    n_annotations=eval_base.meta["annotations"],
                    n_labels=eval_base.meta["labels"],
                    eval_base=eval_base.meta["duration"],
                    eval_base_pr=(
                        eval_pr.meta["duration"]
                        if eval_pr and eval_pr.meta
                        else -1
                    ),
                    eval_base_pr_detail=(
                        eval_detail.meta["duration"]
                        if eval_detail and eval_detail.meta
                        else -1
                    ),
                ).result()
            )

    write_results_to_file(write_path=write_path, results=results)


if __name__ == "__main__":

    # run bounding box benchmark
    run_benchmarking_analysis(
        combinations=[
            (AnnotationType.BOX, AnnotationType.BOX),
        ],
        limits_to_test=[5000, 5000],
    )

    # run polygon benchmark
    run_benchmarking_analysis(
        combinations=[
            (AnnotationType.POLYGON, AnnotationType.POLYGON),
        ],
        limits_to_test=[5000, 5000],
    )
    # run raster benchmark
    run_benchmarking_analysis(
        combinations=[
            (AnnotationType.RASTER, AnnotationType.RASTER),
        ],
        limits_to_test=[500],
        compute_detailed=False,
    )
