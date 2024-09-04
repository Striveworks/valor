import io
import json
import os
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
)
from valor_core import ValorDetectionManager as Manager
from valor_core import enums
from valor_core.enums import AnnotationType


def time_it(fn):
    def wrapper(*args, **kwargs):
        start = time()
        results = fn(*args, **kwargs)
        return (time() - start, results)

    return wrapper


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

    # sort file by datum uid
    with open(file_path, "r") as f:
        lines = [x for x in f]
    with open(file_path, "w") as f:
        for line in sorted(
            lines, key=lambda x: int(json.loads(x)["datum"]["uid"])
        ):
            f.write(line)


def write_results_to_file(write_path: Path, results: list[dict]):
    """Write results to manager_results.json"""
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


def _create_annotation(
    dtype: str,
    ann: dict,
):
    ann.pop("text")
    ann.pop("context_list")

    labels = []
    for label in ann["labels"]:
        labels.append(Label(**label))
    ann["labels"] = labels

    if ann["bounding_box"] and dtype == AnnotationType.BOX:
        ann["bounding_box"] = Box(ann["bounding_box"])
        return Annotation(**ann)

    if ann["polygon"] and dtype == AnnotationType.POLYGON:
        ann["polygon"] = Polygon(ann["polygon"])
        return Annotation(**ann)

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
        return Annotation(**ann)


@time_it
def ingest_and_preprocess(
    manager: Manager,
    gt_type: AnnotationType,
    pd_type: AnnotationType,
    gt_path: Path,
    pd_path: Path,
    limit: int,
    chunk_size: int,
) -> tuple[float, Manager]:
    with open(gt_path, "r") as gf:
        with open(pd_path, "r") as pf:

            count = 0
            groundtruths = []
            predictions = []

            accumulated_runtime = 0.0

            for gline, pline in zip(gf, pf):

                # unpack groundtruth
                gt_dict = json.loads(gline)
                gt_dict["datum"].pop("text")
                gt_dict["datum"] = Datum(**gt_dict["datum"])
                annotations = [
                    _create_annotation(dtype=gt_type, ann=ann)
                    for ann in gt_dict["annotations"]
                ]
                gt_dict["annotations"] = annotations
                gt = GroundTruth(**gt_dict)
                groundtruths.append(gt)

                # unpack prediction
                pd_dict = json.loads(pline)
                pd_dict["datum"].pop("text")
                pd_dict["datum"] = Datum(**pd_dict["datum"])
                annotations = [
                    _create_annotation(dtype=pd_type, ann=ann)
                    for ann in pd_dict["annotations"]
                ]
                pd_dict["annotations"] = annotations
                pd = Prediction(**pd_dict)
                predictions.append(pd)

                count += 1
                if count >= limit and limit > 0:
                    break
                elif len(groundtruths) < chunk_size or chunk_size == -1:
                    continue

                runtime, _ = time_it(manager.add_data)(
                    groundtruths, predictions
                )
                accumulated_runtime += runtime
                groundtruths = []
                predictions = []

            if groundtruths:
                runtime, _ = time_it(manager.add_data)(
                    groundtruths, predictions
                )
                accumulated_runtime += runtime
        return (accumulated_runtime, manager)


def run_base_evaluation(manager: Manager):
    """Run a base evaluation (with no PR curves) using Manager."""
    return manager.evaluate()


def run_pr_curve_evaluation(manager: Manager):
    """Run a base evaluation with PrecisionRecallCurve included using Manager."""
    return manager.evaluate()


def run_detailed_pr_curve_evaluation(manager: Manager):
    """Run a base evaluation with PrecisionRecallCurve and DetailedPrecisionRecallCurve included using Manager."""
    return manager.evaluate()


@dataclass
class Benchmark:
    limit: int
    base_runtime: float
    pr_runtime: float
    detailed_runtime: float
    n_datums: int
    n_annotations: int
    n_labels: int
    gt_type: AnnotationType
    pd_type: AnnotationType
    chunk_size: int
    base_ingest: float
    pr_ingest: float
    detailed_ingest: float
    base_precompute: float
    pr_precompute: float
    detailed_precompute: float
    eval_base: float
    eval_base_pr: float
    eval_base_pr_detail: float

    def result(self) -> dict:
        return {
            "limit": self.limit,
            "n_datums": self.n_datums,
            "n_annotations": self.n_annotations,
            "n_labels": self.n_labels,
            "dtype": {
                "groundtruth": self.gt_type.value,
                "prediction": self.pd_type.value,
            },
            "chunk_size": self.chunk_size,
            "base": {
                "ingestion": f"{round(self.base_ingest - self.base_precompute, 2)} seconds",
                "evaluation": {
                    "preprocessing": f"{round(self.base_precompute, 2)} seconds",
                    "computation": f"{round(self.eval_base, 2)} seconds",
                    "total": f"{round(self.base_precompute + self.eval_base, 2)} seconds",
                },
            },
            "base+pr": {
                "ingestion": f"{round(self.pr_ingest - self.pr_precompute, 2)} seconds",
                "evaluation": {
                    "preprocessing": f"{round(self.pr_precompute, 2)} seconds",
                    "computation": f"{round(self.eval_base_pr, 2)} seconds",
                    "total": f"{round(self.pr_precompute + self.eval_base_pr, 2)} seconds",
                },
            }
            if self.pr_ingest > -1
            else {},
            "base+pr+detailed": {
                "ingestion": f"{round(self.detailed_ingest - self.detailed_precompute, 2)} seconds",
                "evaluation": {
                    "preprocessing": f"{round(self.detailed_precompute, 2)} seconds",
                    "computation": f"{round(self.eval_base_pr_detail, 2)} seconds",
                    "total": f"{round(self.detailed_precompute + self.eval_base_pr_detail, 2)} seconds",
                },
            }
            if self.detailed_ingest > -1
            else {},
        }


def run_benchmarking_analysis(
    limits_to_test: list[int],
    combinations: list[tuple[AnnotationType, AnnotationType]] | None = None,
    results_file: str = "manager_results.json",
    chunk_size: int = -1,
    compute_pr: bool = True,
    compute_detailed: bool = True,
    ingestion_timeout=30,
    evaluation_timeout=30,
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

            # === Base Evaluation ===
            start = time()
            base_evaluation = Manager()

            # ingest + preprocess
            base_ingest, (
                base_precompute,
                base_evaluation,
            ) = ingest_and_preprocess(
                manager=base_evaluation,
                gt_type=gt_type,
                pd_type=pd_type,
                gt_path=current_directory / Path(gt_filename),
                pd_path=current_directory / Path(pd_filename),
                limit=limit,
                chunk_size=chunk_size,
            )  # type: ignore - time_it wrapper
            if base_ingest > ingestion_timeout and ingestion_timeout != -1:
                raise TimeoutError(
                    f"Base precomputation timed out with limit of {limit}."
                )

            # evaluate
            base_results = run_base_evaluation(base_evaluation)
            base_total = time() - start
            assert base_results.meta
            n_datums = base_results.meta["datums"]
            n_annotations = base_results.meta["annotations"]
            n_labels = base_results.meta["labels"]
            base = base_results.meta["duration"]
            if base > evaluation_timeout and evaluation_timeout != -1:
                raise TimeoutError(
                    f"Base evaluation timed out with {n_datums} datums."
                )

            # === PR Evaluation ===
            pr_total = -1
            pr_ingest = -1
            pr_precompute = -1
            pr = -1
            if compute_pr:
                start = time()
                pr_evaluation = Manager(
                    metrics_to_return=[
                        enums.MetricType.AP,
                        enums.MetricType.AR,
                        enums.MetricType.mAP,
                        enums.MetricType.APAveragedOverIOUs,
                        enums.MetricType.mAR,
                        enums.MetricType.mAPAveragedOverIOUs,
                        enums.MetricType.PrecisionRecallCurve,
                    ]
                )

                # ingest + preprocess
                pr_ingest, (
                    pr_precompute,
                    pr_evaluation,
                ) = ingest_and_preprocess(
                    manager=pr_evaluation,
                    gt_type=gt_type,
                    pd_type=pd_type,
                    gt_path=current_directory / Path(gt_filename),
                    pd_path=current_directory / Path(pd_filename),
                    limit=limit,
                    chunk_size=chunk_size,
                )  # type: ignore - time_it wrapper
                if pr_ingest > ingestion_timeout and ingestion_timeout != -1:
                    raise TimeoutError(
                        f"PR precomputation timed out with {n_datums} datums."
                    )

                # evaluate
                pr_results = run_pr_curve_evaluation(pr_evaluation)
                pr_total = time() - start
                assert pr_results.meta
                pr = pr_results.meta["duration"]
                if pr > evaluation_timeout and evaluation_timeout != -1:
                    raise TimeoutError(
                        f"PR evaluation timed out with {n_datums} datums."
                    )

            # === Detailed Evaluation ===
            detailed_total = -1
            detailed_ingest = -1
            detailed_precompute = -1
            detailed = -1
            if compute_detailed:
                start = time()
                detailed_evaluation = Manager(
                    metrics_to_return=[
                        enums.MetricType.AP,
                        enums.MetricType.AR,
                        enums.MetricType.mAP,
                        enums.MetricType.APAveragedOverIOUs,
                        enums.MetricType.mAR,
                        enums.MetricType.mAPAveragedOverIOUs,
                        enums.MetricType.PrecisionRecallCurve,
                        enums.MetricType.DetailedPrecisionRecallCurve,
                    ]
                )

                # ingest + preprocess
                detailed_ingest, (
                    detailed_precompute,
                    detailed_evaluation,
                ) = ingest_and_preprocess(
                    manager=detailed_evaluation,
                    gt_type=gt_type,
                    pd_type=pd_type,
                    gt_path=current_directory / Path(gt_filename),
                    pd_path=current_directory / Path(pd_filename),
                    limit=limit,
                    chunk_size=chunk_size,
                )  # type: ignore - time_it wrapper
                if (
                    detailed_ingest > ingestion_timeout
                    and ingestion_timeout != -1
                ):
                    raise TimeoutError(
                        f"Detailed precomputation timed out with {n_datums} datums."
                    )

                # evaluate
                detailed_results = run_detailed_pr_curve_evaluation(
                    detailed_evaluation
                )
                detailed_total = time() - start

                assert detailed_results.meta
                detailed = detailed_results.meta["duration"]
                if detailed > evaluation_timeout and evaluation_timeout != -1:
                    raise TimeoutError(
                        f"Detailed evaluation timed out with {n_datums} datums."
                    )

            results.append(
                Benchmark(
                    limit=limit,
                    base_runtime=base_total,
                    pr_runtime=pr_total,
                    detailed_runtime=detailed_total,
                    n_datums=n_datums,
                    n_annotations=n_annotations,
                    n_labels=n_labels,
                    gt_type=gt_type,
                    pd_type=pd_type,
                    chunk_size=chunk_size,
                    base_ingest=base_ingest,
                    pr_ingest=pr_ingest,
                    detailed_ingest=detailed_ingest,
                    base_precompute=base_precompute,
                    pr_precompute=pr_precompute,
                    detailed_precompute=detailed_precompute,
                    eval_base=base,
                    eval_base_pr=pr,
                    eval_base_pr_detail=detailed,
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
        compute_detailed=False,
    )

    # run polygon benchmark
    run_benchmarking_analysis(
        combinations=[
            (AnnotationType.POLYGON, AnnotationType.POLYGON),
        ],
        limits_to_test=[5000, 5000],
        compute_detailed=False,
    )

    # run raster benchmark
    run_benchmarking_analysis(
        combinations=[
            (AnnotationType.RASTER, AnnotationType.RASTER),
        ],
        limits_to_test=[500, 500],
        compute_detailed=False,
    )
