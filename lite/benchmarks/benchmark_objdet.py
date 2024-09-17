import json
import os
from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from pathlib import Path
from time import time

import requests
from tqdm import tqdm
from valor_lite.detection import DataLoader


class AnnotationType(str, Enum):
    NONE = "none"
    BOX = "box"
    POLYGON = "polygon"
    MULTIPOLYGON = "multipolygon"
    RASTER = "raster"


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


@time_it
def ingest(
    manager: DataLoader,
    gt_path: Path,
    pd_path: Path,
    limit: int,
    chunk_size: int,
):
    accumulated_time = 0.0
    with open(gt_path, "r") as gf:
        with open(pd_path, "r") as pf:

            count = 0
            groundtruths = []
            predictions = []
            for gline, pline in zip(gf, pf):

                # groundtruth
                gt_dict = json.loads(gline)
                groundtruths.append(gt_dict)

                # prediction
                pd_dict = json.loads(pline)
                predictions.append(pd_dict)

                count += 1
                if count >= limit and limit > 0:
                    break
                elif len(groundtruths) < chunk_size or chunk_size == -1:
                    continue

                timer, _ = time_it(manager.add_data_from_valor_dict)(
                    zip(groundtruths, predictions), True
                )
                accumulated_time += timer
                groundtruths = []
                predictions = []

            if groundtruths:
                timer, _ = time_it(manager.add_data_from_valor_dict)(
                    zip(groundtruths, predictions), True
                )
                accumulated_time += timer

    return accumulated_time


@dataclass
class Benchmark:
    limit: int
    n_datums: int
    n_groundtruths: int
    n_predictions: int
    n_labels: int
    gt_type: AnnotationType
    pd_type: AnnotationType
    chunk_size: int
    ingestion: float
    preprocessing: float
    precomputation: float
    evaluation: float
    detailed_curves: list[tuple[int, float]]

    def result(self) -> dict:
        return {
            "limit": self.limit,
            "n_datums": self.n_datums,
            "n_groundtruths": self.n_groundtruths,
            "n_predictions": self.n_predictions,
            "n_labels": self.n_labels,
            "dtype": {
                "groundtruth": self.gt_type.value,
                "prediction": self.pd_type.value,
            },
            "chunk_size": self.chunk_size,
            "ingestion": {
                "loading_from_file": f"{round(self.ingestion - self.preprocessing, 2)} seconds",
                "numpy_conversion + IoU": f"{round(self.preprocessing, 2)} seconds",
                "ranking_pairs": f"{round(self.precomputation, 2)} seconds",
                "total": f"{round(self.ingestion + self.precomputation, 2)} seconds",
            },
            "base_evaluation": f"{round(self.evaluation, 2)} seconds",
            "detailed_pr_curve": [
                {
                    "n_points": 10,
                    "n_examples": curve[0],
                    "computation": f"{round(curve[1], 2)} seconds",
                }
                for curve in self.detailed_curves
            ],
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
    gt_multipolygon_filename = "gt_objdet_coco_raster_multipolygon.jsonl"
    gt_raster_filename = "gt_objdet_coco_raster_bitmask.jsonl"
    pd_box_filename = "pd_objdet_yolo_bbox.jsonl"
    pd_polygon_filename = "pd_objdet_yolo_polygon.jsonl"
    pd_multipolygon_filename = "pd_objdet_yolo_multipolygon.jsonl"
    pd_raster_filename = "pd_objdet_yolo_raster.jsonl"

    groundtruth_caches = {
        AnnotationType.BOX: gt_box_filename,
        AnnotationType.POLYGON: gt_polygon_filename,
        AnnotationType.MULTIPOLYGON: gt_multipolygon_filename,
        AnnotationType.RASTER: gt_raster_filename,
    }
    prediction_caches = {
        AnnotationType.BOX: pd_box_filename,
        AnnotationType.POLYGON: pd_polygon_filename,
        AnnotationType.MULTIPOLYGON: pd_multipolygon_filename,
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
            manager = DataLoader()

            # ingest + preprocess
            (ingest_time, preprocessing_time,) = ingest(
                manager=manager,
                gt_path=current_directory / Path(gt_filename),
                pd_path=current_directory / Path(pd_filename),
                limit=limit,
                chunk_size=chunk_size,
            )  # type: ignore - time_it wrapper

            finalization_time, evaluator = time_it(manager.finalize)()

            if ingest_time > ingestion_timeout and ingestion_timeout != -1:
                raise TimeoutError(
                    f"Base precomputation timed out with limit of {limit}."
                )

            # test detailed pr curve with no samples
            detailed_pr_curve_time_no_samples, _ = time_it(
                evaluator.compute_detailed_pr_curve
            )()

            # test detailed pr curve with 3 samples
            detailed_pr_curve_time_three_samples, _ = time_it(
                evaluator.compute_detailed_pr_curve
            )(n_samples=3)

            # evaluate
            eval_time, metrics = time_it(evaluator.evaluate)()
            # print(metrics)
            if eval_time > evaluation_timeout and evaluation_timeout != -1:
                raise TimeoutError(
                    f"Base evaluation timed out with {evaluator.n_datums} datums."
                )

            results.append(
                Benchmark(
                    limit=limit,
                    n_datums=evaluator.n_datums,
                    n_groundtruths=evaluator.n_groundtruths,
                    n_predictions=evaluator.n_predictions,
                    n_labels=evaluator.n_labels,
                    gt_type=gt_type,
                    pd_type=pd_type,
                    chunk_size=chunk_size,
                    ingestion=ingest_time,
                    preprocessing=preprocessing_time,
                    precomputation=finalization_time,
                    evaluation=eval_time,
                    detailed_curves=[
                        (0, detailed_pr_curve_time_no_samples),
                        (3, detailed_pr_curve_time_three_samples),
                    ],
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

    # # run polygon benchmark
    # run_benchmarking_analysis(
    #     combinations=[
    #         (AnnotationType.POLYGON, AnnotationType.POLYGON),
    #     ],
    #     limits_to_test=[5000, 5000],
    #     compute_detailed=False,
    # )

    # # run raster benchmark
    # run_benchmarking_analysis(
    #     combinations=[
    #         (AnnotationType.RASTER, AnnotationType.RASTER),
    #     ],
    #     limits_to_test=[500, 500],
    #     compute_detailed=False,
    # )
