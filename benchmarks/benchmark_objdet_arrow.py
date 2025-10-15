import json
import os
from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from pathlib import Path
from time import time

import requests
from tqdm import tqdm

from valor_lite.object_detection import BoundingBox, Detection
from valor_lite.object_detection.loader import Loader


def _get_bbox_extrema(
    data: list,
) -> tuple[float, float, float, float]:
    """Get the bounding box coordinates from a valor Annotation object."""
    x = [point[0] for shape in data for point in shape]
    y = [point[1] for shape in data for point in shape]
    return (min(x), max(x), min(y), max(y))


def _convert_valor_dicts_into_Detection(gt_dict: dict, pd_dict: dict):
    """Convert a groundtruth dictionary and prediction dictionary into a valor_lite Detection object."""
    gts = []
    pds = []

    for gidx, gann in enumerate(gt_dict["annotations"]):
        labels = []
        for valor_label in gann["labels"]:
            # NOTE: we only include labels where the key is "name"
            if valor_label["key"] != "name":
                continue

            labels.append(valor_label["value"])

        # if the annotation doesn't contain any labels that aren't key == 'name', then we skip that annotation
        if not labels:
            continue

        x_min, x_max, y_min, y_max = _get_bbox_extrema(gann["bounding_box"])

        gts.append(
            BoundingBox(
                uid=f"gt_{gidx}",
                xmin=x_min,
                xmax=x_max,
                ymin=y_min,
                ymax=y_max,
                labels=labels,
            )
        )

    for pidx, pann in enumerate(pd_dict["annotations"]):
        labels, scores = [], []
        for valor_label in pann["labels"]:
            if valor_label["key"] != "name":
                continue
            labels.append(valor_label["value"])
            scores.append(valor_label["score"])
        if not labels:
            continue

        x_min, x_max, y_min, y_max = _get_bbox_extrema(pann["bounding_box"])

        pds.append(
            BoundingBox(
                uid=f"pd_{pidx}",
                xmin=x_min,
                xmax=x_max,
                ymin=y_min,
                ymax=y_max,
                labels=labels,
                scores=scores,
            )
        )

    return Detection(
        uid=gt_dict["datum"]["uid"],
        groundtruths=gts,
        predictions=pds,
    )


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
    """Write results to json"""
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
    manager: Loader,
    gt_path: Path,
    pd_path: Path,
    limit: int,
    chunk_size: int,
):
    accumulated_time = 0.0
    with open(gt_path, "r") as gf:
        with open(pd_path, "r") as pf:

            count = 0
            detections = []
            for gline, pline in zip(gf, pf):

                gt_dict = json.loads(gline)
                pd_dict = json.loads(pline)
                detections.append(
                    _convert_valor_dicts_into_Detection(
                        gt_dict=gt_dict, pd_dict=pd_dict
                    )
                )

                count += 1
                if count >= limit and limit > 0:
                    break
                elif len(detections) < chunk_size or chunk_size == -1:
                    continue

                timer, _ = time_it(manager.add_bounding_boxes)(detections)
                accumulated_time += timer
                detections = []

            if detections:
                timer, _ = time_it(manager.add_bounding_boxes)(detections)
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
    finalization: float
    precision_recall_computation: float
    confusion_matrix: float

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
                "add_data": f"{round(self.preprocessing, 2)} seconds",
                "finalization": f"{round(self.finalization, 2)} seconds",
                "total": f"{round(self.ingestion + self.finalization, 2)} seconds",
            },
            "precision_recall_computation": f"{round(self.precision_recall_computation, 2)} seconds",
            "confusion_matrix_computation": f"{round(self.confusion_matrix, 2)} seconds",
        }


def run_benchmarking_analysis(
    limits_to_test: list[int],
    combinations: list[tuple[AnnotationType, AnnotationType]] | None = None,
    results_file: str = "objdet_results.json",
    chunk_size: int = -1,
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
            manager = Loader(
                current_directory / ".valor",
                batch_size=1_000,
                rows_per_file=10_000,
            )

            # ingest + preprocess
            (ingest_time, preprocessing_time,) = ingest(
                manager=manager,
                gt_path=current_directory / Path(gt_filename),
                pd_path=current_directory / Path(pd_filename),
                limit=limit,
                chunk_size=chunk_size,
            )  # type: ignore - time_it wrapper

            print(ingest_time, preprocessing_time)

            finalization_time, evaluator = time_it(manager.finalize)()
            print("final", finalization_time)
            print("n_files", manager._detailed.num_files)

            # evaluate - base metrics only
            eval_time, metrics = time_it(evaluator.evaluate)(
                score_thresholds=[0.5, 0.75, 0.9],
                iou_thresholds=[0.1, 0.5, 0.75],
            )
            print("evaluation time", eval_time)

            if ingest_time > ingestion_timeout and ingestion_timeout != -1:
                raise TimeoutError(
                    f"Base precomputation timed out with limit of {limit}."
                )

            # evaluate - base metrics only
            eval_time, metrics = time_it(evaluator.compute_precision_recall)(
                score_thresholds=[0.5, 0.75, 0.9],
                iou_thresholds=[0.1, 0.5, 0.75],
            )
            if eval_time > evaluation_timeout and evaluation_timeout != -1:
                raise TimeoutError(
                    f"Base evaluation timed out with {evaluator.metadata.number_of_datums} datums."
                )

            # evaluate - base metrics + detailed
            confusion_matrix_time, metrics = time_it(
                evaluator.compute_confusion_matrix
            )(
                score_thresholds=[0.5, 0.75, 0.9],
                iou_thresholds=[0.1, 0.5, 0.75],
            )
            if (
                confusion_matrix_time > evaluation_timeout
                and evaluation_timeout != -1
            ):
                raise TimeoutError(
                    f"Detailed evaluation timed out with {evaluator.metadata.number_of_datums} datums."
                )

            results.append(
                Benchmark(
                    limit=limit,
                    n_datums=0,  # evaluator.metadata.number_of_datums,
                    n_groundtruths=0,  # evaluator.metadata.number_of_ground_truths,
                    n_predictions=0,  # evaluator.metadata.number_of_predictions,
                    n_labels=0,  # evaluator.metadata.number_of_labels,
                    gt_type=gt_type,
                    pd_type=pd_type,
                    chunk_size=chunk_size,
                    ingestion=ingest_time,
                    preprocessing=preprocessing_time,
                    finalization=finalization_time,
                    precision_recall_computation=eval_time,
                    confusion_matrix=confusion_matrix_time,
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
