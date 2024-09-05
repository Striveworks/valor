import json
import os
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from time import time

import requests
from tqdm import tqdm

from valor import Client, Dataset, GroundTruth, Model, Prediction, connect
from valor.enums import AnnotationType
from valor.exceptions import DatasetAlreadyExistsError, ModelAlreadyExistsError

connect("http://0.0.0.0:8000")
client = Client()


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
def ingest_groundtruths(
    dataset: Dataset,
    path: Path,
    limit: int,
    chunk_size: int,
    timeout: int | None,
):
    with open(path, "r") as f:
        count = 0
        chunks = []
        for line in f:
            gt_dict = json.loads(line)
            gt = GroundTruth.decode_value(gt_dict)
            chunks.append(gt)
            count += 1
            if count >= limit and limit > 0:
                break
            elif len(chunks) < chunk_size or chunk_size == -1:
                continue

            dataset.add_groundtruths(chunks, timeout=timeout)
            chunks = []
        if chunks:
            dataset.add_groundtruths(chunks, timeout=timeout)


@time_it
def ingest_predictions(
    dataset: Dataset,
    model: Model,
    path: Path,
    limit: int,
    chunk_size: int,
    timeout: int | None,
):
    with open(path, "r") as f:
        count = 0
        chunks = []
        for line in f:
            pd_dict = json.loads(line)
            pd = Prediction.decode_value(pd_dict)
            chunks.append(pd)
            count += 1
            if count >= limit and limit > 0:
                break
            elif len(chunks) < chunk_size or chunk_size == -1:
                continue

            model.add_predictions(dataset, chunks, timeout=timeout)
            chunks = []
        if chunks:
            model.add_predictions(dataset, chunks, timeout=timeout)


def run_base_evaluation(dset: Dataset, model: Model, timeout: int | None):
    """Run a base evaluation (with no PR curves)."""
    try:
        evaluation = model.evaluate_detection(dset)
        evaluation.wait_for_completion(timeout=timeout)
    except TimeoutError:
        raise TimeoutError(
            f"Base evaluation timed out when processing {evaluation.meta['datums']} datums."  # type: ignore
        )
    return evaluation


def run_pr_curve_evaluation(dset: Dataset, model: Model, timeout: int | None):
    """Run a base evaluation with PrecisionRecallCurve included."""
    try:
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
        evaluation.wait_for_completion(timeout=timeout)
    except TimeoutError:
        raise TimeoutError(
            f"PR evaluation timed out when processing {evaluation.meta['datums']} datums."  # type: ignore
        )
    return evaluation


def run_detailed_pr_curve_evaluation(
    dset: Dataset, model: Model, timeout: int | None
):
    """Run a base evaluation with PrecisionRecallCurve and DetailedPrecisionRecallCurve included."""

    try:
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
        evaluation.wait_for_completion(timeout=timeout)
    except TimeoutError:
        raise TimeoutError(
            f"Detailed evaluation timed out when processing {evaluation.meta['datums']} datums."  # type: ignore
        )
    return evaluation


@dataclass
class Benchmark:
    limit: int
    n_datums: int
    n_annotations: int
    n_labels: int
    gt_type: AnnotationType
    pd_type: AnnotationType
    chunk_size: int
    gt_ingest: float
    gt_finalization: float
    gt_deletion: float
    pd_ingest: float
    pd_finalization: float
    pd_deletion: float
    eval_base: float
    eval_base_pr: float
    eval_base_pr_detail: float

    def result(self) -> dict:
        return {
            "limit": self.limit,
            "chunk_size": self.chunk_size,
            "n_datums": self.n_datums,
            "n_annotations": self.n_annotations,
            "n_labels": self.n_labels,
            "dtype": {
                "groundtruth": self.gt_type.value,
                "prediction": self.pd_type.value,
            },
            "base": {
                "ingestion": {
                    "dataset": f"{round(self.gt_ingest, 2)} seconds",
                    "model": f"{round(self.pd_ingest, 2)} seconds",
                },
                "finalization": {
                    "dataset": f"{round(self.gt_finalization, 2)} seconds",
                    "model": f"{round(self.pd_finalization, 2)} seconds",
                },
                "evaluation": {
                    "preprocessing": "0.0 seconds",
                    "computation": f"{round(self.eval_base, 2)} seconds",
                    "total": f"{round(self.eval_base, 2)} seconds",
                },
                "deletion": {
                    "dataset": f"{round(self.gt_deletion, 2)} seconds",
                    "model": f"{round(self.pd_deletion, 2)} seconds",
                },
            },
            "base+pr": {
                "ingestion": {
                    "dataset": f"{round(self.gt_ingest, 2)} seconds",
                    "model": f"{round(self.pd_ingest, 2)} seconds",
                },
                "finalization": {
                    "dataset": f"{round(self.gt_finalization, 2)} seconds",
                    "model": f"{round(self.pd_finalization, 2)} seconds",
                },
                "evaluation": {
                    "preprocessing": "0.0 seconds",
                    "computation": f"{round(self.eval_base_pr, 2)} seconds",
                    "total": f"{round(self.eval_base_pr, 2)} seconds",
                },
                "deletion": {
                    "dataset": f"{round(self.gt_deletion, 2)} seconds",
                    "model": f"{round(self.pd_deletion, 2)} seconds",
                },
            }
            if self.eval_base_pr > -1
            else {},
            "base+pr+detailed": {
                "ingestion": {
                    "dataset": f"{round(self.gt_ingest, 2)} seconds",
                    "model": f"{round(self.pd_ingest, 2)} seconds",
                },
                "finalization": {
                    "dataset": f"{round(self.gt_finalization, 2)} seconds",
                    "model": f"{round(self.pd_finalization, 2)} seconds",
                },
                "evaluation": {
                    "preprocessing": "0.0 seconds",
                    "computation": f"{round(self.eval_base_pr_detail, 2)} seconds",
                    "total": f"{round(self.eval_base_pr_detail, 2)} seconds",
                },
                "deletion": {
                    "dataset": f"{round(self.gt_deletion, 2)} seconds",
                    "model": f"{round(self.pd_deletion, 2)} seconds",
                },
            }
            if self.eval_base_pr_detail > -1
            else {},
        }


def run_benchmarking_analysis(
    limits_to_test: list[int],
    combinations: list[tuple[AnnotationType, AnnotationType]] | None = None,
    results_file: str = "results.json",
    chunk_size: int = -1,
    ingestion_timeout: int = 30,
    evaluation_timeout: int = 30,
    compute_pr: bool = True,
    compute_detailed: bool = True,
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

    groundtruths = {
        AnnotationType.BOX: gt_box_filename,
        AnnotationType.POLYGON: gt_polygon_filename,
        AnnotationType.MULTIPOLYGON: gt_multipolygon_filename,
        AnnotationType.RASTER: gt_raster_filename,
    }
    predictions = {
        AnnotationType.BOX: pd_box_filename,
        AnnotationType.POLYGON: pd_polygon_filename,
        AnnotationType.MULTIPOLYGON: pd_multipolygon_filename,
        AnnotationType.RASTER: pd_raster_filename,
    }

    # default is to perform all combinations
    if combinations is None:
        combinations = [
            (gt_type, pd_type)
            for gt_type in groundtruths
            for pd_type in predictions
        ]

    # cache data locally
    filenames = [*list(groundtruths.values()), *list(predictions.values())]
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

            gt_filename = groundtruths[gt_type]
            pd_filename = predictions[pd_type]

            try:
                dataset = Dataset.create(name="coco")
                model = Model.create(name="yolo")
            except (
                DatasetAlreadyExistsError,
                ModelAlreadyExistsError,
            ) as e:
                client.delete_dataset("coco")
                client.delete_model("yolo")
                raise e

            # === Ingestion ===
            gt_ingest_time, _ = ingest_groundtruths(
                dataset=dataset,
                path=current_directory / Path(gt_filename),
                limit=limit,
                chunk_size=chunk_size,
                timeout=ingestion_timeout,
            )  # type: ignore - time_it wrapper
            gt_finalization_time, _ = time_it(dataset.finalize)()
            pd_ingest_time, _ = ingest_predictions(
                dataset=dataset,
                model=model,
                path=current_directory / Path(pd_filename),
                limit=limit,
                chunk_size=chunk_size,
                timeout=ingestion_timeout,
            )  # type: ignore - time_it wrapper
            pd_finalization_time, _ = time_it(model.finalize_inferences)(
                dataset
            )

            # === Base Evaluation ===
            base_results = run_base_evaluation(
                dset=dataset, model=model, timeout=evaluation_timeout
            )
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
            pr = -1
            if compute_pr:
                pr_results = run_pr_curve_evaluation(
                    dset=dataset, model=model, timeout=evaluation_timeout
                )
                assert pr_results.meta
                pr = pr_results.meta["duration"]
                if pr > evaluation_timeout and evaluation_timeout != -1:
                    raise TimeoutError(
                        f"PR evaluation timed out with {n_datums} datums."
                    )

            # === Detailed Evaluation ===
            detailed = -1
            if compute_detailed:
                detailed_results = run_detailed_pr_curve_evaluation(
                    dset=dataset, model=model, timeout=evaluation_timeout
                )
                assert detailed_results.meta
                detailed = detailed_results.meta["duration"]
                if detailed > evaluation_timeout and evaluation_timeout != -1:
                    raise TimeoutError(
                        f"Detailed evaluation timed out with {n_datums} datums."
                    )

            # delete model
            start = time()
            client.delete_model(model.name, timeout=30)
            pd_deletion_time = time() - start

            # delete dataset
            start = time()
            client.delete_dataset(dataset.name, timeout=30)
            gt_deletion_time = time() - start

            results.append(
                Benchmark(
                    limit=limit,
                    n_datums=n_datums,
                    n_annotations=n_annotations,
                    n_labels=n_labels,
                    gt_type=gt_type,
                    pd_type=pd_type,
                    chunk_size=chunk_size,
                    gt_ingest=gt_ingest_time,
                    gt_finalization=gt_finalization_time,
                    gt_deletion=gt_deletion_time,
                    pd_ingest=pd_ingest_time,
                    pd_finalization=pd_finalization_time,
                    pd_deletion=pd_deletion_time,
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
        chunk_size=250,
        limits_to_test=[5000, 5000],
    )

    # run polygon benchmark
    run_benchmarking_analysis(
        combinations=[
            (AnnotationType.POLYGON, AnnotationType.POLYGON),
        ],
        chunk_size=250,
        limits_to_test=[5000, 5000],
    )

    # run multipolygon benchmark
    run_benchmarking_analysis(
        combinations=[
            (AnnotationType.MULTIPOLYGON, AnnotationType.RASTER),
        ],
        evaluation_timeout=0,
        chunk_size=10,
        limits_to_test=[1, 1],
        compute_pr=False,
        compute_detailed=False,
    )

    # # run raster benchmark
    # run_benchmarking_analysis(
    #     combinations=[
    #         (AnnotationType.RASTER, AnnotationType.RASTER),
    #     ],
    #     limits_to_test=[100, 100],
    #     ingestion_chunk_size=10,
    #     evaluation_timeout=0,
    #     compute_pr=False,
    #     compute_detailed=False,
    # )
