import json
import os
import re
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from time import time

import numpy as np
import requests
from tqdm import tqdm

from valor import Client, Dataset, GroundTruth, Model, Prediction, connect
from valor.exceptions import DatasetAlreadyExistsError, ModelAlreadyExistsError

connect("http://0.0.0.0:8000")
client = Client()


def time_it(fn, *args, **kwargs):
    start = time()
    fn(*args, **kwargs)
    return time() - start


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
    dataset: Dataset,
    path: Path,
    limit: int,
    chunk_size: int,
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
            elif len(chunks) < chunk_size:
                continue

            dataset.add_groundtruths(chunks, timeout=30)
            chunks = []
        if chunks:
            dataset.add_groundtruths(chunks, timeout=30)


def ingest_predictions(
    dataset: Dataset,
    model: Model,
    datum_uids: list[str],
    path: Path,
    limit: int,
    chunk_size: int,
):
    pattern = re.compile(r'"uid":\s*"(\d+)"')
    with open(path, "r") as f:
        count = 0
        chunks = []
        for line in f:
            match = pattern.search(line)
            if not match:
                continue
            elif match.group(1) not in datum_uids:
                continue
            pd_dict = json.loads(line)
            pd = Prediction.decode_value(pd_dict)
            chunks.append(pd)
            count += 1
            if count >= limit and limit > 0:
                break
            elif len(chunks) < chunk_size:
                continue

            model.add_predictions(dataset, chunks, timeout=30)
            chunks = []
        if chunks:
            model.add_predictions(dataset, chunks, timeout=30)


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


@dataclass
class DataBenchmark:
    dtype: str = "unknown"
    ingestion: list[float] = field(default_factory=list)
    finalization: list[float] = field(default_factory=list)
    deletion: list[float] = field(default_factory=list)

    def add(
        self,
        dtype: str,
        ingestion: float,
        finalization: float,
        deletion: float,
    ):
        self.dtype = dtype
        self.ingestion.append(ingestion)
        self.finalization.append(finalization)
        self.deletion.append(deletion)

    def result(self) -> dict[str, float | str]:
        return {
            "dtype": self.dtype,
            "ingestion": round(float(np.mean(self.ingestion)), 2),
            "finalization": round(float(np.mean(self.finalization)), 2),
            "deletion": round(float(np.mean(self.deletion)), 2),
        }


@dataclass
class EvaluationBenchmark:
    gt_type: str = "unknown"
    pd_type: str = "unknown"
    n_datums: int = 0
    n_annotations: int = 0
    n_labels: int = 0
    eval_base: list[float] = field(default_factory=list)
    eval_base_pr: list[float] = field(default_factory=list)
    eval_base_pr_detail: list[float] = field(default_factory=list)

    def add(
        self,
        gt_type: str,
        pd_type: str,
        eval_base: dict,
        eval_base_pr: dict,
        eval_base_pr_detail: dict,
    ):
        self.gt_type = gt_type
        self.pd_type = pd_type
        self.n_datums = eval_base["labels"]
        self.n_labels = eval_base["labels"]
        self.n_annotations = eval_base["annotations"]
        self.eval_base.append(eval_base["duration"])
        self.eval_base_pr.append(eval_base_pr["duration"])
        self.eval_base_pr_detail.append(eval_base_pr_detail["duration"])

    def result(self) -> dict[str, float | str]:
        return {
            "groundtruth_type": self.gt_type,
            "prediction_type": self.pd_type,
            "number_of_datums": self.n_datums,
            "number_of_annotations": self.n_annotations,
            "number_of_labels": self.n_labels,
            "base": round(float(np.mean(self.eval_base)), 2),
            "base+pr": round(float(np.mean(self.eval_base_pr)), 2),
            "base+pr+detailed": round(
                float(np.mean(self.eval_base_pr_detail)), 2
            ),
        }


def run_benchmarking_analysis(
    limits_to_test: list[int],
    results_file: str = "results.json",
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
        "box": gt_box_filename,
        "polygon": gt_polygon_filename,
        "multipolygon": gt_multipolygon_filename,
        "raster": gt_raster_filename,
    }
    predictions = {
        "box": pd_box_filename,
        "polygon": pd_polygon_filename,
        "multipolygon": pd_multipolygon_filename,
        "raster": pd_raster_filename,
    }

    # cache data locally
    filenames = [*list(groundtruths.values()), *list(predictions.values())]
    for filename in filenames:
        file_path = current_directory / Path(filename)
        url = f"https://pub-fae71003f78140bdaedf32a7c8d331d2.r2.dev/{filename}"
        download_data_if_not_exists(
            file_name=filename, file_path=file_path, url=url
        )

    gt_results = defaultdict(DataBenchmark)
    pd_results = defaultdict(DataBenchmark)
    eval_results = defaultdict(lambda: defaultdict(EvaluationBenchmark))

    # iterate through datum limits
    for limit in limits_to_test:
        for gt_type, gt_filename in groundtruths.items():
            for pd_type, pd_filename in predictions.items():

                print(gt_type, pd_type)

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

                # gt bbox ingestion
                gt_ingest_time = time_it(
                    ingest_groundtruths,
                    dataset=dataset,
                    path=current_directory / Path(gt_filename),
                    limit=limit,
                    chunk_size=1000,
                )

                # gt bbox finalization
                gt_finalization_time = time_it(dataset.finalize)

                # pd bbox ingestion
                box_datum_uids = [datum.uid for datum in dataset.get_datums()]
                pd_ingest_time = time_it(
                    ingest_predictions,
                    dataset=dataset,
                    model=model,
                    datum_uids=box_datum_uids,
                    path=current_directory / Path(pd_filename),
                    limit=limit,
                    chunk_size=1000,
                )

                # model finalization
                pd_finalization_time = time_it(
                    model.finalize_inferences, dataset
                )

                try:
                    eval_base = run_base_evaluation(dset=dataset, model=model)
                    eval_pr = run_pr_curve_evaluation(
                        dset=dataset, model=model
                    )
                    eval_detail = run_detailed_pr_curve_evaluation(
                        dset=dataset, model=model
                    )
                except TimeoutError:
                    raise TimeoutError(
                        f"Evaluation timed out when processing {limit} datums."
                    )

                start = time()
                client.delete_model(model.name, timeout=30)
                pd_deletion_time = time() - start

                start = time()
                client.delete_dataset(dataset.name, timeout=30)
                gt_deletion_time = time() - start

                gt_results[gt_type].add(
                    dtype=gt_type,
                    ingestion=gt_ingest_time,
                    finalization=gt_finalization_time,
                    deletion=gt_deletion_time,
                )
                pd_results[pd_type].add(
                    dtype=pd_type,
                    ingestion=pd_ingest_time,
                    finalization=pd_finalization_time,
                    deletion=pd_deletion_time,
                )
                eval_results[gt_type][pd_type].add(
                    gt_type=gt_type,
                    pd_type=pd_type,
                    eval_base=eval_base.meta,
                    eval_base_pr=eval_pr.meta,
                    eval_base_pr_detail=eval_detail.meta,
                )

    results = {
        "dataset": [result.result() for result in gt_results.values()],
        "model": [result.result() for result in pd_results.values()],
        "evaluation": [
            result.result()
            for permuation in eval_results.values()
            for result in permuation.values()
        ],
    }
    write_results_to_file(write_path=write_path, result_dict=results)


if __name__ == "__main__":
    run_benchmarking_analysis(
        limits_to_test=[12, 12],
    )
