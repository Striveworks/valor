import json
import os
import re
from datetime import datetime
from pathlib import Path
from time import time

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
            if count >= limit:
                break
            elif len(chunks) < chunk_size:
                continue

            dataset.add_groundtruths(chunks)
            chunks = []
        if chunks:
            dataset.add_groundtruths(chunks)


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
            if count >= limit:
                break
            elif len(chunks) < chunk_size:
                continue

            model.add_predictions(dataset, chunks)
            chunks = []
        if chunks:
            model.add_predictions(dataset, chunks)


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

    from dataclasses import dataclass

    @dataclass
    class Dummy:
        meta: dict

    dummy_value = Dummy(meta={"duration": -1, "labels": 0, "annotations": 0})

    # iterate through datum limits
    for limit in limits_to_test:

        gt_box_ingest_time = -1
        pd_box_ingest_time = -1
        gt_bbox_finalization_time = -1
        pd_bbox_finalization_time = -1
        box_deletion_time = -1
        gt_raster_ingest_time = -1
        pd_raster_ingest_time = -1
        gt_raster_finalization_time = -1
        pd_raster_finalization_time = -1
        raster_deletion_time = -1

        eval_base_box = dummy_value
        eval_pr_box = dummy_value
        eval_detail_box = dummy_value
        eval_base_raster = dummy_value
        eval_pr_raster = dummy_value
        eval_detail_raster = dummy_value

        if compute_box:
            try:
                dset_box = Dataset.create(name="coco-box")
                model_box = Model.create(name="yolo-box")
            except (
                DatasetAlreadyExistsError,
                ModelAlreadyExistsError,
            ) as e:
                client.delete_dataset("coco-box")
                client.delete_model("yolo-box")
                raise e

            # gt bbox ingestion
            gt_box_ingest_time = time_it(
                ingest_groundtruths,
                dataset=dset_box,
                path=current_directory / Path(gt_box_filename),
                limit=limit,
                chunk_size=1000,
            )

            # gt bbox finalization
            gt_bbox_finalization_time = time_it(dset_box.finalize)

            # pd bbox ingestion
            box_datum_uids = [datum.uid for datum in dset_box.get_datums()]
            pd_box_ingest_time = time_it(
                ingest_predictions,
                dataset=dset_box,
                model=model_box,
                datum_uids=box_datum_uids,
                path=current_directory / Path(pd_box_filename),
                limit=limit,
                chunk_size=1000,
            )

            # pd bbox finalization
            pd_bbox_finalization_time = time_it(
                model_box.finalize_inferences, dset_box
            )

            try:
                eval_base_box = run_base_evaluation(
                    dset=dset_box, model=model_box
                )
                eval_pr_box = run_pr_curve_evaluation(
                    dset=dset_box, model=model_box
                )
                eval_detail_box = run_detailed_pr_curve_evaluation(
                    dset=dset_box, model=model_box
                )
            except TimeoutError:
                raise TimeoutError(
                    f"Evaluation timed out when processing {limit} datums."
                )

            start = time()
            client.delete_dataset(dset_box.name, timeout=30)
            client.delete_model(model_box.name, timeout=30)
            box_deletion_time = time() - start

        if compute_raster:
            try:
                dset_raster = Dataset.create(name="coco-raster")
                model_raster = Model.create(name="yolo-raster")
            except (
                DatasetAlreadyExistsError,
                ModelAlreadyExistsError,
            ) as e:
                client.delete_dataset("coco-raster")
                client.delete_model("yolo-raster")
                raise e

            # gt raster ingestion
            gt_raster_ingest_time = time_it(
                ingest_groundtruths,
                dataset=dset_raster,
                path=current_directory / Path(gt_raster_filename),
                limit=limit,
                chunk_size=100,
            )

            # gt raster finalization
            gt_raster_finalization_time = time_it(dset_raster.finalize)

            # pd raster ingestion
            raster_datum_uids = [
                datum.uid for datum in dset_raster.get_datums()
            ]
            pd_raster_ingest_time = time_it(
                ingest_predictions,
                dataset=dset_raster,
                model=model_raster,
                datum_uids=raster_datum_uids,
                path=current_directory / Path(pd_raster_filename),
                limit=limit,
                chunk_size=100,
            )

            # pd raster finalization
            pd_raster_finalization_time = time_it(
                model_raster.finalize_inferences, dset_raster
            )

            try:
                eval_base_raster = run_base_evaluation(
                    dset=dset_raster, model=model_raster
                )
                # eval_pr_raster = run_pr_curve_evaluation(
                #     dset=dset_raster, model=model_raster
                # )
                # eval_detail_raster = run_detailed_pr_curve_evaluation(
                #     dset=dset_raster, model=model_raster
                # )
            except TimeoutError:
                raise TimeoutError(
                    f"Evaluation timed out when processing {limit} datums."
                )

            start = time()
            client.delete_dataset(dset_raster.name, timeout=30)
            client.delete_model(model_raster.name, timeout=30)
            raster_deletion_time = time() - start

        results = {
            "box": {
                "info": {
                    "number_of_datums": limit,
                    "number_of_unique_labels": eval_base_box.meta["labels"],
                    "number_of_annotations": eval_base_box.meta["annotations"],
                },
                "ingestion": {
                    "groundtruth": f"{(gt_box_ingest_time):.1f} seconds",
                    "prediction": f"{(pd_box_ingest_time):.1f} seconds",
                },
                "finalization": {
                    "dataset": f"{(gt_bbox_finalization_time):.1f} seconds",
                    "model": f"{(pd_bbox_finalization_time):.1f} seconds",
                },
                "evaluation": {
                    "base": f"{(eval_base_box.meta['duration']):.1f} seconds",
                    "base+pr": f"{(eval_pr_box.meta['duration']):.1f} seconds",
                    "base+pr+detail": f"{(eval_detail_box.meta['duration']):.1f} seconds",
                },
                "deletion": f"{(box_deletion_time):.1f} seconds",
            },
            "raster": {
                "info": {
                    "number_of_datums": limit,
                    "number_of_unique_labels": eval_base_raster.meta["labels"],
                    "number_of_annotations": eval_base_raster.meta[
                        "annotations"
                    ],
                },
                "ingestion": {
                    "groundtruth": f"{(gt_raster_ingest_time):.1f} seconds",
                    "prediction": f"{(pd_raster_ingest_time):.1f} seconds",
                },
                "finalization": {
                    "dataset": f"{(gt_raster_finalization_time):.1f} seconds",
                    "model": f"{(pd_raster_finalization_time):.1f} seconds",
                },
                "evaluation": {
                    "base": f"{(eval_base_raster.meta['duration']):.1f} seconds",
                    "base+pr": f"{(eval_pr_raster.meta['duration']):.1f} seconds",
                    "base+pr+detail": f"{(eval_detail_raster.meta['duration']):.1f} seconds",
                },
                "deletion": f"{(raster_deletion_time):.1f} seconds",
            },
        }
        write_results_to_file(write_path=write_path, result_dict=results)


if __name__ == "__main__":
    run_benchmarking_analysis(
        limits_to_test=[5000, 5000, 5000],
        compute_box=True,
        compute_raster=False,
    )
