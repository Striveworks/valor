import json
import os
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from time import time

import requests
from tqdm import tqdm
from valor_core.enums import AnnotationType
from valor_core.evaluator import DetectionManager as Manager


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
    manager: Manager,
    gt_path: Path,
    pd_path: Path,
    limit: int,
    chunk_size: int,
):
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

                manager.add_data_from_dict(groundtruths, predictions)
                groundtruths = []
                predictions = []
            if groundtruths:
                manager.add_data_from_dict(groundtruths, predictions)


def run_base_evaluation(manager: Manager, timeout: int | None):
    """Run a base evaluation (with no PR curves)."""
    try:
        evaluation = manager.evaluate()
    except TimeoutError:
        raise TimeoutError(
            f"Base evaluation timed out when processing {evaluation.meta['datums']} datums."  # type: ignore
        )
    return evaluation


# def run_pr_curve_evaluation(dset: Dataset, model: Model, timeout: int | None):
#     """Run a base evaluation with PrecisionRecallCurve included."""
#     try:
#         evaluation = model.evaluate_detection(
#             dset,
#             metrics_to_return=[
#                 "AP",
#                 "AR",
#                 "mAP",
#                 "APAveragedOverIOUs",
#                 "mAR",
#                 "mAPAveragedOverIOUs",
#                 "PrecisionRecallCurve",
#             ],
#         )
#         evaluation.wait_for_completion(timeout=timeout)
#     except TimeoutError:
#         raise TimeoutError(
#             f"PR evaluation timed out when processing {evaluation.meta['datums']} datums."  # type: ignore
#         )
#     return evaluation


# def run_detailed_pr_curve_evaluation(
#     dset: Dataset, model: Model, timeout: int | None
# ):
#     """Run a base evaluation with PrecisionRecallCurve and DetailedPrecisionRecallCurve included."""

#     try:
#         evaluation = model.evaluate_detection(
#             dset,
#             metrics_to_return=[
#                 "AP",
#                 "AR",
#                 "mAP",
#                 "APAveragedOverIOUs",
#                 "mAR",
#                 "mAPAveragedOverIOUs",
#                 "PrecisionRecallCurve",
#                 "DetailedPrecisionRecallCurve",
#             ],
#         )
#         evaluation.wait_for_completion(timeout=timeout)
#     except TimeoutError:
#         raise TimeoutError(
#             f"Detailed evaluation timed out when processing {evaluation.meta['datums']} datums."  # type: ignore
#         )
#     return evaluation


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
                "base": round(self.eval_base, 5),
                "base+pr": round(self.eval_base_pr, 5),
                "base+pr+detailed": round(self.eval_base_pr_detail, 5),
            },
        }


def run_benchmarking_analysis(
    limits_to_test: list[int],
    combinations: list[tuple[AnnotationType, AnnotationType]] | None = None,
    results_file: str = "results.json",
    chunk_size: int = -1,
    compute_pr: bool = True,
    compute_detailed: bool = True,
    ingestion_timeout: int = 30,
    evaluation_timeout: int = 30,
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
        # AnnotationType.MULTIPOLYGON: gt_multipolygon_filename,
        AnnotationType.RASTER: gt_raster_filename,
    }
    predictions = {
        AnnotationType.BOX: pd_box_filename,
        AnnotationType.POLYGON: pd_polygon_filename,
        # AnnotationType.MULTIPOLYGON: pd_multipolygon_filename,
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

            manager = Manager()

            # ingestion
            ingest_time, _ = ingest(
                manager=manager,
                gt_path=current_directory / Path(gt_filename),
                pd_path=current_directory / Path(pd_filename),
                limit=limit,
                chunk_size=chunk_size,
            )  # type: ignore - time_it wrapper
            print("ingest", ingest_time)

            finalization_time, _ = time_it(manager.finalize)()
            print("preprocess", finalization_time)

            ap_time, ap_metrics = time_it(manager.compute_ap)()
            print("AP computation (work in progress)", ap_time)

            print(json.dumps(ap_metrics, indent=2))

            print("ingest", ingest_time)
            print("preprocess", finalization_time)
            print("AP computation (work in progress)", ap_time)

            # # run evaluations
            # eval_pr = None
            # eval_detail = None
            # eval_base = run_base_evaluation(
            #     manager=manager, timeout=evaluation_timeout
            # )
            # if compute_pr:
            #     eval_pr = run_pr_curve_evaluation(
            #         dset=dataset, model=model, timeout=evaluation_timeout
            #     )
            # if compute_detailed:
            #     eval_detail = run_detailed_pr_curve_evaluation(
            #         dset=dataset, model=model, timeout=evaluation_timeout
            #     )

    #         results.append(
    #             EvaluationBenchmark(
    #                 limit=limit,
    #                 gt_stats=DataBenchmark(
    #                     dtype=gt_type,
    #                     ingestion=gt_ingest_time,
    #                 ),
    #                 pd_stats=DataBenchmark(
    #                     dtype=pd_type,
    #                     ingestion=pd_ingest_time,
    #                 ),
    #                 n_datums=eval_base.meta["datums"],
    #                 n_annotations=eval_base.meta["annotations"],
    #                 n_labels=eval_base.meta["labels"],
    #                 eval_base=eval_base.meta["duration"],
    #                 eval_base_pr=eval_pr.meta["duration"] if eval_pr else -1,
    #                 eval_base_pr_detail=(
    #                     eval_detail.meta["duration"] if eval_detail else -1
    #                 ),
    #             ).result()
    #         )

    # write_results_to_file(write_path=write_path, results=results)


if __name__ == "__main__":

    # run bounding box benchmark
    run_benchmarking_analysis(
        combinations=[
            (AnnotationType.BOX, AnnotationType.BOX),
        ],
        limits_to_test=[5000],
    )

    # # run polygon benchmark
    # run_benchmarking_analysis(
    #     combinations=[
    #         (AnnotationType.POLYGON, AnnotationType.POLYGON),
    #     ],
    #     limits_to_test=[5000, 5000],
    # )

    # # run multipolygon benchmark
    # run_benchmarking_analysis(
    #     combinations=[
    #         (AnnotationType.MULTIPOLYGON, AnnotationType.MULTIPOLYGON),
    #     ],
    #     limits_to_test=[6, 6],
    #     compute_detailed=False,
    # )

    # # run raster benchmark
    # run_benchmarking_analysis(
    #     combinations=[
    #         (AnnotationType.RASTER, AnnotationType.RASTER),
    #     ],
    #     limits_to_test=[6, 6],
    #     compute_detailed=False,
    # )
