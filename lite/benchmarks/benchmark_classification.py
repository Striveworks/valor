import json
import os
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from time import time

import requests
from tqdm import tqdm
from valor_lite.classification import Classification, DataLoader


def _convert_valor_dicts_into_Classification(gt_dict: dict, pd_dict: dict):
    """Convert a groundtruth dictionary and prediction dictionary into a valor_lite Classification object."""
    pds = []
    scores = []

    # there's only one annotation / label per groundtruth in the benchmarking data
    gt = gt_dict["annotations"][0]["labels"][0]["value"]
    pds = []
    scores = []
    for pann in pd_dict["annotations"]:
        for valor_label in pann["labels"]:
            pds.append(valor_label["value"])
            scores.append(valor_label["score"])

    return Classification(
        uid=gt_dict["datum"]["uid"],
        groundtruth=gt,
        predictions=pds,
        scores=scores,
    )


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


@time_it
def ingest(
    loader: DataLoader,
    gt_path: Path,
    pd_path: Path,
    limit: int,
    chunk_size: int,
):
    accumulated_time = 0.0
    with open(gt_path, "r") as gf:
        with open(pd_path, "r") as pf:
            count = 0
            classifications = []
            for gline, pline in zip(gf, pf):

                gt_dict = json.loads(gline)
                pd_dict = json.loads(pline)
                classifications.append(
                    _convert_valor_dicts_into_Classification(
                        gt_dict=gt_dict, pd_dict=pd_dict
                    )
                )
                count += 1
                if count >= limit and limit > 0:
                    break
                elif len(classifications) < chunk_size or chunk_size == -1:
                    continue

                timer, _ = time_it(loader.add_data)(classifications)
                accumulated_time += timer
                classifications = []

            if classifications:
                timer, _ = time_it(loader.add_data)(classifications)
                accumulated_time += timer

    return accumulated_time


@dataclass
class Benchmark:
    limit: int
    n_datums: int
    n_groundtruths: int
    n_predictions: int
    n_labels: int
    chunk_size: int
    ingestion: float
    preprocessing: float
    precomputation: float
    evaluation: float
    detailed_evaluation: list[tuple[int, float]]

    def result(self) -> dict:
        return {
            "limit": self.limit,
            "n_datums": self.n_datums,
            "n_groundtruths": self.n_groundtruths,
            "n_predictions": self.n_predictions,
            "n_labels": self.n_labels,
            "chunk_size": self.chunk_size,
            "ingestion": {
                "loading_from_file": f"{round(self.ingestion - self.preprocessing, 2)} seconds",
                "numpy_conversion": f"{round(self.preprocessing, 2)} seconds",
                "finalization": f"{round(self.precomputation, 2)} seconds",
                "total": f"{round(self.ingestion + self.precomputation, 2)} seconds",
            },
            "base_evaluation": f"{round(self.evaluation, 2)} seconds",
            "detailed_evaluation": [
                {
                    "n_points": 10,
                    "n_examples": curve[0],
                    "computation": f"{round(curve[1], 2)} seconds",
                }
                for curve in self.detailed_evaluation
            ],
        }


def run_benchmarking_analysis(
    limits_to_test: list[int],
    results_file: str = "clf_results.json",
    chunk_size: int = -1,
    ingestion_timeout=30,
    evaluation_timeout=30,
):
    """Time various function calls and export the results."""
    current_directory = Path(__file__).parent
    write_path = current_directory / Path(results_file)

    gt_filename = "gt_classification.jsonl"
    pd_filename = "pd_classification.jsonl"

    # cache data locally
    for filename in [gt_filename, pd_filename]:
        file_path = current_directory / Path(filename)
        url = f"https://pub-fae71003f78140bdaedf32a7c8d331d2.r2.dev/{filename}"
        download_data_if_not_exists(
            file_name=filename, file_path=file_path, url=url
        )

    # iterate through datum limits
    results = list()
    for limit in limits_to_test:

        # === Base Evaluation ===
        loader = DataLoader()

        # ingest + preprocess
        (ingest_time, preprocessing_time,) = ingest(
            loader=loader,
            gt_path=current_directory / Path(gt_filename),
            pd_path=current_directory / Path(pd_filename),
            limit=limit,
            chunk_size=chunk_size,
        )  # type: ignore - time_it wrapper

        finalization_time, evaluator = time_it(loader.finalize)()

        if ingest_time > ingestion_timeout and ingestion_timeout != -1:
            raise TimeoutError(
                f"Base precomputation timed out with limit of {limit}."
            )

        # evaluate
        eval_time, _ = time_it(evaluator.compute_precision_recall)()
        if eval_time > evaluation_timeout and evaluation_timeout != -1:
            raise TimeoutError(
                f"Base evaluation timed out with {evaluator.n_datums} datums."
            )

        detail_no_examples_time, _ = time_it(
            evaluator.compute_confusion_matrix
        )(
            number_of_examples=0,
        )
        if (
            detail_no_examples_time > evaluation_timeout
            and evaluation_timeout != -1
        ):
            raise TimeoutError(
                f"Base evaluation timed out with {evaluator.n_datums} datums."
            )

        detail_three_examples_time, _ = time_it(
            evaluator.compute_confusion_matrix
        )(
            number_of_examples=3,
        )
        if (
            detail_three_examples_time > evaluation_timeout
            and evaluation_timeout != -1
        ):
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
                chunk_size=chunk_size,
                ingestion=ingest_time,
                preprocessing=preprocessing_time,
                precomputation=finalization_time,
                evaluation=eval_time,
                detailed_evaluation=[
                    (0, detail_no_examples_time),
                    (3, detail_three_examples_time),
                ],
            ).result()
        )

    write_results_to_file(write_path=write_path, results=results)


if __name__ == "__main__":

    run_benchmarking_analysis(
        limits_to_test=[5000, 5000, 5000],
    )
