import json
import os
import shutil
import time
import tracemalloc
from datetime import datetime
from pathlib import Path

import numpy as np
from tqdm import tqdm

from valor_lite.semantic_segmentation import Segmentation
from valor_lite.semantic_segmentation.loader import Loader


def format_bytes(bytes_count, decimal_places=2):
    units = ["B", "KB", "MB", "GB", "TB", "PB", "EB", "ZB", "YB"]
    if bytes_count == 0:
        return f"0 {units[0]}"
    idx = 0
    size = float(bytes_count)
    while size >= 1024 and idx < len(units) - 1:
        size /= 1024
        idx += 1
    return f"{size:.{decimal_places}f} {units[idx]}"


def profile(fn):
    def wrapper(*args, **kwargs):
        tracemalloc.start()
        start = time.perf_counter()
        result = fn(*args, **kwargs)
        end = time.perf_counter()
        elapsed = end - start
        _, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        return result, elapsed, peak

    return wrapper


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


def generate_segmentation(
    datum_uid: str,
    number_of_unique_labels: int,
    mask_height: int,
    mask_width: int,
) -> Segmentation:
    """
    Generates a semantic segmentation annotation.

    Parameters
    ----------
    datum_uid : str
        The datum UID for the generated segmentation.
    number_of_unique_labels : int
        The number of pixel indices, including background at zero.
    mask_height : int
        The height of the mask in pixels.
    mask_width : int
        The width of the mask in pixels.

    Returns
    -------
    Segmentation
        A generated semantic segmentation annotation.
    """

    if not 1 <= number_of_unique_labels <= 65_536:
        raise ValueError("The number of labels must be between 1 and 65,536.")
    probabilities = np.full(
        number_of_unique_labels,
        0.5 / max(number_of_unique_labels - 1, 1),
    )
    probabilities[0] = 0.5 if number_of_unique_labels > 1 else 1.0
    indices = (
        np.random.default_rng()
        .choice(
            number_of_unique_labels,
            size=(mask_height * 2, mask_width),
            p=probabilities,
        )
        .astype(np.uint16)
    )
    return Segmentation(
        uid=datum_uid,
        groundtruths=indices[:mask_height],
        predictions=indices[mask_height:],
        labels=[str(i) for i in range(1, number_of_unique_labels)],
    )


def benchmark(
    label_map_shape: tuple[int, int],
    number_of_unique_labels: int,
    number_of_images: int,
    write_path: Path,
    *_,
    memory_limit: float = 4.0,
    time_limit: float = 10.0,
    repeat: int = 1,
    verbose: bool = False,
):
    """
    Runs a single benchmark.

    Parameters
    ----------
    label_map_shape : tuple[int, int]
        The size (h, w) of each label map.
    number_of_unique_labels : int
        The number of unique labels used in the synthetic example.
    number_of_images : int
        The number of distinct datums that are created.
    memory_limit : float
        The maximum amount of system memory allowed in gigabytes (GB).
    time_limit : float
        The maximum amount of time permitted before killing the benchmark.
    repeat : int
        The number of times to run a benchmark to produce an average runtime.
    verbose : bool, default=False
        Toggles terminal output of benchmark results.
    """
    elapsed_generation = 0
    elapsed_add_data = 0
    elapsed_finalization = 0
    elapsed_evaluation = 0

    peak_generation = 0
    peak_add_data = 0
    peak_finalization = 0
    peak_evaluation = 0

    for _ in range(repeat):

        path = Path(".valor/benchmark_semseg")
        if path.exists():
            shutil.rmtree(path)
        loader = Loader.persistent(
            path=path,
            batch_size=1_000,
            rows_per_file=10_000,
        )

        for i in tqdm(range(number_of_images)):
            data, elapsed, peak = profile(generate_segmentation)(
                datum_uid=f"uid{i}",
                number_of_unique_labels=number_of_unique_labels,
                mask_height=label_map_shape[0],
                mask_width=label_map_shape[1],
            )
            elapsed_generation += elapsed
            peak_generation = max(peak_generation, peak)

            _, elapsed, peak = profile(loader.add_data)([data])
            elapsed_add_data += elapsed
            peak_add_data = max(peak_add_data, peak)

        evaluator, elapsed, peak = profile(loader.finalize)()
        elapsed_finalization += elapsed
        peak_finalization = max(peak_finalization, peak)

        _, elapsed, peak = profile(evaluator.compute_precision_recall_iou)()
        elapsed_evaluation += elapsed
        peak_evaluation = max(peak_evaluation, peak)

    elapsed_generation /= repeat
    elapsed_add_data /= repeat
    elapsed_finalization /= repeat
    elapsed_evaluation /= repeat

    results = {
        "time": {
            "generation": f"{elapsed_generation} s",
            "add_data": f"{elapsed_add_data} s",
            "finalization": f"{elapsed_finalization} s",
            "evaluation": f"{elapsed_evaluation} s",
        },
        "memory": {
            "generation": format_bytes(peak_generation),
            "add_data": format_bytes(peak_add_data),
            "finalization": format_bytes(peak_finalization),
            "evaluation": format_bytes(peak_evaluation),
        },
        "params": {
            "repeated": repeat,
            "label_map_shape": label_map_shape,
            "number_of_unique_labels": number_of_unique_labels,
            "number_of_images": number_of_images,
        },
    }
    write_results_to_file(write_path=write_path, results=[results])


if __name__ == "__main__":

    current_directory = Path(__file__).parent
    write_path = current_directory / Path("seg_results.json")

    benchmark(
        label_map_shape=(100, 100),
        number_of_images=10_000,
        number_of_unique_labels=10,
        memory_limit=4.0,
        time_limit=10.0,
        repeat=1,
        verbose=True,
        write_path=write_path,
    )
