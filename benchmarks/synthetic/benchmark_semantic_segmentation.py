import json
import os
import time
import tracemalloc
from datetime import datetime
from pathlib import Path

import numpy as np
from tqdm import tqdm

from valor_lite.semantic_segmentation import Bitmask, DataLoader, Segmentation


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
        The number of unique labels.
    mask_height : int
        The height of the mask in pixels.
    mask_width : int
        The width of the mask in pixels.

    Returns
    -------
    Segmentation
        A generated semantic segmenatation annotation.
    """

    if number_of_unique_labels > 1:
        common_proba = 0.4 / (number_of_unique_labels - 1)
        min_proba = min(common_proba, 0.1)
        labels = [str(i) for i in range(number_of_unique_labels)] + [None]
        proba = (
            [0.5]
            + [common_proba for _ in range(number_of_unique_labels - 1)]
            + [0.1]
        )
    elif number_of_unique_labels == 1:
        labels = ["0", None]
        proba = [0.9, 0.1]
        min_proba = 0.1
    else:
        raise ValueError(
            "The number of unique labels should be greater than zero."
        )

    probabilities = np.array(proba, dtype=np.float64)
    weights = (probabilities / min_proba).astype(np.int32)

    indices = np.random.choice(
        np.arange(len(weights)),
        size=(mask_height * 2, mask_width),
        p=probabilities,
    )

    N = len(labels)

    masks = np.arange(N)[:, None, None] == indices

    gts = []
    pds = []
    for lidx in range(N):
        label = labels[lidx]
        if label is None:
            continue
        gts.append(
            Bitmask(
                mask=masks[lidx, :mask_height, :],
                label=label,
            )
        )
        pds.append(
            Bitmask(
                mask=masks[lidx, mask_height:, :],
                label=label,
            )
        )

    return Segmentation(
        uid=datum_uid,
        groundtruths=gts,
        predictions=pds,
        shape=(mask_height, mask_width),
    )


def benchmark(
    bitmask_shape: tuple[int, int],
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
    bitmask_shape : tuple[int, int]
        The size (h, w) of the bitmask array.
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

        loader = DataLoader()

        for i in tqdm(range(number_of_images)):
            data, elapsed, peak = profile(generate_segmentation)(
                datum_uid=f"uid{i}",
                number_of_unique_labels=number_of_unique_labels,
                mask_height=bitmask_shape[0],
                mask_width=bitmask_shape[1],
            )
            elapsed_generation += elapsed
            peak_generation = max(peak_generation, peak)

            _, elapsed, peak = profile(loader.add_data)([data])
            elapsed_add_data += elapsed
            peak_add_data = max(peak_add_data, peak)

        evaluator, elapsed, peak = profile(loader.finalize)()
        elapsed_finalization += elapsed
        peak_finalization = max(peak_finalization, peak)

        _, elapsed, peak = profile(evaluator.evaluate)()
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
            "bitmask_shape": bitmask_shape,
            "number_of_unique_labels": number_of_unique_labels,
            "number_of_images": number_of_images,
        },
    }
    write_results_to_file(write_path=write_path, results=[results])


if __name__ == "__main__":

    current_directory = Path(__file__).parent
    write_path = current_directory / Path("seg_results.json")

    benchmark(
        bitmask_shape=(100, 100),
        number_of_images=10_000,
        number_of_unique_labels=10,
        memory_limit=4.0,
        time_limit=10.0,
        repeat=1,
        verbose=True,
        write_path=write_path,
    )
