import numpy as np
from valor_lite.profiling import (
    Benchmark,
    BenchmarkError,
    create_runtime_profiler,
)
from valor_lite.semantic_segmentation import Bitmask, DataLoader, Segmentation


def generate_segmentation(
    uid: str,
    n_labels: int,
    height: int,
    width: int,
) -> Segmentation:
    """
    Generates a list of segmentation annotations.

    Parameters
    ----------
    uid : str
        The datum UID for the generated segmentation.

    Returns
    -------
    Segmenation
        A generated semantic segmenatation annotation.
    """

    if n_labels > 1:
        common_proba = 0.4 / (n_labels - 1)
        min_proba = min(common_proba, 0.1)
        labels = [str(i) for i in range(n_labels)] + [None]
        proba = [0.5] + [common_proba for _ in range(n_labels - 1)] + [0.1]
    elif n_labels == 1:
        labels = ["0", None]
        proba = [0.9, 0.1]
        min_proba = 0.1
    else:
        labels = [None]
        proba = [1.0]
        min_proba = 1.0

    probabilities = np.array(proba, dtype=np.float64)
    weights = (probabilities / min_proba).astype(np.int32)

    indices = np.random.choice(
        np.arange(len(weights)), size=(height * 2, width), p=probabilities
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
                mask=masks[lidx, :height, :],
                label=label,
            )
        )
        pds.append(
            Bitmask(
                mask=masks[lidx, height:, :],
                label=label,
            )
        )

    return Segmentation(
        uid=uid,
        groundtruths=gts,
        predictions=pds,
    )


def benchmark_add_data(
    n_labels: int,
    shape: tuple[int, int],
    time_limit: float | None,
    repeat: int = 1,
):

    profile = create_runtime_profiler(
        time_limit=time_limit,
        repeat=repeat,
    )

    elapsed = 0
    for _ in range(repeat):
        data = generate_segmentation(
            uid="uid",
            n_labels=n_labels,
            height=shape[0],
            width=shape[1],
        )
        loader = DataLoader()
        elapsed += profile(loader.add_data)([data])
    return elapsed / repeat


def benchmark_finalize(
    n_datums: int,
    n_labels: int,
    time_limit: float | None,
    repeat: int = 1,
):

    profile = create_runtime_profiler(
        time_limit=time_limit,
        repeat=repeat,
    )

    elapsed = 0
    for _ in range(repeat):
        loader = DataLoader()
        for datum_idx in range(n_datums):
            data = generate_segmentation(
                uid=str(datum_idx),
                n_labels=n_labels,
                height=100,
                width=100,
            )
            loader.add_data([data])
        elapsed += profile(loader.finalize)()
    return elapsed / repeat


def benchmark_evaluate(
    n_datums: int,
    n_labels: int,
    time_limit: float | None,
    repeat: int = 1,
):

    profile = create_runtime_profiler(
        time_limit=time_limit,
        repeat=repeat,
    )

    elapsed = 0
    for _ in range(repeat):
        loader = DataLoader()
        for datum_idx in range(n_datums):
            data = generate_segmentation(
                uid=str(datum_idx),
                n_labels=n_labels,
                height=100,
                width=100,
            )
            loader.add_data([data])
        evaluator = loader.finalize()
        elapsed += profile(evaluator.evaluate)()
    return elapsed / repeat


def benchmark(
    bitmask_shape: tuple[int, int],
    number_of_unique_labels: int,
    number_of_images: int,
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

    b = Benchmark(
        time_limit=time_limit,
        memory_limit=int(memory_limit * (1024**3)),
        repeat=repeat,
        verbose=verbose,
    )

    _, failed, details = b.run(
        benchmark=benchmark_add_data,
        n_labels=[number_of_unique_labels],
        shape=[bitmask_shape],
    )
    if failed:
        raise BenchmarkError(
            benchmark=details["benchmark"],
            error_type=failed[0]["error"],
            error_message=failed[0]["msg"],
        )

    _, failed, details = b.run(
        benchmark=benchmark_finalize,
        n_datums=[number_of_images],
        n_labels=[number_of_unique_labels],
    )
    if failed:
        raise BenchmarkError(
            benchmark=details["benchmark"],
            error_type=failed[0]["error"],
            error_message=failed[0]["msg"],
        )

    _, failed, details = b.run(
        benchmark=benchmark_evaluate,
        n_datums=[number_of_images],
        n_labels=[number_of_unique_labels],
    )
    if failed:
        raise BenchmarkError(
            benchmark=details["benchmark"],
            error_type=failed[0]["error"],
            error_message=failed[0]["msg"],
        )
