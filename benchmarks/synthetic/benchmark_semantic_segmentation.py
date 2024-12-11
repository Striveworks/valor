from valor_lite.profiling import Benchmark, BenchmarkError
from valor_lite.semantic_segmentation.benchmark import (
    benchmark_add_data,
    benchmark_evaluate,
    benchmark_finalize,
)


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


if __name__ == "__main__":

    benchmark(
        bitmask_shape=(4000, 4000),
        number_of_images=1000,
        number_of_unique_labels=10,
        memory_limit=4.0,
        time_limit=10.0,
        repeat=1,
        verbose=True,
    )
