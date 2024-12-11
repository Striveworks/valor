from valor_lite.profiling import create_runtime_profiler
from valor_lite.semantic_segmentation import DataLoader, generate_segmentation


def benchmark_add_data(
    n_labels: int,
    shape: tuple[int, int],
    time_limit: float | None,
    repeat: int = 1,
) -> float:
    """
    Benchmarks 'Dataloader.add_data' for semantic segmentation.

    Parameters
    ----------
    n_labels : int
        The number of unique labels to generate.
    shape : tuple[int, int]
        The size (h,w) of the mask to generate.
    time_limit : float, optional
        An optional time limit to constrain the benchmark.
    repeat : int
        The number of times to run the benchmark to produce a runtime average.

    Returns
    -------
    float
        The average runtime.
    """

    profile = create_runtime_profiler(
        time_limit=time_limit,
        repeat=repeat,
    )

    elapsed = 0
    for _ in range(repeat):
        data = generate_segmentation(
            datum_uid="uid",
            number_of_unique_labels=n_labels,
            mask_height=shape[0],
            mask_width=shape[1],
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
    """
    Benchmarks 'Dataloader.finalize' for semantic segmentation.

    Parameters
    ----------
    n_datums : int
        The number of datums to generate.
    n_labels : int
        The number of unique labels to generate.
    time_limit : float, optional
        An optional time limit to constrain the benchmark.
    repeat : int
        The number of times to run the benchmark to produce a runtime average.

    Returns
    -------
    float
        The average runtime.
    """

    profile = create_runtime_profiler(
        time_limit=time_limit,
        repeat=repeat,
    )

    elapsed = 0
    for _ in range(repeat):

        data = [
            generate_segmentation(
                datum_uid=str(i),
                number_of_unique_labels=n_labels,
                mask_height=5,
                mask_width=5,
            )
            for i in range(10)
        ]
        loader = DataLoader()
        for datum_idx in range(n_datums):
            segmentation = data[datum_idx % 10]
            segmentation.uid = str(datum_idx)
            loader.add_data([segmentation])
        elapsed += profile(loader.finalize)()
    return elapsed / repeat


def benchmark_evaluate(
    n_datums: int,
    n_labels: int,
    time_limit: float | None,
    repeat: int = 1,
):
    """
    Benchmarks 'Evaluator.evaluate' for semantic segmentation.

    Parameters
    ----------
    n_datums : int
        The number of datums to generate.
    n_labels : int
        The number of unique labels to generate.
    time_limit : float, optional
        An optional time limit to constrain the benchmark.
    repeat : int
        The number of times to run the benchmark to produce a runtime average.

    Returns
    -------
    float
        The average runtime.
    """

    profile = create_runtime_profiler(
        time_limit=time_limit,
        repeat=repeat,
    )

    elapsed = 0
    for _ in range(repeat):

        data = [
            generate_segmentation(
                datum_uid=str(i),
                number_of_unique_labels=n_labels,
                mask_height=5,
                mask_width=5,
            )
            for i in range(10)
        ]
        loader = DataLoader()
        for datum_idx in range(n_datums):
            segmentation = data[datum_idx % 10]
            segmentation.uid = str(datum_idx)
            loader.add_data([segmentation])
        evaluator = loader.finalize()
        elapsed += profile(evaluator.evaluate)()
    return elapsed / repeat
