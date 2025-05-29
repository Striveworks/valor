import numpy as np

from valor_lite.profiling import create_runtime_profiler
from valor_lite.semantic_segmentation import Bitmask, DataLoader, Segmentation


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
