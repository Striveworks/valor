import numpy as np
from valor_lite.semantic_segmentation import Bitmask, DataLoader, Segmentation


def generate_segmentation(
    uid: str,
    height: int,
    width: int,
    labels: list[str | None],
    proba: list[float],
) -> Segmentation:
    """
    Generates a list of segmentation annotations.

    Parameters
    ----------
    height : int
        The height of the bitmask.
    width : int
        The width of the bitmask.
    labels : list[str | None]
        A list of labels with None representing background.
    proba : list[float]
        A list of probabilities for each label that sum to 1.0. Should be given in increments of 0.01.
    Returns
    -------
    Segmenation
        A generated semantic segmenatation annotation.
    """
    if len(labels) != len(proba):
        raise ValueError("Labels and probabilities should be the same length.")

    probabilities = np.array(proba, dtype=np.float64)
    if not np.isclose(probabilities.sum(), 1.0).all():
        raise ValueError("Probabilities should sum to 1.0.")

    weights = (probabilities * 100.0).astype(np.int32)

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


def generate_cache():
    pass


if __name__ == "__main__":

    seg = generate_segmentation(
        uid="uid",
        height=2,
        width=2,
        labels=["a", "b", "c", None],
        proba=[0.25, 0.25, 0.25, 0.25],
    )

    loader = DataLoader()
    loader.add_data([seg])

    print(loader.matrices)

    evaluator = loader.finalize()

    print(evaluator._confusion_matrices)
