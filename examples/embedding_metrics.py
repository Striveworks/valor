from collections import defaultdict
from dataclasses import dataclass
from typing import Callable

import numpy as np
import pandas as pd
from scipy.spatial.distance import cdist
from scipy.stats import cramervonmises_2samp, ks_2samp


@dataclass
class EmbeddingMatrix:
    """
    Maps scores from each query label to each reference label.
    """

    statistics: dict[str, dict[str, float]]
    pvalues: dict[str, dict[str, float]]


def compute_distances(
    queries: list[np.ndarray], references: list[np.ndarray]
) -> np.ndarray:
    """
    Computes the distance of all embeddings in the query set
    to all the embeddings in the reference set.

    Parameters
    ----------
    queries : list[np.ndarray]
        A list of query embeddings.
    references : list[np.ndarray]
        A list of reference embeddings.

    Returns
    -------
    np.ndarray
        An unordered 1-D array of distances.
    """
    return np.array([cdist(a, b) for a in queries for b in references])


def compute_self_distances(references: list[np.ndarray]) -> np.ndarray:
    """
    Computes the distance of all embeddings to other embeddings in the set.

    Parameters
    ----------
    references : list[np.ndarray]
        A list of reference embeddings.

    Returns
    -------
    np.ndarray
        An unordered 1-D array of distances.
    """
    return np.array(
        [
            cdist(a, b)
            for a_idx, a in enumerate(references)
            for b_idx, b in enumerate(references)
            if a_idx != b_idx
        ]
    )


def _compute_metrics(
    data: list[list[np.ndarray]],
    classes: list[str],
    func: Callable,
) -> EmbeddingMatrix:
    """
    Computes metrics using a selectable scipy.stats function.

    Parameters
    ----------
    data : list[list[np.ndarray]]
        A list of distances lists.
    classes : list[str]
        A list of labels mapped to the distance lists.
    func : Callable
        A scipy.stats function.

    Returns
    -------
    EmbeddingMatrix
        A metric containing a confusion matrix for both p-value and distance metric.
    """
    pvalues = defaultdict(dict)
    statistics = defaultdict(dict)

    def cast_and_round(x):
        x = float(x)
        return round(x, 3)

    for i, query in enumerate(data):
        for j, reference in enumerate(data):
            reference_distance = compute_self_distances(reference)
            if i == j:
                # split the set in two and measure how similarly distributed it is.
                split_idx = len(reference_distance) // 2
                metric = func(
                    reference_distance[:split_idx],
                    reference_distance[split_idx:],
                )
            else:
                query_distance = compute_distances(reference, query)
                metric = func(reference_distance, query_distance)

            label_i = classes[i]
            label_j = classes[j]
            pvalues[label_i][label_j] = cast_and_round(metric.pvalue)
            statistics[label_i][label_j] = cast_and_round(metric.statistic)

    return EmbeddingMatrix(pvalues=pvalues, statistics=statistics)


def compute_cvm(
    data: list[list[np.ndarray]], classes: list[str]
) -> EmbeddingMatrix:
    """
    Computes metrics using the Cramer-Von Mises Test.

    Parameters
    ----------
    data : list[list[np.ndarray]]
        A list of distances lists.
    classes : list[str]
        A list of labels mapped to the distance lists.

    Returns
    -------
    EmbeddingMatrix
        A metric containing a confusion matrix for both p-value and distance metric.
    """
    return _compute_metrics(data, classes, func=cramervonmises_2samp)


def compute_ks(
    data: list[list[np.ndarray]], classes: list[str]
) -> EmbeddingMatrix:
    """
    Computes metrics using the Kolmgorov-Smirnov Test.

    Parameters
    ----------
    data : list[list[np.ndarray]]
        A list of distances lists.
    classes : list[str]
        A list of labels mapped to the distance lists.

    Returns
    -------
    EmbeddingMatrix
        A metric containing a confusion matrix for both p-value and distance metric.
    """
    return _compute_metrics(data, classes, func=ks_2samp)


def create_dataframe(x: EmbeddingMatrix, labels: list[str]):
    col_ix = pd.MultiIndex.from_product([["Reference"], labels])
    row_ix = pd.MultiIndex.from_product([["Query"], labels])

    statistic_df = pd.DataFrame(x.statistics)
    statistic_df = statistic_df.set_index(row_ix)
    statistic_df.columns = col_ix

    pvalue_df = pd.DataFrame(x.pvalues)
    pvalue_df = pvalue_df.set_index(row_ix)
    pvalue_df.columns = col_ix

    return (statistic_df, pvalue_df)
