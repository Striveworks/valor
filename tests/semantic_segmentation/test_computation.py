from collections import Counter

import numpy as np

from valor_lite.semantic_segmentation.computation import compute_intermediates


def test_sparse_pair_counts_match_reference():
    rng = np.random.default_rng(42)
    for n_labels in [1, 2, 20, 257, 65536]:
        gt = rng.integers(n_labels, size=(23, 31), dtype=np.uint16)
        pd = rng.integers(n_labels, size=gt.shape, dtype=np.uint16)
        gt[0, 0], pd[0, 0] = n_labels - 1, n_labels - 1
        reference = Counter(zip(gt.ravel().tolist(), pd.ravel().tolist()))
        gt_ids, pd_ids, counts = compute_intermediates(gt, pd, n_labels)
        assert counts.dtype == np.uint64
        assert gt_ids.dtype == pd_ids.dtype == np.int64
        assert (
            dict(zip(zip(gt_ids.tolist(), pd_ids.tolist()), counts.tolist()))
            == reference
        )
        assert int(counts.sum()) == gt.size
