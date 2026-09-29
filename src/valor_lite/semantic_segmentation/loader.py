from itertools import chain

import numpy as np
import pyarrow as pa
from tqdm import tqdm

from valor_lite.cache import FileCacheWriter, MemoryCacheWriter
from valor_lite.semantic_segmentation.annotation import Segmentation
from valor_lite.semantic_segmentation.computation import compute_intermediates
from valor_lite.semantic_segmentation.evaluator import Builder


class Loader(Builder):
    def __init__(
        self,
        writer: MemoryCacheWriter | FileCacheWriter,
        metadata_fields: list[tuple[str, str | pa.DataType]] | None = None,
    ):
        super().__init__(writer=writer, metadata_fields=metadata_fields)
        self._labels: dict[str, int] = {}
        self._datum_count = 0

    def _add_label(self, value: str) -> int:
        idx = self._labels.get(value)
        if idx is None:
            idx = len(self._labels)
            self._labels[value] = idx
        return idx

    def add_data(
        self,
        segmentations: list[Segmentation],
        show_progress: bool = False,
    ):
        """Cache observed pixel pairs, reconciling local labels by name.

        Declared labels with no pixels are preserved in zero-count rows.
        Their metrics are zero when they have no support, and they participate
        in mean IoU. Input arrays are never modified.
        """
        for segmentation in tqdm(segmentations, disable=not show_progress):
            local_to_global = np.array(
                [self._add_label(label) for label in segmentation.labels],
                dtype=np.int64,
            )
            gt_ids, pd_ids, counts = compute_intermediates(
                groundtruths=segmentation.groundtruths,
                predictions=segmentation.predictions,
                n_labels=len(segmentation.labels),
            )
            gt_metadata = segmentation.groundtruth_metadata or {}
            pd_metadata = segmentation.prediction_metadata or {}
            observed = set(gt_ids.tolist()) | set(pd_ids.tolist())
            absent_pairs = (
                (idx, idx, 0)
                for idx in range(len(segmentation.labels))
                if idx not in observed
            )
            rows = [
                {
                    **(segmentation.metadata or {}),
                    **gt_metadata.get(int(gt), {}),
                    **pd_metadata.get(int(pd), {}),
                    "datum_uid": segmentation.uid,
                    "datum_id": self._datum_count,
                    "gt_label": segmentation.labels[int(gt)],
                    "gt_label_id": local_to_global[gt],
                    "pd_label": segmentation.labels[int(pd)],
                    "pd_label_id": local_to_global[pd],
                    "count": count,
                    "gt_valid": True,
                    "pd_valid": True,
                }
                for gt, pd, count in chain(
                    zip(gt_ids, pd_ids, counts), absent_pairs
                )
            ]
            self._writer.write_rows(rows)
            self._datum_count += 1
