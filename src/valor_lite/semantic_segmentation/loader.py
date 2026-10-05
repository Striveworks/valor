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
        """Cache observed pixel pairs, reconciling foreground labels by name.

        Pixel zero is background. Positive pixel i names labels[i - 1].
        Background uses the existing cache ID -1 and matrix row/column zero.
        Declared labels with no pixels are preserved in zero-count rows.
        Their metrics are zero when they have no support, and they participate
        in mean IoU. Input arrays are never modified.
        """
        for segmentation in tqdm(segmentations, disable=not show_progress):
            local_to_global = np.array(
                [
                    -1,
                    *[self._add_label(label) for label in segmentation.labels],
                ],
                dtype=np.int64,
            )
            local_labels = [None, *segmentation.labels]
            gt_ids, pd_ids, counts = compute_intermediates(
                groundtruths=segmentation.groundtruths,
                predictions=segmentation.predictions,
                n_labels=len(segmentation.labels) + 1,
            )
            observed = set(gt_ids.tolist()) | set(pd_ids.tolist())
            absent_pairs = (
                (idx, idx, 0)
                for idx in range(1, len(segmentation.labels) + 1)
                if idx not in observed
            )
            rows = [
                {
                    **(segmentation.metadata or {}),
                    "datum_uid": segmentation.uid,
                    "datum_id": self._datum_count,
                    "gt_label": local_labels[int(gt)],
                    "gt_label_id": local_to_global[gt],
                    "pd_label": local_labels[int(pd)],
                    "pd_label_id": local_to_global[pd],
                    "count": count,
                }
                for gt, pd, count in chain(
                    zip(gt_ids, pd_ids, counts), absent_pairs
                )
            ]
            self._writer.write_rows(rows)
            self._datum_count += 1
