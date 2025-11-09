import numpy as np
import pyarrow as pa
from tqdm import tqdm

from valor_lite.cache.ephemeral import MemoryCacheWriter
from valor_lite.cache.persistent import FileCacheWriter
from valor_lite.classification.annotation import Classification
from valor_lite.classification.evaluator import Builder


class Loader(Builder):
    def __init__(
        self,
        writer: MemoryCacheWriter | FileCacheWriter,
        roc_curve_writer: MemoryCacheWriter | FileCacheWriter,
        intermediate_writer: MemoryCacheWriter | FileCacheWriter,
        metadata_fields: list[tuple[str, pa.DataType]] | None = None,
    ):
        super().__init__(
            writer=writer,
            roc_curve_writer=roc_curve_writer,
            intermediate_writer=intermediate_writer,
            metadata_fields=metadata_fields,
        )

        # internal state
        self._labels: dict[str, int] = {}
        self._index_to_label: dict[int, str] = {}
        self._datum_count = 0

    def _add_label(self, value: str) -> int:
        idx = self._labels.get(value, None)
        if idx is None:
            idx = len(self._labels)
            self._labels[value] = idx
            self._index_to_label[idx] = value
        return idx

    def add_data(
        self,
        classifications: list[Classification],
        show_progress: bool = False,
    ):
        """
        Adds classifications to the cache.

        Parameters
        ----------
        classifications : list[Classification]
            A list of Classification objects.
        show_progress : bool, default=False
            Toggle for tqdm progress bar.
        """

        disable_tqdm = not show_progress
        for classification in tqdm(classifications, disable=disable_tqdm):
            if len(classification.predictions) == 0:
                raise ValueError(
                    "Classifications must contain at least one prediction."
                )

            # prepare metadata
            datum_metadata = (
                classification.metadata if classification.metadata else {}
            )

            # write to cache
            rows = []
            gidx = self._add_label(classification.groundtruth)
            max_score_idx = np.argmax(np.array(classification.scores))
            for idx, (plabel, score) in enumerate(
                zip(classification.predictions, classification.scores)
            ):
                pidx = self._add_label(plabel)
                rows.append(
                    {
                        # datum
                        "datum_uid": classification.uid,
                        "datum_id": self._datum_count,
                        **datum_metadata,
                        # groundtruth
                        "gt_label": classification.groundtruth,
                        "gt_label_id": gidx,
                        # prediction
                        "pd_label": plabel,
                        "pd_label_id": pidx,
                        # pair
                        "score": float(score),
                        "winner": max_score_idx == idx,
                        "match": (gidx == pidx) and pidx >= 0,
                    }
                )
            self._writer.write_rows(rows)

            # update datum count
            self._datum_count += 1
