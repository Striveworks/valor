import numpy as np
from pyarrow import DataType
from tqdm import tqdm

from valor_lite.cache import FileCacheWriter, MemoryCacheWriter
from valor_lite.semantic_segmentation.annotation import Segmentation
from valor_lite.semantic_segmentation.computation import compute_intermediates
from valor_lite.semantic_segmentation.evaluator import Builder


class Loader(Builder):
    def __init__(
        self,
        writer: MemoryCacheWriter | FileCacheWriter,
        metadata_fields: list[tuple[str, DataType]] | None = None,
    ):
        super().__init__(
            writer=writer,
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
        segmentations: list[Segmentation],
        show_progress: bool = False,
    ):
        """
        Adds segmentations to the cache.

        Parameters
        ----------
        segmentations : list[Segmentation]
            A list of Segmentation objects.
        show_progress : bool, default=False
            Toggle for tqdm progress bar.
        """

        disable_tqdm = not show_progress
        for segmentation in tqdm(segmentations, disable=disable_tqdm):

            groundtruth_labels = -1 * np.ones(
                len(segmentation.groundtruths), dtype=np.int64
            )
            for idx, groundtruth in enumerate(segmentation.groundtruths):
                label_idx = self._add_label(groundtruth.label)
                groundtruth_labels[idx] = label_idx

            prediction_labels = -1 * np.ones(
                len(segmentation.predictions), dtype=np.int64
            )
            for idx, prediction in enumerate(segmentation.predictions):
                label_idx = self._add_label(prediction.label)
                prediction_labels[idx] = label_idx

            if segmentation.groundtruths:
                combined_groundtruths = np.stack(
                    [
                        groundtruth.mask.flatten()
                        for groundtruth in segmentation.groundtruths
                    ],
                    axis=0,
                )
            else:
                combined_groundtruths = np.zeros(
                    (1, segmentation.shape[0] * segmentation.shape[1]),
                    dtype=np.bool_,
                )

            if segmentation.predictions:
                combined_predictions = np.stack(
                    [
                        prediction.mask.flatten()
                        for prediction in segmentation.predictions
                    ],
                    axis=0,
                )
            else:
                combined_predictions = np.zeros(
                    (1, segmentation.shape[0] * segmentation.shape[1]),
                    dtype=np.bool_,
                )

            n_labels = len(self._labels)
            counts = compute_intermediates(
                groundtruths=combined_groundtruths,
                predictions=combined_predictions,
                groundtruth_labels=groundtruth_labels,
                prediction_labels=prediction_labels,
                n_labels=n_labels,
            )

            # prepare metadata
            datum_metadata = (
                segmentation.metadata if segmentation.metadata else {}
            )
            gt_metadata = {
                self._labels[gt.label]: gt.metadata
                for gt in segmentation.groundtruths
                if gt.metadata
            }
            pd_metadata = {
                self._labels[pd.label]: pd.metadata
                for pd in segmentation.predictions
                if pd.metadata
            }

            # cache formatting
            rows = []
            for idx in range(n_labels):
                label = self._index_to_label[idx]
                for pidx in range(n_labels):
                    # write non-zero intersections to cache
                    if counts[idx + 1, pidx + 1] > 0:
                        plabel = self._index_to_label[pidx]
                        rows.append(
                            {
                                # datum
                                "datum_uid": segmentation.uid,
                                "datum_id": self._datum_count,
                                **datum_metadata,
                                # groundtruth
                                "gt_label": label,
                                "gt_label_id": idx,
                                **gt_metadata.get(idx, {}),
                                # prediction
                                "pd_label": plabel,
                                "pd_label_id": pidx,
                                **pd_metadata.get(pidx, {}),
                                # pair
                                "count": counts[idx + 1, pidx + 1],
                            }
                        )
                # write all unmatched to preserve labels
                rows.extend(
                    [
                        {
                            # datum
                            "datum_uid": segmentation.uid,
                            "datum_id": self._datum_count,
                            **datum_metadata,
                            # groundtruth
                            "gt_label": label,
                            "gt_label_id": idx,
                            **gt_metadata.get(idx, {}),
                            # prediction
                            "pd_label": None,
                            "pd_label_id": -1,
                            # pair
                            "count": counts[idx + 1, 0],
                        },
                        {
                            # datum
                            "datum_uid": segmentation.uid,
                            "datum_id": self._datum_count,
                            **datum_metadata,
                            # groundtruth
                            "gt_label": None,
                            "gt_label_id": -1,
                            # prediction
                            "pd_label": label,
                            "pd_label_id": idx,
                            **pd_metadata.get(idx, {}),
                            # pair
                            "count": counts[0, idx + 1],
                        },
                    ]
                )
            rows.append(
                {
                    # datum
                    "datum_uid": segmentation.uid,
                    "datum_id": self._datum_count,
                    **datum_metadata,
                    # groundtruth
                    "gt_label": None,
                    "gt_label_id": -1,
                    # prediction
                    "pd_label": None,
                    "pd_label_id": -1,
                    # pair
                    "count": counts[0, 0],
                }
            )
            self._writer.write_rows(rows)

            # update datum count
            self._datum_count += 1
