import json
from pathlib import Path

import numpy as np
import pyarrow as pa
from tqdm import tqdm

from valor_lite.cache import (
    CacheWriter,
    DataType,
    convert_type_mapping_to_schema,
)
from valor_lite.exceptions import EmptyCacheError
from valor_lite.semantic_segmentation.annotation import Segmentation
from valor_lite.semantic_segmentation.computation import compute_intermediates
from valor_lite.semantic_segmentation.evaluator import Evaluator, Filter


class Loader:
    def __init__(
        self,
        directory: str | Path = ".valor",
        name: str = "default",
        batch_size: int = 10_000,
        rows_per_file: int = 100_000,
        compression: str = "snappy",
        datum_metadata_types: dict[str, DataType] | None = None,
        groundtruth_metadata_types: dict[str, DataType] | None = None,
        prediction_metadata_types: dict[str, DataType] | None = None,
    ):
        self._directory = Path(directory)
        self._name = name

        self._path = self._directory / self._name
        self._path.mkdir(parents=True, exist_ok=True)
        self._cache_path = self._path / "counts"
        self._metadata_path = self._path / "metadata.json"

        self._labels: dict[str, int] = {}
        self._index_to_label: dict[int, str] = {}
        self._datum_count = 0

        with open(self._metadata_path, "w") as f:
            types = {
                "datum": datum_metadata_types,
                "groundtruth": groundtruth_metadata_types,
                "prediction": prediction_metadata_types,
            }
            json.dump(types, f, indent=2)

        datum_metadata_schema = convert_type_mapping_to_schema(
            datum_metadata_types
        )
        groundtruth_metadata_schema = convert_type_mapping_to_schema(
            groundtruth_metadata_types
        )
        prediction_metadata_schema = convert_type_mapping_to_schema(
            prediction_metadata_types
        )

        self._null_gt_metadata = {
            s[0]: None for s in groundtruth_metadata_schema
        }
        self._null_pd_metadata = {
            s[0]: None for s in prediction_metadata_schema
        }

        schema = pa.schema(
            [
                ("datum_uid", pa.string()),
                ("datum_id", pa.int64()),
                *datum_metadata_schema,
                # groundtruth
                ("gt_label", pa.string()),
                ("gt_label_id", pa.int64()),
                *groundtruth_metadata_schema,
                # prediction
                ("pd_label", pa.string()),
                ("pd_label_id", pa.int64()),
                *prediction_metadata_schema,
                # pair
                ("count", pa.uint64()),
            ]
        )
        self._cache = CacheWriter(
            where=self._cache_path,
            schema=schema,
            batch_size=batch_size,
            rows_per_file=rows_per_file,
            compression=compression,
        )

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
            self._cache.write_rows(rows)

            # update datum cache
            self._datum_count += 1

    def finalize(self):
        """
        Performs data finalization and some preprocessing steps.

        Returns
        -------
        Evaluator
            A ready-to-use evaluator object.
        """
        self._cache.flush()
        if self._cache.dataset.count_rows() == 0:
            raise EmptyCacheError()

        return Evaluator(
            directory=self._directory,
            name=self._name,
        )

    @classmethod
    def filter(
        cls,
        directory: str | Path,
        name: str,
        evaluator: Evaluator,
        filter_expr: Filter,
    ) -> Evaluator:
        loader = cls(
            directory=directory,
            name=name,
            batch_size=evaluator._detailed_batch_size,
            rows_per_file=evaluator._detailed_rows_per_file,
            compression=evaluator._detailed_compression,
            datum_metadata_types=evaluator.info.datum_metadata_types,
            groundtruth_metadata_types=evaluator.info.groundtruth_metadata_types,
            prediction_metadata_types=evaluator.info.prediction_metadata_types,
        )
        for fragment in evaluator.dataset.get_fragments():
            tbl = fragment.to_table(filter=filter_expr.datums)

            columns = (
                "datum_id",
                "gt_label_id",
                "pd_label_id",
            )
            pairs = np.column_stack([tbl[col].to_numpy() for col in columns])

            n_pairs = pairs.shape[0]
            gt_ids = pairs[:, (0, 1)].astype(np.int64)
            pd_ids = pairs[:, (0, 2)].astype(np.int64)

            if filter_expr.groundtruths is not None:
                mask_valid_gt = np.zeros(n_pairs, dtype=np.bool_)
                gt_tbl = tbl.filter(filter_expr.groundtruths)
                gt_pairs = np.column_stack(
                    [
                        gt_tbl[col].to_numpy()
                        for col in ("datum_id", "gt_label_id")
                    ]
                ).astype(np.int64)
                for gt in np.unique(gt_pairs, axis=0):
                    mask_valid_gt |= (gt_ids == gt).all(axis=1)
            else:
                mask_valid_gt = np.ones(n_pairs, dtype=np.bool_)

            if filter_expr.predictions is not None:
                mask_valid_pd = np.zeros(n_pairs, dtype=np.bool_)
                pd_tbl = tbl.filter(filter_expr.predictions)
                pd_pairs = np.column_stack(
                    [
                        pd_tbl[col].to_numpy()
                        for col in ("datum_id", "pd_label_id")
                    ]
                ).astype(np.int64)
                for pd in np.unique(pd_pairs, axis=0):
                    mask_valid_pd |= (pd_ids == pd).all(axis=1)
            else:
                mask_valid_pd = np.ones(n_pairs, dtype=np.bool_)

            mask_valid = mask_valid_gt | mask_valid_pd
            mask_valid_gt &= mask_valid
            mask_valid_pd &= mask_valid

            pairs[~mask_valid_gt, 1] = -1
            pairs[~mask_valid_pd, 2] = -1

            for idx, col in enumerate(columns):
                tbl = tbl.set_column(
                    tbl.schema.names.index(col), col, pa.array(pairs[:, idx])
                )
            loader._cache.write_table(tbl)

        loader._cache.flush()
        if loader._cache.dataset.count_rows() == 0:
            raise EmptyCacheError()

        evaluator = Evaluator(
            directory=loader._directory,
            name=loader._name,
            labels_override=evaluator._index_to_label,
        )
        return evaluator
