import json
from pathlib import Path

import numpy as np
import pyarrow as pa
from numpy.typing import NDArray
from tqdm import tqdm

from valor_lite.cache import (
    CacheWriter,
    DataType,
    convert_type_mapping_to_schema,
)
from valor_lite.exceptions import EmptyEvaluatorError
from valor_lite.semantic_segmentation.annotation import Bitmask, Segmentation
from valor_lite.semantic_segmentation.computation import (
    compute_intermediate_confusion_matrices,
)
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
        self._detailed_path = self._path / "detailed"
        self._ranked_path = self._path / "ranked"
        self._metadata_path = self._path / "metadata.json"

        self._labels = {}
        self._datum_count = 0
        self._groundtruth_pixel_count = 0
        self._prediction_pixel_count = 0
        self._total_pixel_count = 0

        with open(self._metadata_path, "w") as f:
            types = {
                "datum": datum_metadata_types,
                "groundtruth": groundtruth_metadata_types,
                "prediction": prediction_metadata_types,
            }
            json.dump(types, f, indent=2)

        datum_metadata_schema = convert_type_mapping_to_schema(datum_metadata_types)
        groundtruth_metadata_schema = convert_type_mapping_to_schema(groundtruth_metadata_types)
        prediction_metadata_schema = convert_type_mapping_to_schema(prediction_metadata_types)

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
            where=self._detailed_path,
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
        return idx

    def _write(
        self,
        datum_id: str,
        confusion_matrix: NDArray[np.int64],
    ):
        pass

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
            # update datum cache
            self._datum_count += 1
            self._groundtruth_count +=

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

            mat = compute_intermediate_confusion_matrices(
                groundtruths=combined_groundtruths,
                predictions=combined_predictions,
                groundtruth_labels=groundtruth_labels,
                prediction_labels=prediction_labels,
                n_labels=len(self._evaluator.index_to_label),
            )

    def finalize(
        self,
        rows_per_file: int | None = None,
        compression: str | None = None,
        write_batch_size: int | None = None,
        read_batch_size: int = 1000,
    ):
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

        evaluator = Evaluator(
            directory=self._directory,
            name=self._name,
        )
        evaluator.rank(
            where=self._ranked_path,
            rows_per_file=rows_per_file,
            compression=compression,
            write_batch_size=write_batch_size,
            read_batch_size=read_batch_size,
        )
        return evaluator

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
        for fragment in evaluator.detailed.get_fragments():
            tbl = fragment.to_table(filter=filter_expr.datums)

            columns = (
                "datum_id",
                "gt_id",
                "pd_id",
                "gt_label_id",
                "pd_label_id",
                "iou",
                "score",
            )
            pairs = np.column_stack([tbl[col].to_numpy() for col in columns])

            n_pairs = pairs.shape[0]
            gt_ids = pairs[:, (0, 1)].astype(np.int64)
            pd_ids = pairs[:, (0, 2)].astype(np.int64)

            mask_valid_gt = np.zeros(n_pairs, dtype=np.bool_)
            mask_valid_pd = np.zeros(n_pairs, dtype=np.bool_)

            if filter_expr.groundtruths is not None:
                gt_tbl = tbl.filter(filter_expr.groundtruths)
                gt_pairs = np.column_stack(
                    [gt_tbl[col].to_numpy() for col in ("datum_id", "gt_id")]
                ).astype(np.int64)
                for gt in np.unique(gt_pairs, axis=0):
                    mask_valid_gt |= (gt_ids == gt).all(axis=1)

            if filter_expr.predictions is not None:
                pd_tbl = tbl.filter(filter_expr.predictions)
                pd_pairs = np.column_stack(
                    [pd_tbl[col].to_numpy() for col in ("datum_id", "pd_id")]
                ).astype(np.int64)
                for pd in np.unique(pd_pairs, axis=0):
                    mask_valid_pd |= (pd_ids == pd).all(axis=1)

            mask_valid = mask_valid_gt | mask_valid_pd
            mask_valid_gt &= mask_valid
            mask_valid_pd &= mask_valid

            pairs[np.ix_(~mask_valid_gt, (1, 3))] = -1.0  # type: ignore - numpy ix_
            pairs[np.ix_(~mask_valid_pd, (2, 4, 6))] = -1.0  # type: ignore - numpy ix_
            pairs[~mask_valid_pd | ~mask_valid_gt, 5] = 0.0

            for idx, col in enumerate(columns):
                tbl = tbl.set_column(
                    tbl.schema.names.index(col), col, pa.array(pairs[:, idx])
                )

            mask_invalid = ~mask_valid | (pairs[:, (1, 2)] < 0).all(axis=1)
            filtered_tbl = tbl.filter(pa.array(~mask_invalid))
            loader._cache.write_table(filtered_tbl)

        loader._cache.flush()
        if loader._cache.dataset.count_rows() == 0:
            raise EmptyCacheError()

        evaluator = Evaluator(
            directory=loader._directory,
            name=loader._name,
            labels_override=evaluator._index_to_label,
        )
        evaluator.rank(where=loader._ranked_path)
        return evaluator