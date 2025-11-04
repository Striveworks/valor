import json
from pathlib import Path

import numpy as np
from pyarrow import DataType
from tqdm import tqdm

from valor_lite.cache import FileCacheWriter, MemoryCacheWriter
from valor_lite.exceptions import EmptyCacheError
from valor_lite.semantic_segmentation.annotation import Segmentation
from valor_lite.semantic_segmentation.computation import compute_intermediates
from valor_lite.semantic_segmentation.evaluator import Evaluator
from valor_lite.semantic_segmentation.shared import Base


class Loader(Base):
    def __init__(
        self,
        writer: MemoryCacheWriter | FileCacheWriter,
        datum_metadata_fields: list[tuple[str, DataType]] | None = None,
        groundtruth_metadata_fields: list[tuple[str, DataType]] | None = None,
        prediction_metadata_fields: list[tuple[str, DataType]] | None = None,
        path: str | Path | None = None,
    ):
        self._path = Path(path) if path else None
        self._writer = writer
        self._datum_metadata_fields = datum_metadata_fields
        self._groundtruth_metadata_fields = groundtruth_metadata_fields
        self._prediction_metadata_fields = prediction_metadata_fields

        # internal state
        self._labels: dict[str, int] = {}
        self._index_to_label: dict[int, str] = {}
        self._datum_count = 0

    @classmethod
    def in_memory(
        cls,
        batch_size: int = 10_000,
        datum_metadata_fields: list[tuple[str, DataType]] | None = None,
        groundtruth_metadata_fields: list[tuple[str, DataType]] | None = None,
        prediction_metadata_fields: list[tuple[str, DataType]] | None = None,
    ):
        """
        Create an in-memory evaluator cache.

        Parameters
        ----------
        batch_size : int, default=10_000
            The target number of rows to buffer before writing to the cache. Defaults to 10_000.
        datum_metadata_fields : list[tuple[str, DataType]], optional
            Optional datum metadata field definition.
        groundtruth_metadata_fields : list[tuple[str, DataType]], optional
            Optional ground truth annotation metadata field definition.
        prediction_metadata_fields : list[tuple[str, DataType]], optional
            Optional prediction metadata field definition.
        """
        # create cache
        writer = MemoryCacheWriter.create(
            schema=cls._generate_schema(
                datum_metadata_fields=datum_metadata_fields,
                groundtruth_metadata_fields=groundtruth_metadata_fields,
                prediction_metadata_fields=prediction_metadata_fields,
            ),
            batch_size=batch_size,
        )
        return cls(
            writer=writer,
            datum_metadata_fields=datum_metadata_fields,
            groundtruth_metadata_fields=groundtruth_metadata_fields,
            prediction_metadata_fields=prediction_metadata_fields,
        )

    @classmethod
    def persistent(
        cls,
        path: str | Path,
        batch_size: int = 10_000,
        rows_per_file: int = 100_000,
        compression: str = "snappy",
        datum_metadata_fields: list[tuple[str, DataType]] | None = None,
        groundtruth_metadata_fields: list[tuple[str, DataType]] | None = None,
        prediction_metadata_fields: list[tuple[str, DataType]] | None = None,
        delete_if_exists: bool = False,
    ):
        """
        Create a persistent file-based evaluator cache.

        Parameters
        ----------
        path : str | Path
            Where to store the file-based cache.
        batch_size : int, default=10_000
            The target number of rows to buffer before writing to the cache. Defaults to 10_000.
        rows_per_file : int, default=100_000
            The target number of rows to store per cache file. Defaults to 100_000.
        compression : str, default="snappy"
            The compression methods used when writing cache files.
        datum_metadata_fields : list[tuple[str, DataType]], optional
            Optional datum metadata field definition.
        groundtruth_metadata_fields : list[tuple[str, DataType]], optional
            Optional ground truth annotation metadata field definition.
        prediction_metadata_fields : list[tuple[str, DataType]], optional
            Optional prediction metadata field definition.
        delete_if_exists : bool, default=False
            Option to delete any pre-exisiting cache at the given path.
        """
        path = Path(path)
        if delete_if_exists:
            cls.delete_at_path(path)

        # create cache
        cache = FileCacheWriter.create(
            path=cls._generate_cache_path(path),
            schema=cls._generate_schema(
                datum_metadata_fields=datum_metadata_fields,
                groundtruth_metadata_fields=groundtruth_metadata_fields,
                prediction_metadata_fields=prediction_metadata_fields,
            ),
            batch_size=batch_size,
            rows_per_file=rows_per_file,
            compression=compression,
        )

        # write metadata
        metadata_path = cls._generate_metadata_path(path)
        with open(metadata_path, "w") as f:
            encoded_types = cls._encode_metadata_fields(
                datum_metadata_fields=datum_metadata_fields,
                groundtruth_metadata_fields=groundtruth_metadata_fields,
                prediction_metadata_fields=prediction_metadata_fields,
            )
            json.dump(encoded_types, f, indent=2)

        return cls(
            path=path,
            writer=cache,
            datum_metadata_fields=datum_metadata_fields,
            groundtruth_metadata_fields=groundtruth_metadata_fields,
            prediction_metadata_fields=prediction_metadata_fields,
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
            self._writer.write_rows(rows)

            # update datum count
            self._datum_count += 1

    def finalize(
        self,
        index_to_label_override: dict[int, str] | None = None,
    ):
        """
        Performs data finalization and some preprocessing steps.

        Parameters
        ----------
        index_to_label_override : dict[int, str], optional
            Pre-configures label mapping. Used when operating over filtered subsets.

        Returns
        -------
        Evaluator
            A ready-to-use evaluator object.
        """
        self._writer.flush()
        if self._writer.count_rows() == 0:
            raise EmptyCacheError()

        reader = self._writer.to_reader()

        # build evaluator meta
        (
            index_to_label,
            confusion_matrix,
            info,
        ) = self._generate_meta(reader, index_to_label_override)
        info.datum_metadata_fields = self._datum_metadata_fields
        info.groundtruth_metadata_fields = self._groundtruth_metadata_fields
        info.prediction_metadata_fields = self._prediction_metadata_fields

        return Evaluator(
            reader=reader,
            info=info,
            index_to_label=index_to_label,
            confusion_matrix=confusion_matrix,
            path=self._path,
        )

    def delete(self):
        """Delete any cached files."""
        if self._path and self._path.exists():
            self.delete_at_path(self._path)
