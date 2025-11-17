import numpy as np
import pytest

from valor_lite.exceptions import EmptyCacheError
from valor_lite.semantic_segmentation import Bitmask, Loader, Segmentation


def test_no_data(loader: Loader):
    with pytest.raises(EmptyCacheError):
        loader.finalize()


def test_empty_input(loader: Loader):
    loader.add_data(
        segmentations=[
            Segmentation(
                uid="0", groundtruths=[], predictions=[], shape=(10, 10)
            ),
        ]
    )
    evaluator = loader.finalize()
    assert evaluator.info.number_of_datums == 1
    assert evaluator.info.number_of_pixels == 100
    assert evaluator.info.number_of_groundtruth_pixels == 0
    assert evaluator.info.number_of_prediction_pixels == 0


def test_add_data_metadata_handling(loader: Loader):
    loader.add_data(
        segmentations=[
            Segmentation(
                uid="0",
                metadata={"datum_uid": "c"},
                groundtruths=[
                    Bitmask(
                        mask=np.ones((10, 10), dtype=np.bool_),
                        label="dog",
                        metadata={"datum_uid": "b"},
                    )
                ],
                predictions=[
                    Bitmask(
                        mask=np.ones((10, 10), dtype=np.bool_),
                        label="dog",
                        metadata={"datum_uid": "c"},
                    )
                ],
                shape=(10, 10),
            ),
            Segmentation(
                uid="1",
                metadata={"datum_uid": "c"},
                groundtruths=[
                    Bitmask(
                        mask=np.ones((10, 10), dtype=np.bool_),
                        label="dog",
                        metadata={"datum_uid": "b"},
                    )
                ],
                predictions=[
                    Bitmask(
                        mask=np.ones((10, 10), dtype=np.bool_),
                        label="dog",
                        metadata={"datum_uid": "c"},
                    )
                ],
                shape=(10, 10),
            ),
        ]
    )
    loader._writer.flush()
    reader = loader._writer.to_reader()

    datum_uids = set()
    for tbl in reader.iterate_tables():
        assert set(tbl.column_names) == {
            "datum_uid",
            "datum_id",
            "gt_label",
            "gt_label_id",
            "pd_label",
            "pd_label_id",
            "count",
            "gt_xmin",
            "pd_xmin",
        }
        for uid in tbl["datum_uid"].to_pylist():
            datum_uids.add(uid)
    assert datum_uids == {"0", "1"}
