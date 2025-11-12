from pathlib import Path

import pytest

from valor_lite.classification import Classification, Loader
from valor_lite.classification.shared import (
    generate_cache_path,
    generate_metadata_path,
)
from valor_lite.exceptions import EmptyCacheError


def test_loader_no_data(tmp_path: Path):
    loader = Loader.persistent(tmp_path)
    with pytest.raises(EmptyCacheError):
        loader.finalize()


def test_finalization_no_data(tmp_path: Path):
    loader = Loader.persistent(tmp_path)
    with pytest.raises(EmptyCacheError):
        loader.finalize()


def test_unmatched_ground_truths(
    tmp_path: Path,
    classifications_no_predictions: list[Classification],
):
    loader = Loader.persistent(tmp_path)

    with pytest.raises(ValueError) as e:
        loader.add_data(classifications_no_predictions)
    assert "Classifications must contain at least one prediction" in str(e)


def test_evaluator_deletion(
    tmp_path: Path, basic_classifications: list[Classification]
):
    loader = Loader.persistent(tmp_path)
    loader.add_data(basic_classifications)

    # check both caches exist
    assert tmp_path.exists()
    assert generate_cache_path(tmp_path).exists()
    assert generate_metadata_path(tmp_path).exists()


def test_add_data_metadata_handling(loader: Loader):
    loader.add_data(
        classifications=[
            Classification(
                uid="0",
                metadata={"datum_uid": "a"},
                groundtruth="dog",
                predictions=["dog"],
                scores=[1.0],
            ),
            Classification(
                uid="1",
                metadata={"datum_uid": "b"},
                groundtruth="dog",
                predictions=["dog"],
                scores=[1.0],
            ),
        ]
    )
    loader._writer.flush()
    reader = loader._writer.to_reader()

    datum_uids = set()
    for tbl in reader.iterate_tables():
        assert set(tbl.column_names) == {
            "datum_id",
            "datum_uid",
            "gt_label",
            "gt_label_id",
            "match",
            "pd_label",
            "pd_label_id",
            "pd_score",
            "pd_winner",
            "test",
        }
        for uid in tbl["datum_uid"].to_pylist():
            datum_uids.add(uid)
    assert datum_uids == {"0", "1"}
