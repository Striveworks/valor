import json

import pytest
from valor_lite.classification import Classification, DataLoader


def test_no_data():
    loader = DataLoader()
    with pytest.raises(ValueError):
        loader.finalize()


def test_valor_integration():

    gt_json = '{"datum": {"uid": "2hVjLfwXSkbWmFhaISWOT2mHTbx", "metadata": {"width": {"type": "integer", "value": 224}, "height": {"type": "integer", "value": 224}}}, "annotations": [{"task_type": "classification", "metadata": {}, "labels": [{"key": "class_label", "value": "RED TAILED HAWK", "score": null}], "bounding_box": null, "polygon": null, "raster": null, "embedding": null}]}'
    pd_json = '{"datum": {"uid": "2hVjLfwXSkbWmFhaISWOT2mHTbx", "metadata": {"width": {"type": "integer", "value": 224}, "height": {"type": "integer", "value": 224}}}, "annotations": [{"task_type": "classification", "metadata": {}, "labels": [{"key": "class_label", "value": "AMERICAN AVOCET", "score": 1.7822019060531602e-07}, {"key": "class_label", "value": "AVADAVAT", "score": 0.0005651581450365484}, {"key": "class_label", "value": "BLACK TAIL CRAKE", "score": 0.00014690875832457095}, {"key": "class_label", "value": "BLUE MALKOHA", "score": 0.07310464978218079}, {"key": "class_label", "value": "CAMPO FLICKER", "score": 0.9117385149002075}, {"key": "class_label", "value": "CHUKAR PARTRIDGE", "score": 0.0014892773469910026}, {"key": "class_label", "value": "CRESTED KINGFISHER", "score": 0.012080634944140911}, {"key": "class_label", "value": "DOUBLE BRESTED CORMARANT", "score": 0.0004424828803166747}, {"key": "class_label", "value": "EMU", "score": 2.4317616407643072e-05}, {"key": "class_label", "value": "GREAT KISKADEE", "score": 0.00040792522486299276}], "bounding_box": null, "polygon": null, "raster": null, "embedding": null}]}'

    gt = json.loads(gt_json)
    pd = json.loads(pd_json)

    loader = DataLoader()
    loader.add_data_from_valor_dict([(gt, pd)])
    loader.finalize()

    assert loader._evaluator._detailed_pairs.shape == (10, 5)

    assert len(loader._evaluator.index_to_label) == 11
    assert loader._evaluator.n_datums == 1


def test_missing_predictions(
    classifications_no_predictions: list[Classification],
):
    loader = DataLoader()

    with pytest.raises(ValueError) as e:
        loader.add_data(classifications_no_predictions)
    assert "Classifications must contain at least one prediction" in str(e)
