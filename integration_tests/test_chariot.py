import tempfile

import pytest


def test_idk():
    assert 1 == 3


chariot = pytest.importorskip("chariot")
print(chariot)


# @TODO Write test using the chariot datasets test as reference
# https://github.com/Striveworks/chariot/blob/main/py/libs/sdk/integration_tests/test_datasets_cv.py


def test_chariot_ds_to_velour_ds():

    from chariot.client import connect
    from chariot.datasets.upload import upload_annotated_data

    connect(host="https://production.chariot.striveworks.us")

    # Create a temporary file
    with tempfile.NamedTemporaryFile(mode="w+b") as f:

        # Image Classification
        jsonl = (
            '{"path": "'
            + str(f.name)
            + '", "annotations": [{"class_label": "dog"}]}\n{"path": "'
            + str(f.name)
            + '", "annotations": [{"class_label": "cat"}]}'
        )

        # Write to tempfile
        f.write(jsonl.encode("utf-8"))
        f.flush()
        f.seek(0)

        # Upload to Chariot
        upload_annotated_data(
            name="Integration Test",
            description="Integration test.",
            project_name="OnBoarding",
            train_annotation_file=f.name,
        )
