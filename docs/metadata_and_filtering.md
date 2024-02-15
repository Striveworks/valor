# Metadata and filtering

## Metadata

Velour offers rich support for attaching metadata to almost any object, which can then be used to filter, group, and organize objects in Velour.

The metadata types supported are:

- simple data types (strings, numerics)
- datetimes (via `datetime.datetime` in the Velour client)
- geometry and geographies (via GeoJSON)

Metadata is added on object creation. For example, if you want to use metadata to organize models that come from training run checkpoints. This may look like

```python
run_name: str
ckpt: int

Model.create(name=f"{run_name}-ckpt{ckpt}", metadata={"run_name": run_name, "ckpt": ckpt})
```

or if a datum has an associated datetime of capture, that can be added in the creation stage

```python
from datetime import datetime

Datum(uid=fname, metadata={"capture_day": datetime.datetime(day=1, month=1, year=2021)})
```

## Filtering

Velour supports filtering objects based off of metadata or other attributes (such as labels or bounding boxes). One of the most important use cases of filtering is to define a subset of a dataset to evaluate a model on.

### Filtering by metadata

For example, using the above example where `capture_day` was added as metadata, one way to to test model drift could be to evaluate the model over different time periods. Such a workflow may look like:

```python
import datetime

import Velour

model: Velour.Model # classification model
dset: Velour.Dataset # dataset to evaluate on

# compare performance on data captured before and after 2020
d = datetime.datetime(day=5, month=10, year=2020)
eval1 = model.evaluate_classification(dset, filter_by=[Datum.metadata["capture_day"] < d])
eval2 = model.evaluate_classification(dset, filter_by=[Datum.metadata["capture_day"] > d])
```

### Filtering by geometric attributes

As an example for filtering by geometric attributes, consider evaluating an object detection model's performance on small objects, where we define small as being less than 500 square pixels in area. This can be achieved via

```python
import Velour

model: Velour.Model # object detection model
dset: Velour.Dataset # dataset to evaluate on

dset.evaluate_detection(dset, filter_by=[Velour.Annotation.bounding_box.area < 500])
```

### Filtering in queries

Filtering can also be used when querying for different objects. For example, taking the model section checkpoint example from above, we could query model checkpoints from a training run based on the checkpoint number greater than 100 by

```python
from Velour import client

run_name: str # run name to query for

client.get_models([Model.metadata["run_name"] == run_name, Model.metadata["ckpt"] > 100])
```
