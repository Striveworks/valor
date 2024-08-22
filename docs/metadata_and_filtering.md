# Metadata and Filtering

## Metadata

Valor offers rich support for attaching metadata to almost any object, which can then be used to filter, group, and organize objects in Valor.

The metadata types supported are:

- simple data types (strings, numerics, boolean)
- datetimes (via `datetime.datetime`, `datetime.date`, `datetime.time`, and `datetime.timedelta` in the Valor client)
- geometries and geographies (via GeoJSON)

Metadata is added on object creation. For example, if you want to use metadata to organize models that come from training run checkpoints, this may look like:

```python
run_name: str
ckpt: int

Model.create(name=f"{run_name}-ckpt{ckpt}", metadata={"run_name": run_name, "ckpt": ckpt})
```

or if a datum has an associated datetime of capture, that can be added in the creation stage:

```python
from datetime import datetime

Datum(uid=fname, metadata={"capture_day": datetime.datetime(day=1, month=1, year=2021)})
```

## Filtering

Valor supports filtering objects based on metadata or other attributes (such as labels or bounding boxes). One of the most important use cases of filtering is to define a subset of a dataset to evaluate a model on.

### Filtering by metadata

For example, using the above example where `capture_day` was added as metadata, one way to test model drift could be to evaluate the model over different time periods. Such a workflow may look like:

```python
import datetime

from valor.schemas import Filter, Or

...

before_filter = Filter(
    datums=(
        Datum.metadata["capture_day"] < d
    )
)
after_filter = Filter(
    datums=(
        Datum.metadata["capture_day"] > d
    )
)

# compare performance on data captured before and after 2020
d = datetime.datetime(day=5, month=10, year=2020)
eval1 = model.evaluate_classification(dset, filter_by=before_filter)
eval2 = model.evaluate_classification(dset, filter_by=after_filter)
```

### Filtering by geometric attributes

As an example for filtering by geometric attributes, consider evaluating an object detection model's performance on small objects, where we define small as being less than 500 square pixels in area. This can be achieved via:

```python
from valor.schemas import Filter

...

f = Filter(
    annotations=(
        valor.Annotation.bounding_box.area < 500
    )
)

dset.evaluate_detection(dset, filter_by=f)
```

### Filtering in queries

Filtering can also be used when querying for different objects. For example, taking the model section checkpoint example from above, we could query model checkpoints from a training run based on the checkpoint number greater than 100 by:

```python
from valor import client
from valor.schemas import Filter, And

run_name: str # run name to query for

f = Filter(
    models=And(
        Model.metadata["run_name"] == run_name,
        Model.metadata["ckpt"] > 100,
    )
)

client.get_models(f)
```
