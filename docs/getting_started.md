# Local QuickStart

## Setting up the Backend

### Helm

```shell
helm repo add velour https://striveworks.github.io/velour-charts/
helm install velour velour/velour
# Velour should now be avaiable at velour.namespace.svc.local
```

### Docker

```shell
make dev-env
```

## Python Client

### Installation

```shell
pip install velour-client
```

### Import Dependencies

```py
from velour import Dataset, Model, Datum, Annotation, GroundTruth, Prediction, Label
from velour.client import Client
from velour.enums import TaskType
```

## Connect to the Client

The `velour.Client` class gives an object that is used to communicate with the `velour` backend.

> In the case that the host uses authentication, then the argument `access_token` should also be passed to `Client`.
```py
client = Client(HOST_URL)
```


The `Dataset` object uses a staticmethod for creation.

```py
dataset = Dataset.create(client, "myDataset")
```

`Datum` is a `CoreType` and is initialized.

```py
datum = Datum(uid="uid")
```

### Create `GroundTruth`
```py
# create groundtruth annotation
groundtruth_annotation = Annotation(
    task_type = TaskType.CLASSIFICATION,
    labels = [
        schemas.Label(key="class", value="dog"),
        schemas.Label(key="category", value="animal"),
    ]
)

# create prediction annotation
groundtruth_annotation = Annotation(
    task_type = TaskType.CLASSIFICATION,
    labels = [
        schemas.Label(key="class", value="dog"),
        schemas.Label(key="category", value="animal"),
    ]
)
```

```py
# create groundtruth
groundtruth = GroundTruth(
    datum=datum,
    annotations=[groundtruth_annotations],
)

#
```

## Create a Datum

```py
datum = Datum(uid="myDatum")
```

## Create a GroundTruth

```py
dataset.add_groundtruth(
    GroundTruth(
        datum=
    )
)
```

## Final Code

```py
from velour import Dataset, Model, Datum, Annotation, GroundTruth, Prediction, Label
from velour.client import Client
from velour.enums import TaskType

# connect to client
client = Client(HOST_URL)

# create dataset
dataset = Dataset.create(client, "myDataset")

# create datum
datum = Datum(uid="uid")

# create groundtruth annotation
groundtruth_annotation = Annotation(
    task_type = TaskType.CLASSIFICATION,
    labels = [
        schemas.Label(key="class", value="dog"),
        schemas.Label(key="category", value="animal"),
    ]
)

# create groundtruth
groundtruth = GroundTruth(
    datum=datum,
    annotations=[groundtruth_annotations],
)

# add groundtruth to dataset
dataset.add_groundtruth(groundtruth)

# prepare dataset for evaluation
dataset.finalize()

# create model
model = Model.create(client, "myModel")

# create prediction annotation
prediction_annotation = Annotation(
    task_type = TaskType.CLASSIFICATION,
    labels = [
        schemas.Label(key="class", value="dog", score=0.6),
        schemas.Label(key="class", value="car", score=0.4),
        schemas.Label(key="category", value="animal", score=0.8),
        schemas.Label(key="category", value="vehicle", score=0.2),
    ]
)

# create prediction
prediction = Prediction(
    model=model.name,
    datum=datum,
    annoations=[prediction_annotation],
)

# add prediction to model
model.add_prediction(prediction)

# prepare model for evaluation over dataset
model.finalize(dataset)

# run evaluation
evaluation = model.evaluate_classification(dataset)

# wait for completion
evaluation.wait_for_completion()

print(evaluation.metrics["metrics"])
```
