# Getting Started

Velour is a centralized evaluation store which makes it easy to measure, explore, and rank model performance. For an overview of what Velour is and why it's important, please refer to our [high-level overview](index.md).

On this page, we'll describe how to get up and running with Velour.

## 1. Install Docker

As a first step, be sure your machine has Docker installed. [Click here](https://docs.docker.com/engine/install/) for basic installation instructions.


## 2. Clone the repo and open the directory

Choose a file in which to store Velour, then run:

```shell
git clone https://github.com/striveworks/velour
cd velour
```


## 3. Start services

There are multiple ways to start the Velour API service.

### a. Helm Chart

When deploying Velour on k8s via Helm, you can use our pre-built chart using the following commands:

```shell
helm repo add velour https://striveworks.github.io/velour-charts/
helm install velour velour/velour
# Velour should now be avaiable at velour.namespace.svc.local
```

### b. Docker

You can download the latest Velour image from `ghcr.io/striveworks/velour/velour-service`.

### c. Manual Deployment

If you would prefer to build your own image or want a debug console for the backend, please see the deployment instructions in ["Contributing to Velour"](contributing.md).

## 4. Use Velour

There's two ways to access Velour: by leveraging our Python client, or by calling our REST endpoints directly.

### 4a. Using the Python client

Let's walk-through a hypothetical example where we're trying to classify dogs and cats in a series of images. Note that all of the code below is pseudo-code for clarity; please see our ["Getting Started"](https://github.com/Striveworks/velour/blob/main/examples/getting_started.ipynb) notebook for a working example.

#### Install the client

To install the Python client, you can run:

```shell
pip install velour-client
```

#### Import dependencies

Import dependencies directly from the client module using:

```py
from velour import (
    Client,
    Dataset,
    Model,
    Datum,
    Annotation,
    GroundTruth,
    Prediction,
    Label,
)
from velour.schemas import (
    BoundingBox,
    Polygon,
    BasicPolygon,
    Point,
)
from velour.enums import TaskType
```

#### Connect to the Client

The `velour.Client` class gives an object that is used to communicate with the `velour` backend.

```py
client = Client("http://0.0.0.0:8000")
```

In the event that the host uses authentication, the argument `access_token` should also be passed to `Client`.

#### Pass your groundtruths into Velour

First, we define our `Dataset` object using `Dataset.create()`.

```py
dataset = Dataset(
    client=client,
    name="myDataset",
    metadata={        # optional, metadata can take `str`, `int`, `float` value types.
        "some_string": "hello_world",
        "some_number": 1234,
        "a_different_number": 1.234,
    },
    geospatial=None,  # optional, define a GeoJSON
)
```

Next, we add one or more `GroundTruths` to our `Dataset`. These objects help Velour understand "What is the correct classification for this particular image?".

```py

# create a groundtruth for a set of images that we know all depict a dog
dog_images = [
    {"path": "a/b/c/img3.png", "annotations": [{"class_label": "dog", "bbox": {"xmin": 16, "ymin": 130, "xmax": 70, "ymax": 150}}, {"class_label": "person", "bbox": {"xmin": 89, "ymin": 10, "xmax": 97, "ymax": 110}}]},
    {"path": "a/b/c/img4.png", "annotations": [{"class_label": "cat", "bbox": {"xmin": 500, "ymin": 220, "xmax": 530, "ymax": 260}}]},
    {"path": "a/b/c/img5.png", "annotations": []}
]

for image in dog_images:

    # each image will have its own Datum. this object will help us connect groundtruths and predictions when it's time for evaluation
    image.datum = Datum(
        uid=image.name # a unique ID for each image
        metadata={
            "path": element["path"]
        }
    )

    annotations = [
        Annotation(
            task_type=TaskType.DETECTION,
            labels=[Label(key="class_label", value=annotation["class_label"])],
            bounding_box=BoundingBox.from_extrema(
                xmin=annotation["bbox"]["xmin"],
                xmax=annotation["bbox"]["xmax"],
                ymin=annotation["bbox"]["ymin"],
                ymax=annotation["bbox"]["ymax"],
            )
        )
        for annotation in element["annotations"]
        if len(annotation) > 0
    ]

    groundtruth = GroundTruth(
        datum=image.datum,
        annotations=annotations,
    )

    # add it to your dataset
    dataset.add_groundtruth(groundtruth)

# now that we've added all our groundtruths, we can finalize our dataset for evaluation
dataset.finalize()
```

#### Pass your predictions into Velour

Now that we've passed several images of dogs into Velour, we need to pass in model predictions before we can evaluate whether those predictions were correct or not. To accomplish this task, we start by defining our `Model`:

```py
# create model
model = Model(
    client=client,
    name="myModel",
    metadata={
        "foo": "bar",
        "some_number": 4321,
    },
    geospatial=None,
)
```

Next, we tell Velour what our model predicted for each image by attaching `Predictions` to our `Model`:

```py
# populate a dictionary mapping Datum UIDs to datums for all of the datums in our dataset
datums_by_uid = {
    datum.uid: datum
    for datum in dataset.get_datums()
}

def create_prediction_from_object_detection_dict(element: dict, datums_by_uid:dict) -> Prediction:

    # get datum from dataset using filename
    uid=Path(element["path"]).stem
    datum = datums_by_uid[uid]

    # create Annotations
    annotations = [
        Annotation(
            task_type=TaskType.DETECTION,
            labels=[
                Label(key="class_label", value=label["class_label"], score=label["score"])
                for label in annotation["labels"]
            ],
            bounding_box=BoundingBox.from_extrema(
                xmin=annotation["bbox"]["xmin"],
                xmax=annotation["bbox"]["xmax"],
                ymin=annotation["bbox"]["ymin"],
                ymax=annotation["bbox"]["ymax"],
            )
        )
        for annotation in element["annotations"]
        if len(annotation) > 0
    ]

    # create and return Prediction
    return Prediction(
        datum=datum,
        annotations=annotations,
    )

object_detections = [
    {"path": "a/b/c/img3.png", "annotations": [
        {"labels": [{"class_label": "dog", "score": 0.8}, {"class_label": "cat", "score": 0.1}, {"class_label": "person", "score": 0.1}], "bbox": {"xmin": 16, "ymin": 130, "xmax": 70, "ymax": 150}},
        {"labels": [{"class_label": "dog", "score": 0.05}, {"class_label": "cat", "score": 0.05}, {"class_label": "person", "score": 0.9}], "bbox": {"xmin": 89, "ymin": 10, "xmax": 97, "ymax": 110}}
    ]},
    {"path": "a/b/c/img4.png", "annotations": [
        {"labels": [{"class_label": "dog", "score": 0.8}, {"class_label": "cat", "score": 0.1}, {"class_label": "person", "score": 0.1}], "bbox": {"xmin": 500, "ymin": 220, "xmax": 530, "ymax": 260}}
    ]},
    {"path": "a/b/c/img5.png", "annotations": []}
]

for element in object_detections:
    # create prediction
    prediction = create_prediction_from_object_detection_dict(element, datums_by_uid=datums_by_uid)

    # add prediction to model
    model.add_prediction(prediction)
```

#### Run your evaluation and print metrics

Now that both our `Dataset` and `Model` are finalized, we can evaluate how well our hypothetical model did at predicting whether or not each image contained a dog.

```py
# run evaluation
evaluation = model.evaluate_classification(
    dataset=dataset,
    filters=[
        Label.key == "animal" # with this filter, we're asking Velour to only evaluate how well our model predicted animals in each image
    ]
).wait_for_completion() # wait for the job to finish

# print our classification metrics
print(evaluation.metrics)
```

For more examples, please see our [sample notebooks](https://github.com/Striveworks/velour/tree/main/sample_notebooks).


### 4b. Using API endpoints
You can also leverage Velour's API without using the Python client. [Click here](endpoints.md) to read up on all of our API endpoints.


# Next Steps

For more examples, we'd recommend reviewing our [sample notebooks on GitHub](https://github.com/Striveworks/velour/blob/main/examples/getting_started.ipynb). For more detailed explainations of Velour's technical underpinnings, see our [technical concepts guide](technical_concepts.md).
