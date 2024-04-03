# Basic Usage

Valor is a centralized evaluation store that makes it easy to measure, explore, and rank model performance. For an overview of what Valor is and why it's important, please refer to our [high-level overview](index.md).

On this page, we'll describe the basic usage of Valor. For how to install Valor, please see our [installation guide](installation.md).

There are two ways to access Valor: by leveraging our Python client (the typical way users of Valor will interact with the service), or by calling our REST endpoints directly (e.g. for integrating Valor into other services). This guide covers the former way of interacting with Valor. For the latter, please see our [API endpoints documentation](endpoints.md).

## Using the Python client

Let's walk through a hypothetical example where we're trying to classify dogs and cats in a series of images. Note that all of the code below is pseudo-code for clarity; please see our [Getting Started](https://github.com/Striveworks/valor/blob/main/examples/getting_started.ipynb) notebook for a working example.

### Import dependencies

Import basic objects directly from the client module using:

```py
from valor import (
    connect
    Client,
    Dataset,
    Model,
    Datum,
    Annotation,
    GroundTruth,
    Prediction,
    Label,
)
from valor.schemas import (
    BoundingBox,
    Polygon,
    BasicPolygon,
    Point,
)
from valor.enums import TaskType
```

### Connect to the Client

The `valor.Client` class gives an object that is used to communicate with the `valor` back end.

```py
connect("http://0.0.0.0:8000")
client = Client()
```

In the event that the host uses authentication, the argument `access_token` should also be passed to `Client`.

### Pass your ground truths into Valor

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

Next, we add one or more `GroundTruths` to our `Dataset`. These objects help Valor understand: "What is the correct classification for this particular image?".

```py

# We start with an example of what third-party annotations could look like.
# img3.png contains a bounding box annotation with label "dog".
# img4.png contains a bounding box annotation with label "cat"
# img5.png contains no annotations
groundtruth_annotations = [
    {"path": "a/b/c/img3.png", "annotations": [{"class_label": "dog", "bbox": {"xmin": 16, "ymin": 130, "xmax": 70, "ymax": 150}}, {"class_label": "person", "bbox": {"xmin": 89, "ymin": 10, "xmax": 97, "ymax": 110}}]},
    {"path": "a/b/c/img4.png", "annotations": [{"class_label": "cat", "bbox": {"xmin": 500, "ymin": 220, "xmax": 530, "ymax": 260}}]},
    {"path": "a/b/c/img5.png", "annotations": []}
]

for image in groundtruth_annotations:

    # each image is represented by a Valor Datum.
    # this is used to connect ground truths and predictions when it's time for evaluation.
    datum = Datum(
        uid=Path(image["path"]).stem, # strip the filename for use as Datum UID.
        metadata={
            "path": image["path"],  # store the path in metadata
        }
    )

    # a Valor Annotation consists of a task_type, labels, and, optionally, a geometry.
    annotations = [
        Annotation(
            task_type=TaskType.OBJECT_DETECTION,
            labels=[Label(key="class_label", value=annotation["class_label"])],
            box=BoundingBox.from_extrema(
                xmin=annotation["bbox"]["xmin"],
                xmax=annotation["bbox"]["xmax"],
                ymin=annotation["bbox"]["ymin"],
                ymax=annotation["bbox"]["ymax"],
            )
        )
        for annotation in image["annotations"]
        if len(annotation) > 0
    ]

    # the datum and annotations we created are then used to form a GroundTruth.
    groundtruth = GroundTruth(
        datum=datum,
        annotations=annotations,
    )

    # add it to your dataset
    dataset.add_groundtruth(groundtruth)

# now that we've added all our ground truths, we can finalize our dataset for evaluation
dataset.finalize()
```

### Pass your predictions into Valor

Now that we've passed several images of dogs into Valor, we need to pass in model predictions before we can evaluate whether those predictions were correct or not. To accomplish this task, we start by defining our `Model`:

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

Next, we tell Valor what our model predicted for each image by attaching `Predictions` to our `Model`:

```py

def create_prediction_from_object_detection_dict(element: dict, datums_by_uid:dict) -> Prediction:

    # get datum from dataset using filename
    uid=Path(element["path"]).stem
    groundtruth = dataset.get_groundtruth(uid)

    # create Annotations
    annotations = [
        Annotation(
            task_type=TaskType.OBJECT_DETECTION,
            labels=[
                Label(key="class_label", value=label["class_label"], score=label["score"])
                for label in annotation["labels"]
            ],
            box=BoundingBox.from_extrema(
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
        datum=groundtruth.datum,
        annotations=annotations,
    )

# let's represent the simulated model output in a similar format to the ground truths:
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

### Run your evaluation and print metrics

Now that both our `Dataset` and `Model` are finalized, we can evaluate how well our hypothetical model performed.

```py
# run evaluation
evaluation = model.evaluate_classification(
    dataset=dataset,
    filters=[
        Annotation.labels.in_(
            [
                Label(key="class_label", value="dog"),
                Label(key="class_label", value="cat"),
            ]
         # with this filter, we're asking Valor to only evaluate how well our model predicted cats and dogs in our images.
    ]
)
evaluation.wait_for_completion() # wait for the job to finish

# get the result of our evaluation
result = evaluation.get_result()

# print our classification metrics
print(result.metrics)
```

### Run a filtered evaluation and print metrics

Valor offers more than just 1:1 evaluations; it allows the creation of metadata filters to stratify the dataset ground truths and model predictions. This enables the user to ask complex questions about their data.

With this in mind, let's pose the question: _"How well did the model perform on animal prediction?"_

We can ask this question with the following evaluation statement:

```py
# run evaluation
animal_evaluation = model.evaluate_classification(
    dataset=dataset,
    filters=[
        # with this filter, we're asking Valor to only evaluate how well our model performed on predicting cats and dogs.
        Annotation.labels.in_(
            [
                Label(key="class_label", value="dog"),
                Label(key="class_label", value="cat"),
            ]
        ),
    ]
)

animal_evaluation.wait_for_completion() # wait for the job to finish

# get the result of our evaluation
result = animal_evaluation.get_result()

# print our classification metrics
print(result.metrics)
```

For more examples, please see our [sample notebooks](https://github.com/Striveworks/valor/tree/main/sample_notebooks).

## Next Steps

For more examples, we'd recommend reviewing our [sample notebooks on GitHub](https://github.com/Striveworks/valor/blob/main/examples/getting_started.ipynb). For more detailed explanations of Valor's technical underpinnings, see our [technical concepts guide](technical_concepts.md).
