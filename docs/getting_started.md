# Getting Started

Velour is a centralized evaluation store which makes it easy to measure, explore, and rank model performance. For an overview of what Velour is and why it's important, [please our overview here](index.md).

On this page, we'll describe how to get up and running with Velour.

## Installation

## 1. Install Docker

As a first step, be sure your machine has Docker installed. [Click here](https://docs.docker.com/engine/install/) for basic installation instructions.


## 2. Clone the repo and open the directory

Choose a file in which to store Velour, then run:

```shell
git clone https://github.com/striveworks/velour
cd velour
```


## 3. Start the container

Start by setting the environment variable `POSTGRES_PASSWORD` to your liking, then start Docker and build the container:

```shell
export POSTGRES_PASSWORD="my-password"
docker compose up
```

## 4. Use Velour

There's two ways to access Velour: by leveraging our Python client, or by calling our REST endpoints directly.

### 4a. Using the Python client

Let's walk-through a hypothetical example where we're trying to classify dogs and cats in a series of images. Note that all of the code below is pseudo-code for clarity; please see our ["Getting Started"](#TODO) notebook for a working example.

#### Install the client

To install the Python client, you can run:

```shell
pip install velour-client
```

#### Import dependencies

Import dependencies directly from the client module using:

```py
from velour import Dataset, Model, Datum, Annotation, GroundTruth, Prediction, Label
from velour.client import Client
from velour.enums import TaskType
```

#### Connect to the Client

The `velour.Client` class gives an object that is used to communicate with the `velour` backend.

```py
client = Client(HOST_URL)
```

In the event that the host uses authentication, the argument `access_token` should also be passed to `Client`.

#### Pass your groundtruths into Velour

First, we define our `Dataset` object using `Dataset.create()`.

```py
dataset = Dataset.create(client, "my_dog_dataset")
```

Next, we add one or more `Groundtruths` to our `Dataset`. These objects help Velour understand "What is the correct classification for this particular image?".

```py

# create a groundtruth for a set of images that we know all depict a dog
for image in dog_images:

    # each image will have its own Datum. this object will help us connect groundtruths and predictions when it's time for evaluation
    image.datum = Datum(
            uid=image.name # a unique ID for each image
            metadata=[ # the metadata we want to use to describe our image
                schemas.MetaDatum(
                    name="type",
                    value="image",
                ),
                schemas.MetaDatum(
                    name="height",
                    value=image.height,
                ),
                schemas.MetaDatum(
                    name="width",
                    value=image.width,
                ),
            ],
    )

    groundtruth = GroundTruth(
        datum=datum,
        annotations=[ # a list of annotations to add to the image
            Annotation(
                task_type = TaskType.CLASSIFICATION,
                labels = [
                    schemas.Label(key="class", value="dog"),
                    schemas.Label(key="category", value="animal"),
                ]
            )
        ],
    )

    # add it to your dataset
    dataset.add_groundtruth(groundtruth)

# now that we've added all our groundtruths, we can finalize our dataset for evaluation
dataset.finalize()
```

#### Pass your prediction into Velour

Now that we've passed several images of dogs into Velour, we need to pass in model predictions before we can evaluate whether those predictions were correct or not. To accomplish this task, we start by defining our `Model`:

```py
# create model
model = Model.create(client, "my_model")
```

Next, we tell Velour what our model predicted for each image by attaching `Predictions` to our `Model`:

```py

# pass a prediction for each image into Velour
for image in dog_images:
    prediction = Prediction(
        model=model.name,
        datum=image.datum, # note that we use the same datums we created before
        annoations=[
            Annotation(
                task_type = TaskType.CLASSIFICATION,
                labels = [
                    schemas.Label(key="class", value="dog", score=image.dog_score),
                    schemas.Label(key="class", value="cat", score=image.cat_score,
                    schemas.Label(key="category", value="animal", score=image.animal_score),
                    schemas.Label(key="category", value="vehicle", score=image.vehicle_score),
                ]
            )
        ],
    )

# prepare model for evaluation over dataset
model.finalize(dataset)
```

#### Run your evaluation and print metrics

Now that both our `Dataset` and `Model` are finalized, we can evaluate how well our hypothetical model did at predicting whether or not each image contained a dog.

```py
# run evaluation
evaluation = model.evaluate_classification(
    dataset=dataset,
    filters=[
        Label.value = "dog" # with this filter, we're asking Velour to only evaluate how well our model predicted dogs in each image
    ]
    timeout=30, # use this argument to wait up to thirty seconds for the evaluation to complete
)

# print our classification metrics
print(evaluation.metrics)
```

For more examples, please see our [sample notebooks](https://github.com/Striveworks/velour/tree/main/sample_notebooks).



### 4b. Using API endpoints
You can also leverage Velour's API without using the Python client. [Click here](references/API/Endpoints.md) to read up on all of our API endpoints.


# Next Steps

For more examples, we'd recommend reviewing our [sample notebooks on GitHub](#TODO). For more detailed explainations of Velour's technical underpinnings, see our [technical concepts guide](technical_concepts.md).