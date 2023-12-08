
# Technical Concepts

On this page, we'll describe many of the technical concepts underpinning Velour.

## High-Level Workflow

The typical Velour workflow involves POSTing groundtruth annotations (e.g., class labels, bounding boxes, segmentation masks, etc.) and model predictions to our API service. The service leverages these groundtruths and predictions to compute evaluation metrics and stores them centrally in Postgres. Users can also attach metadata to their `Datasets`, `Models`, `GroundTruths`, and `Annotations`; this metadata makes it easy to query for specific subsets of evaluations at a later date. Once an evaluation is stored in Velour, users can query those evaluations from Postgres via `GET` requests to the Velour API.

Note that Velour does _not_ store raw data (such as underlying images) or facilitate model inference. Only the following items are stored in Postgres:

- GroundTruth annotations
- Predictions outputted from a model
- Metadata from any of Velour's various classes
- Evaluation metrics computed by Velour
- The state of user-supplied datasets, models, and evaluation jobs


## Supported Task Types

As of December 2023, Velour supports the following types of supervised learning tasks:

- Image classification (including multi-label classification)
- Object detection
- Segmentation (including both instance and semantic segmentation)

We expect the Velour framework to extend well to other types of supervised learning tasks, and plan to expand our supported task types in future releases.


## Components

We can think of Velour in terms of four orthogonal components:

### API

The core of Velour is a backend REST API service. Users can call the API's endpoints directly (e.g., `POST /datasets`), or they can use our Python client to handle the handle the API calls from their Python scripts.  All of Velour's state is stored in Postgres and/or Redis; the API itself is completely stateless.

Note that, after you start the API service, you'll be able to view FastAPI's automatically generated API documentation at `https://<your host>/docs`.

### Redis

Redis is an open source, in-memory data store commonly used for caching. We use Redis to cache state and speed-up our evaluation calculations.

### PostgreSQL

PostgreSQL (a.k.a., Postgres or psql) is an open-source relational database management system. We use Postgres store all of Velour's various objects and states.

One of the most important reasons we chose Postgres was because of its PostGIS extension, which adds support for storing, indexing and querying geographic data. PostGIS enables Velour to quickly filter prior evaluations using geographic coordinates, which is a critically important feature for any computer vision tasks involving satellite data.

### Python Client

Finally, we created a client to make it easier for our users to play with Velour from their Python environment. All of Velour's heavy lifting is done by our API: our Python client is basically just a wrapper to make it easier to call our endpoints.

## Classes

The Velour API and Client both make use of six core classes. [Click here](/references/API/Schemas/Core/) for technical references on each of these API classes.

### `Dataset`

The highest-level class is a `Dataset`, which stores metadata and annotations associated with a particular set of data. Note that `Dataset` is an abstraction: you can have multiple `Datasets` which reference the exact same input data, which is useful if you need to update or version your data over time.

`Datasets` require a name at instantation, and can optionally take in various types of metadata that you want to associate with your data.

### `Model`

`Models` describe a particular instantition of a machine learning model. We use the `Model` object to delineate between different models runs, or between the same model run over time. Note that `Models` aren't children of `Datasets`; you can have one `Model` contain predictions for multiple `Datasets`.


`Models` require a name at instantation, and can optionally take in various types of metadata that you want to associate with your model.


### `GroundTruth`

A `GroundTruth` object clarifies what the correct prediction should be for a given piece of data (e.g., an image). For an object detection task, for example, the `GroundTruth` would store a human-drawn bounding box that, when overlayed over an object, would correctly enclose the object that we're trying to predict.

`GroundTruths` take `Datums` and `Annotations` as arguments.

### `Prediction`

A `Prediction` object describes the output of a machine learning model. For an object detection task, for example, the `Prediction` would describe a machine-generated bounding box enclosing the area where a computer vision model believes a certain class of object can be found.

`Predictions` take one `Datum` and a list of `Annotations` as arguments.


### `Datum`

`Datums` are used to store metadata about your `GroundTruths` or `Predictions`. This metadata can include user-supplied metadata (e.g., JSONs filled with config details) or geospatial coordinates (via the `geospatial` argument).

A `Datum` requires a universal ID (UID) and dataset name at instantiation, along with any `metadata` or `geospatial` dictionaries that you want to associate with your `GroundTruth` or `Prediction`.
`Datums` provide the vital link between `GroundTruths` / `Predictions` and `Datasets`, and are useful when filtering your evaluations on specific conditions.


### `Annotation`

`Annotations` attach to both `GroundTruths` and `Predictions`, enabling users to add textual annotations to these objects. If a `GroundTruth` depicts a bounding box around a cat, for example, the `Annotation` would be passed into the `GroundTruth` to clarify the correct label for the `GroundTruth` (e.g., `class=cat`) and any other labels the user wants to specify for that bounding box (e.g., `breed=tabby`).

`Annotations` require the user to specify their task type, labels, and metadata at instantition. Users can also pass-in various output representations tailored to their specific task, such as bounding boxes, segmentations, or image rasters.





Each case has the notion of a label, which is a key/value pair. This is used (instead of forcing labels to be strings) to support things such as

- multi-label classification. e.g. a dataset of cropped vehicles that have make, model, and year labels
- additional attributes, such as COCO's `isCrowd` attribute.

#TODO graph diagram?

## Authentication

The API can be run without authentication (by default), or with authentication provided by [auth0](https://auth0.com/). To enable authentication, you can either:

- Set the environment variables `AUTH_DOMAIN`, `AUTH_AUDIENCE`, and `AUTH_ALGORITHMS` manually (e.g., `export AUTH_DOMAIN=<your domain>`)
- Set these env variables in a file named `.env.auth`, and place that file in the `api` directory. An example of such a file would look like:

```
AUTH0_DOMAIN="velour.us.auth0.com"
AUTH0_AUDIENCE="https://velour.striveworks.us/"
AUTH0_ALGORITHMS="RS256"
```

You can use the tests in `integration_tests/test_client_auth.py` to check whether your authenticator is running correctly.

## Deployment Settings

When deploying behind a proxy or with external routing, the environment variable `API_ROOT_PATH` environmental variable should be used to set the `root_path` arguement to `fastapi.FastAPI` (see https://fastapi.tiangolo.com/advanced/behind-a-proxy/#setting-the-root_path-in-the-fastapi-app).



# MetaTypes

Velour uses a robust metadata system that supports client-side types with no need for any work on the backend to provide support. In the velour Python client these are referred to as metatypes. A metatype is a mapping of a complex data type into a velour Datum type.

An example of such a metatype is ImageMetadata which encodes image height and width into Datum’s metadata attribute.

<details>
<summary>ImageMetadata</summary>

An image consists of specifying the following information:

| name | type | description |
| - | - | - |
| uid | `str` | A unique identifier for the image. This is up to the enduser but some typical options are a filename/path in object store or a dataset specific image id (such as the image id in the COCO dataset) |
| height | `int` | The height of the image. This is necessary for certain operations in the backend, such as converting a polygon contour to a mask |
| width | `int` | The width of the image. This is necessary for certain operations in the backend, such as converting a polygon contour to a mask |
| frame (optional) | `int` | The frame number in the case that the image is a frame of a video |

For example:

```py
from velour.data_types import Image

img = Image(uid="abc123", height=128, width=256)
```

> **Note:** This creates an image object but is not yet uploaded to the backend service.

</details>

<details>
<summary>VideoFrameMetadata</summary>

| name | type | description |
| - | - | - |
| image | `ImageMetadata` | Image corresponding to video frame. |
| frame | `int` | Video frame number. |

</details>

<br>

# Creating a Dataset

### Adding annotations

Data structures and methods are provided for adding annotations to an image and then adding the image metadata and annotations to the backend service. We now go over the different types of supported annotations

<br>

# Creating a Model

<br>

# Running an Evaluation

#### Image classification

An image classification is specified by a `velour.data_types.Image` object and a list of labels. Each label is of type `velour.data_types.Label` which consists of a key/value pairs of strings. For example:

```py
from velour.data_types import Label, GroundTruthImageClassification

label1 = Label(key="occluded", value="yes")
label2 = Label(key="class_name", value="truck")

gt_cls = GroundTruthImageClassification(image=img, labels=[label1, label2])
```

To associate ground truth image classifications (and in particular also the underlying `Image` objects) to a dataset, we use the `velour.Dataset.add_groundtruth_classifications` method which takes a list of image classifications

```py
dataset.add_groundtruth_classifications([gt_cls])
```

This will post the annotations to the backend velour service.

#### Object Detection

#### Semantic Segmentation

## Model

Velour has the notion of a model that stores inferences of a machine learning model; Velour does not need access to the model itself to evaluate, it just needs the predictions to be sent to it.

## Evaluation

Supported Tasks:
- Classification
    - F1
    - AUCROC
    - Accuracy
    - Precision
    - Recall
- Object Detection
    - AP
    - mAP
    - AP Averaged Over IOU's
    - mAP Averaged Over IOU's
- Semantic Segmentation
    - IOU
    - mIOU



An evaluation job sends a request to the backend to evaluate a model against a dataset. This will result in the computation of a host of metrics.

## Metric


# Development

![badge](https://img.shields.io/endpoint?url=https://gist.githubusercontent.com/ekorman/501428c92df8d0de6805f40fb78b1363/raw/velour-coverage.json)

This repo contains the python [client](client) and [backend api](api) packages for velour. For velour's user documentation, [click here](https://striveworks.github.io/velour/).

# Getting Started

## 1. Install Docker

As a first step, be sure your machine has Docker installed. [Click here](https://docs.docker.com/engine/install/) for basic installation instructions.

## 2. Install Dependencies

Create a Python enviroment using your preferred method.

```bash
# venv
python3 -m venv .env-velour
source .env-velour/bin/activate

# conda
conda create --name velour python=3.11
conda activate velour
```

To ensure formatting consistency, we use [pre-commit](https://pre-commit.com/) to manage git hooks. To install pre-commit, run:

```bash
pip install pre-commit
pre-commit install
```

Install the client module.

```bash
pip install client/[test]
```

# API Development

Velour offers multiple methods of deploying the backend. If you do not require the ability to debug the API, please skip this section and follow the instructions in `Setting up the Backend`.

<details>
<summary>Deploy for development.</summary>

1. Install dependencies.

```bash
# install the api module.
pip install api/[test]
```

2. Launch the containers.

```bash
# launch PostgreSQL in background.
make start-postgis

# launch Redis in background.
make start-redis

# launch the Server.
make start-server
```

</details>

# (Optional) Setup pgAdmin to debug postgis

You can use the pgAdmin utility to debug your postgis tables as you code. Start by [installing pgAdmin](https://www.pgadmin.org/download/), then select `Object > Register > Server` to connect to your postgis container:
- *Host name/address*: 0.0.0.0
- *Port*: 5432
- *Maintenance database*: postgres
- *Username*: postgres

# Try it out!

We'd recommend starting with the notebooks in `sample_notebooks/*.ipynb`.

# Release process

A release is made by publishing a tag of the form `vX.Y.Z` (e.g. `v0.1.0`). This will trigger a GitHub action that will build and publish the python client to [PyPI](https://pypi.org/project/velour-client/). These releases should be created using the [GitHub UI](https://github.com/Striveworks/velour/releases).

## Tests

There are integration tests, backend unit tests, and backend functional tests.

## CI/CD

All tests are run via GitHub actions on every push.

## Running locally

Tests can be run locally using Pytest as follows.

```shell
# install pytest
pip install pytest
```

### Integration tests

```shell
pytest integration_tests
```

### Backend unit tests

```shell
pytest api/tests/unit-tests
```

### Backend functional tests

> **Note:** Functional tests require a running instance of PostgreSQL.

```shell
POSTGRES_PASSWORD=password \
POSTGRES_HOST=localhost \
pytest api/tests/functional-tests/
```