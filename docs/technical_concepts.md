
# Technical Concepts

On this page, we'll describe many of the technical concepts underpinning Velour.

## High-Level Workflow

The typical Velour workflow involves POSTing groundtruth annotations (e.g., class labels, bounding boxes, segmentation masks, etc.) and model predictions to our API service. The service leverages these groundtruths and predictions to compute evaluation metrics, and then stores the groundtruths, predictions, and evaluation metrics centrally in Postgres. Users can also attach metadata to their `Datasets`, `Models`, `GroundTruths`, and `Annotations`; this metadata makes it easy to query for specific subsets of evaluations at a later date. Once an evaluation is stored in Velour, users can query those evaluations from Postgres via `GET` requests to the Velour API.

Note that Velour does _not_ store raw data (such as underlying images) or facilitate model inference. Only the following items are stored in Postgres:

- GroundTruth annotations
- Predictions outputted from a model
- Metadata from any of Velour's various classes
- Evaluation metrics computed by Velour
- State related to any of the above


## Supported Task Types

As of January 2024, Velour supports the following types of computer vision tasks and associated metrics:

- Image classification (including multi-label classification)
    - F1
    - AUCROC
    - Accuracy
    - Precision
    - Recall
- Object detection
    - AP
    - mAP
    - AP Averaged Over IOU's
    - mAP Averaged Over IOU's
- Segmentation (including both instance and semantic segmentation)
    - IOU
    - mIOU


We expect the Velour framework to extend well to other types of supervised learning tasks, and plan to expand our supported task types in future releases.


## Components

We can think of Velour in terms of four orthogonal components:

### API

The core of Velour is a backend REST API service. Users can call the API's endpoints directly (e.g., `POST /datasets`), or they can use our Python client to handle the API calls in their Python environment.  All of Velour's state is stored in Postgres and/or Redis; the API itself is completely stateless.

Note that, after you start the API service in Dockers, you'll be able to view FastAPI's automatically generated API documentation at `https://<your host>/docs`.

### Redis

Redis is an open source, in-memory data store commonly used for caching. We use Redis to cache state and speed-up our evaluation calculations.

### PostgreSQL

PostgreSQL (a.k.a., Postgres or psql) is an open-source relational database management system. We use Postgres to store all of Velour's various objects and states.

One of the most important reasons we chose Postgres was because of its PostGIS extension, which adds support for storing, indexing and querying geographic data. PostGIS enables Velour to quickly filter prior evaluations using geographic coordinates, which is a critically important feature for any computer vision tasks involving satellite data.

### Python Client

Finally, we created a client to make it easier for our users to play with Velour from their Python environment. All of Velour's validations and computations are handled by our API: the Python client simply provides convenient methods to call the API's endpoints.

## Classes

The Velour API and Python client both make use of six core classes. [Click here](/references/API/Schemas/Core/) for technical references on each of these classes in our API reference docs.

### `Dataset`

The highest-level class is a `Dataset`, which stores metadata and annotations associated with a particular set of data. Note that `Dataset` is an abstraction: you can have multiple `Datasets` which reference the exact same input data, which is useful if you need to update or version your data over time.

`Datasets` require a name at instantation, and can optionally take in various types of metadata that you want to associate with your data.

### `Model`

`Models` describe a particular instantiation of a machine learning model. We use the `Model` object to delineate between different models runs, or between the same model run over time. Note that `Models` aren't children of `Datasets`; you can have one `Model` contain predictions for multiple `Datasets`.


`Models` require a name at instantation, and can optionally take in various types of metadata that you want to associate with your model.


### `GroundTruth`

A `GroundTruth` object clarifies what the correct prediction should be for a given piece of data (e.g., an image). For an object detection task, for example, the `GroundTruth` would store a human-drawn bounding box that, when overlayed over an object, would correctly enclose the object that we're trying to predict.

`GroundTruths` take one `Datum` and a list of `Annotations` as arguments.

### `Prediction`

A `Prediction` object describes the output of a machine learning model. For an object detection task, for example, the `Prediction` would describe a machine-generated bounding box enclosing the area where a computer vision model believes a certain class of object can be found.

`Predictions` take one `Datum` and a list of `Annotations` as arguments.


### `Datum`

`Datums` are used to store metadata about `GroundTruths` or `Predictions`. This metadata can include user-supplied metadata (e.g., JSONs filled with config details) or geospatial coordinates (via the `geospatial` argument). `Datums` provide the vital link between `GroundTruths` / `Predictions` and `Datasets`, and are useful when filtering your evaluations on specific conditions.


A `Datum` requires a universal ID (UID) and dataset name at instantiation, along with any `metadata` or `geospatial` dictionaries that you want to associate with your `GroundTruth` or `Prediction`.


### `Annotation`

`Annotations` attach to both `GroundTruths` and `Predictions`, enabling users to add textual labels to these objects. If a `GroundTruth` depicts a bounding box around a cat, for example, the `Annotation` would be passed into the `GroundTruth` to clarify the correct label for the `GroundTruth` (e.g., `class=cat`) and any other labels the user wants to specify for that bounding box (e.g., `breed=tabby`).

`Annotations` require the user to specify their task type, labels, and metadata at instantition. Users can also pass-in various visual representations tailored to their specific task, such as bounding boxes, segmentations, or image rasters.


## Authentication

The API can be run without authentication (by default), or with authentication provided by [auth0](https://auth0.com/). To enable authentication, you can either:

- Set the environment variables `AUTH_DOMAIN`, `AUTH_AUDIENCE`, and `AUTH_ALGORITHMS` manually (e.g., `export AUTH_DOMAIN=<your domain>`)
- Set these env variables in a file named `.env.auth`, and place that file in the `api` directory. An example of such a file would look like:

```
AUTH0_DOMAIN="your_domain.auth0.com"
AUTH0_AUDIENCE="https://your_domain.com/"
AUTH0_ALGORITHMS="RS256"
```

You can use the tests in `integration_tests/test_client_auth.py` to check whether your authenticator is running correctly.

## Deployment Settings

When deploying behind a proxy or with external routing, the environment variable `API_ROOT_PATH` environmental variable should be used to set the `root_path` arguement to `fastapi.FastAPI` (see https://fastapi.tiangolo.com/advanced/behind-a-proxy/#setting-the-root_path-in-the-fastapi-app).


## Release Process

A release is made by publishing a tag of the form `vX.Y.Z` (e.g. `v0.1.0`). This will trigger a GitHub action that will build and publish the python client to [PyPI](https://pypi.org/project/velour-client/). These releases should be created using the [GitHub UI](https://github.com/Striveworks/velour/releases).

