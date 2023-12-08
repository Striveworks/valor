
## How it works
The core of _velour_ is a backend REST API service. user's will typically interact with this service via a python client. there is also a lightweight web interface. At a high-level, the typical workflow involves posting groundtruth annotations (class labels, bounding boxes, segmentation masks, etc.) and model predictions to the service. Velour, on the backend, then handles the computation of metrics, stores them centrally, and allows them to be queried. Velour does _not_ store raw data (such as underlying images) or facilitate model inference. It only stores groundtruth annotations and the predictions outputted from a model.

Some highlights:

- The service handles the computation of metrics. This help makes them trustworthy and auditable, and is also useful when metric computations can be computationally expensive (e.g. for object detection).
- Metrics are centralized and queryable. In particular, the service facilicates comparing performance of multiple models against multiple datasets.
- Since inferences and groundtruths are stored, additional metrics can be computed without having to redo model inferences. For example, maybe you run default AP metric settings for object detection but later decide you want to know AP at lower IOU thresholds.


# API

The backend consists of three components

1. A `velour` REST API service.
2. A PostgreSQL instance with the PostGIS extention.
3. A redis instance

FastAPI generates API documentation which can be found at:

`http://<url>:<port>/docs#/`


# Authentication

The API can be run without authentication (by default) or with authentication provided by [auth0](https://auth0.com/). A small react app (code at `web/`)

## Backend

To enable authentication for the backend either set the environment variables `AUTH_DOMAIN`, `AUTH_AUDIENCE`, and `AUTH_ALGORITHMS` or put them in a file named `.env.auth` in the `api` directory. An example of such a file is

```
AUTH0_DOMAIN="velour.us.auth0.com"
AUTH0_AUDIENCE="https://velour.striveworks.us/"
AUTH0_ALGORITHMS="RS256"
```


## Testing auth

All tests mentioned above run without authentication except for `integration_tests/test_client_auth.py`. Running this test requires setting the envionment variables `AUTH0_DOMAIN`, `AUTH0_AUDIENCE`, `AUTH0_CLIENT_ID`, and `AUTH0_CLIENT_SECRET` accordingly.

# Deployment settings

For deploying behind a proxy or with external routing, the environment variable `API_ROOT_PATH` can be set in the backend, which sets the `root_path` arguement to `fastapi.FastAPI` (see https://fastapi.tiangolo.com/advanced/behind-a-proxy/#setting-the-root_path-in-the-fastapi-app)

# Schemas

# Geometry

<details>
<summary><strong>Point</strong></summary>

## Description

Briefly describe the purpose and functionality of the class.

## Attributes

>| name | type | description |
>| - | - | - |
>| x | `float` |  |
>| y | `float` |  |

## Methods

><details>
><summary><b>resize</b></summary>
>
>**Description**\
>Initialize the class instance.
>
>**Parameters**
>| name | type | description |
>| - | - | - |
>| og_img_h | `int` |  |
>| og_img_w | `int` |  |
>| new_img_h | `int` |  |
>| new_img_w | `int` |  |
>
>**Returns**\
>None.
></details>

## Usage

```python
# Creating an instance of MyClass
my_instance = MyClass(param1=value1, param2=value2)
```

</details>

<details>
<summary><strong>Box</strong></summary>

## Description

Briefly describe the purpose and functionality of the class.

## Attributes

>| name | type | description |
>| - | - | - |
>| min | `Point` |  |
>| max | `Point` |  |
>

## Usage

```python
# Creating an instance of MyClass
my_instance = MyClass(param1=value1, param2=value2)
```
</details>

<details>
<summary><strong>BasicPolygon</strong></summary>

## Description

Briefly describe the purpose and functionality of the class.

## Attributes

>| name | type | description |
>| - | - | - |
>| points | `List[Point]` |  |

## Methods

><details>
><summary><b>xy_list</b></summary>
>
>**Description**\
>Initialize the class instance.
>
>**Returns**\
>`List[Point]`
></details>

><details>
><summary><b>tuple_list</b></summary>
>
>**Description:**\
>Initialize the class instance.
>
>**Returns:**\
>`List[Tuple[float,float]]`
></details>

## Usage

```python
# Creating an instance of MyClass
my_instance = MyClass(param1=value1, param2=value2)
```

</details>

<details>
<summary><strong>Polygon</strong></summary>
</details>

<details>
<summary><strong>BoundingBox</strong></summary>
</details>

<details>
<summary><strong>MultiPolygon</strong></summary>
</details>

<details>
<summary><strong>Raster</strong></summary>
</details>




## Supported tasks

Currently `velour` supports the following groundtruth label types:

- image classifications
- object detections
- instance segmentations
- semantic segmentations
- tabular data classifications

Each case has the notion of a label, which is a key/value pair. This is used (instead of forcing labels to be strings) to support things such as

- multi-label classification. e.g. a dataset of cropped vehicles that have make, model, and year labels
- additional attributes, such as COCO's `isCrowd` attribute.


# CoreTypes

The Velour python client supports a small set of object types that facilitate the creation of a unlimited set of user-defined types. These “atomic” types construct and transport the underlying annotation, label and score as well as any associated metadata.

<details>
<summary>Dataset</summary>

| attribute | type | description |
| - | - | - |
| id | `int` |  |
| name | `str` |  |
| metadata | `dict[str, Union[float, str]]`|  |
| geospatial | `dict` | GeoJSON format. |

`velour` stores metadata and annotations associated to a machine learning dataset. For example, in the case of a computer vision dataset, `velour` needs unique identifiers for images, height and width of images, and annotations (such as image classifications, bounding boxes, segmentation masks, etc.) but the underlying images themselves are not stored or needed by velour.

The process of creating a new dataset to be used in velour is to first create an empty dataset via

```py
dataset = client.create_dataset(DATASET_NAME) # DATASET_NAME a string.
```

`dataset` is then a `velour.Dataset` object and can be used to add groundtruth labels.

</details>

<details>
<summary>Model</summary>

| attribute | type | description |
| - | - | - |
| id | `int` |  |
| name | `str` |  |
| metadata | `dict[str, Union[float, str]]`|  |
| geospatial | `dict` | GeoJSON format. |

</details>

<details>
<summary>Datum</summary>

| attribute | type | description |
| - | - | - |
| uid | `str` |  |
| dataset | `str` |  |
| metadata | `dict[str, Union[float, str]]`|  |
| geospatial | `dict` | GeoJSON format. |

</details>

<details>
<summary>Annotation</summary>

| attribute | type | description |
| - | - | - |
| task_type | `enums.TaskType` |
| labels | `list[Label]` | |
| metadata | `dict[str, Union[float, str]]`||
| bounding_box | `schemas.BoundingBox` ||
| polygon | `schemas.Polygon` ||
| multipolygon | `schemas.MultiPolygon` ||
| raster | `schemas.Raster` ||
| jsonb | todo ||


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

</details>

<details>
<summary>GroundTruth</summary>

| attribute | type | description |
| - | - | - |
| datum | `Datum` | |
| annotations | `list[Annotation]` | |

```py
# create groundtruth
groundtruth = GroundTruth(
    datum=datum,
    annotations=[groundtruth_annotations],
)
```

</details>

<details>
<summary>Prediction</summary>

| attribute | type | description |
| - | - | - |
| model | `str` |
| datum | `Datum` | |
| annotations | `list[Annotation]` | |

</details>

<details>
<summary>Label</summary>

| attribute | type | description |
| - | - | - |
| key | `str` | |
| value | `int` | |
| score | `Optional[float]` | 0-1 |

</details>

<br>

# Schemas

<details>
<summary>Filtering</summary>

> <details>
> <summary>ValueFilter</summary>
>
> | attribute | type | description |
> | - | - | - |
> | value | `Union[int, float, str]` |  |
> | operator | `str` |  Valid string operators: `{"==","!="}`. Numeric operators can draw from the set `{">","<",">=","<=","==","!="}`. |
>
> </details>

> <details>
> <summary>GeospatialFilter</summary>
>
> | attribute | type | description |
> | - | - | - |
> | geodict | `dict` | GeoJSON |
> | operator | `str` |  Valid operators: `{"inside", "outside", "intersect"}` |
>
> </details>

> <details>
> <summary>Filter</summary>
>
> | attribute | type | description |
> | - | - | - |
> | dataset_names | `list[str]` |  |
> | dataset_metadata | `list[schemas.ValueFilter]` |  |
> | dataset_geospatial | `list[schemas.GeospatialFilter]`|  |
> | models_names | `list[str]` |  |
> | models_metadata | `list[schemas.ValueFilter]` |  |
> | models_geospatial | `list[schemas.GeospatialFilter]`|  |
> | datum_uids | `list[str]` |  |
> | datum_metadata | `list[schemas.ValueFilter]` |  |
> | datum_geospatial | `list[schemas.GeospatialFilter]`|  |
> | task_types | `list[enums.TaskType]`|  |
> | annotation_types | `list[enums.AnnotationType]`|  |
> | annotation_geometric_area | `list[schemas.ValueFilter]`|  |
> | annotation_metadata | `list[schemas.ValueFilter]` |  |
> | annotation_geospatial | `list[schemas.GeospatialFilter]`|  |
> | prediction_scores | `list[schemas.ValueFilter]`|  |
> | labels | `list[dict[str,str]]`|  |
> | label_ids | `list[int]`|  |
> | label_keys | `list[str]`|  |
>
> </details>

</details>

<details>
<summary>Geometry</summary>

> <details>
> <summary>Point</summary>
>
> | attribute | type | description |
> | - | - | - |
> | x | `float` |  |
> | y | `float` |  |
>
> | method | args | type |
> | - | - | - |
> | resize |  | `Point` |
> |  | og_img_h | `int` |
> |  | og_img_w | `int` |
> |  | new_img_h | `int` |
> |  | new_img_w | `int` |
>
> </details>

> <details>
> <summary>Box</summary>
>
> | attribute | type | description |
> | - | - | - |
> | min | `Point` |  |
> | max | `Point` |  |
>
> </details>

> <details>
> <summary>BasicPolygon</summary>
>
> | attribute | type | description |
> | - | - | - |
> | points | `list[Point]` |  |
>
> | method | args | type |
> | - | - | - |
> | xy_list |  | `list[Point]` |
> | tuple_list |  | `int` |
> | xmin |  | `Point` |
> | xmax |  | `Point` |
> | ymin |  | `Point` |
> | ymax |  | `Point` |
> | from_box | | `BasicPolygon` |
> |  | box | `Box` |
>
> </details>

> <details>
> <summary>BoundingBox</summary>
>
> | attribute | type | description |
> | - | - | - |
> | polygon | `BasicPolygon` |  |
>
> </details>

> <details>
> <summary>Polygon</summary >
>
> | attribute | type | description |
> | - | - | - |
> | boundary | `BasicPolygon` |  |
> | holes | `list[BasicPolygon]` |  |
>
> </details>

> <details>
> <summary>MultiPolygon</summary>
>
> | attribute | type | description |
> | - | - | - |
> | polygons | `list[Polygon]` |  |
>
> </details>

> <details>
> <summary>Raster</summary>
>
> | attribute | type | description |
> | - | - | - |
> | mask | `str` |  |
>
> </details>

</details>

<br>

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

`velour` has the notion of a model that stores inferences of a machine learning model; `velour` does not need access to the model itself to evaluate, it just needs the predictions to be sent to it.

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