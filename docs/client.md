# Python Client

Here we cover the basic concepts involved in interacting with the `velour` service via the Python client.

Additionally, sample Jupyter notebooks are available [here](https://github.com/Striveworks/velour/tree/main/sample_notebooks).

## Installation

The client is hosted on PyPI and can be installed via

```shell
pip install velour-client
```

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

## Client

The `velour.Client` class gives an object that is used to communicate with the `velour` backend. It can be instantiated via

```py
from velour.client import Client

client = Client(HOST_URL)
```

In the case that the host uses authentication, then the argument `access_token` should also be passed to `Client`.

<br>

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
