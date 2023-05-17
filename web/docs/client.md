# Working with the Python client

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

## Dataset

`velour` stores metadata and annotations associated to a machine learning dataset. For example, in the case of a computer vision dataset, `velour` needs unique identifiers for images, height and width of images, and annotations (such as image classifications, bounding boxes, segmentation masks, etc.) but the underlying images themselves are not stored or needed by velour.

The process of creating a new dataset to be used in velour is to first create an empty dataset via

```py
dataset = client.create_dataset(DATASET_NAME) # DATASET_NAME a string.
```

`dataset` is then a `velour.Dataset` object and can be used to add groundtruth labels.

### Image

An image consists of specifying the following information:

| name             | type    | description                                                                                                                                                                                         |
| ---------------- | ------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| uid              | string  | A unique identifier for the image. This is up to the enduser but some typical options are a filename/path in object store or a dataset specific image id (such as the image id in the COCO dataset) |
| height           | integer | The height of the image. This is necessary for certain operations in the backend, such as converting a polygon contour to a mask                                                                    |
| width            | integer | The width of the image. This is necessary for certain operations in the backend, such as converting a polygon contour to a mask                                                                     |
| frame (optional) | integer | The frame number in the case that the image is a frame of a video                                                                                                                                   |

For example:

```py
from velour.data_types import Image

img = Image(uid="abc123", height=128, width=256)
```

Note: this creates an image object but is not yet uploaded to the backend service.

### Adding annotations

Data structures and methods are provided for adding annotations to an image and then adding the image metadata and annotations to the backend service. We now go over the different types of supported annotations

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

## Evaluation job

An evaluation job sends a request to the backend to evaluate a model against a dataset. This will result in the computation of a host of metrics.

## Metric
