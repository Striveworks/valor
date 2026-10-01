# Valor: Fast and Efficient ML Evaluation

![badge](https://img.shields.io/endpoint?url=https://gist.githubusercontent.com/ekorman/c002bbe8c0e479921401b884f4b9fc53/raw/valor-coverage.json)

**valor-lite** is a lightweight, numpy-based library designed for fast and seamless evaluation of machine learning models. It is optimized for environments where quick, responsive evaluations are essential, whether as part of a larger service or embedded within user-facing tools. Valor empowers data scientists and engineers to evaluate the performance of their machine learning pipelines and use those evaluations to make better modeling decisions in the future.

Valor is maintained by Striveworks, a cutting-edge MLOps company based out of Austin, Texas. It's core developers are [Charles Zaloom](https://github.com/czaloom), [Nick Lind](https://github.com/ntlind), [Benjamin Nativi](https://github.com/bnativi), and [Eric Korman](https://github.com/ekorman).

We'd love to learn more about your interest in Valor and answer any questions you may have; please don't hesitate to reach out to us on [Slack](https://striveworks-public.slack.com/join/shared_invite/zt-1a0jx768y-2J1fffN~b4fXYM8GecvOhA#/shared-invite/email) or [GitHub](https://github.com/striveworks/valor).

For more information, please see our [user docs](https://striveworks.github.io/valor/).

## Installation

### PyPi
```shell
pip install valor-lite
```

### Source
```shell
git clone https://github.com/Striveworks/valor.git
cd valor
make install
```

## Usage

### Classification

```python
from valor_lite.classification import DataLoader, Classification, MetricType

classifications = [
    Classification(
        uid="uid0",
        groundtruth="dog",
        predictions=["dog", "cat", "bird"],
        scores=[0.75, 0.2, 0.05],
    ),
    Classification(
        uid="uid1",
        groundtruth="cat",
        predictions=["dog", "cat", "bird"],
        scores=[0.41, 0.39, 0.1],
    ),
]

loader = DataLoader()
loader.add_data(classifications)
evaluator = loader.finalize()

metrics = evaluator.evaluate()

assert metrics[MetricType.Precision][0].to_dict() == {
    'type': 'Precision',
    'value': [0.5],
    'parameters': {
        'score_thresholds': [0.0],
        'hardmax': True,
        'label': 'dog'
    }
}
```

### Object Detection

```python
from valor_lite.object_detection import DataLoader, Detection, BoundingBox, MetricType

detections = [
    Detection(
        uid="uid0",
        groundtruths=[
            BoundingBox(
                xmin=0, xmax=10,
                ymin=0, ymax=10,
                labels=["dog"]
            ),
            BoundingBox(
                xmin=20, xmax=30,
                ymin=20, ymax=30,
                labels=["cat"]
            ),
        ],
        predictions=[
            BoundingBox(
                xmin=1, xmax=11,
                ymin=1, ymax=11,
                labels=["dog", "cat", "bird"],
                scores=[0.85, 0.1, 0.05]
            ),
            BoundingBox(
                xmin=21, xmax=31,
                ymin=21, ymax=31,
                labels=["dog", "cat", "bird"],
                scores=[0.34, 0.33, 0.33]
            ),
        ],
    ),
]

loader = DataLoader()
loader.add_bounding_boxes(detections)
evaluator = loader.finalize()

metrics = evaluator.evaluate()

assert metrics[MetricType.Precision][0].to_dict() == {
    'type': 'Precision',
    'value': 0.5,
    'parameters': {
        'iou_threshold': 0.5,
        'score_threshold': 0.5,
        'label': 'dog'
    }
}
```

### Semantic Segmentation

Pixel value 0 is background; positive value `i` refers to `labels[i - 1]`.
Provide foreground names only. Background counts toward accuracy, not mean IoU.

```python
import numpy as np
from valor_lite.semantic_segmentation import Loader, MetricType, Segmentation

segmentation = Segmentation(
    uid="image-1",
    labels=["sky", "road", "car"],
    groundtruths=np.array([[0, 1], [2, 3]], dtype=np.uint16),
    predictions=np.array([[0, 1], [1, 3]], dtype=np.uint16),
)

loader = Loader.in_memory()
loader.add_data([segmentation])
evaluator = loader.finalize()
metrics = evaluator.compute_precision_recall_iou()
print(metrics[MetricType.Precision][0])
```
