# valor-lite: Fast, local machine learning evaluation.

valor-lite is a lightweight, numpy-based library designed for fast and seamless evaluation of machine learning models. It is optimized for environments where quick, responsive evaluations are essential, whether as part of a larger service or embedded within user-facing tools.

valor-lite is maintained by Striveworks, a cutting-edge MLOps company based in Austin, Texas. If you'd like to learn more or have questions, we invite you to connect with us on [Slack](https://striveworks-public.slack.com/join/shared_invite/zt-1a0jx768y-2J1fffN~b4fXYM8GecvOhA#/shared-invite/email) or explore our [GitHub repository](https://github.com/striveworks/valor).

For additional details, be sure to check out our user [documentation](https://striveworks.github.io/valor/). We're excited to support you in making the most of Valor!

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
