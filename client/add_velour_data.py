import time

from velour.client import Client
from velour.data_types import (
    BoundingBox,
    GroundTruthDetection,
    Image,
    Label,
    PredictedDetection,
    ScoredLabel,
)
from velour.metrics import Task


client = Client(host="http://localhost:8000")

# add image data
img1 = Image(uid="uid1", height=900, width=300)
img2 = Image(uid="uid2", height=400, width=300)

gt_dets = [
    GroundTruthDetection(
        bbox=BoundingBox(xmin=10, ymin=10, xmax=60, ymax=40),
        labels=[Label(key="k1", value="v1")],
        image=img1,
    ),
    GroundTruthDetection(
        bbox=BoundingBox(xmin=15, ymin=0, xmax=70, ymax=20),
        labels=[Label(key="k1", value="v1")],
        image=img2,
    ),
]
dataset = client.create_image_dataset("image-dataset")
dataset.add_groundtruth(gt_dets)
dataset.finalize()

pred_dets = [
    PredictedDetection(
        bbox=BoundingBox(xmin=10, ymin=10, xmax=60, ymax=40),
        scored_labels=[
            ScoredLabel(label=Label(key="k1", value="v1"), score=0.3)
        ],
        image=img1,
    ),
    PredictedDetection(
        bbox=BoundingBox(xmin=15, ymin=0, xmax=70, ymax=20),
        scored_labels=[
            ScoredLabel(label=Label(key="k2", value="v2"), score=0.98)
        ],
        image=img2,
    ),
]
model = client.create_image_model("detection-model")
model.add_predictions(dataset, pred_dets)
model.finalize_inferences(dataset)

eval_job = model.evaluate_ap(
    dataset=dataset,
    model_pred_task_type=Task.BBOX_OBJECT_DETECTION,
    dataset_gt_task_type=Task.BBOX_OBJECT_DETECTION,
    iou_thresholds=[0.1, 0.6],
    ious_to_keep=[0.1, 0.6],
    label_key="k1",
    min_area=10,
    max_area=2000,
)

# tabular data
dataset = client.create_tabular_dataset(name="tabular-dataset")
dataset.add_groundtruth(
    [
        [Label(key="class", value=str(t))]
        for t in [1, 1, 2, 0, 0, 0, 1, 1, 1, 1]
    ]
)
dataset.finalize()

tabular_preds = [
    [0.37, 0.35, 0.28],
    [0.24, 0.61, 0.15],
    [0.03, 0.88, 0.09],
    [0.97, 0.03, 0.0],
    [1.0, 0.0, 0.0],
    [1.0, 0.0, 0.0],
    [0.01, 0.96, 0.03],
    [0.28, 0.02, 0.7],
    [0.78, 0.21, 0.01],
    [0.45, 0.11, 0.44],
]
model = client.create_tabular_model(name="tabular-model")
model.add_predictions(
    dataset,
    [
        [
            ScoredLabel(Label(key="class", value=str(i)), score=pred[i])
            for i in range(len(pred))
        ]
        for pred in tabular_preds
    ],
)
model.finalize_inferences(dataset)
job = model.evaluate_classification(dataset=dataset)