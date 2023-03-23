from enum import Enum


class Task(Enum):
    OBJECT_DETECTION = "Object Detection"
    INSTANCE_SEGMENTATION = "Instance Segmentation"
    IMAGE_CLASSIFICATION = "Image Classification"
    SEMANTIC_SEGMENTATION = "Semantic Segmentation"


class MetricType(Enum):
    AP = "Average Precision"
    mAP = "Mean Average Precision"
