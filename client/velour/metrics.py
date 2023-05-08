from enum import Enum


class Task(Enum):
    BBOX_OBJECT_DETECTION = "Bounding Box Object Detection"
    POLY_OBJECT_DETECTION = "Polygon Object Detection"
    INSTANCE_SEGMENTATION = "Instance Segmentation"
    IMAGE_CLASSIFICATION = "Image Classification"
    SEMANTIC_SEGMENTATION = "Semantic Segmentation"
