from .classification import compute_clf_metrics
from .detection import compute_detection_metrics
from .segmentation import compute_semantic_segmentation_metrics
from .text_generation import compute_text_generation_metrics

__all__ = [
    "compute_clf_metrics",
    "compute_detection_metrics",
    "compute_semantic_segmentation_metrics",
    "compute_text_generation_metrics",
]
