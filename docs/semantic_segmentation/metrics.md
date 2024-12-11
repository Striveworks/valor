# Semantic Segmentation Metrics

| Name | Description | Equation |
| :- | :- | :- |
| Precision | The number of true positives divided by the total number of positive predictions (i.e., the number of true positives plus the number of false positives). | $\dfrac{\|TP\|}{\|TP\|+\|FP\|}$ |
| Recall | The number of true positives divided by the total count of the class of interest (i.e., the number of true positives plus the number of true negatives). | $\dfrac{\|TP\|}{\|TP\|+\|FN\|}$ |
| F1 | A weighted average of precision and recall. | $\frac{2 * Precision * Recall}{Precision + Recall}$ |
| Intersection Over Union (IOU) | A ratio between the ground truth and predicted regions of an image, measured as a percentage, grouped by class. |$\dfrac{area( prediction \cap groundtruth )}{area( prediction \cup groundtruth )}$ |
| Mean IOU | The average of IOU across labels, grouped by label key. | $\dfrac{1}{\text{number of labels}} \sum\limits_{label \in labels} IOU_{c}$ |
| Confusion Matrix | | See [Confusion Matrix](#confusion-matrix) |

# Appendix: Metric Calculations

## Confusion Matrix

Description

### Unmatched Predictions

### Unmatched Ground Truths