# Object Detection Metrics

| Name | Description | Equation |
| :- | :- | :- |
| Counts | A dictionary containing counts of true positives, false positives, and false negatives for each label. | See [Counts](#counts)|
| Accuracy | The number of true positive predictions divided by the total number of predictions across all classes. | $\dfrac{\|TP\|+\|TN\|}{\|TP\|+\|TN\|+\|FP\|+\|FN\|}$ |
| Precision | The number of true positives divided by the total number of positive predictions (i.e., the number of true positives plus the number of false positives). | $\dfrac{\|TP\|}{\|TP\|+\|FP\|}$ |
| Recall | The number of true positives divided by the total count of the class of interest (i.e., the number of true positives plus the number of true negatives). | $\dfrac{\|TP\|}{\|TP\|+\|FN\|}$ |
| F1 | A weighted average of precision and recall. | $\frac{2 * Precision * Recall}{Precision + Recall}$ |
| Average Precision (AP) | The weighted mean of precisions achieved at several different recall thresholds for a single Intersection over Union (IOU), grouped by class. | See [Average Precision](#average-precision-ap). |
| Mean Average Precision (mAP) 	| The average of several AP metrics, grouped by label keys and IOU thresholds. | $\dfrac{1}{\text{number of labels}} \sum\limits_{label \in labels} AP_{c}$ |
| Average Recall (AR) | The average of several recall metrics across IOU thresholds, grouped by class labels. | See [Average Recall](#average-recall-ar). |
| Mean Average Recall (mAR) | The average of several AR metrics, grouped by label keys. | $\dfrac{1}{\text{number of labels}} \sum\limits_{label \in labels} AR_{class}$ |
| Precision-Recall Curves | | See [Precision-Recall Curve](#precision-recall-curve)|
| Confusion Matrix | | See [Confusion Matrix](#confusion-matrix)|

# Appendix: Metric Calculations

## Counts

## Average Precision (AP)

For object detection and instance segmentation tasks, average precision is calculated from the intersection-over-union (IOU) of geometric predictions and ground truths.

### Multiclass Precision and Recall

Tasks that predict geometries (such as object detection or instance segmentation) use the ratio intersection-over-union (IOU) to calculate precision and recall. IOU is the ratio of the intersecting area over the joint area spanned by the two geometries, and is defined in the following equation.

$Intersection \ over \ Union \ (IOU) = \dfrac{Area( prediction \cap groundtruth )}{Area( prediction \cup groundtruth )}$

Using different IOU thresholds, we can determine whether we count a pairing between a prediction and a ground truth pairing based on their overlap.

| Case | Description |
| :- | :- |
| True Positive (TP) | Prediction-GroundTruth pair exists with IOU >= threshold. |
| False Positive (FP) | Prediction-GroundTruth pair exists with IOU < threshold. |
| True Negative (TN) | Unused in multi-class evaluation.
| False Negative (FN) | No Prediction with a matching label exists for the GroundTruth. |

- $Precision = \dfrac{|TP|}{|TP| + |FP|} = \dfrac{\text{Number of True Predictions}}{|\text{Predictions}|}$

- $Recall = \dfrac{|TP|}{|TP| + |FN|} = \dfrac{\text{Number of True Predictions}}{|\text{Groundtruths}|}$

### Matching Ground Truths with Predictions

To properly evaluate a detection, we must first find the best pairings of predictions to ground truths. We start by iterating over our predictions, ordering them by highest scores first. We pair each prediction with the ground truth that has the highest calculated IOU. Both the prediction and ground truth are now considered paired and removed from the pool of choices.

```python
def rank_ious(
    groundtruths: list,
    predictions: list,
) -> list[float]:
    """Ranks ious by unique pairings."""

    retval = []
    groundtruths = set(groundtruths)
    for prediction in sorted(predictions, key=lambda x : -x.score):
        groundtruth = max(groundtruths, key=lambda x : calculate_iou(groundtruth, prediction))
        groundtruths.remove(groundtruth)
        retval.append(calculate_iou(groundtruth, prediction))
```

### Precision-Recall Curve

We can now compute the precision-recall curve using our previously ranked IOU's. We do this by iterating through the ranked IOU's and creating points cumulatively using recall and precision.

```python
def create_precision_recall_curve(
    number_of_groundtruths: int,
    ranked_ious: list[float],
    threshold: float
) -> list[tuple[float, float]]:
    """Creates the precision-recall curve from a list of IOU's and a threshold."""

    retval = []
    count_tp = 0
    for i in range(ranked_ious):
        if ranked_ious[i] >= threshold:
            count_tp += 1
        precision = count_tp / (i + 1)
        recall = count_tp / number_of_groundtruths
        retval.append((recall, precision))
```

### Calculating Average Precision

Average precision is defined as the area under the precision-recall curve.

We will use a 101-point interpolation of the curve to be consistent with the COCO evaluator. The intent behind interpolation is to reduce the fuzziness that results from ranking pairs.

$AP = \frac{1}{101} \sum\limits_{r\in\{ 0, 0.01, \ldots , 1 \}}\rho_{interp}(r)$

$\rho_{interp} = \underset{\tilde{r}:\tilde{r} \ge r}{max \ \rho (\tilde{r})}$

### References
- [MS COCO Detection Evaluation](https://cocodataset.org/#detection-eval)
- [The PASCAL Visual Object Classes (VOC) Challenge](https://link.springer.com/article/10.1007/s11263-009-0275-4)
- [Mean Average Precision (mAP) Using the COCO Evaluator](https://pyimagesearch.com/2022/05/02/mean-average-precision-map-using-the-coco-evaluator/)

## Average Recall (AR)

To calculate Average Recall (AR), we:

1. Find the count of true positives above specified IOU and confidence thresholds for all images containing a ground truth of a particular class.
2. Divide that count of true positives by the total number of ground truths to get the recall value per class and IOU threshold. Append that recall value to a list.
3. Repeat steps 1 & 2 for multiple IOU thresholds (e.g., [.5, .75])
4. Take the average of our list of recalls to arrive at the AR value per class.

Note that this metric differs from COCO's calculation in two ways:

- COCO averages across classes while calculating AR, while we calculate AR separately for each class. Our AR calculations matches the original FAIR definition of AR, while our mAR calculations match what COCO calls AR.
- COCO calculates three different AR metrics (AR@1, AR@5, AR@100) by considering only the top 1/5/100 most confident predictions during the matching process. Valor, on the other hand, allows users to input a `recall_score_threshold` value that will prevent low-confidence predictions from being counted as true positives when calculating AR.

## Precision-Recall Curve
Precision-recall curves offer insight into which confidence threshold you should pick for your production pipeline. The `PrecisionRecallCurve` metric includes the true positives, false positives, true negatives, false negatives, precision, recall, and F1 score for each (label key, label value, confidence threshold) combination. When using the Valor Python client, the output will be formatted as follows:

```python

pr_evaluation = evaluate_detection(
    data=dataset,
)
print(pr_evaluation)

[...,
{
    "type": "PrecisionRecallCurve",
    "parameters": {
        "label_key": "class", # The key of the label.
        "pr_curve_iou_threshold": 0.5, # Note that this value will be None for classification tasks. For detection tasks, we use 0.5 as the default threshold, but allow users to pass an optional `pr_curve_iou_threshold` parameter in their evaluation call.
    },
    "value": {
        "cat": { # The value of the label.
            "0.05": { # The confidence score threshold, ranging from 0.05 to 0.95 in increments of 0.05.
                "fn": 0,
                "fp": 1,
                "tp": 3,
                "recall": 1,
                "precision": 0.75,
                "f1_score": .857,
            },
            ...
        },
    }
}]
```

It's important to note that these curves are computed slightly differently from our other aggregate metrics above:

The `PrecisionRecallCurve` values differ from the precision-recall curves used to calculate [Average Precision](#average-precision-ap) in two subtle ways:

- The `PrecisionRecallCurve` values visualize how precision and recall change as confidence thresholds vary from 0.05 to 0.95 in increments of 0.05. In contrast, the precision-recall curves used to calculate Average Precision are non-uniform; they vary over the actual confidence scores for each ground truth-prediction match.
- If your pipeline predicts a label on an image, but that label doesn't exist on any ground truths in that particular image, then the `PrecisionRecallCurve` values will consider that prediction to be a false positive, whereas the other detection metrics will ignore that particular prediction.

## Confusion Matrix

Valor also includes a more detailed version of `PrecisionRecallCurve` which can be useful for debugging your model's false positives and false negatives. When calculating `DetailedPrecisionCurve`, Valor will classify false positives as either `hallucinations` or `misclassifications` and your false negatives as either `missed_detections` or `misclassifications` using the following logic:

#### Object Detection Tasks
  - A **false positive** is a `misclassification` if the following conditions are met:
    1. There is a qualified prediction with the same `Label.key` as the ground truth on the datum, but the `Label.value` is incorrect
    2. The qualified prediction and ground truth have an IOU >= `pr_curve_iou_threshold`.
  - A **false positive** that does not meet the `misclassification` criteria is considered to be a part of the `hallucinations` set.
  - A **false negative** is determined to be a `misclassification` if the following criteria are met:
    1. There is a qualified prediction with the same `Label.key` as the ground truth on the datum, but the `Label.value` is incorrect.
    2. The qualified prediction and ground truth have an IOU >= `pr_curve_iou_threshold`.
  - For a **false negative** that does not meet this criteria, we consider it to have `no_predictions`.
  - **Example**: if there's a photo with one ground truth label on it (e.g., `Label(key='animal', value='dog')`), and we predicted another bounding box directly over that same object (e.g., `Label(key='animal', value='cat')`), we'd say it's a `misclassification`.

The `DetailedPrecisionRecallOutput` also includes up to `n` examples of each type of error, where `n` is set using `pr_curve_max_examples`. An example output is as follows:


```python
# To retrieve more detailed examples for each `fn`, `fp`, and `tp`, look at the `DetailedPrecisionRecallCurve` metric
detailed_evaluation = evaluate_detection(
    data=dataset,
    pr_curve_max_examples=1 # The maximum number of examples to return for each observation type (e.g., hallucinations, misclassifications, etc.)
    metrics_to_return=[..., 'DetailedPrecisionRecallCurve'] # DetailedPrecisionRecallCurve isn't returned by default; the user must ask for it explicitly
)
print(detailed_evaluation)

[...,
{
    "type": "DetailedPrecisionRecallCurve",
    "parameters": {
        "label_key": "class", # The key of the label.
        "pr_curve_iou_threshold": 0.5,
    },
    "value": {
        "cat": { # The value of the label.
            "0.05": { # The confidence score threshold, ranging from 0.05 to 0.95 in increments of 0.05.
                "fp": {
                    "total": 1,
                    "observations": {
                        'hallucinations': {
                            "count": 1,
                            "examples": [
                                (
                                    'test_dataset',
                                     1,
                                    '{"type":"Polygon","coordinates":[[[464.08,105.09],[495.74,105.09],[495.74,146.99],[464.08,146.99],[464.08,105.91]]]}'
                               ) # There's one false positive for this (key, value, confidence threshold) combination as indicated by the one tuple shown here. This tuple contains that observation's dataset name, datum ID, and coordinates in the form of a GeoJSON string. For classification tasks, this tuple will only contain the given observation's dataset name and datum ID.
                            ],
                        }
                    },
                },
                "tp": {
                    "total": 3,
                    "observations": {
                        'all': {
                            "count": 3,
                            "examples": [
                                (
                                    'test_dataset',
                                     2,
                                    '{"type":"Polygon","coordinates":[[[464.08,105.09],[495.74,105.09],[495.74,146.99],[464.08,146.99],[464.08,105.91]]]}'
                               ) # We only return one example since `pr_curve_max_examples` is set to 1 by default; update this argument at evaluation time to store and retrieve an arbitrary number of examples.
                            ],
                        },
                    }
                },
                "fn": {...},
            },
        },
    }
}]
```