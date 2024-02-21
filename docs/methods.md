# ROC AUC for Classification

## Determining Binary Truth

| Element | Description |
| ------- | ------------ |
| True Positive (TP) | Prediction returns True and is correct. |
| False Positive (FP) | Prediction returns True and is incorrect. |
| True Negative (TN) | Prediction returns False and is correct. |
| False Negative (FN) | Prediction returns False and is incorrect. |

- $\text{True Positive Rate} = \dfrac{|TP|}{|TP| + |FN|}$

- $\text{False Positive Rate} = \dfrac{|FP|}{|FP| + |TN|}$

- $\text{Precision} = \dfrac{|TP|}{|TP| + |FP|}$

- $\text{Recall} = \dfrac{|TP|}{|TP| + |FN|}$

## Receiver Operating Characteristic (ROC)

WIP

## Area under the ROC curve (ROC AUC)

WIP

## References
- [Classification: ROC Curve and AUC](https://developers.google.com/machine-learning/crash-course/classification/roc-and-auc)


# Average Precision (AP) for Object Detection

For object-detection and instance segmentation tasks, average-precision is calculated from the intersection-over-union (IoU) of geometric annotations.

## What is Intersection-over-Union (IoU)?

The overlap between the groundtruth and predicted regions of an image, measured as a percentage, grouped by class. IOUs are calculated by a) fetching the groundtruth and prediction rasters for a particular image and class, b) counting the true positive pixels (e.g., the number of pixels that were selected in both the groundtruth masks and prediction masks), and c) dividing the sum of true positives by the total number of pixels in both the groundtruth and prediction masks.

$$Intersection \ over \ Union \ (IoU) = \dfrac{Area( prediction \cap groundtruth )}{Area( prediction \cup groundtruth )}$$

## Multiclass Precision and Recall

| Case | Description |
| :- | :- |
| True Positive (TP) | Prediction meets IoU threshold requirements. |
| False Positive (FP) | Prediction fails IoU threshold requirements. |
| True Negative (TN) | Unused in multi-class evaluation.
| False Negative (FN) | No prediction exists. |

- $Precision = \dfrac{|TP|}{|TP| + |FP|} = \dfrac{\text{Number of True Predictions}}{|\text{Predictions}|}$

- $Recall = \dfrac{|TP|}{|TP| + |FN|} = \dfrac{\text{Number of True Predictions}}{|\text{Groundtruths}|}$

## Finding the best prediction for a groundtruth.

To properly evaluate a detection we must first find the best matches of predictions to ground truths. We start by iterating over predictions by score from highest to lowest. For each prediction we assign the ground truth with the highest IoU value. Both the prediction and ground truth are now considered paired and removed from the pool of choices.

Note: For simplicity, the following algorithm assumes operation over the same datum and label.

$$
\begin{aligned}
&\underline{\large\textbf{Algorithm 1} \hspace{0.5em} \text{Rank IOU's}} \\
&\textbf{Data: }\text{Lists of groundtruths and predictions sharing the same label for a datum.} \\
&\textbf{Results: }\text{Ranked list of IoU's.} \\
&\textbf{Note: }\text{Assume }argsort\text{ algorithm sorts in descending order.} \\
\\
&1 \hspace{1.5em} groundtruths \gets \text{list of geometries} \\
&2 \hspace{1.5em} predictions \gets \text{list of geometries} \\
&3 \hspace{1.5em} scores \gets \text{list of prediction scores} \\
\\
&4 \hspace{1.5em} k \gets \text{length of } groundtruths\\
\\
&5 \hspace{1.5em} visited \gets \text{new empty set} \\
&6 \hspace{1.5em} ious \gets \text{new list of size }k\\
&7 \hspace{1.5em} ranked\_ious \gets \text{new empty list} \\
\\
&8 \hspace{1.5em} I \gets \textbf{argsort}(scores) \\
&9 \hspace{1.5em} \textbf{foreach }i\text{ in }I\textbf{ do} \\
&10 \hspace{1em} | \quad p \gets predictions[i] \\
&11 \hspace{1em} | \quad \textbf{for }j=1\text{ to }k\textbf{ do} \\
&12 \hspace{1em} | \quad | \quad g \gets groundtruths[j] \\
&13 \hspace{1em} | \quad | \quad ious[j] \gets IoU(p, g) \\
&14 \hspace{1em} | \quad \textbf{end} \\
&15 \hspace{1em} | \quad J \gets \textbf{argsort}(ious) \\
&16 \hspace{1em} | \quad \textbf{foreach }j\text{ in }J\textbf{ do} \\
&17 \hspace{1em} | \quad | \quad \textbf{if }j\text{ not in }visited\_groundtruths \textbf{ then} \\
&18 \hspace{1em} | \quad | \quad | \quad visited.add(j) \\
&19 \hspace{1em} | \quad | \quad | \quad ranked\_ious.append(ious[j]) \\
&20 \hspace{1em} | \quad | \quad | \quad \textbf{break} \\
&21 \hspace{1em} | \quad | \quad \textbf{end} \\
&22 \hspace{1em} | \quad \textbf{end} \\
&23 \hspace{1em} \textbf{end} \\
\end{aligned}
$$

## Precision-Recall Curve

We can now compute the precision-recall curve using our previously ranked IoU's.

$$
\begin{aligned}
&\underline{\large\textbf{Algorithm 2} \hspace{0.5em} \text{Precision-Recall Curve}} \\
&\textbf{Data: }\text{Ranked list of IoU's for a label and a IoU threshold between 0 and 1.} \\
&\textbf{Results: }\text{List of points on a curve.} \\
\\
&1 \hspace{1.5em} ranked\_ious \gets \text{list of IoU's} \\
&2 \hspace{1.5em} threshold \gets \text{IoU Threshold} \\
\\
&3 \hspace{1.5em} n \gets \text{total number of groundtruths.} \\
&4 \hspace{1.5em} count\_tp \gets 0 \\
\\
&5 \hspace{1.5em} curve \gets \text{new empty list} \\
\\
&6 \hspace{1.5em} \textbf{for }i=0\text{ to }n-1\textbf{ do} \\
&7 \hspace{1.5em} | \quad \textbf{if } ranked\_ious[i] \ge threshold \textbf{ then} \\
&8 \hspace{1.5em} | \quad | \quad count\_tp \gets count\_tp + 1 \\
&9 \hspace{1.5em} | \quad \textbf{end} \\
&10 \hspace{1em} | \quad precision \gets count\_tp \mathrel{{/}} (i+1) \\
&11 \hspace{1em} | \quad recall \gets count\_tp \mathrel{{/}} n \\
&12 \hspace{1em} | \quad curve.append((recall, precision)) \\
&13 \hspace{1em} \textbf{end} \\
\end{aligned}
$$

## Calculating Average Precision

Average precision is defined as the integration of the precision-recall curve. However, due to the varying nature of datasets it has been shown that interpolating this curve with a fixed set of points helps to reduce inconsistencies between dataset splits. The defacto standard has been to use a 101-point interpolation of the precision-recall curve to compute this integral.

$$
AP = \frac{1}{101} \sum\limits_{r\in\{ 0, 0.01, \ldots , 1 \}}\rho_{interp}(r)
$$

$$
\rho_{interp} = \underset{\tilde{r}:\tilde{r} \ge r}{max \ \rho (\tilde{r})}
$$

## References
- [MS COCO Detection Evaluation](https://cocodataset.org/#detection-eval)
- [Mean Average Precision (mAP) Using the COCO Evaluator](https://pyimagesearch.com/2022/05/02/mean-average-precision-map-using-the-coco-evaluator/)

## Notes
- When calculating IOUs for object detection metrics, Valor handles the necessary conversion between different types of image annotations. For example, if your model prediction is a polygon and your groundtruth is a raster, then the raster will be converted to a polygon prior to calculating the IOU.
