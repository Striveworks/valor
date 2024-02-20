

# TOC

- [Binary ROCAUC](#binary-rocauc) (WIP)
- [Average Precision (AP)](#average-precision-ap)
    - Classification (TBD)
    - [Object-Detection](#object-detection)
    - Semantic-Segmentation (WIP)

# Binary ROCAUC

## Determining Binary Truth

| Element | Description |
| ------- | ------------ |
| True Positive (TP) | Prediction returns True and is correct. |
| False Positive (FP) | Prediction returns True and is incorrect. |
| True Negative (TN) | Prediction returns False and is correct. |
| False Negative (FN) | Prediction returns False and is incorrect. |

# Average Precision (AP)

## Object-Detection

For object-detection and segmentation tasks, average-precision is calculated from the intersection-over-union (IoU) of geometric annotations.

### Intersection-over-Union (IoU)

$$Intersection \ over \ Union \ (IoU) = \dfrac{Area( prediction \cap groundtruth )}{Area( prediction \cup groundtruth )}$$

#### Relevant PostGIS Functions

- [ST_INTERSECTION](https://postgis.net/docs/ST_Intersection.html)

- [ST_UNION](https://postgis.net/docs/ST_Union.html)

- [ST_AREA](https://postgis.net/docs/ST_Area.html) for Polygon Area

    ```sql
    SELECT ST_Area(annotation.bounding_box) FROM annotation;
    SELECT ST_Area(annotation.polygon) FROM annotation;
    SELECT ST_Area(annotation.multipolygon) FROM annotation;
    ```

- [ST_COUNT](https://postgis.net/docs/RT_ST_Count.html) for Raster Area

    ```sql
    SELECT ST_Count(annotation.raster) FROM annotation;
    ```

#### Example - Polygon IoU Calculation in PostGIS

```sql
CREATE OR REPLACE FUNCTION calculate_iou(groundtruth geometry, prediction geometry)
RETURNS numeric AS $$
BEGIN
    RETURN (
        SELECT COALESCE(ST_AREA(ST_INTERSECTION(groundtruth, prediction)) / ST_AREA(ST_UNION(groundtruth, prediction)), 0)
    );
END;
$$;
```

### Finding the best prediction for a groundtruth.

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

#### Example - PostgreSQL Implementation

```sql
SELECT
prediction_subquery.id AS p_id,
groundtruth _subquery.id AS g_id,
groundtruth_subquery.label_id AS label_id
prediction_subquery.score AS score,
calculate_iou(groundtruth_subquery.geom, prediction_subquery.geom) AS iou
FROM (
    SELECT
    groundtruth.id,
    groundtruth.datum_id,
    groundtruth.label_id,
    annotation.box AS geom
    FROM groundtruth
    JOIN annotation
    ON groundtruth.annotation_id = annotation.id
) AS groundtruth_subquery
CROSS JOIN (
    SELECT
    prediction.id AS id,
    prediction.datum_id AS datum_id,
    prediction.label_id AS label_id,
    prediciion.score AS score,
    annotation.box AS geom
    FROM prediction
    JOIN annotation
    ON prediction.annotation_id = annotation.id
) AS prediction_subquery
WHERE groundtruth_subquery.datum_id = prediction_subquery.datum_id
AND groundtruth_subquery.label_id = prediction_subquery.label_id
ORDER BY -score, -iou
```

### What are Precision and Recall?

| Case | Description |
| :- | :- |
| True Positive (TP) | Prediction meets IoU threshold requirements. |
| False Positive (FP) | Prediction fails IoU threshold requirements. |
| True Negative (TN) | Unused in multi-class evaluation.
| False Negative (FN) | No prediction exists. |

- $Precision = \dfrac{|TP|}{|TP| + |FP|} = \dfrac{\text{Number of True Predictions}}{|\text{Predictions}|}$

- $Recall = \dfrac{|TP|}{|TP| + |FN|} = \dfrac{\text{Number of True Predictions}}{|\text{Groundtruths}|}$

### Precision-Recall Curve

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

### Calculate Average Precision

Average precision is defined as the integration of the precision-recall curve. However, due to the varying nature of datasets it has been shown that interpolating this curve with a fixed set of points helps to reduce inconsistencies between dataset splits. The defacto standard has been to use a 101-point interpolation of the precision-recall curve to compute this integral.

$$
AP = \frac{1}{101} \sum\limits_{r\in\{ 0, 0.01, \ldots , 1 \}}\rho_{interp}(r)
$$

$$
\rho_{interp} = \underset{\tilde{r}:\tilde{r} \ge r}{max \ \rho (\tilde{r})}
$$

### References
- [MS COCO Detection Evaluation](https://cocodataset.org/#detection-eval)
- [Mean Average Precision (mAP) Using the COCO Evaluator](https://pyimagesearch.com/2022/05/02/mean-average-precision-map-using-the-coco-evaluator/)

# Mean Average Precision (mAP)

$mAP = \dfrac{1}{|classes|} \sum\limits_{c \in classes} AP_{c}$




