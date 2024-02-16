### NOTES - Average Precision (Model -> Dataset)

1. Aggregate data.
    1. Aggregate groundtruths.
    1. Aggregate predictions.
1. Create joint table.
    1. Cartesian (Cross) Join
        1. Group by datum.
        1. Constrain by $label_{groundtruth} == label_{prediction}$
1. Compute IoU.
    1. Use PostGIS to compute intersection-over-union (IoU).
1. Create Ranked Pairs
    1. Sort Predictions by score.
    1. Sort Prediction-Groundtruths pairings by IoU.
    1. Greedily create pairs from the sorted list.
        1. No groundtruth or prediction can be repeated.
1. Get number of groundtruths.
    1. Label mappping stuff???
1. Compute Average-Precision
    1. Setup
        1. User chooses IoU Threshold
        1. Initialize precision and recall arrays.
        1. Initialize counters for True-Positive, False-Positive and False-Negative.
    1. Iterate through all labels.
        1. Count True-Positive
            1. $RankedPair_{score} > 0 \land RankedPair_{iou} >= IoU_{threshold}$
        1. Count False-Positive (else-condition)
            1. $RankedPair_{score} == 0 \lor RankedPair_{score} < IoU_{threshold}$
                1. This seems like an issue? The score == 0 condition should probably not be counted.
        1. Count False-Negative
            1. $Count_{groundtruths} - Count_{TP}$
        1. Append $\dfrac{TP}{TP + FP}$ to precision array.
        1. Append $\dfrac{TP}{TP + FN}$ to recall array.
    1. Integrate Precision over Recall
        1. 101-point interpolation standard.

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

Requirements
1. For a pairing to exist both elements must share the same
    1. Datum
    1. Label

### Calculate Intersection-over-Union (IoU)

$$Intersection \ over \ Union \ (IoU) = \dfrac{Area( prediction \cap groundtruth )}{Area( prediction \cup groundtruth )}$$

#### PostGIS Functions

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

### Ranking IOU's

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

### Calculate Average Precision.

- Calculate Precision and Recall

    | Case | Description |
    | ------- | ------------ |
    | True Positive (TP) | Prediction meets IoU threshold requirements. |
    | False Positive (FP) | Prediction fails IoU threshold requirements. |
    | True Negative (TN) | Unused.
    | False Negative (FN) | No prediction exists. |

    - $Precision = \dfrac{|TP|}{|TP| + |FP|}$

    - $Recall = \dfrac{|TP|}{|TP| + |FN|}$

- $\text{Average Precision} = \sum\limits_{x} (R_x - R_{x-1}) P_x \ where \ R_0 = 0$

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


$$
\begin{aligned}
&\underline{\large\textbf{Algorithm 3} \hspace{0.5em} \text{Average Precision using 101-Point Interpolation}} \\
&\textbf{Data: }\text{List of points representing the precision-recall curve.} \\
&\textbf{Results: }\text{The AP metric.} \\
\\
&1 \hspace{1.5em} curve \gets \text{list of points with form } [(x_1,y_1),\ldots,(x_n,y_n)]\\
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

### Example - Bounding Box

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




# Mean Average Precision (mAP)

$mAP = \dfrac{1}{|classes|} \sum\limits_{c \in classes} AP_{c}$

# SQL Examples

## Object-Detection

### AP of Bounding Boxes


