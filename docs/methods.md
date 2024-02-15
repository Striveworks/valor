---
header-includes:
  - \usepackage[ruled,vlined,linesnumbered]{algorithm2e}
---
# Algorithm 1
Just a sample algorithmn
\begin{algorithm}[H]
\DontPrintSemicolon
\SetAlgoLined
\KwResult{Write here the result}
\SetKwInOut{Input}{Input}\SetKwInOut{Output}{Output}
\Input{Write here the input}
\Output{Write here the output}
\BlankLine
\While{While condition}{
    instructions\;
    \eIf{condition}{
        instructions1\;
        instructions2\;
    }{
        instructions3\;
    }
}
\caption{While loop with If/Else condition}
\end{algorithm}

# Evaluation Methods

## Classification


## Object-Detection


### Average Precision (Model -> Dataset)

1. Aggregate data.
    1. Aggregate groundtruths.
    1. Aggregate predictions.
1. Create joint table.
    1. Cartesian (Cross) Join
        1. Group by datum.
        1. Constrain by $label_{groundtruth} == label_{prediction}$
1. Compute IOU.
    1. Use PostGIS to compute intersection-over-union (IOU).
1. Create Ranked Pairs
    1. Sort Predictions by score.
    1. Sort Prediction-Groundtruths pairings by IOU.
    1. Greedily create pairs from the sorted list.
        1. No groundtruth or prediction can be repeated.
1. Get number of groundtruths.
    1. Label mappping stuff???
1. Compute Average-Precision
    1. Setup
        1. User chooses IOU Threshold
        1. Initialize precision and recall arrays.
        1. Initialize counters for True-Positive, False-Positive and False-Negative.
    1. Iterate through all labels.
        1. Count True-Positive
            1. $RankedPair_{score} > 0 \land RankedPair_{iou} >= IOU_{threshold}$
        1. Count False-Positive (else-condition)
            1. $RankedPair_{score} == 0 \lor RankedPair_{score} < IOU_{threshold}$
                1. This seems like an issue? The score == 0 condition should probably not be counted.
        1. Count False-Negative
            1. $Count_{groundtruths} - Count_{TP}$
        1. Append $\dfrac{TP}{TP + FP}$ to precision array.
        1. Append $\dfrac{TP}{TP + FN}$ to recall array.
    1. Integrate Precision over Recall
        1. 101-point interpolation standard.


| 1.1 Groundtruths |
| ---------------- |
| $Groundtruth$ |
| $Annotation$ |
| $Datum $|
| $Label$ |

| 1.2 Predictions |
| --------------- |
| $Prediction$ |
| $Annotation$ |
| $Datum $|
| $Label$ |

| 2.1 Joint Table |
| --------------- |
| $Datum$ |
| $Groundtruth$ |
| $Prediction$ |
| $Label_{groundtruth}$ |
| $Label_{prediction}$ |
| $Annotation_{groundtruth}$ |
| $Annotation_{prediction}$ |
| $Prediction_{score}$ |

| 3.1 IOU Table |
| --------------- |
| $Datum$ |
| $Groundtruth$ |
| $Prediction$ |
| $Label_{groundtruth}$ |
| $Label_{prediction}$ |
| $Prediction_{score}$ |
| $IOU$ |

| 6.1 Ranked Table |
| --------------- |
| $Groundtruth$ |
| $Prediction$ |
| $Prediction_{score}$ |
| $IOU$ |


### Binary ROCAUC


### Determining Binary Truth

| Element | Description |
| ------- | ------------ |
| True Positive (TP) | Prediction returns True and is correct. |
| False Positive (FP) | Prediction returns True and is incorrect. |
| True Negative (TN) | Prediction returns False and is correct. |
| False Negative (FN) | Prediction returns False and is incorrect. |


### Determining Multi-Class Truth



### Precision

$Precision = \dfrac{|TP|}{|TP| + |FP|}$

### Recall

$Recall = \dfrac{|TP|}{|TP| + |FN|}$

# Average Precision (AP)

## Object Detection

1. Definitions

    - $Groundtruths \ (G) = \{ g_1, \ g_2, \ \ldots, \ g_k \}$

    - $Predictions \ (P) = \{ p_1, \ p_2, \ \ldots, \ p_n \}\text{ where }(score_1 > score_2 > \cdots > score_n)$

1. Create prediction pairings.

    - $ Pairings = \left\{ \ (p,g) \in P \times G \ | \ p_{datum} \neq g_{datum} \ \right\} $

1. Compute IOU's

    - $Intersection \ Over \ Union \ (IOU) = \dfrac{Area( p_{geom} \cap g_{geom} )}{Area( p_{geom} \cup g_{geom} )}$

    - $ iou_{ij} = IOU( p_i, \ g_j ) $

1. Order the prediction pairings by IOU.

    ```python
    ordered_pairings = sort(pairings, lambda pairing : -pairing.iou)
    ```

1. Greedily select pairings.

    ```python
    g_set = set()
    p_set = set()
    valid_list = list()

    for p, g in ordered_pairing:
        if p in p_set or g in g_set:
            continue
        p_set.add(p)
        g_set.add(g)
        valid_list.append(pairing)
    ```

    - $\text{Define valid list }V$

    - $\text{For each }(p_i,g_j)\text{ in Ordered Pairings:}\\\text{If }(p_i \not\in V) \land (g_j \not\in V)\text{ then append }(p_i,g_j)\text{ to }V$

1. Compute AP.

    - Calculate Precision and Recall

        | Case | Description |
        | ------- | ------------ |
        | True Positive (TP) | Prediction meets IOU threshold requirements. |
        | False Positive (FP) | Prediction fails IOU threshold requirements. |
        | True Negative (TN) | Unused.
        | False Negative (FN) | No prediction exists. |

        - $Precision = \dfrac{|TP|}{|TP| + |FP|}$

        - $Recall = \dfrac{|TP|}{|TP| + |FN|}$

    - Calculate the cumulative sum for precision and recall.

        - $\text{Number of groundtruths} = |G| = |TP| + |FN|$

        - $|TP|_x = \sum\limits_{i=1}^x \begin{cases} 1 \text{ if }iou>threshold \\ 0\text{ otherwise} \end{cases} $

        - $\text{Cumulative Precision: } P_x = \dfrac{|TP|_x}{x}$

        - $\text{Cumulative Recall: } R_x = \dfrac{|TP|_x}{|G|}$

    - $\text{Average Precision} = \sum\limits_{x} (R_x - R_{x-1}) P_x \ where \ R_0 = 0$

# Mean Average Precision (mAP)

$mAP = \dfrac{1}{|classes|} \sum\limits_{c \in classes} AP_{c}$
