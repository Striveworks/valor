# Semantic segmentation metrics

Input value 0 is background; positive value `i` identifies `labels[i - 1]`.
Precision, recall, F1, IoU, and mean IoU cover foreground classes only. Pixel
accuracy includes background matches. Declared foreground classes with no
support receive zero per-class metrics and remain in mean IoU.

Datum filtering removes complete rows. Annotation filtering remaps excluded
sides to background:

- A retained foreground prediction against background contributes false positives.
- A retained foreground ground truth against background contributes false negatives.
- An excluded side paired with background becomes a correct background match.
- A row failing both side filters is discarded from all counts and metrics.

Classes excluded from both sides disappear from per-class metrics and mean IoU.
`get_info` reports total pixels including background; ground truth and prediction
pixel counts cover only foreground on that side. Background-only input with
`labels=[]` has accuracy 1.0, no per-class metrics, and mean IoU 0.0. Metrics are
zero when only zero-count rows remain. Materializing a selection with no rows
raises `EmptyCacheError`.

The existing computation is preserved: confusion-matrix index 0 is background,
per-class metrics use the foreground submatrix, and accuracy uses the full
diagonal. Background-versus-foreground pairs contribute to the unmatched ratios.
Metadata fields named `gt_valid` or `pd_valid` have no special meaning.

::: valor_lite.semantic_segmentation.metric
