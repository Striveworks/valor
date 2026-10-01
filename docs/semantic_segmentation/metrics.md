# Semantic segmentation metrics

Input value 0 is reserved for background. For N pixel indices, `labels` contains
N-1 foreground names: pixel value `i > 0` identifies `labels[i - 1]`. Foreground
classes participate in precision, recall, F1, IoU, and mean IoU. Background is
excluded from these metrics, but matching background pixels count toward
accuracy. Declared foreground classes with zero support are preserved in
zero-count rows, receive zero per-class metrics, and remain in the mean.

Datum filtering removes complete pixel-pair rows. Annotation filtering masks
ground truth and prediction independently by remapping excluded sides to
background:

- If both sides remain, the pair contributes its usual match or misclassification.
- If only a foreground prediction remains, the pair contributes false-positive pixels.
- If only a foreground ground truth remains, the pair contributes false-negative pixels.
- If one side is excluded and the other is background, the pair is a background
  match and contributes correct pixels to accuracy.
- If both sides are excluded, the pair contributes nothing to pixel totals,
  metric numerators, or metric denominators.

One-sided foreground pairs remain in the accuracy denominator as errors.
Pairs excluded on both sides are never counted as correct pixels. `get_info`
applies the same filtering semantics: total pixels include background, while
ground truth and prediction pixel counts include only foreground on that side.
A selection with no pixels never reports perfect accuracy: metrics are zero
when only zero-count rows remain; materializing a selection with no rows raises
`EmptyCacheError`. Background-only input with `labels=[]` has accuracy 1.0,
no per-class metrics, and mean IoU 0.0.

The vocabulary is recovered from included annotations only. A class excluded
from both ground truth and prediction is removed from per-class metrics and
the mean IoU denominator. A class retained on either side remains in the
vocabulary. Included classes represented by zero-count rows still receive
zero per-class metrics and remain in mean IoU.

Background-versus-foreground pairs contribute to the corresponding unmatched
ratios. Background and excluded sides share cache ID `-1` and a null label name.
Both map to confusion-matrix row or column 0. The existing metric computation
uses the foreground submatrix for per-class metrics and the full diagonal,
including background matches, for accuracy.

Legacy caches can still contain implicit-background and missing-annotation
counts. Their original interpretation is preserved when loading those caches.
Inclusion is determined by label IDs; metadata fields named `gt_valid` or
`pd_valid` have no special meaning.

::: valor_lite.semantic_segmentation.metric
