# Semantic segmentation metrics

Every named class, including the class at local index zero, participates in
precision, recall, F1, IoU, and mean IoU. Declared classes with zero support are
preserved in zero-count rows, receive zero per-class metrics, and remain in the
mean. Pixel accuracy counts matching labels across evaluated pixels.

Datum filtering removes complete pixel-pair rows. Annotation filtering masks
ground truth and prediction independently:

- If both sides remain, the pair contributes its usual match or misclassification.
- If only the prediction remains, the pair contributes false-positive pixels.
- If only the ground truth remains, the pair contributes false-negative pixels.
- If both sides are excluded, the pair contributes nothing to pixel totals,
  metric numerators, or metric denominators.

One-sided pairs remain in the accuracy denominator as errors. Pairs excluded
on both sides are never counted as correct pixels. `get_info` applies the same
masking semantics to pixel counts. A selection with no pixels never reports
perfect accuracy: metrics are zero when only zero-count rows remain;
materializing a selection with no rows raises `EmptyCacheError`.

The vocabulary is recovered from retained row labels, including labels on
masked sides. Classes with zero remaining support receive zero per-class
metrics and remain in mean IoU while their names occur in retained rows.
Classes absent from every retained row are excluded from the vocabulary.

Unfiltered, fully labelled data has matching total, ground truth and prediction
pixel counts and zero unmatched ratios. One-sided masks contribute to the
corresponding unmatched ratios. Original labels and IDs remain in the cache;
`gt_valid` and `pd_valid` store the masks.

Legacy caches can still contain implicit-background and missing-annotation
counts. Their original interpretation is preserved when loading those caches.
Validity columns are optional for legacy caches; absent or null values mean
that the side has not been filtered.

::: valor_lite.semantic_segmentation.metric
