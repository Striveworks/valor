import time

from test_utils import *
from valor_implementation import _calculate_pr_curves
from optimized_implementation import _calculate_pr_curves_optimized

n = 1000
n_class = 3
predictions_per_datum = 3

groundtruth_df = generate_groundtruth(n, n_class)
prediction_df = generate_predictions(n, n_class, 3)

print(f"Groundtruth Dataframe Memory Usage: {groundtruth_df.memory_usage(index=True).sum()}")
print(f"Prediction Dataframe Memory Usage: {groundtruth_df.memory_usage(index=True).sum()}")

## Randomly drop some samples.
## groundtruth_df = groundtruth_df.sample(frac=0.9)
## prediction_df = prediction_df.sample(frac=0.8)

start = time.time()
valor_df = _calculate_pr_curves(
    groundtruth_df=groundtruth_df,
    prediction_df=prediction_df,
    metrics_to_return=["PrecisionRecallCurve"],
    pr_curve_max_examples=0,
)
end = time.time()
print(f"Valor completed {n} in {end - start}")


start = time.time()
fast_df = _calculate_pr_curves_optimized(
    groundtruth_df=groundtruth_df,
    prediction_df=prediction_df,
    metrics_to_return=["PrecisionRecallCurve"],
    pr_curve_max_examples=0,
)
end = time.time()
print(f"New code completed {n} in {end - start}")

print(label_values[:n_class])

def print_compare(label_value):
    print(valor_df[valor_df['label_value'] == label_value][fast_df.columns])
    print(fast_df[fast_df['label_value'] == label_value][fast_df.columns])
