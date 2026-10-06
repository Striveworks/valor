"""Benchmark metric computation on an existing evaluator without ingestion.

Example:
    python benchmarks/benchmark_objdet_computation.py testdata --repeats 3
"""

import argparse
import json
from pathlib import Path
from statistics import median
from time import perf_counter

from valor_lite.object_detection import Evaluator


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("path", type=Path)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument(
        "--iou-thresholds", type=float, nargs="+", default=[0.1, 0.5, 0.75]
    )
    parser.add_argument(
        "--score-thresholds", type=float, nargs="+", default=[0.5, 0.75, 0.9]
    )
    parser.add_argument("--output", type=Path)
    parser.add_argument("--reference-precision-recall", type=Path)
    parser.add_argument("--reference-confusion", type=Path)
    args = parser.parse_args()
    if args.repeats <= 0:
        parser.error("--repeats must be positive")

    evaluator = Evaluator.load(args.path)
    report = {
        "path": str(args.path.resolve()),
        "iou_thresholds": args.iou_thresholds,
        "score_thresholds": args.score_thresholds,
        "repeats": args.repeats,
        "results": {},
    }
    operations = [
        (
            "precision_recall",
            evaluator.compute_precision_recall,
            args.reference_precision_recall,
        ),
        (
            "confusion_matrix",
            evaluator.compute_confusion_matrix,
            args.reference_confusion,
        ),
    ]
    for name, operation, reference in operations:
        expected = json.loads(reference.read_text()) if reference else None
        samples = []
        for run in range(args.repeats):
            start = perf_counter()
            metrics = operation(
                iou_thresholds=args.iou_thresholds,
                score_thresholds=args.score_thresholds,
            )
            elapsed = perf_counter() - start
            if isinstance(metrics, dict):
                actual = {
                    key.value: [metric.to_dict() for metric in values]
                    for key, values in metrics.items()
                }
            else:
                actual = [metric.to_dict() for metric in metrics]
            if expected is None:
                expected = actual
            elif actual != expected:
                raise RuntimeError(f"{name} results differ from reference")
            samples.append(elapsed)
            print(f"{name} run {run + 1}: {elapsed:.3f}s", flush=True)
        report["results"][name] = {
            "samples_seconds": samples,
            "median_seconds": median(samples),
            "results_match": True,
            "external_reference": str(reference) if reference else None,
        }

    encoded = json.dumps(report, indent=2) + "\n"
    if args.output:
        args.output.write_text(encoded)
    print(encoded)


if __name__ == "__main__":
    main()
