import time
import random
import numpy as np
import pandas as pd

import os
import linecache
import tracemalloc

random.seed(time.time())

label_values = ["cat", "dog", "bee", "rat", "cow", "fox", "ant", "owl", "bat"]

def generate_groundtruth(n, n_class=3):
    n_class = min(n_class, len(label_values))
    n_class - max(1, n_class)

    classes = label_values[:n_class]

    df = pd.DataFrame({
        "datum_uid": [f"uid{i}" for i in range(n)],
        "datum_id": [f"img{i}" for i in range(n)],
        "id": [f"gt{i}" for i in range(n)],
        "label_key": "class_label",
        "label_value": [random.choice(classes) for _ in range(n)],
    })

    return df

def generate_predictions(n, n_class=3, preds_per_datum=1):
    n_class = min(n_class, len(label_values))
    n_class - max(1, n_class)

    classes = label_values[:n_class]

    preds_per_datum = min(1, preds_per_datum)
    preds_per_datum = max(n_class, preds_per_datum)

    all_labels = []
    all_scores = []
    for _ in range(n):
        labels = random.sample(classes, preds_per_datum)
        scores = [random.uniform(0,1) for _ in range(preds_per_datum)]
        total = sum(scores)
        for i in range(len(scores)):
            scores[i] /= total

        all_labels += labels
        all_scores += scores

    df = pd.DataFrame({
        "datum_uid": np.repeat([f"uid{i}" for i in range(n)], preds_per_datum),
        "datum_id": np.repeat([f"img{i}" for i in range(n)], preds_per_datum),
        "id": [f"pd{i}" for i in range(n*preds_per_datum)],
        "label_key": "class_label",
        "label_value": all_labels,
        "score": all_scores,
    })

    return df

def pretty_print_tracemalloc_snapshot(snapshot, key_type='lineno', limit=3):
    snapshot = snapshot.filter_traces((
        tracemalloc.Filter(False, "<frozen importlib._bootstrap>"),
        tracemalloc.Filter(False, "<unknown>"),
    ))
    top_stats = snapshot.statistics(key_type)

    print("Top %s lines" % limit)
    for index, stat in enumerate(top_stats[:limit], 1):
        frame = stat.traceback[0]
        # replace "/path/to/module/file.py" with "module/file.py"
        filename = os.sep.join(frame.filename.split(os.sep)[-2:])
        print("#%s: %s:%s: %.1f KiB"
              % (index, filename, frame.lineno, stat.size / 1024))
        line = linecache.getline(frame.filename, frame.lineno).strip()
        if line:
            print('    %s' % line)

    other = top_stats[limit:]
    if other:
        size = sum(stat.size for stat in other)
        print("%s other: %.1f KiB" % (len(other), size / 1024))
    total = sum(stat.size for stat in top_stats)
    print("Total allocated size: %.1f KiB" % (total / 1024))