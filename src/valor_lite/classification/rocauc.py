import heapq

import numpy as np
import pyarrow.parquet as pq

from valor_lite.cache import CacheReader, CacheWriter


def iterate_sorted_pairs(
    reader: CacheReader,
    rows_per_chunk: int,
    read_batch_size: int,
):
    id_columns = [
        "datum_id",
        "gt_label_id",
        "pd_label_id",
    ]
    if reader.num_dataset_files == 1:
        pf = pq.ParquetFile(reader.dataset_files[0])
        tbl = pf.read()
        ids = np.column_stack([tbl[col].to_numpy() for col in id_columns])
        scores = tbl["score"].to_numpy()
        winners = tbl["winner"].to_numpy()

        n_pairs = ids.shape[0]
        n_chunks = n_pairs // rows_per_chunk
        for i in range(n_chunks):
            lb = i * rows_per_chunk
            ub = i * (rows_per_chunk + 1)
            yield ids[lb:ub], scores[lb:ub], winners[lb:ub]
        lb = n_chunks * rows_per_chunk
        yield ids[lb:], scores[lb:], winners[lb:]
    else:

        def generate_heap_item(batches, batch_idx, row_idx):
            score = batches[batch_idx]["score"][row_idx].as_py()
            gidx = batches[batch_idx]["gt_label_id"][row_idx].as_py()
            pidx = batches[batch_idx]["pd_label_id"][row_idx].as_py()
            return (
                -score,
                pidx,
                gidx,
                batch_idx,
                row_idx,
            )

        # merge sorted rows
        heap = []
        batch_iterators = []
        batches = []
        for batch_idx, path in enumerate(reader.dataset_files):
            pf = pq.ParquetFile(path)
            batch_iter = pf.iter_batches(batch_size=read_batch_size)
            batch_iterators.append(batch_iter)
            batches.append(next(batch_iterators[batch_idx], None))
            if batches[batch_idx] is not None and len(batches[batch_idx]) > 0:
                heapq.heappush(heap, generate_heap_item(batches, batch_idx, 0))

        ids_buffer = []
        scores_buffer = []
        winners_buffer = []
        while heap:
            _, _, _, batch_idx, row_idx = heapq.heappop(heap)
            row_table = batches[batch_idx].slice(row_idx, 1)

            ids_buffer.append(
                np.column_stack(
                    [row_table[col].to_numpy() for col in id_columns]
                )
            )
            scores_buffer.append(row_table["score"].to_numpy())
            winners_buffer.append(
                row_table["winner"].to_numpy(zero_copy_only=False)
            )
            if len(ids_buffer) >= rows_per_chunk:
                ids = np.concatenate(ids_buffer, axis=0)
                scores = np.concatenate(scores_buffer, axis=0)
                winners = np.concatenate(winners_buffer, axis=0)
                yield ids, scores, winners
                ids_buffer, scores_buffer, winners_buffer = [], [], []

            row_idx += 1
            if row_idx < len(batches[batch_idx]):
                heapq.heappush(
                    heap,
                    generate_heap_item(batches, batch_idx, row_idx),
                )
            else:
                batches[batch_idx] = next(batch_iterators[batch_idx], None)
                if (
                    batches[batch_idx] is not None
                    and len(batches[batch_idx]) > 0
                ):
                    heapq.heappush(
                        heap, generate_heap_item(batches, batch_idx, 0)
                    )
        if len(ids_buffer) > 0:
            ids = np.concatenate(ids_buffer, axis=0)
            scores = np.concatenate(scores_buffer, axis=0)
            winners = np.concatenate(winners_buffer, axis=0)
            yield ids, scores, winners


def compute_rocauc(
    cache: CacheReader,
    n_labels: int,
):
    """
    Compute ROCAUC and mean ROCAUC.
    """

    positive_count = label_metadata[:, 0]
    negative_count = label_metadata[:, 1] - label_metadata[:, 0]

    true_positives = np.zeros((n_labels, n_datums), dtype=np.int32)
    false_positives = np.zeros_like(true_positives)
    scores = np.zeros_like(true_positives, dtype=np.float64)

    for label_idx in range(n_labels):
        if label_metadata[label_idx, 1] == 0:
            continue

        mask_pds = pd_labels == label_idx
        true_positives[label_idx] = mask_matching_labels[mask_pds]
        false_positives[label_idx] = ~mask_matching_labels[mask_pds]
        scores[label_idx] = data[mask_pds, 3]

    cumulative_fp = np.cumsum(false_positives, axis=1)
    cumulative_tp = np.cumsum(true_positives, axis=1)

    # sort by -tpr, -score
    indices = np.lexsort((-cumulative_tp, -scores), axis=1)
    cumulative_fp = np.take_along_axis(cumulative_fp, indices, axis=1)
    cumulative_tp = np.take_along_axis(cumulative_tp, indices, axis=1)

    # running max of cumulative_tp
    np.maximum.accumulate(cumulative_tp, axis=1, out=cumulative_tp)

    fpr = np.zeros_like(true_positives, dtype=np.float64)
    np.divide(
        cumulative_fp,
        negative_count[:, np.newaxis],
        where=negative_count[:, np.newaxis] > 1e-9,
        out=fpr,
    )
    tpr = np.zeros_like(true_positives, dtype=np.float64)
    np.divide(
        cumulative_tp,
        positive_count[:, np.newaxis],
        where=positive_count[:, np.newaxis] > 1e-9,
        out=tpr,
    )

    # compute rocauc
    rocauc = npc.trapezoid(x=fpr, y=tpr, axis=1)

    # compute mean rocauc
    mean_rocauc = rocauc.mean()

    return rocauc, mean_rocauc  # type: ignore[reportReturnType]
