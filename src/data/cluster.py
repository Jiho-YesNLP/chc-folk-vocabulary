"""Greedy cosine "leader" clustering for synonym-merging short descriptor strings.

Used by Stage-4 Step B to merge near-duplicate extracted spans (and conceptual-schema
phrases) within a single ability. Embeddings are BGE-M3 dense vectors (already
L2-normalized), so cosine == dot product.

Algorithm (deterministic): visit items in priority order (most frequent first); each
unassigned item opens a new cluster and becomes its representative/canonical; every
remaining unassigned item within `threshold` cosine of that representative joins it.
n is small (unique strings per ability — hundreds at most), so the full n×n similarity
is fine.
"""

from __future__ import annotations

import numpy as np


def cluster_by_cosine(
    vectors: np.ndarray, priority: np.ndarray, threshold: float
) -> tuple[list[int], list[int]]:
    """Cluster row-vectors by greedy leader assignment.

    Args:
        vectors:   (n, d) L2-normalized embeddings.
        priority:  (n,) score; higher = considered first (becomes representative).
        threshold: cosine cutoff for joining a representative's cluster.

    Returns:
        labels: list[int] length n — cluster id per item.
        reps:   list[int] — representative item index per cluster id (reps[c] is the
                seed of cluster c), in cluster-id order.
    """
    n = len(vectors)
    if n == 0:
        return [], []
    order = sorted(range(n), key=lambda i: (-float(priority[i]), i))
    sims = vectors @ vectors.T
    labels = [-1] * n
    reps: list[int] = []
    for i in order:
        if labels[i] != -1:
            continue
        cid = len(reps)
        reps.append(i)
        labels[i] = cid
        row = sims[i]
        for j in order:
            if labels[j] == -1 and row[j] >= threshold:
                labels[j] = cid
    return labels, reps
