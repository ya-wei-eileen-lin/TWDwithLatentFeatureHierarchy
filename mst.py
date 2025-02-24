import numpy as np

def mst(dists, n):
    """Maximum Spanning Tree of a dense adjacency matrix using Prim's Algorithm."""
    ij = np.empty((n - 1, 2), dtype=int)
    l = np.empty(n - 1)

    merged = np.zeros(n, dtype=int)
    D = np.full(n, -np.inf)
    j = np.empty(n, dtype=int)

    x = 0
    for k in range(n - 1):
        merged[x] = 1
        current_max = -np.inf
        for i in range(n):
            if merged[i] == 1:
                continue

            dist = dists[x, i]
            if D[i] < dist:
                D[i] = dist
                j[i] = x

            if current_max < D[i]:
                current_max = D[i]
                y = i

        ij[k, 0] = j[y]
        ij[k, 1] = y
        l[k] = current_max
        x = y

    order = np.argsort(l, kind='mergesort')[::-1]
    ij = ij[order]
    l = l[order]

    return ij, l

def reorder(A, idx, n):
    """Reorder matrix A based on index idx."""
    B = np.empty((n, n))
    for i in range(n):
        k = idx[i]
        for j in range(n):
            B[i, j] = A[k, idx[j]]
    return B
