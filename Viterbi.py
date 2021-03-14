# Viterbi's algorithm

__authors__ = "Pedro Pablo"


import numpy as np


def viterbi(A, B, p, signal):
    A = np.array(A)
    B = np.array(B)
    p = np.array(p)
    signal = np.array(signal)

    N = A.shape[0]
    T = len(signal)

    V = np.zeros((N, T))
    G = np.zeros((N, T))

    for i in range(N):
        V[i, 0] = p[i] * B[i, signal[0]]

    for k in range(1, T):
        for j in range(N):
            values = [V[i, k - 1] * A[i, j] for i in range(N)]
            V[j, k] = (max(values[i] * B[j, signal[k]] for i in range(N)))
            G[j, k] = np.argmax(values)


    prob = max(V[:, k])

    path = np.zeros(T, dtype=int)
    path[T - 1] = np.argmax(V[:, k])
    for t in range(T - 1, 0, -1):
        path[t - 1] = G[path[t], t]

    return path, prob
