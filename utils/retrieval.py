import numpy as np


def euclidean(a, b):
    # compute and return the euclidean distance between two vectors
    return np.linalg.norm(a - b)


def perform_search(query_features, indexed_train, max_results=5):
    retrieved = []
    for idx in range(0, len(indexed_train["features"])):
        distance = euclidean(query_features, indexed_train["features"][idx])
        retrieved.append((distance, idx))
    retrieved = sorted(retrieved)[:max_results]
    return retrieved
