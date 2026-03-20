import numpy as np

def random_forest_vote(predictions):
    """
    Compute the majority vote from multiple tree predictions.
    """
    # Write code here
    predictions = np.asarray(predictions)
    n_trees, n_samples = predictions.shape
    out = []
    for j in range(n_samples):
        pred = predictions[:, j]
        values, count = np.unique(pred, return_counts=True)
        majority = values[np.argmax(count)]
        out.append(majority)
    return out